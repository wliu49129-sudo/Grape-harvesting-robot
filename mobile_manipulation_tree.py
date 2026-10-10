from typing import Any

import operator
import py_trees

from bt_dsl.node_factory import get_node_factory
from rpp_behaviours import (
    ActionGripper,
    ArmMoveEndInCartesian,
    ArmMoveJoints,
    ArmScanUntilTarget,
    ArmMovetoPoseInCartesian,
    ChassisMove,
    GetGraspPoseByTF,
    NavigationToPose,
    StartRackSearch,
    StartYoloDetection,
    StopYoloDetection,
    WaitForRackSearchTarget,
)


def _sequence(name: str, *children: py_trees.behaviour.Behaviour, memory: bool = True):
    # 顺序节点：子节点按顺序执行，前一个失败/返回 False 时后续不会继续。
    return py_trees.composites.Sequence(name=name, memory=memory, children=list(children))


def _selector(name: str, *children: py_trees.behaviour.Behaviour, memory: bool = True):
    # 选择节点：从左到右尝试子节点，直到有一个成功或返回 Running。
    return py_trees.composites.Selector(name=name, memory=memory, children=list(children))


def _best_effort(name: str, child: py_trees.behaviour.Behaviour):
    """Run cleanup even when the wrapped action fails, then continue the cleanup chain."""
    return py_trees.decorators.FailureIsSuccess(name=name, child=child)


class _RepeatUntilNoTarget(py_trees.decorators.Decorator):
    """Repeat successful pick cycles and stop normally after an empty scan."""

    def __init__(self, name, child, scan_behaviour):
        super().__init__(name=name, child=child)
        self.scan_behaviour = scan_behaviour

    def update(self):
        child_status = self.decorated.status
        if child_status == py_trees.common.Status.FAILURE:
            if self.scan_behaviour.no_target_found:
                self.feedback_message = "scan completed without a target"
                return py_trees.common.Status.SUCCESS
            self.feedback_message = "pick cycle failed"
            return py_trees.common.Status.FAILURE
        if child_status == py_trees.common.Status.SUCCESS:
            self.feedback_message = "pick completed; scanning again"
            return py_trees.common.Status.RUNNING
        return child_status


def _moveit_config(config: dict[str, Any]) -> dict[str, Any]:
    # 统一整理 MoveIt 的执行参数，供各个 ArmMoveJoints / ArmMoveEndInCartesian 节点复用。
    arm = config["arm"]
    return {
        "group_name": arm["group_name"],
        "base_link_name": arm["base_link_name"],
        "joint_names": arm["joint_names"],
        "end_effector_name": arm["end_effector_name"],
        "planner_id": arm.get("planner_id", ""),
        "max_velocity": arm.get("max_velocity", 0.3),
        "max_acceleration": arm.get("max_acceleration", 0.3),
        "execution_timeout": arm.get("execution_timeout", 30.0),
        "cartesian_fraction_threshold": arm.get("cartesian_fraction_threshold", 1.0),
        "cartesian_max_step": arm.get("cartesian_max_step", 0.0025),
    }


def _topics(config: dict[str, Any]) -> dict[str, str]:
    return config.get("topics", {}) or {}


def _gripper(config: dict[str, Any], name: str, goal: str) -> ActionGripper:
    settings = config.get("gripper", {}) or {}
    return ActionGripper(
        name=name,
        goal=goal,
        action_name=settings.get("action_name", "/gripper_controller/gripper_cmd"),
        open_position=settings.get("open_position", 2450),
        close_position=settings.get("close_position", 1660),
    )


def _build_scan_detection(
    config: dict[str, Any],
    moveit_config: dict[str, Any],
) -> tuple[py_trees.behaviour.Behaviour, ArmScanUntilTarget]:
    # 视觉检测 + 目标抓取位姿计算的核心逻辑。
    # 关键顺序：移动到扫描起点 -> 打开 YOLO -> 扫描寻找目标 -> 获取稳定抓取位姿 -> 关闭 YOLO。
    frames = config["frames"]
    grasp = config["grasp"]
    poses = config["poses"]
    scan = config.get("scan", {}) or {}
    yolo_service = _topics(config).get("yolo_enable", "/yolo/enable")
    scan_behaviour = ArmScanUntilTarget(
        name="ScanFromStartToEndUntilTarget",
        moveit_config=moveit_config,
        goal=poses["detect_end"],
        arm_base_frame=frames["base"],
        target_frame=frames["target"],
        target_max_age=scan.get("target_max_age", 0.8),
        start_time_tolerance=scan.get("start_time_tolerance", 0.25),
        max_velocity=scan.get("max_velocity"),
        max_acceleration=scan.get("max_acceleration"),
    )

    normal = _sequence(
        "DetectGrapeWhileArmScanning",
        # 先把机械臂移到开始扫描的位置，保证相机/手臂视野覆盖检测区域。
        ArmMoveJoints(
            name="MoveToScanStartPose",
            moveit_config=moveit_config,
            goal=poses["detect_start"],
        ),
        # 打开目标检测服务，开始识别目标果实。
        StartYoloDetection(
            name="StartYoloDetectionBeforeScan",
            service_name=yolo_service,
        ),
        # 沿着扫描终点运动，并在运动过程中等待目标框出现，直到满足抓取条件。
        scan_behaviour,
        # 将检测到的目标在坐标系之间转换成机械臂末端抓取位姿。
        GetGraspPoseByTF(
            name="GetStableGraspPoseAfterScan",
            arm_base_frame=frames["base"],
            target_frame=frames["target"],
            grasp_frame=frames["grasp"],
            eef_xyz=grasp["eef_xyz"],
            eef_rpy=grasp["eef_rpy"],
            feed_in_xyz=grasp["feed_in_xyz"],
            detection_timeout=grasp.get("detection_timeout", 5.0),
            max_tf_age=grasp.get("max_tf_age", 4.0),
        ),
        # 成功完成检测后关闭 YOLO，避免持续占用视觉资源。
        StopYoloDetection(
            name="StopYoloDetectionAfterScan",
            service_name=yolo_service,
        ),
    )
    cleanup = _sequence(
        "ScanFailureCleanup",
        # 如果扫描检测失败，则尝试关闭 YOLO，并把失败向上传递给上层重试逻辑。
        py_trees.decorators.FailureIsSuccess(
            name="StopYoloDetectionOnScanFailure",
            child=StopYoloDetection(
                name="StopYoloDetectionAfterScanFailure",
                service_name=yolo_service,
            ),
        ),
        py_trees.behaviours.Failure(name="ReturnFailureAfterScanFailure"),
    )
    # 先尝试正常检测流程；如果失败，走清理分支并返回失败。
    return _selector("DetectGrapeByScanning", normal, cleanup), scan_behaviour


def _build_place_action(config: dict[str, Any], moveit_config: dict[str, Any], name: str = "DoPlace"):
    # 放置动作：先回到放置位姿，打开夹爪，再回到安全复位位姿。
    poses = config["poses"]
    normal = _sequence(
        "PlaceNormal",
        # 回到放置点，准备将抓到的物体放下。
        ArmMoveJoints(name="BackToPlacePose", moveit_config=moveit_config, goal=poses["place"]),
        # 放置时打开夹爪，释放物体。
        _gripper(config, "OpenGripperAtPlace", "open"),
        # 完成放置后回到复位位姿，准备下一次抓取或等待下一命令。
        ArmMoveJoints(name="BackToResetPose", moveit_config=moveit_config, goal=poses["reset"]),
    )
    recovery = _sequence(
        "PlaceRecoverAndReturnIdle",
        # 如果放置阶段失败，至少保证机械臂返回到安全位姿，避免卡在危险状态。
        ArmMoveJoints(
            name="BackToResetPoseOnPlaceFailure",
            moveit_config=moveit_config,
            goal=poses["reset"],
        ),
        py_trees.behaviours.Failure(name="ReturnFailureAfterPlaceRecovery"),
    )
    return _selector(name, normal, recovery)


def _build_pick_action(config: dict[str, Any], moveit_config: dict[str, Any]):
    # 一次抓取动作：视觉检测候选 -> 夹持 -> 放置 -> 回到下一轮扫描起点。
    # 扫描到终点仍无目标时返回 FAILURE，由外层 RepeatUntilNoTarget 将其
    # 解释为“当前货架没有更多目标”，而不是执行失败返航分支。
    poses = config["poses"]
    grasp = config["grasp"]
    yolo_service = _topics(config).get("yolo_enable", "/yolo/enable")

    # 视觉候选失败时只执行检测清理并把结果交给外层循环；真正的
    # “送入/夹紧/放置”动作只有在单次扫描和抓取位姿均成功后才会继续。
    scan_detection, scan_behaviour = _build_scan_detection(config, moveit_config)
    candidate_attempt = _sequence(
        "GraspCandidateAttempt",
        # 先确保夹爪打开，避免抓取前状态干扰。
        _gripper(config, "OpenGripper", "open"),
        # 视觉检测并计算稳态抓取位姿；检测失败时进入 cleanup 分支。
        scan_detection,
        # 把末端移动到抓取位姿，准备闭合夹爪。
        ArmMovetoPoseInCartesian(
            name="MoveToGraspPose",
            moveit_config=moveit_config,
            goal_name=config["frames"]["grasp"],
        ),
    )
    candidate_cleanup = _sequence(
        "FalseCandidateCleanup",
        # 视觉候选失败后关闭 YOLO，再把失败结果交给外层循环判断。
        py_trees.decorators.FailureIsSuccess(
            name="StopYoloDetectionAfterCandidateFailure",
            child=StopYoloDetection(
                name="StopYoloDetectionForCandidateRetry",
                service_name=yolo_service,
            ),
        ),
        py_trees.behaviours.Failure(name="CandidateAttemptFailed"),
    )
    candidate_with_cleanup = _selector(
        "CandidateAttemptWithCleanup",
        candidate_attempt,
        candidate_cleanup,
    )
    candidate_retries = int(grasp.get("candidate_max_retries", 0))
    if candidate_retries < 0:
        candidate_retries = 0

    normal = _sequence(
        "PickNormal",
        # 对“误检候选”允许重试一定次数，避免因为一帧误判导致整个抓取失败。
        py_trees.decorators.Retry(
            name="RetryFalseGraspCandidate",
            child=candidate_with_cleanup,
            num_failures=candidate_retries,
        ),
        # 成功拿到候选后，沿末端方向送入抓取位姿。
        ArmMoveEndInCartesian(
            name="FeedIn",
            moveit_config=moveit_config,
            goal=grasp["feed_in_xyz"],
        ),
        # 夹住目标。
        _gripper(config, "CloseGripper", "close"),
        # 抓住后把目标从抓取位姿拉出（或提升）至安全位置。
        ArmMoveEndInCartesian(
            name="FeedOut",
            moveit_config=moveit_config,
            goal=grasp["feed_out_xyz"],
        ),
        # 抓取成功后开始放置动作。
        _build_place_action(config, moveit_config),
        # 放置结束后回到扫描起点，等待下一轮可能的抓取。
        ArmMoveJoints(
            name="ReturnToScanStartPoseAfterPlace",
            moveit_config=moveit_config,
            goal=poses["detect_start"],
        ),
    )
    return normal, scan_behaviour


def _build_reset_action(config: dict[str, Any], moveit_config: dict[str, Any]):
    return ArmMoveJoints(
        name="ResetActionRoot",
        moveit_config=moveit_config,
        goal=config["poses"]["reset"],
    )


def _build_task_loop(config: dict[str, Any], moveit_config: dict[str, Any]):
    # 任务循环支持两种入口：传统的导航取货流程，以及先由货架搜索控制器
    # 把底盘送到货架前、再进入葡萄剪取流程的移动操作流程。
    navigation = config["navigation"]
    topics = _topics(config)
    mission = config.get("mission", {}) or {}
    rack_search = config.get("rack_search", {}) or {}
    pick_action, scan_behaviour = _build_pick_action(config, moveit_config)
    pick_and_place = _RepeatUntilNoTarget(
        name="PickAndPlaceUntilNoTarget",
        child=pick_action,
        scan_behaviour=scan_behaviour,
    )

    if bool(rack_search.get("enabled", False)):
        rack_yolo_service = rack_search.get(
            "yolo_enable", "/rack/yolo/enable"
        )
        rack_status_topic = rack_search.get(
            "status_topic", "/rack_search/status"
        )
        rack_start_service = rack_search.get(
            "start_service", "/rack_search/start"
        )
        return_station = navigation.get(
            "return_station", navigation.get("place_station")
        )
        poses = config["poses"]
        if return_station is None:
            raise ValueError("navigation.return_station or place_station is required")

        normal_nav_and_pick = _sequence(
            "RackToGrapeMission",
            StartRackSearch(
                name="StartRackSearch",
                service_name=rack_start_service,
                wait_timeout=rack_search.get("start_timeout", 10.0),
            ),
            WaitForRackSearchTarget(
                name="WaitForRackSearchTarget",
                status_topic=rack_status_topic,
                wait_timeout=rack_search.get("target_timeout", 180.0),
            ),
            # The rack controller also disables this service on success.  This
            # best-effort call makes the model hand-off explicit and idempotent.
            _best_effort(
                "StopRackYoloBeforeGrapeDetection",
                StopYoloDetection(
                    name="StopRackYolo",
                    service_name=rack_yolo_service,
                ),
            ),
            pick_and_place,
            # No-target scanning ends at detect_end. Return the arm to a safe
            # pose before the chassis starts navigating away from the rack.
            ArmMoveJoints(
                name="ResetArmBeforeReturn",
                moveit_config=moveit_config,
                goal=poses["reset"],
            ),
            NavigationToPose(
                name="NavigateToReturnStation",
                goal=return_station,
            ),
            ArmMoveJoints(
                name="ResetArmAtReturnStation",
                moveit_config=moveit_config,
                goal=poses["reset"],
            ),
        )
        recovery = _sequence(
            "RackToGrapeMissionRecovery",
            _best_effort(
                "StopGrapeYoloOnMissionFailure",
                StopYoloDetection(
                    name="StopGrapeYoloAfterMissionFailure",
                    service_name=topics.get("yolo_enable", "/yolo/enable"),
                ),
            ),
            _best_effort(
                "ResetArmOnMissionFailure",
                ArmMoveJoints(
                    name="ResetArmOnMissionFailure",
                    moveit_config=moveit_config,
                    goal=poses["reset"],
                ),
            ),
            _best_effort(
                "ReturnToStationOnMissionFailure",
                NavigationToPose(
                    name="ReturnToStationOnMissionFailure",
                    goal=return_station,
                ),
            ),
            _best_effort(
                "ResetArmAtStationAfterMissionFailure",
                ArmMoveJoints(
                    name="ResetArmAtStationAfterMissionFailure",
                    moveit_config=moveit_config,
                    goal=poses["reset"],
                ),
            ),
            py_trees.behaviours.Failure(name="ReturnFailureAfterMissionRecovery"),
        )
        nav_and_pick = _selector(
            "RackToGrapeMissionWithRecovery",
            normal_nav_and_pick,
            recovery,
        )
    else:
        nav_and_pick = _sequence(
            "NavAndPickActionRoot",
            # 传统模式：先导航到取货站点，再执行抓放循环。
            NavigationToPose(name="NavigateToPickStation", goal=navigation["pick_station"]),
            ChassisMove(
                name="ChassisFeedIn",
                direction="forward",
                distance=navigation.get("chassis_feed_depth", 0),
                speed=navigation.get("chassis_speed", 0.1),
                cmd_topic=topics.get("cmd_vel", "/cmd_vel"),
            ),
            pick_and_place,
            ChassisMove(
                name="ChassisRetreat",
                direction="backward",
                distance=navigation.get("chassis_feed_depth", 0),
                speed=navigation.get("chassis_speed", 0.1),
                cmd_topic=topics.get("cmd_vel", "/cmd_vel"),
            ),
        )

    task_chain = _sequence("TaskChainActionRoot", nav_and_pick)
    # 对整个任务链再套一层 Repeat，实现多轮任务循环。
    task_repeat_num = mission.get("task_repeat_num", mission.get("repeat_num", -1))
    return py_trees.decorators.Repeat(
        name="TaskLoopActionRoot",
        child=task_chain,
        num_success=task_repeat_num,
    )


def _event_node(name: str, topic_name: str, variable_name: str):
    # 将外部事件（如 start/cancel 话题）写入黑板，供行为树判定是否启动或取消任务。
    factory = get_node_factory("EventToBlackboard")
    return factory(
        name,
        {"topic_name": topic_name, "variable_name": variable_name},
        lambda child: child,
    )


def build_mobile_manipulation_tree(config: dict[str, Any]):
    """Build the mobile manipulation tree from task parameters."""
    # 顶层入口：并行监听事件，并将任务派发到“工作状态”或“空闲状态”。
    moveit_config = _moveit_config(config)
    topics = _topics(config)
    task_loop = _build_task_loop(config, moveit_config)
    reset = _build_reset_action(config, moveit_config)

    # 取消命令：如果黑板中的 cancel 事件为 True，则优先中断当前任务。
    cancel_requested = py_trees.behaviours.CheckBlackboardVariableValue(
        name="CancelRequested",
        check=py_trees.common.ComparisonExpression(
            variable="event_cancel_button",
            value=True,
            operator=operator.eq,
        ),
    )
    # 启动命令：需要等到 start 事件为 True 才开始工作。
    start_requested = py_trees.behaviours.CheckBlackboardVariableValue(
        name="StartRequested",
        check=py_trees.common.ComparisonExpression(
            variable="event_start_button",
            value=True,
            operator=operator.eq,
        ),
    )
    work = _sequence(
        "Work",
        # 第一步必须确认已收到 start 事件，否则不允许执行任务。
        start_requested,
        # start 后若已触发 cancel，则停止工作并跳过任务循环；否则执行主任务。
        _selector("WorkOrBeCancelled", cancel_requested, task_loop, memory=False),
        # 任务流程完成后，回到安全复位姿态，等待下一次开始命令。
        reset,
    )
    # 顶层 dispatch：如果 work 成功/运行，则持续执行任务；否则保持 Idle。
    dispatch = _selector("TaskDispatch", work, py_trees.behaviours.Running(name="Idle"), memory=False)
    return py_trees.composites.Parallel(
        name="MobileManipulationMission",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False),
        children=[
            # 监听 start 事件并写入 blackboard。
            _event_node(
                "StartEvent",
                topics.get("start", "/bt/dashboard/start"),
                "event_start_button",
            ),
            # 监听 cancel 事件并写入 blackboard。
            _event_node(
                "CancelEvent",
                topics.get("cancel", "/bt/dashboard/cancel"),
                "event_cancel_button",
            ),
            # 真正的任务分发节点。
            dispatch,
        ],
    )
