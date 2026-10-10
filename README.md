# Embodied Perception-Action Closed-Loop for Autonomous Grape Harvesting

This repository is the official companion code and documentation for the paper:
**"Embodied Perception-Action Closed-Loop for Autonomous Grape Harvesting: Visual Servoing and Mobile Manipulation"**

## 📖 Overview

This work presents a four-layer embodied perception-action framework for autonomous grape harvesting, built on ROS 2. It integrates global navigation, visual-servo base alignment, perception-driven manipulation, and behavior-tree scheduling with recovery. The system uses RGB-D imaging and a YOLOv8-seg model to continuously update target localization during motion, closing the perception-action loop.

## ✨ Highlights

- **Mask-Guided RGB-D Perception:** YOLOv8-seg instance masks and robust depth filtering for 3D target localization.
- **Closed-Loop Visual Servoing:** Repeated visual observations guide base yaw correction and odometry-assisted approach.
- **Embodied Manipulation:** Joint-space scanning, temporal TF validation, and behavior-tree scheduling support harvesting and recovery.

## 📂 Repository Structure

- `System_architecture.png` - Four-layer system architecture diagram (Fig. 1).
- `embodied_perception_manipulation_pipeline.png` - Active perception and manipulation pipeline (Fig. 2).
- `mobile_manipulation_tree.py` - Behavior tree implementation for task scheduling and recovery.

## 📊 System-Level Performance

Evaluated across 50 field trials, the proposed embodied framework achieves:

| Metric | Success Rate |
|---|---|
| Navigation & Alignment | 96.0% |
| Harvesting | 98.0% |
| Placement | 92.0% |
| Return to Home | 98.0% |

## 🛠️ Setup & Dependencies

- **OS:** Ubuntu 22.04
- **Middleware:** ROS 2 Humble
- **Deep Learning:** PyTorch, Ultralytics YOLOv8
- **Hardware:** Intel RealSense D435i, xArm6, MR1000 mobile base

*(Detailed setup instructions will be added soon.)*
## 🎥 Demo Video

[![Bilibili](https://img.shields.io/badge/Video-Bilibili-00A1D6?logo=bilibili&logoColor=white)](https://www.bilibili.com/video/BV1VYpG6UEaU/)

Watch the full demonstration of the embodied perception-action closed-loop for autonomous grape harvesting.

