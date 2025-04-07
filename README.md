
# 🤖 EKF-SLAM Implementation – IFRoS Master's Project

This repository contains an Extended Kalman Filter (EKF)-based Simultaneous Localization and Mapping (SLAM) implementation developed as part of the **IFRoS Master’s program**.

---

## 📘 Project Overview

**Simultaneous Localization and Mapping (SLAM)** is a fundamental task in mobile robotics, where a robot builds a map of an unknown environment while simultaneously estimating its own position within it. This project focuses on the **EKF-SLAM** method using simulated differential drive robots and a variety of sensor and motion models.

---

## 📂 Key Components

- **Differential Drive Simulation**: Simulated robot with control over input displacement and constant velocity
- **EKF Filter Core**: Implements prediction and correction steps using probabilistic models
- **Feature-Based SLAM**: Uses 2D Cartesian features and both Cartesian/Polar observation models
- **Modular Architecture**: Classes for robot motion, map features, pose, Gaussian filtering, and SLAM logic
- **Visualization**: Figures and simulation results for trajectories, maps, and uncertainty ellipses

---

## 🧠 Algorithms & Techniques

- **Extended Kalman Filter (EKF)**
- **Differential Drive Kinematics**
- **2D Feature Tracking and Association**
- **Uncertainty Propagation (Covariance Ellipses)**
- **Localization vs Full SLAM** distinction
- **Feature Management and Initialization**

---

## 🗂️ Directory Structure

- `EKF.py`, `GaussianFilter.py`, `KF.py`: Core filtering logic
- `Pose.py`, `Feature.py`, `MapFeature.py`: State representation
- `DifferentialDriveSimulatedRobot.py`: Robot simulation
- `FEKFSLAM_*.py`, `MBL_*.py`: EKF SLAM and motion models
- `Figures/`: Visual outputs from simulations
- `img/`, `pyreverse_output/`: Diagrams and class structure visuals

---

## ▶️ How to Run

Open any of the EKF-SLAM example scripts in:

```bash
EKF-SLAM-IFRoS-Master/
```

Run the simulations using:

```bash
python FEKFSLAM_3DOFDD_InputVelocityMM_2DCartesianFeatureOM.py
```

Make sure required packages like `numpy`, `matplotlib`, etc., are installed.

---

> If this helped your learning or project, consider sharing or citing the repository!
