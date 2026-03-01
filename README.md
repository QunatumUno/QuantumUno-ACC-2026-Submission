# Quantum Uno ACC 2026 Submission

## Welcome to Our Esteemed Panel

We are excited to present our autonomous navigation system developed for the **ACC 2026 Self-Driving Car Competition**. This document provides an overview of our project, detailing the architecture, control strategies, and setup instructions.

---

## Project Overview

**Team Quantum Uno** has designed a self-driving car system that integrates vehicle control and perception. Our primary objectives include:

- Implementing a **Pure Pursuit Controller** for lateral control.
- Using a custom-trained **YOLO5** model for real-time detection of various traffic elements.
- Employing the **Breadth-First Search (BFS)** algorithm for optimal path planning in dynamic environments.

### Control Strategies

1. **Pure Pursuit Controller:** This controller allows the vehicle to follow a designated path smoothly by calculating the required steering angle based on the vehicle's position relative to the path.
2. **PI Controller:** For speed control.

### Perception with YOLO5

Our perception module employs a custom-trained **YOLO5** model, which is integral to the vehicle's ability to navigate safely and efficiently. The model is designed to detect and classify various traffic elements in real-time, enhancing situational awareness. Key features include:

- **Traffic Light Detection:** Identifies the state of traffic lights (red, yellow, green) to make informed stopping decisions.
- **Sign Recognition:** Detects stop signs, yield signs, and roundabout signs, allowing the vehicle to adhere to traffic rules.
- **Obstacle Detection:** Recognizes cones and other obstacles in the environment, facilitating safe navigation.
- **Pedestrian Detection:** Identifies pedestrians in the vicinity to ensure the vehicle can react appropriately to prevent accidents.

The YOLO5 model enhances the vehicle's situational awareness and decision-making processes in real-time, contributing to a safer driving experience.

---

## Path Planning Using BFS

Our path planning module employs the **BFS algorithm**, which systematically explores all possible paths to determine the most efficient route from any starting point to a destination. The BFS approach ensures that the vehicle can navigate effectively while avoiding obstacles and adhering to traffic rules.

### BFS Implementation Steps

- **Adjacency Mapping:** Convert the map's node sequence into an adjacency list representation.
- **Queue-Based Exploration:** Begin from the initial position and explore all neighboring nodes at the current depth before advancing to the next level.
- **Backtracking for Path Reconstruction:** Once the destination is reached, backtrack using parent pointers to reconstruct the optimal path for the vehicle to follow.

---

## System Requirements

### Hardware & Software Specifications

- **Operating System:** Ubuntu 24.04
- **GPU:** NVIDIA (RTX 3060 or higher recommended)
- **Architecture:** x86_64
- **ROS 2 Version:** Humble (implemented via Docker container)
- **Python Versions:**
  - Host system: Python 3.11.4
  - Docker container: Python 3.8.10

---

## File Management in Development Environment

When you place files in the `python_dev` folder, they will be accessible in the following directory once you start the Development Docker Container:
```
admin@username:/workspaces/isaac_ros-dev/python_dev
```

---

## Setup Instructions

1. **Start the development container. Follow official software setup guide:**  
   [ACC 2026 Software Setup Instructions](https://github.com/quanser/student-competition-resources-ros/blob/main/Virtual_ROS_Resources/Virtual_ROS_Software_Setup.md)

2. **Inside the Running Docker Container, run the following commands.**

   Open Terminal 1 and execute the following commands:

   ```bash
   git clone https://github.com/QunatumUno/QuantumUno-ACC-2026-Submission.git
   git clone https://github.com/ultralytics/yolov5


   mv /workspaces/isaac_ros-dev/python_dev/QuantumUno-ACC-2026-Submission/*/ \
      /workspaces/isaac_ros-dev/python_dev/

   mv /workspaces/isaac_ros-dev/python_dev/QuantumUno-ACC-2026-Submission/* \
      /workspaces/isaac_ros-dev/python_dev/yolov5/

   pip install -r requirements.txt

   pip install onnxruntime
   ```


---

## Running the System

To run our navigation algorithm, run the following command from python_dev in the development container:
```
cd yolov5
python3 navigation.py
```

---
## Results

Watch our simulation results on [YouTube](https://youtu.be/LntGkPBh0PA?si=91LTpXD3jFUzR42A).


## Conclusion

We appreciate your time in reviewing our project. If you have any questions or require further assistance, please feel free to reach out through our support channels.

---

## Additional Resources

- [YOLO5 Model for Real-Time Detection](link-to-yolo5)
- [Path Planning Using BFS](https://github.com/Quantum-Uno/2026-ACC-Self-Driving-Car-Competition1/tree/main/path_planning)
- [Pure Pursuit Controller Documentation](https://github.com/Quantum-Uno/2026-ACC-Self-Driving-Car-Competition1/tree/main/purepursuit)

Thank you for your evaluation!
