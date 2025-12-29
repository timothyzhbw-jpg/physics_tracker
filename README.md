# physics_tracker
Real-time kinematics analysis engine utilizing Computer Vision (OpenCV) and Finite Difference methods to extract physical vectors from video feeds.

# Kinematics-CV: Real-Time Physics Analysis Engine

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![Physics](https://img.shields.io/badge/Domain-Applied%20Physics-orange)

> **A computational tool designed to bridge the gap between theoretical mechanics and real-world experimental data acquisition.**

## 🚀 Overview
**Kinematics-CV** is a high-performance computer vision engine built to track projectile motion in real-time. Unlike standard tracking scripts, this system implements a **physics-first architecture**, converting raw pixel data into meaningful kinematic vectors (Velocity $\vec{v}$ and Acceleration $\vec{a}$) with industrial-grade signal processing.

I developed this tool to address a common challenge in undergraduate physics labs: the disconnect between idealized textbook formulas and noisy real-world measurements.

---

## ✨ Key Features (Engineering & Math)

### 1. Robust Computer Vision Pipeline
* **HSV Color Space Segmentation:** Utilizes Hue-Saturation-Value masking instead of RGB to ensure robustness against varying lighting conditions in the lab.
* **Morphological Noise Reduction:** Implements erosion and dilation algorithms to filter out high-frequency visual noise before processing.
* **Centroid Calculation:** Computes Image Moments ($M_{00}, M_{10}, M_{01}$) to determine the precise center of mass of the tracked object.

### 2. Physics & Signal Processing Core
* **Finite Difference Method:** Approximates instantaneous derivatives using discrete time-steps:
    $$v \approx \frac{\Delta x}{\Delta t}, \quad a \approx \frac{\Delta v}{\Delta t}$$
* **Low-Pass Filter (Signal Smoothing):** Raw sensor data is inherently noisy. I implemented an **Exponential Weighted Moving Average (EWMA)** filter to stabilize acceleration readouts without introducing significant latency.
    * *Algorithm:* $v_{smooth} = \alpha \cdot v_{new} + (1 - \alpha) \cdot v_{prev}$

### 3. Scientific Visualization (HUD)
* Renders a real-time **Heads-Up Display** overlaying physical data on the video feed.
* Visualizes trajectory paths with a dynamic "comet-tail" buffer for immediate path analysis.

---

## 🛠️ System Architecture

The project follows a modular **Object-Oriented Programming (OOP)** structure to ensure scalability for future lab instrumentation integration.

* **Encapsulation:** The `PhysicsTracker` class manages state variables (`velocity`, `acceleration`, `buffers`) locally, preventing global namespace pollution.
* **Time Complexity Optimization:** Utilizes `collections.deque` for the trajectory buffer, achieving **O(1)** time complexity for append and pop operations, which is critical for maintaining high frame rates during real-time analysis.
* **Scalability:** The code is designed to support ArUco marker calibration and multi-object tracking in future iterations.

---

## ⚡ Quick Start

### Prerequisites
* Python 3.x
* Webcam (Built-in or USB)

### Dependencies
* `opencv-python`
* `numpy`

### Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/timothyzhbw-jpg/physics_tracker.git
    cd physics_tracker
    ```
2.  Install required packages:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the engine:
    ```bash
    python physics_tracker.py
    ```

---

## 🔭 Future Roadmap

This project is currently a prototype for a larger **"Automated Lab Assistant"** ecosystem. Future goals include:
* **ArUco Marker Integration:** Auto-calibration of the `PIXELS_PER_METER` constant for precise metric measurements.
* **Kalman Filter:** Implementing predictive tracking to handle object occlusion.
* **Data Export:** Auto-save kinematic data to `.csv` for analysis in Python/Matlab.

---

### Author
**Bowen Zhang**
*University of California, Santa Barbara*
