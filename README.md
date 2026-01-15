# physics_tracker
Real-time kinematics analysis engine utilizing Computer Vision (OpenCV) and Finite Difference methods to extract physical vectors from video feeds.

![Live Demo](output.gif)

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

### 2. Physics & State Estimation Core: Linear Algebra
* **Linear Kalman Filter (LKF):** Instead of simple smoothing, I implemented a stochastic state estimator.
    * **State Space Model:** The system tracks the state vector $\mathbf{x} = [x, y]^T$ using a transition matrix $\mathbf{F}$ (Identity) and measurement matrix $\mathbf{H}$.
    * **Covariance Tuning:** By manipulating the Process Noise Covariance ($\mathbf{Q}$) and Measurement Noise Covariance ($\mathbf{R}$) matrices, the engine dynamically weights the "trust" between the physical inertia model and the camera sensor data.
* **Finite Difference Method:** Approximates instantaneous derivatives from the *smoothed* state estimate:
    $$v \approx \frac{\Delta x_{kalman}}{\Delta t}, \quad a \approx \frac{\Delta v}{\Delta t}$$

### 3. Scientific Visualization (HUD)
* Renders a real-time **Heads-Up Display** overlaying physical data on the video feed.
* Visualizes trajectory paths with a dynamic "comet-tail" buffer (`deque`) for immediate path analysis.
* **Visual Debugging:** Displays both the Raw Sensor Input (Red) and the Kalman Estimate (Green) to visualize the noise reduction in real-time.

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
    python physics_tracker_roi.py
    ```

---

## 🔭 Future Roadmap: From 2D to 3D Lab Assistant

The current version serves as a proof-of-concept. The next development phase focuses on evolving the engine into a **Stereoscopic Physics Station** to handle complex 3D mechanics.

### 1. Stereo Vision & Depth Perception (The "3D Upgrade")
* **Dual-Camera Implementation:** Implementing **Epipolar Geometry** algorithms to fuse video feeds from two calibrated cameras.
* **3D Triangulation:** By calculating the **disparity map** between left and right channels, the system will extract the $z$-coordinate (Depth), upgrading the tracking state vector from 2D $[x, y]$ to 3D $[x, y, z]$.

### 2. Automated Calibration System
* **Intrinsic & Extrinsic Matrix Calculation:** Developing a routine to automatically compute camera parameters (focal length, optical center) and relative poses using a standard checkerboard pattern (Zhang's Method).
* **Distortion Correction:** Applying radial and tangential distortion coefficients to rectify wide-angle lens curvature for high-precision measurement.

### 3. Data Visualization & Serialization
* **3D Trajectory Plotting:** Integration with `matplotlib` (mplot3d) or `Plotly` to automatically generate interactive **3D Scatter Plots** of the projectile's path post-experiment.
* **Universal Data Export:** Auto-serializing kinematic data (Timestamp, Position $\vec{r}$, Velocity $\vec{v}$, Acceleration $\vec{a}$) into structured `.csv` or `.json` datasets, ready for immediate analysis in MATLAB or Python/Pandas.
---

### Author
**Bowen Zhang**
*University of California, Santa Barbara*
