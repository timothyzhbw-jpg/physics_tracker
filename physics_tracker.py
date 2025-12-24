import cv2
import numpy as np
import time
from collections import deque

class PhysicsTracker:
    """
    Real-time Kinematics Analysis Engine.
    
    This module integrates Computer Vision (OpenCV) with classical mechanics 
    to track object trajectories and calculate instantaneous physical 
    properties (Velocity, Acceleration) in real-time.
    
    Features:
    - HSV Color Space Segmentation
    - Finite Difference Method for Kinematics
    - Low-Pass Signal Smoothing (Noise Reduction)
    - Real-time Data Visualization (HUD)
    
    Author: Bowen
    """

    def __init__(self, buffer_size=64):
        # --- 1. CONFIGURATION ---
        # Define HSV color range for the tracking object (Default: Red)
        self.color_lower1 = np.array([0, 120, 70])
        self.color_upper1 = np.array([10, 255, 255])
        self.color_lower2 = np.array([170, 120, 70])
        self.color_upper2 = np.array([180, 255, 255])

        # Physics Calibration
        # Currently assuming 100 pixels = 0.1 meters. 
        # Scalable for real-world lab setups.
        self.PIXELS_PER_METER = 1000.0 
        
        # Data Buffers & State Variables
        self.points = deque(maxlen=buffer_size) 
        self.times = deque(maxlen=buffer_size)
        self.velocity = 0.0      # m/s
        self.acceleration = 0.0  # m/s^2
        self.prev_time = time.time()
        self.prev_pos = None

    def process_frame(self, frame):
        """
        Main Processing Pipeline:
        Frame -> Denoise -> Masking -> Contour -> Physics Calc -> Render
        """
        current_time = time.time()
        
        # A. Preprocessing (Gaussian Blur)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # B. Color Segmentation
        mask1 = cv2.inRange(hsv, self.color_lower1, self.color_upper1)
        mask2 = cv2.inRange(hsv, self.color_lower2, self.color_upper2)
        mask = mask1 | mask2
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # C. Object Detection
        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            
            if M["m00"] > 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                if radius > 10:
                    # Visual Markers
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)
                    
                    # --- D. PHYSICS UPDATE ---
                    self.update_physics(center, current_time)

        # Update Trajectory
        self.points.appendleft(center)
        self.times.appendleft(current_time)
        
        # E. Visualization
        self.draw_trajectory(frame)
        self.draw_dashboard(frame)

        self.prev_time = current_time
        return frame

    def update_physics(self, current_pos, current_time):
        """
        Calculates Velocity & Acceleration with Signal Smoothing.
        """
        if self.prev_pos is not None:
            # Displacement
            dx = current_pos[0] - self.prev_pos[0]
            dy = current_pos[1] - self.prev_pos[1]
            dist_pixels = np.sqrt(dx**2 + dy**2)
            dist_meters = dist_pixels / self.PIXELS_PER_METER
            
            # Time Delta
            dt = current_time - self.times[0] if len(self.times) > 0 else 0.03
            if dt == 0: dt = 0.01

            # Instantaneous Calculations
            raw_velocity = dist_meters / dt
            dv = raw_velocity - self.velocity
            raw_acceleration = dv / dt
            
            # Low-Pass Filter (Smoothing Algorithm)
            # Essential for stable readings in noisy sensor environments
            self.velocity = 0.7 * self.velocity + 0.3 * raw_velocity
            self.acceleration = 0.9 * self.acceleration + 0.1 * raw_acceleration

        self.prev_pos = current_pos

    def draw_trajectory(self, frame):
        """Renders the motion path."""
        for i in range(1, len(self.points)):
            if self.points[i - 1] is None or self.points[i] is None:
                continue
            thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
            cv2.line(frame, self.points[i - 1], self.points[i], (0, 0, 255), thickness)

    def draw_dashboard(self, frame):
        """
        Renders the Scientific Data Interface.
        """
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 150), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Generic Professional Header
        cv2.putText(frame, "KINEMATICS ENGINE", (25, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Velocity:     {self.velocity:.2f} m/s", (25, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        
        cv2.putText(frame, f"Acceleration: {abs(self.acceleration):.2f} m/s^2", (25, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        cv2.putText(frame, "Ref. Frame: Inertial (Lab)", (25, 135), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    tracker = PhysicsTracker()

    # Changed print message to be generic and professional
    print("--- Physics Kinematics Engine Initialized ---")
    print("System Ready. Please present a RED object.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        processed_frame = tracker.process_frame(frame)
        
        # Generic Window Title
        cv2.imshow("Real-time Physics Analysis", processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()