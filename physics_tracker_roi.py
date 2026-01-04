import cv2
import numpy as np
import time
from collections import deque

class PhysicsTracker:
    """
    Real-time Kinematics Engine with OpenCV Kalman Filter
    - User selects object in initial ROI
    - Kalman Filter smoothes the position data (2D)
    - Physics engine calculates velocity/acceleration from smoothed data
    """

    def __init__(self, buffer_size=64):
        self.PIXELS_PER_METER = 1000.0
        self.points = deque(maxlen=buffer_size)
        self.times = deque(maxlen=buffer_size)
        self.velocity = 0.0
        self.acceleration = 0.0
        self.prev_pos = None

        # --- KALMAN FILTER SETUP (Minimal 2D) ---
        self.kalman = cv2.KalmanFilter(2, 2, 0)
        self.kalman.transitionMatrix = np.eye(2, dtype=np.float32)      
        self.kalman.measurementMatrix = np.eye(2, dtype=np.float32)
        self.kalman.processNoiseCov = np.eye(2, dtype=np.float32) * 1e-3
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kalman.errorCovPost = np.eye(2, dtype=np.float32)

        self.kalman_initialized = False
        # ----------------------------------------

        # Initial ROI and HSV
        self.roi = None
        self.hsv_lower = None
        self.hsv_upper = None

    def process_frame(self, frame):
        current_time = time.time()

        # --- 1. Initial ROI selection ---
        if self.roi is None:
            self.roi = cv2.selectROI("Select Object", frame, fromCenter=False)
            cv2.destroyWindow("Select Object")

            x, y, w, h = self.roi
            roi_frame = frame[y:y+h, x:x+w]
            hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

            h_mean = np.mean(hsv_roi[:,:,0])
            s_mean = np.mean(hsv_roi[:,:,1])
            v_mean = np.mean(hsv_roi[:,:,2])

            delta_h, delta_s, delta_v = 10, 60, 60
            self.hsv_lower = np.array([max(0, h_mean - delta_h),
                                       max(0, s_mean - delta_s),
                                       max(0, v_mean - delta_v)], dtype=np.uint8)
            self.hsv_upper = np.array([min(179, h_mean + delta_h),
                                       min(255, s_mean + delta_s),
                                       min(255, v_mean + delta_v)], dtype=np.uint8)

        # --- 2. Process full frame ---
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        filtered_center = None 

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((cx, cy), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            if M["m00"] > 0:
                
                raw_cx = int(M["m10"]/M["m00"])
                raw_cy = int(M["m01"]/M["m00"])
                
                # --- KALMAN UPDATE START ---
                measurement = np.array([[np.float32(raw_cx)], 
                                        [np.float32(raw_cy)]])


                if not self.kalman_initialized:
                    self.kalman.statePost = measurement
                    self.kalman_initialized = True


                prediction = self.kalman.predict()
                
                estimated = self.kalman.correct(measurement)
                
                fx = int(estimated[0][0])
                fy = int(estimated[1][0])
                filtered_center = (fx, fy)
                # --- KALMAN UPDATE END ---

                if radius > 5:

                    cv2.circle(frame, (raw_cx, raw_cy), 5, (0, 0, 255), 1)
                    cv2.circle(frame, filtered_center, 6, (0, 255, 0), -1)

                    self.update_physics(filtered_center, current_time)

        # Update trajectory 
        if filtered_center is not None:
            self.points.appendleft(filtered_center)
            self.times.appendleft(current_time)

        self.draw_trajectory(frame)
        self.draw_dashboard(frame)

        return frame

    def update_physics(self, current_pos, current_time):
        if self.prev_pos is not None:
            dx = current_pos[0] - self.prev_pos[0]
            dy = current_pos[1] - self.prev_pos[1]
            dist_pixels = np.sqrt(dx**2 + dy**2)
            dist_meters = dist_pixels / self.PIXELS_PER_METER

            dt = current_time - self.times[0] if len(self.times) > 0 else 0.03
            if dt == 0: dt = 0.01

            raw_velocity = dist_meters / dt
            dv = raw_velocity - self.velocity
            raw_acceleration = dv / dt

            self.velocity = 0.7 * self.velocity + 0.3 * raw_velocity
            self.acceleration = 0.9 * self.acceleration + 0.1 * raw_acceleration

        self.prev_pos = current_pos

    def draw_trajectory(self, frame):
        for i in range(1, len(self.points)):
            if self.points[i-1] is None or self.points[i] is None:
                continue
            thickness = int(np.sqrt(64 / float(i+1)) * 2.5)
            
            cv2.line(frame, self.points[i-1], self.points[i], (0, 255, 0), thickness)

    def draw_dashboard(self, frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (10,10), (320,150), (20,20,20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)


        cv2.putText(frame, "KALMAN SMOOTHED TRACKER", (25,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.putText(frame, f"Velocity:     {self.velocity:.2f} m/s", (25,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)
        cv2.putText(frame, f"Acceleration: {abs(self.acceleration):.2f} m/s^2", (25,110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)
        cv2.putText(frame, "Algorithm: cv2.KalmanFilter", (25,135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150,150,150), 1)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    tracker = PhysicsTracker()
    print("System Ready. Select ROI.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        processed_frame = tracker.process_frame(frame)
        cv2.imshow("Tracker", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()