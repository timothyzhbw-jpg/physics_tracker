import cv2
import numpy as np
from typing import Tuple, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KalmanFilterTracker:
    """
    Kalman Filter-based tracker for physics objects using OpenCV's cv2.KalmanFilter.
    Tracks position and velocity of objects in 2D space.
    """
    
    def __init__(self, 
                 process_noise: float = 0.03,
                 measurement_noise: float = 0.5,
                 initial_error: float = 1.0):
        """
        Initialize the Kalman Filter tracker.
        
        Args:
            process_noise: Process noise covariance (Q matrix)
            measurement_noise: Measurement noise covariance (R matrix)
            initial_error: Initial estimation error covariance
        """
        # State vector: [x, y, vx, vy] (position and velocity)
        self.kf = cv2.KalmanFilter(4, 2)
        
        # Transition matrix (state transition)
        # Models constant velocity: x_new = x + vx, y_new = y + vy
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix (maps state to measurements)
        # We only measure position, not velocity
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance matrix (Q)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        
        # Measurement noise covariance matrix (R)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        
        # Initial state error covariance matrix (P)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * initial_error
        
        # Initial state (will be set on first measurement)
        self.kf.statePost = np.array([[0], [0], [0], [0]], dtype=np.float32)
        
        self.is_initialized = False
        self.measurements = []
        
        logger.info("Kalman Filter Tracker initialized")
    
    def initialize(self, x: float, y: float) -> None:
        """
        Initialize the Kalman Filter with first measurement.
        
        Args:
            x: Initial x position
            y: Initial y position
        """
        self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
        self.is_initialized = True
        self.measurements.append((x, y))
        logger.info(f"Kalman Filter initialized at position ({x}, {y})")
    
    def predict(self) -> Tuple[float, float]:
        """
        Predict next state using Kalman Filter prediction step.
        
        Returns:
            Tuple of (predicted_x, predicted_y)
        """
        if not self.is_initialized:
            logger.warning("Kalman Filter not initialized. Call initialize() first.")
            return (0, 0)
        
        prediction = self.kf.predict()
        predicted_x = float(prediction[0][0])
        predicted_y = float(prediction[1][0])
        
        return (predicted_x, predicted_y)
    
    def update(self, x: float, y: float) -> Tuple[float, float]:
        """
        Update Kalman Filter with measurement and return corrected state.
        
        Args:
            x: Measured x position
            y: Measured y position
            
        Returns:
            Tuple of (corrected_x, corrected_y)
        """
        if not self.is_initialized:
            self.initialize(x, y)
        
        # Create measurement vector
        measurement = np.array([[x], [y]], dtype=np.float32)
        
        # Correct (update) the filter with measurement
        self.kf.correct(measurement)
        
        # Get corrected state
        state = self.kf.statePost
        corrected_x = float(state[0][0])
        corrected_y = float(state[1][0])
        
        self.measurements.append((corrected_x, corrected_y))
        
        return (corrected_x, corrected_y)
    
    def get_velocity(self) -> Tuple[float, float]:
        """
        Get estimated velocity from state vector.
        
        Returns:
            Tuple of (vx, vy)
        """
        if not self.is_initialized:
            return (0, 0)
        
        state = self.kf.statePost
        vx = float(state[2][0])
        vy = float(state[3][0])
        
        return (vx, vy)
    
    def get_state(self) -> np.ndarray:
        """
        Get full state vector [x, y, vx, vy].
        
        Returns:
            State vector as numpy array
        """
        return self.kf.statePost.copy()
    
    def reset(self) -> None:
        """Reset the Kalman Filter to uninitialized state."""
        self.is_initialized = False
        self.measurements.clear()
        self.kf.statePost = np.array([[0], [0], [0], [0]], dtype=np.float32)
        logger.info("Kalman Filter reset")


class PhysicsTrackerROI:
    """
    Physics Tracker for Region of Interest (ROI) using Kalman Filter.
    Tracks objects within a defined ROI and estimates their trajectory.
    """
    
    def __init__(self, 
                 roi_bounds: Tuple[int, int, int, int],
                 process_noise: float = 0.03,
                 measurement_noise: float = 0.5):
        """
        Initialize the Physics Tracker for ROI.
        
        Args:
            roi_bounds: Tuple of (x1, y1, x2, y2) defining the ROI
            process_noise: Process noise for Kalman Filter
            measurement_noise: Measurement noise for Kalman Filter
        """
        self.roi_bounds = roi_bounds  # (x1, y1, x2, y2)
        self.trackers: dict = {}  # Dictionary of KalmanFilterTracker objects by ID
        self.next_id = 0
        
        # Parameters
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        logger.info(f"Physics Tracker ROI initialized with bounds: {roi_bounds}")
    
    def is_in_roi(self, x: float, y: float) -> bool:
        """
        Check if a point is within the ROI.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            True if point is within ROI, False otherwise
        """
        x1, y1, x2, y2 = self.roi_bounds
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def add_tracker(self, object_id: Optional[int] = None) -> int:
        """
        Add a new object tracker.
        
        Args:
            object_id: Optional custom ID for the object. If None, auto-generates.
            
        Returns:
            ID of the created tracker
        """
        if object_id is None:
            object_id = self.next_id
            self.next_id += 1
        
        self.trackers[object_id] = KalmanFilterTracker(
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise
        )
        
        logger.info(f"Added tracker with ID: {object_id}")
        return object_id
    
    def remove_tracker(self, object_id: int) -> bool:
        """
        Remove an object tracker.
        
        Args:
            object_id: ID of the tracker to remove
            
        Returns:
            True if tracker was removed, False if not found
        """
        if object_id in self.trackers:
            del self.trackers[object_id]
            logger.info(f"Removed tracker with ID: {object_id}")
            return True
        return False
    
    def update_tracker(self, object_id: int, x: float, y: float) -> Tuple[float, float]:
        """
        Update a tracker with new measurement.
        
        Args:
            object_id: ID of the tracker to update
            x: Measured x position
            y: Measured y position
            
        Returns:
            Tuple of (corrected_x, corrected_y)
        """
        if object_id not in self.trackers:
            logger.warning(f"Tracker with ID {object_id} not found")
            return (x, y)
        
        return self.trackers[object_id].update(x, y)
    
    def predict_tracker(self, object_id: int) -> Tuple[float, float]:
        """
        Get prediction for a tracker.
        
        Args:
            object_id: ID of the tracker
            
        Returns:
            Tuple of (predicted_x, predicted_y)
        """
        if object_id not in self.trackers:
            logger.warning(f"Tracker with ID {object_id} not found")
            return (0, 0)
        
        return self.trackers[object_id].predict()
    
    def get_tracker_velocity(self, object_id: int) -> Tuple[float, float]:
        """
        Get estimated velocity of a tracker.
        
        Args:
            object_id: ID of the tracker
            
        Returns:
            Tuple of (vx, vy)
        """
        if object_id not in self.trackers:
            logger.warning(f"Tracker with ID {object_id} not found")
            return (0, 0)
        
        return self.trackers[object_id].get_velocity()
    
    def get_all_trackers(self) -> dict:
        """
        Get all active trackers.
        
        Returns:
            Dictionary of active trackers with their IDs
        """
        return self.trackers.copy()
    
    def clear_all_trackers(self) -> None:
        """Clear all trackers."""
        self.trackers.clear()
        logger.info("All trackers cleared")


def draw_roi(frame: np.ndarray, roi_bounds: Tuple[int, int, int, int], 
             color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    """
    Draw ROI rectangle on frame.
    
    Args:
        frame: Input frame
        roi_bounds: Tuple of (x1, y1, x2, y2)
        color: Color in BGR format
        thickness: Line thickness
        
    Returns:
        Frame with ROI drawn
    """
    x1, y1, x2, y2 = roi_bounds
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame


def draw_tracker_trajectory(frame: np.ndarray, 
                           tracker: KalmanFilterTracker,
                           color: Tuple[int, int, int] = (0, 255, 255),
                           radius: int = 3,
                           max_history: int = 30) -> np.ndarray:
    """
    Draw tracker trajectory on frame.
    
    Args:
        frame: Input frame
        tracker: KalmanFilterTracker object
        color: Color in BGR format
        radius: Circle radius for points
        max_history: Maximum number of history points to draw
        
    Returns:
        Frame with trajectory drawn
    """
    measurements = tracker.measurements[-max_history:]
    
    if len(measurements) > 1:
        for i in range(len(measurements) - 1):
            x1, y1 = int(measurements[i][0]), int(measurements[i][1])
            x2, y2 = int(measurements[i + 1][0]), int(measurements[i + 1][1])
            cv2.line(frame, (x1, y1), (x2, y2), color, 1)
            cv2.circle(frame, (x1, y1), radius, color, -1)
    
    if measurements:
        x, y = int(measurements[-1][0]), int(measurements[-1][1])
        cv2.circle(frame, (x, y), radius, color, -1)
    
    return frame


if __name__ == "__main__":
    # Example usage
    roi_bounds = (100, 100, 500, 400)
    tracker_roi = PhysicsTrackerROI(roi_bounds)
    
    # Add a tracker
    tracker_id = tracker_roi.add_tracker()
    
    # Simulate measurements
    measurements = [(150, 150), (155, 152), (160, 155), (165, 157), (170, 160)]
    
    for x, y in measurements:
        corrected_x, corrected_y = tracker_roi.update_tracker(tracker_id, float(x), float(y))
        predicted_x, predicted_y = tracker_roi.predict_tracker(tracker_id)
        vx, vy = tracker_roi.get_tracker_velocity(tracker_id)
        
        print(f"Measurement: ({x}, {y})")
        print(f"Corrected: ({corrected_x:.2f}, {corrected_y:.2f})")
        print(f"Predicted: ({predicted_x:.2f}, {predicted_y:.2f})")
        print(f"Velocity: ({vx:.4f}, {vy:.4f})")
        print("-" * 50)
