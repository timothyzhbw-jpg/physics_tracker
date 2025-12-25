import unittest
import numpy as np

class TestPhysicsEngine(unittest.TestCase):
    def test_velocity_calculation(self):
        dx = 10.0
        dt = 2.0
        velocity = dx / dt
        self.assertEqual(velocity, 5.0)
        print("Velocity calculation test passed!")

if __name__ == '__main__':
    unittest.main()