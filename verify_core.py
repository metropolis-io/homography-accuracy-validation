import numpy as np
import cv2
from src.data_loader import load_camera_params
from src.homography_evaluator import project_points

def test_projection():
    params = load_camera_params('resources/camera-1-parameters.json')
    H = params['H']
    
    # Test point (pixel center)
    test_points = np.array([[640, 360]], dtype=np.float32)
    projected = project_points(test_points, H)
    
    print(f"H matrix shape: {H.shape}")
    print(f"Original point: {test_points}")
    print(f"Projected point: {projected}")
    
    assert projected.shape == (1, 2)
    print("Projection test passed!")

if __name__ == "__main__":
    test_projection()
