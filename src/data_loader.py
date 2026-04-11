import json
import cv2
import numpy as np

def load_camera_params(json_path):
    with open(json_path, 'r') as f:
        params = json.load(f)
    
    def parse_matrix(key):
        if key not in params:
            return None
        data = params[key]
        return np.array(data['data']).reshape(data['rows'], data['cols'])

    k_matrix = parse_matrix('K-matrix')
    lens_coeffs = parse_matrix('lens-coeffs')
    homography = parse_matrix('homography')
    extrinsics = parse_matrix('extrinsics')
    image_size = parse_matrix('image-size')
    
    return {
        'K': k_matrix,
        'dist': lens_coeffs,
        'H': homography,
        'RT': extrinsics,
        'size': (int(image_size[0, 0]), int(image_size[1, 0])) if image_size is not None else None
    }

def load_and_undistort(img_path, camera_params, do_undistort=True):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    if do_undistort and camera_params['dist'] is not None:
        h, w = img.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_params['K'], camera_params['dist'], (w, h), 1, (w, h)
        )
        img = cv2.undistort(img, camera_params['K'], camera_params['dist'], None, new_camera_matrix)
    
    return img
