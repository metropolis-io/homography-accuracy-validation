import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

def project_points(points, homography):
    """Project camera points into world space (satellite) coordinates."""
    if len(points) == 0:
        return np.empty((0, 2))
    
    # Reshape for cv2.perspectiveTransform
    points_reshaped = points.reshape(-1, 1, 2).astype(np.float32)
    projected = cv2.perspectiveTransform(points_reshaped, homography)
    return projected.reshape(-1, 2)

def calculate_chamfer_error(projected_points, sat_mask):
    """Calculate the error for each projected point based on distance to nearest ground truth point in sat_mask."""
    # Distance transform of satellite mask (distance to nearest '1' pixel)
    # The sat_mask should be 255 for feature, 0 otherwise
    dt = distance_transform_edt(1 - (sat_mask / 255.0))
    
    errors = []
    valid_points = []
    
    for p in projected_points:
        x, y = int(round(p[0])), int(round(p[1]))
        if 0 <= x < dt.shape[1] and 0 <= y < dt.shape[0]:
            errors.append(dt[y, x])
            valid_points.append(p)
        else:
            # Point projected outside satellite image
            pass
            
    return np.array(errors), np.array(valid_points)

def stratify_by_distance(points_world, errors, extrinsics, bin_size=10, max_dist=10000):
    """
    Slice errors by distance in front of the camera.
    
    Assumptions:
    - Extrinsics define the camera position and orientation in world coordinates.
    - Camera center in world coordinates is simply T (4th column of extrinsics).
    - Camera forward direction is the 3rd column of the rotation matrix R.
    """
    if extrinsics is None:
        return {}
    
    R = extrinsics[:, :3]
    T = extrinsics[:, 3].flatten()
    
    # Camera center in world
    cam_center_world = T[:2] # Using 2D for world points projection
    
    # Forward vector in world coordinates (3rd column of R)
    # For a top-down view or 2D projection, we focus on the direction in the XY plane
    forward_vector = R[:2, 2] 
    if np.linalg.norm(forward_vector) > 0:
        forward_vector /= np.linalg.norm(forward_vector)
    
    # Vector from camera to world points
    vectors = points_world - cam_center_world
    
    # Project vectors onto the forward direction to get 'depth' or 'distance in front'
    dists = np.dot(vectors, forward_vector)
    
    bins = np.arange(0, max_dist + bin_size, bin_size)
    stratified_errors = {}
    
    for i in range(len(bins) - 1):
        idx = (dists >= bins[i]) & (dists < bins[i+1])
        if np.any(idx):
            stratified_errors[bins[i]] = errors[idx]
            
    return stratified_errors

def calculate_parallel_divergence(line1_points, line2_points, homography):
    """Calculate angular error and width variance for parallel lines."""
    # Project both lines
    p1 = project_points(line1_points, homography)
    p2 = project_points(line2_points, homography)
    
    # Fit lines to projected points
    vx1, vy1, x1, y1 = cv2.fitLine(p1, cv2.DIST_L2, 0, 0.01, 0.01)
    vx2, vy2, x2, y2 = cv2.fitLine(p2, cv2.DIST_L2, 0, 0.01, 0.01)
    
    # Angle between vectors
    angle = np.abs(np.arctan2(vy1, vx1) - np.arctan2(vy2, vx2))
    angle = np.degrees(angle) % 180
    if angle > 90: angle = 180 - angle
    
    # Width variance
    # For each point on line 1, find distance to line 2
    # Line 2 eq: (y - y2) * vx2 - (x - x2) * vy2 = 0
    widths = []
    norm = np.sqrt(vx2**2 + vy2**2)
    for p in p1:
        w = np.abs((p[1] - y2) * vx2 - (p[0] - x2) * vy2) / norm
        widths.append(w)
        
    return angle, np.var(widths)
