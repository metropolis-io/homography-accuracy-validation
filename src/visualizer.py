import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_stratified_error(stratified_errors, title="Error by Distance"):
    plt.figure(figsize=(10, 6))
    bins = sorted(stratified_errors.keys())
    means = [np.mean(stratified_errors[b]) for b in bins]
    stds = [np.std(stratified_errors[b]) for b in bins]
    
    plt.bar(bins, means, yerr=stds, width=100, alpha=0.7, color='b', capsize=5)
    plt.xlabel('Distance (units)')
    plt.ylabel('Chamfer Error (pixels in world)')
    plt.title(title)
    plt.grid(True)
    plt.show()

def render_error_heatmap(sat_img, valid_points, errors, max_display_dim=1000):
    """Render an error heatmap onto the satellite image and resize for display."""
    heatmap = sat_img.copy()
    
    if len(errors) > 0:
        min_err = np.min(errors)
        max_err = np.max(errors)
        range_err = max_err - min_err
        norm_errors = (errors - min_err) / (range_err + 1e-6)
        
        for p, err in zip(valid_points, norm_errors):
            x, y = int(round(p[0])), int(round(p[1]))
            if 0 <= x < heatmap.shape[1] and 0 <= y < heatmap.shape[0]:
                color = (0, int(255 * (1 - err)), int(255 * err))
                cv2.circle(heatmap, (x, y), 6, color, -1)
    
    return resize_for_display(heatmap, max_display_dim)

def resize_for_display(img, max_dim=1000):
    h, w = img.shape[:2]
    scale = min(max_dim / w, max_dim / h, 1.0)
    if scale < 1.0:
        return cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def show_comparison_view(cam_img, cam_mask, sat_img, projected_points, max_dim=800):
    """Show camera mask and its projection on satellite image side-by-side."""
    # 1. Overlay mask on camera image
    cam_view = cam_img.copy()
    cam_view[cam_mask > 0] = (0, 255, 0)
    
    # 2. Overlay projected points on satellite image
    sat_view = sat_img.copy()
    for p in projected_points:
        x, y = int(round(p[0])), int(round(p[1]))
        if 0 <= x < sat_view.shape[1] and 0 <= y < sat_view.shape[0]:
            cv2.circle(sat_view, (x, y), 5, (0, 0, 255), -1)
            
    # Resize to match heights for side-by-side
    h_cam, w_cam = cam_view.shape[:2]
    h_sat, w_sat = sat_view.shape[:2]
    
    target_h = max_dim
    scale_cam = target_h / h_cam
    scale_sat = target_h / h_sat
    
    cam_resized = cv2.resize(cam_view, (int(w_cam * scale_cam), target_h))
    sat_resized = cv2.resize(sat_view, (int(w_sat * scale_sat), target_h))
    
    comparison = np.hstack((cam_resized, sat_resized))
    cv2.imshow("1x2 Comparison: Camera Mask (Left) vs Projected (Right)", comparison)

def overlay_features(img, mask, color=(0, 255, 0)):
    """Overlay binary mask features on an image."""
    overlay = img.copy()
    overlay[mask > 0] = color
    return overlay
