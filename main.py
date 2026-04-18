import argparse
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import logging
from src.data_loader import load_camera_params, load_and_undistort
from src.feature_extractor import PointPicker, densify_points, create_binary_mask
from src.homography_evaluator import project_points, calculate_chamfer_error, stratify_by_distance, calculate_parallel_divergence
from src.visualizer import plot_stratified_error, render_error_heatmap, overlay_features, show_comparison_view

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Homography Accuracy Validation")
    parser.add_argument("--camera-json", type=str, required=True, help="Path to camera-x-parameters.json")
    parser.add_argument("--camera-img", type=str, required=True, help="Path to camera image")
    parser.add_argument("--sat-img", type=str, required=True, help="Path to satellite image")
    parser.add_argument("--mode", choices=["reference", "self"], default="reference", help="Validation mode")
    parser.add_argument("--undistort", action="store_true", help="Undistort image using lens coeffs")
    parser.add_argument("--interval", type=int, default=5, help="Interpolation interval in pixels")
    
    args = parser.parse_args()
    
    # 1. Load Data
    logger.info(f"Loading camera parameters from {args.camera_json}")
    camera_params = load_camera_params(args.camera_json)
    
    logger.info(f"Loading and undistorting camera image: {args.camera_img}")
    cam_img = load_and_undistort(args.camera_img, camera_params, do_undistort=args.undistort)
    
    logger.info(f"Loading satellite image: {args.sat_img}")
    sat_img = cv2.imread(args.sat_img)
    
    if cam_img is None or sat_img is None:
        logger.error("Could not load images.")
        return

    if args.mode == "reference":
        # 2. Extract Features (Camera)
        picker = PointPicker("Pick Camera Features")
        cam_points = picker.pick_points(cam_img)
        logger.info(f"Number of points picked on camera image: {len(cam_points)}")
        
        if len(cam_points) == 0:
            logger.warning("No camera points picked. Exiting.")
            return
            
        dense_cam_points = densify_points(cam_points, interval=args.interval)
        logger.info(f"Number of points after densification: {len(dense_cam_points)}")
        cam_mask = create_binary_mask(cam_img.shape, dense_cam_points)
        
        # 3. Project points for visualization
        projected_points = project_points(dense_cam_points, camera_params['H'])
        logger.info(f"Number of projected points: {len(projected_points)}")
        
        # Show side-by-side comparison
        show_comparison_view(cam_img, cam_mask, sat_img, projected_points)

        # 4. Extract Features (Satellite)
        picker_sat = PointPicker("Pick Satellite Features")
        sat_points = picker_sat.pick_points(sat_img)
        logger.info(f"Number of points picked on satellite image: {len(sat_points)}")
        
        if len(sat_points) == 0:
            logger.warning("No satellite points picked. Exiting.")
            return
            
        dense_sat_points = densify_points(sat_points, interval=args.interval)
        logger.info(f"Number of satellite points after densification: {len(dense_sat_points)}")
        sat_mask = create_binary_mask(sat_img.shape, dense_sat_points)
        
        # 5. Evaluate (Chamfer)
        errors, valid_proj_points = calculate_chamfer_error(projected_points, sat_mask)
        logger.info(f"Number of valid projected points (within satellite bounds): {len(valid_proj_points)}")
        
        # 6. Stratify Error
        RT = camera_params['RT']
        stratified = stratify_by_distance(valid_proj_points, errors, RT)
        logger.info(f"Number of distance bins for stratification: {len(stratified)}")
        
        # 7. Visualize
        if stratified:
            plot_stratified_error(stratified, title=f"Accuracy Stratified by Distance ({args.mode})")
        
        heatmap = render_error_heatmap(sat_img, valid_proj_points, errors)
        cv2.imshow("Error Heatmap (Satellite View)", heatmap)
        cv2.waitKey(0)
        
    elif args.mode == "self":
        # self-consistency mode: Parallel lines
        logger.info("Self-consistency mode: Pick two sets of points for parallel lines.")
        picker_l1 = PointPicker("Parallel Line 1")
        l1_points = picker_l1.pick_points(cam_img)
        logger.warning(f"Points for Line 1: {len(l1_points)}")
        
        picker_l2 = PointPicker("Parallel Line 2")
        l2_points = picker_l2.pick_points(cam_img)
        logger.warning(f"Points for Line 2: {len(l2_points)}")
        
        if len(l1_points) > 1 and len(l2_points) > 1:
            # Create separate densified points to avoid joining them
            dense_l1 = densify_points(l1_points, interval=args.interval)
            dense_l2 = densify_points(l2_points, interval=args.interval)

            logger.warning(f"Densified Line 1 points: {len(dense_l1)}, Line 2 points: {len(dense_l2)}")
            
            # Combine for combined mask and visualization
            dense_line_points = np.vstack((dense_l1, dense_l2))
            line_mask = create_binary_mask(cam_img.shape, dense_line_points)
            projected_lines = project_points(dense_line_points, camera_params['H'])
            
            # Show side-by-side comparison for parallel lines
            show_comparison_view(cam_img, line_mask, sat_img, projected_lines)
            
            angle_err, var_err = calculate_parallel_divergence(l1_points, l2_points, camera_params['H'])
            logger.info(f"Parallel Line Divergence: Angle Error = {float(angle_err):.4f} deg, Width Std Dev = {float(math.sqrt(var_err)):.4f}")
            cv2.waitKey(0)
        else:
            logger.warning("Not enough points for parallel lines.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
