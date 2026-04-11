import cv2
import numpy as np
from scipy.interpolate import interp1d

class PointPicker:
    def __init__(self, window_name="Pick Points", max_display_dim=1000):
        self.window_name = window_name
        self.points = []
        self.max_display_dim = max_display_dim
        self.scale = 1.0

    def _mouse_callback(self, event, x, y, flags, param):
        display_img = param
        if event == cv2.EVENT_LBUTTONDOWN:
            # Map back to original coordinates
            orig_x = int(x / self.scale)
            orig_y = int(y / self.scale)
            self.points.append((orig_x, orig_y))
            
            # Draw on display image
            cv2.circle(display_img, (x, y), 3, (0, 0, 255), -1)
            if len(self.points) > 1:
                # Get the previous point in display coordinates
                prev_display_x = int(self.points[-2][0] * self.scale)
                prev_display_y = int(self.points[-2][1] * self.scale)
                cv2.line(display_img, (prev_display_x, prev_display_y), (x, y), (0, 255, 0), 1)
            cv2.imshow(self.window_name, display_img)

    def pick_points(self, img):
        self.points = []
        h, w = img.shape[:2]
        
        # Calculate scale to fit in max_display_dim
        self.scale = min(self.max_display_dim / w, self.max_display_dim / h, 1.0)
        display_w = int(w * self.scale)
        display_h = int(h * self.scale)
        
        display_img = cv2.resize(img, (display_w, display_h))
        cv2.imshow(self.window_name, display_img)
        cv2.setMouseCallback(self.window_name, self._mouse_callback, display_img)
        
        print(f"[{self.window_name}] Resized to {display_w}x{display_h} (Scale: {self.scale:.2f})")
        print(f"Pick points. Press 'q' to finish, 'c' to clear.")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.points = []
                display_img = cv2.resize(img, (display_w, display_h))
                cv2.imshow(self.window_name, display_img)

        cv2.destroyWindow(self.window_name)
        return np.array(self.points)

def densify_points(points, interval=5):
    """Interpolate between points to get a dense set of points."""
    if len(points) < 2:
        return points
    
    dense_points = []
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i+1]
        dist = np.linalg.norm(p2 - p1)
        num_points = max(int(dist / interval), 2)
        
        t = np.linspace(0, 1, num_points)
        line_points = (1 - t[:, None]) * p1 + t[:, None] * p2
        dense_points.extend(line_points[:-1])
    
    dense_points.append(points[-1])
    return np.array(dense_points)

def create_binary_mask(img_shape, dense_points):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for p in dense_points:
        x, y = int(round(p[0])), int(round(p[1]))
        if 0 <= x < img_shape[1] and 0 <= y < img_shape[0]:
            mask[y, x] = 255
    return mask
