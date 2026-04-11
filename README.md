The implementation follows these mathematical steps:

  1. Coordinate Projection
  Each set of camera points $P_{cam} = (u, v)$ is transformed into the world plane coordinates $P_{world} = (x, y)$ using the homography matrix $H$:
  $$\begin{bmatrix} x' \\ y' \\ w' \end{bmatrix} = H \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}, \quad x = \frac{x'}{w'}, \quad y = \frac{y'}{w'}$$

  2. Line Fitting
  For each set of projected points, a line is fitted using the $L_2$ distance (least squares). A line is represented by a unit direction vector $\vec{v}$ and a point $\vec{p}_0$ on
  the line:
  $$\text{Line}_1: \vec{v}_1 = \begin{bmatrix} v_{x1} \\ v_{y1} \end{bmatrix}, \vec{p}_1 = \begin{bmatrix} x_1 \\ y_1 \end{bmatrix} \quad \text{Line}_2: \vec{v}2 = \begin{bmatrix}
  v{x2} \\ v_{y2} \end{bmatrix}, \vec{p}_2 = \begin{bmatrix} x_2 \\ y_2 \end{bmatrix}$$

  3. Angular Error ($\theta$)
  The angular error represents the divergence in orientation between the two lines. It is calculated as the absolute difference of their heading angles:
  $$\alpha_1 = \text{atan2}(v_{y1}, v_{x1}), \quad \alpha_2 = \text{atan2}(v_{y2}, v_{x2})$$
  $$\theta = |\alpha_1 - \alpha_2| \pmod{180^\circ}$$
  In a perfect homography, $\theta = 0^\circ$. Values $> 0^\circ$ indicate a "shear" or "rotation" error in the ground plane estimation.

  4. Width Consistency ($\sigma^2$)
  This metric detects "pinching" or "flaring" (perspective distortion) that hasn't been correctly flattened. We calculate the perpendicular distance $W_i$ from every projected
  point $P_i = (x_i, y_i)$ of the first line to the fitted second line:
  $$W_i = \frac{|(y_i - y_2)v_{x2} - (x_i - x_2)v_{y2}|}{\sqrt{v_{x2}^2 + v_{y2}^2}}$$
  Since $\vec{v}_2$ is a unit vector, the denominator is 1. The final metric is the variance of these widths:
  $$\sigma^2 = \frac{1}{n} \sum_{i=1}^n (W_i - \bar{W})^2$$
   * Low Variance: The lines are equidistant (parallel), even if $\theta \neq 0$.
   * High Variance: The distance between the lines changes as they move away from the camera, indicating that the ground plane is not correctly modeled as flat.