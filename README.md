# Introduction
The evaluation is done based on two categories:

Man Made feature comparison to satellite image(like bollards, curbs) i.e using top down image from satellite/drone as a reference and seeing how well the same features extracted from an image align

Geometric characteristics/Self Consistency i.e. parallel lines in real world should be projected as parallel

The feature extraction itself is conducted in two modes

Manual: Interactive window to pick points and interoplate to obtain a dense set of points for binary masking

Auto: Features extracted automatically by a Segmentation model

# Validation Criteria

## Parallel Line Divergence (Self-Consistency)

     This method evaluates the internal logic of the homography without needing a satellite map.

## Chamfer Distance (Reference-Based comparison)**

     This method evaluates how well the camera "aligns" with a known map (the satellite image).
