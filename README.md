# LeRobot
#Extract_Contour(Phase_1)
Phase 1 extracts a clean contour from a raw image and converts it into a normalized trajectory that can be used by a robot for imitation learning. The steps include:

1. loading the image

2. converting to grayscale

3. thresholding to isolate the object

4. extracting the largest contour

5. sampling ~300 points

6. normalizing coordinates to (0–1)

7. saving them as a .npy file

This contour will be used as the target trajectory for robot movement in Phase 2.
