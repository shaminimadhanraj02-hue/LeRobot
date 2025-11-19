# LeRobot
#Extract_Contour(Phase_1)
Phase 1 extracts a clean contour from a raw image and converts it into a normalized trajectory that can be used by a robot for imitation learning. The steps include:

loading the image

converting to grayscale

thresholding to isolate the object

extracting the largest contour

sampling ~300 points

normalizing coordinates to (0–1)

saving them as a .npy file

This contour will be used as the target trajectory for robot movement in Phase 2.