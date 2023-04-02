import cv2
import numpy as np

# Load the image
image = cv2.imread('Entrenamiento2.png')

# Convert to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Threshold the image based on the color range you want to segment
lower_range = np.array([0, 0, 0])
upper_range = np.array([144, 144, 25])
mask = cv2.inRange(hsv_image, lower_range, upper_range)

# Smooth the edges of the binary mask
kernel = np.ones((5,5),np.uint8)
mask = cv2.erode(mask, kernel, iterations=1)
mask = cv2.dilate(mask, kernel, iterations=1)

# Apply the watershed algorithm
dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
ret, markers = cv2.connectedComponents(np.uint8(dist_transform))
markers = markers+1
markers[mask==0] = 0
markers = cv2.watershed(image, markers)

# Assign a unique color to each region to create a color mask
color_mask = np.zeros_like(image)
for i in range(1, np.max(markers)+1):
    color_mask[markers == i] = np.random.randint(0, 255, size=(1,3))
    
# Show the color mask
cv2.imshow('Color Mask', color_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
