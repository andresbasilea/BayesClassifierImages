import cv2
import numpy as np

# Load the images
img2 = cv2.imread('output.png')
img1 = cv2.imread('Entrenamiento3.png')


# Convert the first image to grayscale
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# Threshold the grayscale image to obtain a binary mask
ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

# Find the contours in the binary mask
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy of the second image with only the pixels that correspond to the contours
output = np.zeros_like(img2)
for contour in contours:
    x,y,w,h = cv2.boundingRect(contour)
    crop = img2[y:y+h, x:x+w]
    output[y:y+h, x:x+w][cv2.cvtColor(contour, cv2.COLOR_BGR2GRAY) > 0] = crop[cv2.cvtColor(contour, cv2.COLOR_BGR2GRAY) > 0]

# Display the result
cv2.imshow('Output', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
