# WORKING V3. CREACION DE MASCARA Y CROPPED 

import cv2
import numpy as np
import os
import FiltroGaussiano




# Load the image
img = cv2.imread('Entrenamiento2.jpg')

# Create a window to display the image
cv2.namedWindow('image')

# Define the mouse event function
drawing = False
points = []
contours = []
def draw(event, x, y, flags, param):
    global drawing, points, img, contours

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points = [(x, y)]

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(img, points[-1], (x, y), (0, 0, 255), 2)
            points.append((x, y))

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        points.append((x, y))
        contour = np.array(points)
        contours.append(contour)
        cv2.drawContours(img, [contour], 0, (0, 0, 0), 2)

# Bind the mouse event function to the window
cv2.setMouseCallback('image', draw)

# Display the image and wait for the user to draw on it
while True:
    cv2.imshow('image', img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('r'):
        img = cv2.imread('Entrenamiento2.jpg')
        points = []
        contours = []

# Segment the selected regions and save as separate images
for i, contour in enumerate(contours):
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    result = cv2.bitwise_and(img, img, mask=mask)
    x, y, w, h = cv2.boundingRect(contour)
    cropped_image = result[y:y+h, x:x+w]
    
    # Create a mask with only the cropped area
    mask = np.zeros(cropped_image.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour - contour.min(axis=0)], 0, 255, -1)
    
    # Apply the mask to the cropped image
    masked_cropped_image = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)
    
    # Create binary mask with cropped region as white
    binary_mask = np.zeros(img.shape[:2], np.uint8)
    binary_mask[y:y+h, x:x+w] = mask
    
    # Save cropped image and binary mask
    cv2.imwrite(f"output_image_{i+1}.png", masked_cropped_image)
    cv2.imwrite(f"output_mask_{i+1}.png", binary_mask)

# Close all windows
cv2.destroyAllWindows()
