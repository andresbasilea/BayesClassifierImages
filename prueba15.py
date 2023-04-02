import cv2
import numpy as np

def prueba():
    # Load the image
    img = cv2.imread('OutputMonkeys/All_Masks_1.png')

    # Define the green color range
    lower_green = np.array([0, 255, 0])
    upper_green = np.array([0, 255, 0])

    # Create a mask for the green object
    mask_green = cv2.inRange(img, lower_green, upper_green)

    # Perform morphological operations to create a 10-pixel contour around the green object
    kernel = np.ones((10, 10), np.uint8)
    mask_surround = cv2.morphologyEx(mask_green, cv2.MORPH_DILATE, kernel) - mask_green

    # Merge the green object and the surrounding contour into a single image
    contours, hierarchy = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, (0, 150, 0), 20)


    # Save the output image
    cv2.imwrite('OutputMonkeys/All_Masks_1_plus_halo.png', img)
