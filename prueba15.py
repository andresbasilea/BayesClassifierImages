import cv2
import numpy as np
import os 

def prueba():
    # Load the image
    i = 1
    for filename in os.listdir("OutputMonkeys/"):
            if "All_Masks_" in filename:
                # Load the image
                img = cv2.imread(f'OutputMonkeys/All_Masks_{i}.png')
                print(i,img)

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
                cv2.drawContours(img, contours, -1, (0, 0, 255), 20)


                # Save the output image
                cv2.imwrite(f'OutputMonkeys/All_halo_Masks_{i}.png', img)
                i+=1
