import cv2
import numpy as np
import numpy as np
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
from PIL import Image, ImageFilter
import cv2
import numpy as np
import os


def GaussianFilter(image):
    image_obj = Image.open(image)
    newsize = (600,600)
    image_obj = image_obj.resize(newsize)     
    image_obj = image_obj.filter(ImageFilter.GaussianBlur(radius=5))
    image_obj.save(image.rstrip(".jpg") + ".png")
    return image_obj




NUM_CLASSES = 3

# Define the colors for each class
COLORS = [
    (0, 255, 0),  # Green
    (0, 0, 255),  # Red
    (255, 0, 0)

]


# Define the variables for the user drawing
drawing = False
points = []
contours = [[] for _ in range(NUM_CLASSES)]
class_num = 0

def draw(event, x, y, flags, param):
    global drawing, points

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            points.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        points.append((x, y))
        contours[class_num].append(np.array(points))
        points = []

def Fruit_Mask(filename, image_num):
    global class_num, contours

    # Load the image
    img = cv2.imread(filename)

    # Create a black mask for each class
    masks = [np.zeros(img.shape[:2], np.uint8) for _ in range(NUM_CLASSES)]

    class_num = 0
    contours = [[] for _ in range(NUM_CLASSES)]
    # Display the image and let the user draw contours
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw)
    while True:
        # Draw the contours for the current class
        mask = masks[class_num]
        for contour in contours[class_num]:
            cv2.drawContours(mask, [contour], -1, 255, -1)

        # Display the image with the contours drawn
        # masked_img = cv2.bitwise_and(img, img, mask=mask)
        # cv2.imshow('image', masked_img)

        img_with_contours = img.copy()
        for i, mask in enumerate(masks):
            color = COLORS[i]
            img_with_contours[mask > 0] = color
        cv2.imshow('image', img_with_contours)


        # Handle user input
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Clear the current class
            contours[class_num] = []
            masks[class_num] = np.zeros(img.shape[:2], np.uint8)
        elif key == ord('n'):
            # Switch to the next class
            class_num = (class_num + 1) % NUM_CLASSES
        elif key == ord('p'):
            # Switch to the previous class
            class_num = (class_num - 1) % NUM_CLASSES
        elif key == ord('r'):
            img = cv2.imread(filename)
            points = []
            contours = []
        elif key == ord('s'):
            # Save the results
            output = np.zeros(img.shape, dtype=np.uint8)
            for i, mask in enumerate(masks):
                color = COLORS[i]
                output += cv2.bitwise_and(img, img, mask=mask)
                output[mask > 0] = color
            cv2.imwrite(f'OutputFruits/All_Masks_{image_num+1}.png', output)

            # Save the results
            for i, contour_list in enumerate(contours):
                color = COLORS[i]
                for j, contour in enumerate(contour_list):
                    mask = np.zeros(img.shape[:2], np.uint8)
                    cv2.drawContours(mask, [contour], -1, 255, -1)
                    masked_img = cv2.bitwise_and(img, img, mask=mask)
                    x, y, w, h = cv2.boundingRect(contour)
                    cropped_img = masked_img[y:y+h, x:x+w]
                    cv2.imwrite(f"OutputFruits/Class_{i+1}_Contour_{j+1}_{image_num+1}.png", cropped_img)


    # Clean up
    cv2.destroyAllWindows()


import prueba16
import prueba15
def Monkey_Mask(filename, image_num):
    global class_num, contours

    # Load the image
    img = cv2.imread(filename)

    # Create a black mask for each class
    masks = [np.zeros(img.shape[:2], np.uint8) for _ in range(NUM_CLASSES)]

    class_num = 0
    contours = [[] for _ in range(NUM_CLASSES)]
    # Display the image and let the user draw contours
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw)
    while True:
        # Draw the contours for the current class
        mask = masks[class_num]
        for contour in contours[class_num]:
            cv2.drawContours(mask, [contour], -1, 255, -1)

        # Display the image with the contours drawn
        # masked_img = cv2.bitwise_and(img, img, mask=mask)
        # cv2.imshow('image', masked_img)

        img_with_contours = img.copy()
        for i, mask in enumerate(masks):
            color = COLORS[i]
            img_with_contours[mask > 0] = color
        cv2.imshow('image', img_with_contours)


        # Handle user input
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Clear the current class
            contours[class_num] = []
            masks[class_num] = np.zeros(img.shape[:2], np.uint8)
        elif key == ord('n'):
            # Switch to the next class
            class_num = (class_num + 1) % NUM_CLASSES
        elif key == ord('p'):
            # Switch to the previous class
            class_num = (class_num - 1) % NUM_CLASSES
        elif key == ord('r'):
            img = cv2.imread(filename)
            points = []
            contours = []
        elif key == ord('s'):
            # Save the results
            output = np.zeros(img.shape, dtype=np.uint8)
            for i, mask in enumerate(masks):
                color = COLORS[i]
                output += cv2.bitwise_and(img, img, mask=mask)
                output[mask > 0] = color
            cv2.imwrite(f'OutputMonkeys/All_Masks_{image_num+1}.png', output)

            # Save the results
            for i, contour_list in enumerate(contours):
                color = COLORS[i]
                for j, contour in enumerate(contour_list):
                    mask = np.zeros(img.shape[:2], np.uint8)
                    cv2.drawContours(mask, [contour], -1, 255, -1)
                    masked_img = cv2.bitwise_and(img, img, mask=mask)
                    x, y, w, h = cv2.boundingRect(contour)
                    cropped_img = masked_img[y:y+h, x:x+w]
                    cv2.imwrite(f"OutputMonkeys/Class_{i+1}_Contour_{j+1}_{image_num+1}.png", cropped_img)


    # Clean up
    cv2.destroyAllWindows()
    prueba15.prueba()       
    prueba16.prueba()

# Mask_Colors("Entrenamiento2.png")