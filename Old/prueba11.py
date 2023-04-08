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





# Define the colors for each class
COLORS = [
    (255, 0, 0),  # Blue
    (0, 255, 0),  # Green
    (0, 0, 255),  # Red
]

drawing = False
points = []


def draw(event, x, y, flags, param):
    global drawing, points, img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            points.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        points.append((x, y))
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask, [np.array(points)], -1, 255, -1)
        object_mask = mask.copy()
        surrounding_mask = cv2.dilate(mask, np.ones((10, 10), np.uint8)) - mask
        background_mask = 255 - object_mask - surrounding_mask
        masks = [object_mask, surrounding_mask, background_mask]

def Fruit_Mask(filename, image_num):
    # Load the image
    global img
    img = cv2.imread(filename)

    # Display the image and let the user draw a contour
    global drawing, points
    drawing = False
    points = []
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw)

    while True:
        # Draw the contour
        if points:
            mask = np.zeros(img.shape[:2], np.uint8)
            cv2.drawContours(mask, [np.array(points)], -1, 255, -1)
            object_mask = mask.copy()
            surrounding_mask = cv2.dilate(mask, np.ones((10, 10), np.uint8)) - mask
            background_mask = 255 - object_mask - surrounding_mask
            masks = [object_mask, surrounding_mask, background_mask]
            img_with_contours = img.copy()
            for i, mask in enumerate(masks):
                color = COLORS[i]
                img_with_contours[mask > 0] = color
            cv2.imshow('image', img_with_contours)

        # Handle user input
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('r'):
            points = []
        elif key == ord('s'):
            # Save the results
            output = np.zeros(img.shape, dtype=np.uint8)
            for i, mask in enumerate(masks):
                color = COLORS[i]
                output += cv2.bitwise_and(img, img, mask=mask)
                output[mask > 0] = color
            cv2.imwrite(f'Object_Mask_{image_num+1}.png', masks[0])
            cv2.imwrite(f'Surrounding_Mask_{image_num+1}.png', masks[1])
            cv2.imwrite(f'Background_Mask_{image_num+1}.png', masks[2])
            cv2.imwrite(f'All_Masks_{image_num+1}.png', output)

    # Clean up
    cv2.destroyAllWindows()



# def GaussianFilter(image):
#     image_obj = Image.open(image)
#     newsize = (600,600)
#     image_obj = image_obj.resize(newsize)     
#     image_obj = image_obj.filter(ImageFilter.GaussianBlur(radius=5))
#     image_obj.save(image.rstrip(".jpg") + ".png")
#     return image_obj


# imgage = GaussianFilter("D:/UNAM_NUEVO_Y_PROYECTOS/2023-2/ReconPatrones/Practica2/ImagenesEntrenamiento/monos/n019.jpg")
Fruit_Mask(,1)