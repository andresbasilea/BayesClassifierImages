import cv2
import numpy as np

# Define the colors for each class
COLORS = [
    (255, 0, 0),  # Blue
    (0, 255, 0),  # Green
    (0, 0, 255),  # Red
]

def draw(event, x, y, flags, param):
    global drawing, points, img, img_with_contours

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
        img_with_contours = img.copy()
        for i, mask in enumerate(masks):
            color = COLORS[i]
            img_with_contours[mask > 0] = color

def Fruit_Mask(filename, image_num):
    # Load the image
    global img, drawing, points, img_with_contours
    img = cv2.imread(filename)
    img_with_contours = img.copy()
    cv2.imshow('image', img)

    # Display the image and let the user draw a contour
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
            img_with_contours = img.copy()
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

Fruit_Mask("n019.png", 1)