import cv2
import numpy as np

# Define the color for the class
COLOR = (0, 255, 0)  # Green

# Define the variables for the user drawing
drawing = False
points = []
contour = None
surrounding_contour = None


def draw(event, x, y, flags, param):
    global drawing, points, contour, surrounding_contour

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            points.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        points.append((x, y))
        contour = np.array(points)

        # Create a surrounding contour with 10 pixels thickness
        epsilon = 10
        surrounding_contour = cv2.approxPolyDP(contour, epsilon, True)
        surrounding_contour = surrounding_contour.squeeze()


def Monkey_Mask(filename, image_num):
    global contour, surrounding_contour

    # Load the image
    img = cv2.imread(filename)

    # Create a black mask for the class
    mask = np.zeros(img.shape[:2], np.uint8)

    # Display the image and let the user draw the contour
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw)
    while True:
        # Draw the contour and the surrounding contour
        img_with_contours = img.copy()
        if contour is not None:
            cv2.drawContours(img_with_contours, [contour], -1, COLOR, 2)
        if surrounding_contour is not None:
            cv2.drawContours(img_with_contours, [surrounding_contour], -1, COLOR, 2)

        cv2.imshow('image', img_with_contours)

        # Handle user input
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Clear the contour and the surrounding contour
            contour = None
            surrounding_contour = None
        elif key == ord('r'):
            img = cv2.imread(filename)
            contour = None
            surrounding_contour = None
        elif key == ord('s'):
            # Save the results
            masked_img = cv2.bitwise_and(img, img, mask=mask)
            cv2.drawContours(mask, [surrounding_contour], -1, 255, -1)
            cv2.imwrite(f'Mask_{image_num+1}.png', mask)
            cv2.imwrite(f'Contour_{image_num+1}.png', masked_img)

    # Clean up
    cv2.destroyAllWindows()


Monkey_Mask("n019.png",1)