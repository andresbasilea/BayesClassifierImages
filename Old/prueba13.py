import cv2
import numpy as np

COLORS = [
    (0, 255, 0),  # Green
    (0, 0, 255),  # Red
    (255, 0, 0)

]

mask=[np.zeros(img.shape[:2], np.uint8)]

def Monkey_Mask(filename, image_num):

    # Load the image
    img = cv2.imread(filename)

    # Create a black mask
    mask = np.zeros(img.shape[:2], np.uint8)

    # Display the image and let the user draw contours
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_contour)

    while True:
        # Display the image with the contour drawn
        img_with_contour = img.copy()
        img_with_contour[mask > 0] = COLORS[0]
        cv2.imshow('image', img_with_contour)

        # Handle user input
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Clear the contour
            mask = np.zeros(img.shape[:2], np.uint8)
        elif key == ord('s'):
            # Save the masked image
            masked_img = cv2.bitwise_and(img, img, mask=mask)
            cv2.imwrite(f'Masked_Image_{image_num+1}.png', masked_img)

            # Save the contour as a separate image
            contour_img = img.copy()
            contour_img[mask == 0] = 0
            cv2.imwrite(f'Contour_{image_num+1}.png', contour_img)

    # Clean up
    cv2.destroyAllWindows()

def draw_contour(event, x, y, flags, param):
    global mask
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(mask, (x, y), 5, 255, -1)
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        cv2.circle(mask, (x, y), 5, 255, -1)

Monkey_Mask("n019.png",1)
