import cv2
import numpy as np

# Define the number of classes
NUM_CLASSES = 3

# Define the colors for each class
COLORS = [
    (0, 255, 0),  # Green
    (0, 0, 255),  # Red
    (255, 0, 0)   # Blue
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

def main():
    global class_num

    # Load the image
    img = cv2.imread('Entrenamiento3.png')

    # Create a black mask for each class
    masks = [np.zeros(img.shape[:2], np.uint8) for _ in range(NUM_CLASSES)]

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
        elif key == ord('s'):
            # Save the results
            output = np.zeros(img.shape, dtype=np.uint8)
            for i, mask in enumerate(masks):
                color = COLORS[i]
                output += cv2.bitwise_and(img, img, mask=mask)
                output[mask > 0] = color
            cv2.imwrite('output.png', output)

    # Clean up
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
