import cv2
import numpy as np

# Define the variables for the user drawing
drawing = False
points = []

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

        # Draw the second contour
        second_contour = [(x + 10, y + 10) for (x, y) in points]
        cv2.drawContours(img, [np.array(second_contour)], 0, (0, 255, 0), 2)

        # Draw the third contour
        third_contour = [(0, 0), (img.shape[1], 0), (img.shape[1], img.shape[0]), (0, img.shape[0])]
        third_contour = [pt for pt in third_contour if pt not in points and pt not in second_contour]
        cv2.drawContours(img, [np.array(third_contour)], 0, (0, 0, 255), 2)

def main():
    # Load the image and resize it
    img = cv2.imread('n019.jpg')
    img = cv2.resize(img, (600, 600))

    # Display the image and let the user draw contours
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw)
    while True:
        cv2.imshow('image', img)

        # Handle user input
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Clear the drawing
            img = cv2.resize(cv2.imread('n019.jpg'), (600, 600))
        elif key == ord('s'):
            # Save the result
            cv2.imwrite('result.jpg', img)

    # Clean up
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
