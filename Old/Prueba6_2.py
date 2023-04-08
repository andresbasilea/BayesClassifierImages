def main():
    global class_num

    # Load the image
    img = cv2.imread('Entrenamiento3.png')

    # Create a black mask for each class
    masks = [np.zeros(img.shape[:2], np.uint8) for _ in range(NUM_CLASSES)]

    # Display the image and let the user draw contours
    while True:
        # Draw the contours for the current class
        mask = masks[class_num]
        for contour in contours[class_num]:
            cv2.drawContours(mask, [contour], -1, 255, -1)

        # Display the original image with the contours drawn
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
