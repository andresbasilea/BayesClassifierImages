def CropClasses(image, classes):
    global drawing, points, img, contours

    # Load the image
    img = cv2.imread("Entrenamiento3.png")

    # Create a black mask to draw the selected regions on
    mask = np.zeros(img.shape[:2], np.uint8)

    # Create a list to hold the images of each class
    class_images = [np.zeros_like(img) for _ in classes]

    # Bind the mouse event function to the window
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw)

    # Display the image and wait for the user to draw on it
    while True:
        cv2.imshow('image', img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('r'):
            img = cv2.imread(image)
            points = []
            contours = []

    # Segment the selected regions and save as separate images
    for i, contour in enumerate(contours):
        # Determine which class the contour belongs to
        class_num = -1
        for j, class_ in enumerate(classes):
            if contour in class_:
                class_num = j
                break
        if class_num == -1:
            continue

        # Draw the contour on the mask using a different color for each class
        cv2.drawContours(mask, [contour], -1, class_num + 1, -1)

        # Draw the contour on the class image using the class color
        color = tuple(int(c) for c in np.random.choice(range(256), size=3))
        cv2.drawContours(class_images[class_num], [contour], -1, color, -1)

    # Merge the class images into a single image
    result = np.zeros_like(img)
    for i, class_image in enumerate(class_images):
        result[class_image != 0] = (i + 1) * 50

    # Display the result and wait for the user to press a key
    cv2.imshow('result', result)
    cv2.waitKey(0)

    # Save the result image
    cv2.imwrite('result.png', result)

    # Release all windows
    cv2.destroyAllWindows()
