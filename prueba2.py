# import numpy as np
# from skimage import io, color, filters, segmentation

# # Load the image
# image = io.imread('ImagenesEntrenamiento/Entrenamiento2.jpg')

# # Perform segmentation to get the labels
# labels = segmentation.slic(image, n_segments=4)

# # Convert the labels to a RGB mask
# mask = color.label2rgb(labels, image=image, colors=['red', 'green', 'blue'])

# # Save the mask
# io.imsave('MaskCreated.jpg', mask)


import numpy as np
from skimage import io, color, filters, morphology, measure

# Load the image
image = io.imread('Entrenamiento2.png')

# Convert the image to grayscale
gray = color.rgb2gray(image)

# Threshold the image to create a binary mask
thresh = filters.threshold_otsu(gray)
mask = gray > thresh

# Apply morphological operations to clean up the mask
mask = morphology.remove_small_objects(mask, min_size=300)
mask = morphology.closing(mask, morphology.square(5))

# Find the contours of each object in the mask
contours = measure.find_contours(mask, 0.5)

# Create a mask for each object
masks = []
for contour in contours:
    mask_i = np.zeros_like(mask)
    mask_i[np.round(contour[:, 0]).astype(int), np.round(contour[:, 1]).astype(int)] = 1
    masks.append(mask_i)

# Save the masks as separate images
for i, mask_i in enumerate(masks):
    io.imsave(f'mask{i+1}.jpg', np.uint8(mask_i * 255))
