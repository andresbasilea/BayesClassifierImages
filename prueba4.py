import numpy as np
from skimage import io, color, filters, morphology, measure, feature

# Load the image
image = io.imread('Entrenamiento2.png')

# Convert the image to grayscale
gray = color.rgb2gray(image)

# Compute the local binary pattern (LBP) feature for the grayscale image
lbp = feature.local_binary_pattern(gray, 8, 1, method='uniform')

# Create a mask for bananas
banana_mask = np.logical_and(lbp > 40, lbp < 70)

# Create a mask for white eggs
egg_mask = np.logical_and(lbp > 190, lbp < 220)

# Create a mask for green chillies
green_chilli_mask = np.logical_and(lbp > 100, lbp < 130)

# Combine the masks
mask = np.logical_or(np.logical_or(banana_mask, egg_mask), green_chilli_mask)

# Apply morphological operations to clean up the mask
mask = morphology.remove_small_objects(mask, min_size=500)
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
