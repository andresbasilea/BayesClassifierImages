import numpy as np
from skimage import io, filters, morphology, measure

# Load the image
image = io.imread('Entrenamiento2.png')

# Threshold the red channel to create a mask for the background
background_mask = image[:,:,0] < 200

# Threshold the green channel to create a mask for the bananas
banana_mask = (image[:,:,1] > 150) & (image[:,:,1] < 220)

# Threshold the blue channel to create a mask for the eggs
egg_mask = image[:,:,2] < 150

# Threshold the green channel again to create a mask for the green chillies
green_chilli_mask = (image[:,:,1] > 200) & (image[:,:,1] < 255)

# Combine the masks into a single binary mask
mask = background_mask | banana_mask | egg_mask | green_chilli_mask

# Apply morphological operations to clean up the mask
mask = morphology.remove_small_objects(mask, min_size=100)
mask = morphology.closing(mask, morphology.square(5))

# Label each object in the mask
labels = measure.label(mask)

# Extract the masks for each object
objects = []
for i in range(1, labels.max()+1):
    object_i = (labels == i)
    objects.append(object_i)

# Save the masks as separate images
for i, object_i in enumerate(objects):
    io.imsave(f'mask{i+1}.jpg', np.uint8(object_i * 255))
