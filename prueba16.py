import cv2
import numpy as np
from PIL import Image

# def prueba():
#     # Load the image
#     img = cv2.imread('OutputMonkeys/All_Masks_1_plus_halo.png')

#     # Extract the surrounding pixels
#     surrounding_pixels = np.where(np.all(img == [0, 150, 0], axis=-1))

#     print(f"Surrounding pixels: {surrounding_pixels}")



#     img2 = cv2.imread('n019.png')


#     # Get the pixels corresponding to surrounding_pixels
#     surrounding_pixels_img2 = np.zeros_like(img2)
    
#     for pixel in surrounding_pixels:
#         print(pixel)
#         surrounding_pixels_img2[pixel[0], pixel[1]] = img2[pixel[0], pixel[1]]

#     # Save the images
#     cv2.imwrite('mask1.png', green_pixels_img2)
#     cv2.imwrite('mask2.png', surrounding_pixels_img2)


# def prueba():

#     # Load the first image into a numpy array
#     image = np.array(Image.open('OutputMonkeys/All_Masks_1_plus_halo.png'))

#     # Get the positions of all pixels with the color (0, 150, 0)
#     mask = np.all(image == [0, 150, 0], axis=-1)
#     positions = np.transpose(np.nonzero(mask))

#     # Load the second image into a numpy array
#     image2 = np.array(Image.open('n019.png'))

#     # Get the pixels at the positions of the first image
#     pixels = image2[positions[:,0], positions[:,1]]

#     # Create a new image from the pixels
#     new_image = Image.fromarray(pixels.astype('uint8'))

#     # Save the new image
#     new_image.save('OutputMonkeys/All_Masks_1_plus_halo_contour.png')



def prueba():

    # Load the first image into a numpy array
    image = np.array(Image.open('OutputMonkeys/All_Masks_1_plus_halo.png'))

    # Get the positions of all pixels with the color (0, 150, 0)
    mask = np.all(image == [0, 150, 0], axis=-1)
    positions = np.transpose(np.nonzero(mask))

    # Load the second image into a numpy array
    image2 = np.array(Image.open('n019.png'))

    # Create a new numpy array for the output image
    output = np.zeros_like(image2)

    # Copy the pixels at the positions of the first image from the second image to the output image
    output[positions[:,0], positions[:,1], :] = image2[positions[:,0], positions[:,1], :]

    # Create a new PIL image object from the output array
    new_image = Image.fromarray(output.astype('uint8'))

    # Save the new image
    new_image.save('OutputMonkeys/All_Masks_1_plus_halo_contour.png')

