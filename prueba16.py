import cv2
import numpy as np
from PIL import Image
import os

def prueba():
    i = 1
    for filename in os.listdir("OutputMonkeys/"):
            
            if "All_Masks_" in filename:
                # Load the first image into a numpy array
                image = np.array(Image.open(f'OutputMonkeys/All_halo_Masks_{i}.png'))

                # Get the positions of all pixels with the color (0, 150, 0)
                mask = np.all(image == [255, 0, 0], axis=-1)
                positions = np.transpose(np.nonzero(mask))

                # Load the second image into a numpy array
                image2 = np.array(Image.open(f'Monos/Entrenamiento{i}.png'))

                # Create a new numpy array for the output image
                output = np.zeros_like(image2)

                # Copy the pixels at the positions of the first image from the second image to the output image
                output[positions[:,0], positions[:,1], :] = image2[positions[:,0], positions[:,1], :]

                # Create a new PIL image object from the output array
                new_image = Image.fromarray(output.astype('uint8'))

                # Save the new image
                new_image.save(f'OutputMonkeys/Class_2_{i}_plus_halo_contour.png')
                i+=1
