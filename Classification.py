import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import cv2
import skimage
from skimage import io
from skimage.color import *
import numpy as np
import pprint
import pandas as pd



def BayesRGB(images, num_classes):
    
    
    num_images_file = num_classes * 2


    dic = {}
    for x in range(num_classes):
        x = x+1
        dic['Clase_%s_RGB' %x] = []
        dic['Clase_%s_Mask' %x] = []

    


    i = 0
    for imageRGB, imageMask in zip(images[0::2], images[1::2]):
        pixelsRGB, pixelsMask = list(imageRGB.getdata()),list(imageMask.getdata())
        #pixelsRGB, pixelsMask = imageRGB,imageMask
        dic[f'Clase_{i%num_classes+1}_RGB'].append(pixelsRGB)
        dic[f'Clase_{i%num_classes+1}_Mask'].append(pixelsMask)
        i += 1

    
    print(len([v for k,v in dic.items() if 'RGB' in k]))

    RGB_class_list = []
    for x in range(num_classes):
        RGB_class = [v for k,v in dic.items() if f'{x+1}_RGB' in k]
        Mask_class = [v for k,v in dic.items() if f'{x+1}_Mask' in k]
        df_rgb_means = []
        for image in RGB_class:
            image_array = np.array(image)
            reshaped_array = image_array.reshape(-1,3)
            df = pd.DataFrame(reshaped_array, columns=['R','G','B'])
            df_rgb_means.append(df)

        df_rgb_means = pd.concat(df_rgb_means)
        df_rgb_means = df_rgb_means.loc[~(df_rgb_means==0).all(axis=1)]   # Delete rows with '0' in R, G and B
        RGB_class_list.append([df_rgb_means.mean(),df_rgb_means.cov()])

        print(f"\n\nClase {x+1} RGB DataFrame:\n ",df_rgb_means)
        print(f"\nClass {x+1} mean matrix: \n", df_rgb_means.mean())
        print(f"\nClass {x+1} cov matrix: \n", df_rgb_means.cov())
        




    # RGB_images = [v for k, v in dic.items() if 'RGB' in k]
    # Mask_images = [v for k, v in dic.items() if 'Mask' in k]


    # df_rgb_means = []
    # for image in RGB_images:
    #     image_array = np.array(image)
    #     reshaped_array = image_array.reshape(-1,3)
    #     df = pd.DataFrame(reshaped_array, columns=['R','G','B'])
    #     df_rgb_means.append(df)

    # df_rgb_means = pd.concat(df_rgb_means)
    # print(df_rgb_means)


    # df1 = pd.DataFrame([p for p in RGB_images[0] if sum(p) != 0], columns=['R','G','B', 'A'])


    # df_RGB = []
    # for image in RGB_images:
    #   df_RGB1 = pd.DataFrame([p for p in image if sum(p) != 0], columns=['R','G','B', 'A'])
    #   df_RGB = pd.concat([df_RGB, df_RGB_1])
        
    # print(df1)



    # df1 = pd.DataFrame([p for p in pixeles if sum(p) != 0], columns=['R', 'G', 'B', 'A'])
    # df2 = pd.DataFrame([p for p in pixeles2 if sum(p) != 0], columns=['R', 'G', 'B', 'A'])
    # dfRGB = pd.concat([df1, df2], axis=0)
    # dfRGB = dfRGB.drop(dfRGB.columns[-1], axis=1)

    # df_filtered = dfRGB[(dfRGB['R'] == 255) & (dfRGB['G'] == 255) & (dfRGB['B'] == 255)]
    # print(len(df_filtered))

    # i = 0
    # j = 1
    # for image in images:
    #   # pixels = list(image.getdata())

    #   pixels = i
    #   print(j)
    #   if i%2 == 0: # RGB image
    #       if j%num_classes == 0:
    #           j = 1
    #       dic[f'Clase_{j%num_classes}_RGB'].append(pixels)
    #   else:  # Mask image
    #       if j%num_classes == 0:
    #           j = 1
    #       dic[f'Clase_{j%num_classes}_Mask'].append(pixels)

    #   if i%2 == 0:
    #       j += 1
    #   i += 1
        

    
    

# img=mpimg.imread(repositorioPractica2 + '/Entrenamiento1.jpg') 
# imgplot = plt.imshow(img)

# imgP1=mpimg.imread(repositorioPractica2 + '/Platano1.PNG') 
# imgplotP1 = plt.imshow(imgP1)

# imgP2=mpimg.imread(repositorioPractica2 + '/Platano2.PNG') 
# imgplotP2 = plt.imshow(imgP2)

# print("Tamaño de la Imagen 1:\n",imgP1.shape)

# print("Tamaño de la Imagen 2:\n", imgP2.shape)

# print("Estos valores representan altura, ancho y número de canales de color")

# print("Altura de los plátanos de la imagen Entrenamiento1:",imgP1.shape[0],imgP2.shape[0])
# print("Ancho de los plátanos de la imagen Entrenamiento1:",imgP1.shape[1],imgP2.shape[1])


# dfsize = pd.DataFrame(columns=["Altura","Ancho"])
# dfsize.loc[len(dfsize)] = [imgP1.shape[0],imgP1.shape[1]]
# dfsize.loc[len(dfsize)] = [imgP2.shape[0],imgP2.shape[1]]
# print("Data Frame de los anchos y alturas de los plátanos en imagen Entrenamiento1:\n\n",dfsize)


# # Esta es una prueba de que los pixeles blancos tiene valor 0,0,0,0
# # Carga la imagen
# imagen = Image.open(repositorioPractica2 + '/Platano1.PNG')

# # Obtiene el valor del pixel superior izquierdo
# valor_pixel = imagen.getpixel((10, 10))

# # Imprime el resultado
# print(valor_pixel)

# # Abre la imagen 1
# imagen = Image.open(repositorioPractica2 + '/Platano1.PNG')

# # Obtiene una lista de tuplas de píxeles (R, G, B)
# pixeles = list(imagen.getdata())

# # Abre la imagen 2
# imagen2 = Image.open(repositorioPractica2 + '/Platano2.PNG')

# # Obtiene una lista de tuplas de píxeles (R, G, B)
# pixeles2 = list(imagen2.getdata())

# # Convierte la lista de tuplas en un DataFrame de pandas
# df1 = pd.DataFrame([p for p in pixeles if sum(p) != 0], columns=['R', 'G', 'B', 'A'])
# df2 = pd.DataFrame([p for p in pixeles2 if sum(p) != 0], columns=['R', 'G', 'B', 'A'])
# dfRGB = pd.concat([df1, df2], axis=0)
# dfRGB = dfRGB.drop(dfRGB.columns[-1], axis=1)

# df_filtered = dfRGB[(dfRGB['R'] == 255) & (dfRGB['G'] == 255) & (dfRGB['B'] == 255)]
# print(len(df_filtered))

# # La media
# media = dfRGB.mean()
# print("\nValor de las Medias:","\n", media)


# #Covarianza
# covariance = dfRGB.cov()
# print("\nMatriz de Covarianza:\n", covariance)

# # La media
# media = dfsize.mean()
# print("\nValor de las Medias:","\n", media)


# #Covarianza
# covariance = dfsize.cov()
# print("\nMatriz de Covarianza:\n", covariance)
