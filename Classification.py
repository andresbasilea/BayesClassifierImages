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
import os




def BayesRGB(monkey_fruit):
    
    if monkey_fruit == 1: # monkey images to classify

        print("ho")


    elif monkey_fruit == 2: # fruit images to classify

        print('h')

    else:
        print("Error")






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





import glob
import numpy as np
import pandas as pd

# def obtenerProbAPriori(datos_y, n_clases : int):
#     data_shape = datos_y.shape
#     print(data_shape)
#     total_datos = np.multiply.reduce(data_shape)
#     print("total_datos: " , total_datos)
#     hist = np.histogram(datos_y.reshape(1, total_datos), 
#                     bins=n_clases, 
#                     range=[0,n_clases-1])
#     prob = hist[0].astype(np.long) / total_datos
#     print(prob)

#     plt.bar(range(n_clases), hist[0], align='center')
#     plt.xticks(range(n_clases))
#     plt.xlabel('Class')
#     plt.ylabel('Count')
#     plt.title('Histogram of class frequencies')
#     plt.show()


#     return prob


# filelist = glob.glob("OutputFruits/All_Masks_*.png")
# x = np.array([np.array(Image.open(fname)) for fname in filelist])
# nz = np.nonzero(x)
# rows = np.unique(nz[0])
# print(rows)
# print(len(rows), "x: ",len(x))
# # x_flat = x.reshape(8,-1)
# # df = pd.DataFrame(x_flat)
# # df.to_excel("data.xlsx", index=False)

# image =  Image.open("OutputFruits/All_Masks_1.png")



# data_img = np.array(image)

# prob = obtenerProbAPriori(x, 4)
# print(prob)





# PROBABILIDAD A PRIORI FUNCIONANDO

import numpy as np
import cv2
import os

# Define the path to your training images
path = "OutputFruits/"

# Define the number of classes
num_classes = 3

# Create an array to store the count of pixels for each class
class_counts = np.zeros(num_classes)

# Loop through each training image
for filename in os.listdir(path):
    if "All_Masks_" in filename:
        # Load the image
        img = cv2.imread(os.path.join(path, filename))

        # Extract the mask of each color
        red_mask = ((img[:, :, 2] == 255).astype(int))/2
        blue_mask = ((img[:, :, 0] == 255).astype(int))/3
        green_mask = ((img[:, :, 1] == 255).astype(int))/3

        # Add the count of pixels for each class
        class_counts[0] += np.sum(red_mask)
        class_counts[1] += np.sum(blue_mask)
        class_counts[2] += np.sum(green_mask)

# Calculate the total number of pixels in all masks
total_pixels = np.sum(class_counts)

# Calculate the prior probability of each class
class_probs = class_counts / total_pixels

print("Prior probability of red class:", class_probs[0])
print("Prior probability of blue class:", class_probs[1])
print("Prior probability of green class:", class_probs[2])




# import glob
# import cv2
# import numpy as np

# # Define the file pattern to match
# file_pattern = "OutputFruits/Class_*_*"

# # Get the file list
# file_list = glob.glob(file_pattern)

# # Define the number of classes
# n_classes = 3

# # Initialize the mean matrix for each class
# mean_matrices = [np.zeros((3, 3), dtype=np.float32) for _ in range(n_classes)]

# # Loop over each file and calculate the mean matrix for each class
# for file_path in file_list:
#     # Extract the class number from the file name
#     class_num = int(file_path.split("_")[1]) - 1
    
#     # Read the image
#     img = cv2.imread(file_path)
    
#     # Calculate the mean matrix for the image
#     mean_matrix = np.mean(img, axis=(0, 1))
    
#     # Add the mean matrix to the corresponding class
#     mean_matrices[class_num] += mean_matrix

# # Divide each mean matrix by the number of images in the corresponding class to get the final mean matrix for each class
# for i in range(n_classes):
#     mean_matrices[i] /= len(file_list)

# # Print the final mean matrices for each class
# for i in range(n_classes):
#     print(f"Mean matrix for Class {i+1}:")
#     print(mean_matrices[i])






# OBTENCION DE LA MATRIZ DE MEDIAS FUNCIONANDO 


file_pattern = "OutputFruits/Class_*_*"

# Get the file list
file_list = glob.glob(file_pattern)

# print(file_list)

c1 = []
c2 = []
c3 = []
for image in file_list:
    if "Class_1" in image:
        c1.append(image)
    elif "Class_2" in image:
        c2.append(image)
    elif "Class_3" in image:
        c3.append(image)

num_images_per_class = [c1, c2, c3]
#print(num_images_per_class)
n_classes = 3

# Initialize the mean matrix for each class
# mean_matrices = [np.zeros((1, 3), dtype=np.float32) for _ in range(n_classes)]


RGB_class_list = []
x=0
for class_list in num_images_per_class:
    print(class_list)
    df_rgb_means = []
    for image in class_list:
        #print(image)
        image = image.replace("\\", "/")
        # print(image)
        image = cv2.imread(image)
        image_array = np.array(image)
        # print(image_array)
        reshaped_array = image_array.reshape(-1,3)
        df = pd.DataFrame(reshaped_array, columns=['B','G','R'])
        df_rgb_means.append(df)

    df_rgb_means = pd.concat(df_rgb_means)
    df_rgb_means = df_rgb_means.loc[~(df_rgb_means==0).all(axis=1)]   # Delete rows with '0' in R, G and B
    RGB_class_list.append([df_rgb_means.mean(),df_rgb_means.cov()])
    x+=1

    print(f"\n\nClase {x} RGB DataFrame:\n ",df_rgb_means)
    print(f"\nClass {x} mean matrix: \n", df_rgb_means.mean())
    print(f"\nClass {x} cov matrix: \n", df_rgb_means.cov())

















#     class_num = int(file_path.split('_')[1])-1

#     img = cv2.imread(file_path)

#     non_black = img.any(axis=-1)
#     mean_matrix = np.mean(img[non_black], axis=(0))

#     if np.sum(mean_matrix)>0:
#         mean_matrices[class_num] += mean_matrix


# for i in range(n_classes):
#     mean_matrices[i] /= num_images_per_class[i]


# for i in range(n_classes):
#     print(f'Mean matrix for class {i+1}:')
#     print(mean_matrices[i])
 


# for imageRGB, imageMask in zip(images[0::2], images[1::2]):
#     pixelsRGB, pixelsMask = list(imageRGB.getdata()),list(imageMask.getdata())
#     #pixelsRGB, pixelsMask = imageRGB,imageMask
#     dic[f'Clase_{i%num_classes+1}_RGB'].append(pixelsRGB)
#     dic[f'Clase_{i%num_classes+1}_Mask'].append(pixelsMask)
#     i += 1


# print(len([v for k,v in dic.items() if 'RGB' in k]))

# RGB_class_list = []
# for x in range(num_classes):
#     RGB_class = [v for k,v in dic.items() if f'{x+1}_RGB' in k]
#     Mask_class = [v for k,v in dic.items() if f'{x+1}_Mask' in k]
#     df_rgb_means = []
#     for image in RGB_class:
#         image_array = np.array(image)
#         reshaped_array = image_array.reshape(-1,3)
#         df = pd.DataFrame(reshaped_array, columns=['R','G','B'])
#         df_rgb_means.append(df)

#     df_rgb_means = pd.concat(df_rgb_means)
#     df_rgb_means = df_rgb_means.loc[~(df_rgb_means==0).all(axis=1)]   # Delete rows with '0' in R, G and B
#     RGB_class_list.append([df_rgb_means.mean(),df_rgb_means.cov()])

#     print(f"\n\nClase {x+1} RGB DataFrame:\n ",df_rgb_means)
#     print(f"\nClass {x+1} mean matrix: \n", df_rgb_means.mean())
#     print(f"\nClass {x+1} cov matrix: \n", df_rgb_means.cov())








































# import numpy as np
# import glob
# from PIL import Image

# # Define the classes you want to classify
# classes = ["Fruit1", "Fruit2"]

# # Load the dataset of images composed of masks for each class
# data = []
# for class_name in classes:
#     file_list = glob.glob(f"OutputFruits/All_Masks_*.png")
#     for file_path in file_list:
#         img = Image.open(file_path)
#         data.append((class_name, img))

# # Preprocess the dataset by converting each image to a feature vector
# X = []
# y = []
# for class_name, img in data:
#     pixels = np.array(img)
#     mask = (pixels[:, :, 1] == 255)
#     freqs = np.bincount(mask.ravel(), minlength=2)
#     feature_vec = freqs / np.sum(freqs)
#     X.append(feature_vec)
#     y.append(class_name)
# X = np.array(X)
# y = np.array(y)

# # Split the preprocessed dataset into training and testing sets
# n_train = int(0.8 * len(X))
# X_train, y_train = X[:n_train], y[:n_train]
# X_test, y_test = X[n_train:], y[n_train:]

# # Estimate the prior probability of each class
# prior = {}
# for class_name in classes:
#     prior[class_name] = np.mean(y_train == class_name)

# # Estimate the conditional probability of each pixel value given each class
# cond_prob = {}
# for class_name in classes:
#     mask = (y_train == class_name)
#     freqs = np.mean(X_train[mask], axis=0)
#     cond_prob[class_name] = (freqs + 1) / (np.sum(mask) + 2)  # Laplace smoothing

# # Classify each image in the testing set using Bayesian Naive inference
# y_pred = []
# for i in range(len(X_test)):
#     probs = {}
#     for class_name in classes:
#         p = prior[class_name]
#         for j in range(len(X_test[i])):
#             if X_test[i][j] == 0:
#                 p *= 1 - cond_prob[class_name][j]
#             else:
#                 p *= cond_prob[class_name][j]
#         probs[class_name] = p
#     y_pred.append(max(probs, key=probs.get))

# # Evaluate the performance of the classifier
# print(y_pred)
# accuracy = np.mean(y_pred == y_test)
# print(f"Accuracy: {accuracy}")
