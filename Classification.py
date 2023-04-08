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
import glob
import time
import datetime


def BayesRGB(monkey_fruit):
    
    if monkey_fruit == 1: # monkey images to classify

        with open("BayesianClassifierResults.txt", "a") as output_file:

            start_time = time.time()

            num_classes = 3
            apriori = apriori_probability(1, num_classes)
            mean_matrix, cov_matrix, det_covs, covs_inv = mean_cov_matrix(1, num_classes, apriori)

            classes = {0: 'monkey', 1: 'halo', 2: 'fondo'}
            now = datetime.datetime.now()

            output_file.write("\n\n##############################################################\n\nBayesian classifier for monkey images. \nRan on ")
            output_file.write(now.strftime("%Y-%m-%d %H:%M:%S"))
            output_file.write("\nTrain time: ")
            output_file.write(str(time.time() - start_time))
            output_file.write("\n")

            output_file.write("\n\nTraining: ")
            output_file.write("\n\nPrior probabilities: \n")
            output_file.write("Prior probability of blue class (halo):")
            output_file.write(str(apriori[0]))
            output_file.write("\nPrior probability of green class (monkey):")
            output_file.write(str(apriori[1]))
            output_file.write("\nPrior probability of black class (background):")
            output_file.write(str(apriori[2]))
            
            output_file.write("\n\nMeans: \n")
            output_file.write(str(mean_matrix))
            output_file.write("\n\nCovs: \n")
            output_file.write(str(cov_matrix))
            output_file.write("\n\nDet-covs: \n")
            output_file.write(str(det_covs))
            output_file.write("Inverse-cov: \n")
            output_file.write(str(covs_inv))

            modelo = {
                'classes':classes,
                'apriori': apriori,
                'mean':mean_matrix,
                'cov': cov_matrix,
                'cov_det': det_covs,
                'inverse_cov': covs_inv
            }



            file_pattern = "Monos/Prueba*.jpg"

            file_list = glob.glob(file_pattern)
            # print(file_list)


            for img in file_list:
                img_ = cv2.imread(img)
                new_shape = (600,600)
                resized_img = cv2.resize(img_, new_shape)
                cv2.imwrite(img, resized_img)


            image_data = np.array([np.array(Image.open(file)) for file in file_list])

            # print(image_data[0].shape)



            imgs = np.flip(image_data[:,:,:,0:2], 3)

            prediction = np.zeros_like(image_data[:,:,:,0].astype(np.uint8))

            start_time = time.time()

            output_file.write("\n\nTesting: ")
            # ttime = 0
            for i, image_data_ in enumerate(image_data):
                prediction[i] = predict(image_data_, modelo)
                # ttime = ttime+time_
                # output_file.write("\nTime spent on Bayes discriminator on each image: ")
                # output_file.write(str(ttime))

            # print(prediction)
            
            # output_file.write("\nTotal discriminator time: ")
            # output_file.write(str(ttime))
            output_file.write("\nTest time (image preparation and discriminator): ")
            output_file.write(str(time.time() - start_time))
            # Create images based on prediction

            cont=0
            for x in prediction:
                print(x)
                img = Image.new('RGB', (x.shape[1], x.shape[0]))#, color='black')
                for i in range(x.shape[0]):
                    for j in range(x.shape[1]):
                        if x[i, j] == 1:
                            img.putpixel((j, i), (255, 255, 255))  # white for monkey
                        elif x[i, j] == 0:
                            img.putpixel((j, i), (180, 180, 180))  # grey for halo
                        elif x[i, j] == 2:
                            img.putpixel((j, i), (70, 70, 70))  # darker grey for background

                img.show()
                cont += 1
                img.save(f'resultado_monkey{cont}.png')










    elif monkey_fruit == 2: # fruit images to classify
        with open("BayesianClassifierResults.txt", "a") as output_file:
            start_time = time.time()

            num_classes = 4
            apriori = apriori_probability(2, num_classes)
            mean_matrix, cov_matrix, det_covs, covs_inv = mean_cov_matrix(2, num_classes, apriori)

            now = datetime.datetime.now()

            output_file.write("\n\n##############################################################\n\nBayesian classifier for fruit images. \nRan on ")
            output_file.write(now.strftime("%Y-%m-%d %H:%M:%S"))
            output_file.write("\nTrain time: ")
            output_file.write(str(time.time() - start_time))
            output_file.write("\n")

            output_file.write("\n\nTraining: ")
            output_file.write("\n\nPrior probabilities: \n")
            output_file.write("Prior probability of red class (eggs):")
            output_file.write(str(apriori[0]))
            output_file.write("\nPrior probability of blue class (chillies):")
            output_file.write(str(apriori[1]))
            output_file.write("\nPrior probability of green class (bananas):")
            output_file.write(str(apriori[2]))
            output_file.write("\nPrior probability of black class (background):")
            output_file.write(str(apriori[2]))
            
            output_file.write("\n\nMeans: \n")
            output_file.write(str(mean_matrix))
            output_file.write("\n\nCovs: \n")
            output_file.write(str(cov_matrix))
            output_file.write("\n\nDet-covs: \n")
            output_file.write(str(det_covs))
            output_file.write("Inverse-cov: \n")
            output_file.write(str(covs_inv))


            classes = {0:'banana', 1:'huevo', 2:'chile', 3:'fondo'}

            modelo = {  
                'classes':classes,
                'apriori': apriori,
                'mean':mean_matrix,
                'cov': cov_matrix,
                'cov_det': det_covs,
                'inverse_cov': covs_inv
            }

            file_pattern = "ImagenesPrueba/Prueba*.jpg"

            file_list = glob.glob(file_pattern)
            print(file_list)

            image_data = np.array([np.array(Image.open(file)) for file in file_list])

            imgs = np.flip(image_data[:,:,:,0:2], 3)

            prediction = np.zeros_like(image_data[:,:,:,0].astype(np.uint8))

            start_time = time.time()

            output_file.write("\n\nTesting: ")
            for i, image_data_ in enumerate(image_data):
                prediction[i] = predict(image_data_, modelo)

            
            output_file.write("\nTest time (image preparation and discriminator): ")
            output_file.write(str(time.time() - start_time))
            # Create images based on prediction

            cont=0
            for x in prediction:
                print(x)
                img = Image.new('RGB', (x.shape[1], x.shape[0]))#, color='black')
                for i in range(x.shape[0]):
                    for j in range(x.shape[1]):
                        if x[i, j] == 0:
                            img.putpixel((j, i), (255, 255, 0))  # yellow
                        elif x[i, j] == 1:
                            img.putpixel((j, i), (255, 255, 255))  # white
                        elif x[i, j] == 2:
                            img.putpixel((j, i), (0, 255, 0))  # green
                        elif x[i, j] == 3:
                            img.putpixel((j, i), (255, 0, 0))  # red : background

                img.show()
                cont += 1
                img.save(f'resultado{cont}.png')


    else:
        print("Error")



def apriori_probability(monkey_fruit, num_classes):
    # A priori probability
    
    if monkey_fruit == 1:
        path = "OutputMonkeys/"
        num_classes = num_classes

        # Create an array to store the count of pixels for each class
        class_counts = np.zeros(num_classes)

        # Loop through each training image mask
        for filename in os.listdir(path):
            if "All_Masks_" in filename:
                # Load the image
                img = cv2.imread(os.path.join(path, filename))

                # Extract the mask of each color
                blue_mask = ((img[:, :, 2] == 255).astype(int))   # halo
                green_mask = ((img[:, :, 1] == 255).astype(int))  # monkey
                black_mask = ((img[:,:,-1] == 0).astype(int))     # backggound

                class_counts[0] += np.sum(blue_mask)
                class_counts[1] += np.sum(green_mask)
                class_counts[2] += np.sum(black_mask)

        total_pixels = np.sum(class_counts)
        class_probs = class_counts / total_pixels

        # print("Prior probability of blue class (halo):", class_probs[0])
        # print("Prior probability of green class (monkey):", class_probs[1])
        # print("Prior probability of black class (background):", class_probs[2])

        return class_probs




    elif monkey_fruit == 2:
        path = "OutputFruits/"
        num_classes = num_classes

        # Create an array to store the count of pixels for each class
        class_counts = np.zeros(num_classes)

        # Loop through each training image mask
        for filename in os.listdir(path):
            if "All_Masks_" in filename:
                # Load the image
                img = cv2.imread(os.path.join(path, filename))

                # Extract the mask of each color
                red_mask = ((img[:, :, 0] == 255).astype(int))/3    # eggs
                blue_mask = ((img[:, :, 2] == 255).astype(int))/3   # chillies
                green_mask = ((img[:, :, 1] == 255).astype(int))/2  # bananas
                black_mask = ((img[:,:,-1] == 0).astype(int))/1     # background

                class_counts[0] += np.sum(red_mask)
                class_counts[1] += np.sum(blue_mask)
                class_counts[2] += np.sum(green_mask)
                class_counts[3] += np.sum(black_mask)

        total_pixels = np.sum(class_counts)
        class_probs = class_counts / total_pixels

        print("Prior probability of red class (eggs):", class_probs[0])
        print("Prior probability of blue class (chillies):", class_probs[1])
        print("Prior probability of green class (bananas):", class_probs[2])
        print("Prior probability of black class (background):", class_probs[3])

        return class_probs


def mean_cov_matrix(monkey_fruit, num_classes, apriori):

    if monkey_fruit == 1:
        file_pattern = "OutputMonkeys/All_halo_Masks_*.png*"
        file_list = glob.glob(file_pattern)

        x = 0
        for image in file_list:
            image1 = cv2.imread(image)
            image2 = cv2.imread(f"Monos/Entrenamiento{x+1}.png")
            x+=1
            mask = cv2.inRange(image1, (0,0,0), (0,0,0))

            new = np.zeros_like(image2)
            new[mask!=0] = image2[mask!=0]
            cv2.imwrite(f"OutputMonkeys/Class_3_Background_{x}.png", new)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


        # Obtaining the mean matrix for each class

        file_pattern = "OutputMonkeys/Class_*_*"
        file_list = glob.glob(file_pattern)

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
        n_classes = num_classes

        RGB_class_list_covs = []
        RGB_class_list_means = []
        x=0
        for class_list in num_images_per_class:
            # print(class_list)
            df_rgb_means = []
            for image in class_list:
                image = image.replace("\\", "/")
                image = cv2.imread(image)
                image_array = np.array(image)
                reshaped_array = image_array.reshape(-1,3)
                reshaped_array = reshaped_array[:,[2,1,0]]
                df = pd.DataFrame(reshaped_array, columns=['R','G','B'])
                df_rgb_means.append(df)

            df_rgb_means = pd.concat(df_rgb_means)
            df_rgb_means = df_rgb_means.loc[~(df_rgb_means==0).all(axis=1)]   # Delete rows with '0' in R, G and B
            
            x+=1

            # print(f"\n\nClase {x} RGB DataFrame:\n ",df_rgb_means)
            # print(f"\nClass {x} mean matrix: \n", df_rgb_means.mean())
            # print(f"\nClass {x} cov matrix: \n", df_rgb_means.cov())

            rgb_means = df_rgb_means.to_numpy()

            cov_matrix = np.cov(rgb_means, rowvar=False)
            mean_matrix = np.mean(rgb_means, axis=0)



            RGB_class_list_covs.append(cov_matrix)
            RGB_class_list_means.append(mean_matrix)


        # print("Covs: \n\n", RGB_class_list_covs)
        # print("\n\nMeans: \n\n", RGB_class_list_means)


        # Covariance Matrix

        det_covs = np.array([np.linalg.det(RGB_class_list_covs[k]) for k in range(n_classes)])

        covs_inv = np.zeros_like(RGB_class_list_covs)
        for k in range(n_classes):
            covs_inv[k] = np.linalg.inv(np.matrix(RGB_class_list_covs[k]))


        # print("\n\ndet-covs", det_covs,"\n\n\n")
        # print("inverse-cov", covs_inv, "\n\n\n")

        return RGB_class_list_means, RGB_class_list_covs, det_covs, covs_inv      




    if monkey_fruit == 2:

        # Obtaining the images of only the background, to get the mean of background class. 

        file_pattern = "OutputFruits/All_Masks_*"
        file_list = glob.glob(file_pattern)

        x = 0
        for image in file_list:
            image1 = cv2.imread(image)
            image2 = cv2.imread(f"ImagenesEntrenamiento/Entrenamiento{x+1}.png")
            x+=1
            mask = cv2.inRange(image1, (0,0,0), (0,0,0))

            new = np.zeros_like(image2)
            new[mask!=0] = image2[mask!=0]
            cv2.imwrite(f"OutputFruits/Class_4_Background_{x}.png", new)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


        # Obtaining the mean matrix for each class

        file_pattern = "OutputFruits/Class_*_*"
        file_list = glob.glob(file_pattern)

        c1 = []
        c2 = []
        c3 = []
        c4 = []
        for image in file_list:
            if "Class_1" in image:
                c1.append(image)
            elif "Class_2" in image:
                c2.append(image)
            elif "Class_3" in image:
                c3.append(image)
            elif "Class_4" in image:
                c4.append(image)

        num_images_per_class = [c1, c2, c3, c4]
        n_classes = num_classes

        RGB_class_list_covs = []
        RGB_class_list_means = []
        x=0
        for class_list in num_images_per_class:
            print(class_list)
            df_rgb_means = []
            for image in class_list:
                image = image.replace("\\", "/")
                image = cv2.imread(image)
                image_array = np.array(image)
                reshaped_array = image_array.reshape(-1,3)
                reshaped_array = reshaped_array[:,[2,1,0]]
                df = pd.DataFrame(reshaped_array, columns=['R','G','B'])
                df_rgb_means.append(df)

            df_rgb_means = pd.concat(df_rgb_means)
            df_rgb_means = df_rgb_means.loc[~(df_rgb_means==0).all(axis=1)]   # Delete rows with '0' in R, G and B
            
            x+=1

            print(f"\n\nClase {x} RGB DataFrame:\n ",df_rgb_means)
            print(f"\nClass {x} mean matrix: \n", df_rgb_means.mean())
            print(f"\nClass {x} cov matrix: \n", df_rgb_means.cov())

            rgb_means = df_rgb_means.to_numpy()

            cov_matrix = np.cov(rgb_means, rowvar=False)
            mean_matrix = np.mean(rgb_means, axis=0)



            RGB_class_list_covs.append(cov_matrix)
            RGB_class_list_means.append(mean_matrix)


        print("Covs: \n\n", RGB_class_list_covs)
        print("\n\nMeans: \n\n", RGB_class_list_means)


        # Covariance Matrix

        det_covs = np.array([np.linalg.det(RGB_class_list_covs[k]) for k in range(n_classes)])

        covs_inv = np.zeros_like(RGB_class_list_covs)
        for k in range(n_classes):
            covs_inv[k] = np.linalg.inv(np.matrix(RGB_class_list_covs[k]))


        print("\n\ndet-covs", det_covs,"\n\n\n")
        print("inverse-cov", covs_inv, "\n\n\n")

        return RGB_class_list_means, RGB_class_list_covs, det_covs, covs_inv



def bayes_disc(vector, mean, inverse_cov, cov_det, class_P):
    step = np.matrix(vector-mean)
    inverse_cov = np.matrix(inverse_cov)
    step2 = np.matmul(step, inverse_cov)
    step3 = (step2.dot(step.T))[0,0].item()
    bayes_disc = -(1.0/2.0)*step3-(1.0/2.0)*np.log(cov_det)+np.log(class_P)
    return bayes_disc.item()

def predict(image_data, model):
    image_data_shape = image_data.shape
    prediction = np.empty_like(image_data[:,:,0]).astype(np.uint8)
    discr = np.empty(len(model['classes'])).astype(np.float64)
    for i in range(image_data_shape[0]):
        for j in range(image_data_shape[1]):
            for k in model['classes']:
                vector=image_data[i,j]
                mean=model['mean'][k]
                cov=model['cov'][k]
                inverse_cov=model['inverse_cov'][k]
                cov_det=model['cov_det'][k]
                class_P=model['apriori'][k]
                discr[k] = bayes_disc(vector, mean, inverse_cov, cov_det, class_P)
            prediction[i,j] = discr.argmax()
    return prediction


