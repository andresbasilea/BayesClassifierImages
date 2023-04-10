import cv2
import numpy as np

# Load the image
image = cv2.imread('ImagenValidacionModelo/All_Masks_1.png')

# Create a mask that selects only the green pixels with the exact RGB value of 0, 255, 0
mask = cv2.inRange(image, np.array([0, 255, 0]), np.array([0, 255, 0]))

# Extract the green pixels from the image using the mask
green_pixels = cv2.bitwise_and(image, image, mask=mask)

# Convert the green pixels to a numpy array
green_pixels_array = np.asarray(green_pixels)

# Reshape the array to a 2D matrix with one row for each pixel and three columns for R,G,B values
green_pixels_array = green_pixels_array.reshape((-1, 3))

# Save the green pixels as ground truth or true labels


# # Load the image
# image = cv2.imread('resultado_monkey3.png')

# # Create a mask that selects only the green pixels with the exact RGB value of 0, 255, 0
# mask = cv2.inRange(image, np.array([255, 255, 255]), np.array([180, 180, 180]))

# # Extract the green pixels from the image using the mask
# green_pixels_2 = cv2.bitwise_and(image, image, mask=mask)

# # Convert the green pixels to a numpy array
# green_pixels_array_2 = np.asarray(green_pixels_2)

# # Reshape the array to a 2D matrix with one row for each pixel and three columns for R,G,B values
# green_pixels_array_2 = green_pixels_array.reshape((-1, 3))

# print(green_pixels_array_2)
# np.savetxt('ground_truth_labels.txt', green_pixels_array_2)


# # Load the image
# image = cv2.imread('resultado_monkey3.png')

# # Define the lower and upper color ranges for the two colors you want to extract
# lower_color_range_1 = np.array([255, 255, 255])
# upper_color_range_1 = np.array([255, 255, 255])
# lower_color_range_2 = np.array([180, 180, 180])
# upper_color_range_2 = np.array([180, 180, 180])

# # Create a mask that selects only the pixels with the two color ranges
# mask1 = cv2.inRange(image, lower_color_range_1, upper_color_range_1)
# mask2 = cv2.inRange(image, lower_color_range_2, upper_color_range_2)
# mask = cv2.bitwise_or(mask1, mask2)

# # Extract the pixels from the image using the mask
# pixels = cv2.bitwise_and(image, image, mask=mask)

# # Convert the pixels to a numpy array
# pixels_array = np.asarray(pixels)

# # Reshape the array to a 2D matrix with one row for each pixel and three columns for R,G,B values
# pixels_array = pixels_array.reshape((-1, 3))

# # Save the pixels as ground truth or true labels
# np.savetxt('ground_truth_labels.txt', pixels_array)




### IMAGEN RESULTADO DEL MODELO 


image = cv2.imread('resultado_monkey3.png')

# Define the lower and upper color ranges for the two colors you want to extract
lower_color_range_1 = np.array([255, 255, 255])
upper_color_range_1 = np.array([255, 255, 255])
lower_color_range_2 = np.array([180, 180, 180])
upper_color_range_2 = np.array([180, 180, 180])

# Create a mask that selects only the pixels with the two color ranges
mask1 = cv2.inRange(image, lower_color_range_1, upper_color_range_1)
mask2 = cv2.inRange(image, lower_color_range_2, upper_color_range_2)
mask = cv2.bitwise_or(mask1, mask2)

# Extract the pixels from the image using the mask
pixels = cv2.bitwise_and(image, image, mask=mask)

# Create a binary mask with 1s for pixels with the two color ranges and 0s for others
binary_mask = np.zeros(image.shape[:2], dtype=np.uint8)
binary_mask[mask > 0] = 1

# Save the binary mask as ground truth or true labels
np.savetxt('model_labels.txt', binary_mask.reshape((-1, 1)), fmt='%d')




### IMAGEN GROUND TRUTH


image = cv2.imread('ImagenValidacionModelo/All_Masks_1.png')


# Define the lower and upper color range for the color you want to extract
lower_color_range = np.array([0, 255, 0])
upper_color_range = np.array([0, 255, 0])

# Create a mask that selects only the pixels with the desired color range
mask = cv2.inRange(image, lower_color_range, upper_color_range)

# Extract the pixels from the image using the mask
pixels = cv2.bitwise_and(image, image, mask=mask)

# Create a binary mask with 1s for pixels with the desired color and 0s for others
binary_mask = np.zeros(image.shape[:2], dtype=np.uint8)
binary_mask[mask > 0] = 1

# Save the binary mask as ground truth or true labels
np.savetxt('ground_truth_labels.txt', binary_mask.reshape((-1, 1)), fmt='%d')




# # Load the ground truth labels for the green pixel image and the 255,255,255/180,180,180 pixel image
# gt_other = np.loadtxt('ground_truth_labels.txt')
# model_output = np.loadtxt('model_labels.txt')

# # Calculate True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN)
# TP = np.sum(np.logical_and(gt_other == 1, model_output == 1))
# FP = np.sum(np.logical_and(gt_other == 0, model_output == 1))
# TN = np.sum(np.logical_and(gt_other == 0, model_output == 0))
# FN = np.sum(np.logical_and(gt_other == 1, model_output == 0))

# # Calculate precision, recall, F1 score, and accuracy
# precision = TP / (TP + FP)
# recall = TP / (TP + FN)
# f1_score = 2 * precision * recall / (precision + recall)
# accuracy = (TP + TN) / (TP + TN + FP + FN)

# # Print the results
# print('Precision:', precision)
# print('Recall:', recall)
# print('F1 Score:', f1_score)
# print('Accuracy:', accuracy)



# import numpy as np
# from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# # load ground truth values from text file
# gt = np.loadtxt('ground_truth_labels.txt')
# print(gt.shape)
# # load predicted values from text file
# pred = np.loadtxt('model_labels.txt')
# print(pred.shape)

# # flatten the arrays to compare individual values
# gt_flat = gt.flatten()
# pred_flat = pred.flatten()

# # calculate metrics
# precision = precision_score(gt_flat, pred_flat)
# recall = recall_score(gt_flat, pred_flat, zero_division=1)

# f1 = f1_score(gt_flat, pred_flat)
# accuracy = accuracy_score(gt_flat, pred_flat)

# # print the results
# print('Precision:', precision)
# print('Recall:', recall)
# print('F1 Score:', f1)
# print('Accuracy:', accuracy)








import numpy as np

# Load ground truth and predicted labels from txt files
gt_labels = np.loadtxt('ground_truth_labels.txt')
pred_labels = np.loadtxt('model_labels.txt')

print(gt_labels, pred_labels)
print(gt_labels.shape, pred_labels.shape)
# Calculate true positives, false positives, false negatives, and true negatives
tp = np.sum(np.logical_and(gt_labels == 1, pred_labels == 1))
fp = np.sum(np.logical_and(gt_labels == 0, pred_labels == 1))
fn = np.sum(np.logical_and(gt_labels == 1, pred_labels == 0))
tn = np.sum(np.logical_and(gt_labels == 0, pred_labels == 0))

print(tp, fp, fn, tn)

# Calculate precision, recall, accuracy, and F1-score
precision = tp / (tp + fp)
recall = tp / (tp + fn)
accuracy = (tp + tn) / (tp + fp + tn + fn)
f1_score = 2 * precision * recall / (precision + recall)

# Print the metrics
print('Precision:', precision)
print('Recall:', recall)
print('Accuracy:', accuracy)
print('F1-score:', f1_score)

