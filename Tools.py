import numpy as np
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
from PIL import Image, ImageFilter
import cv2
import numpy as np
import os



def GaussianFilter(image):
	image_obj = Image.open(image)  
	image_obj = image_obj.filter(ImageFilter.GaussianBlur(radius=5))
	image_obj.save(image.rstrip(".jpg") + ".png")
	return image_obj




img = None

drawing = False
points = []
contours = []


def draw(event, x, y, flags, param):
	    global drawing, points, img, contours

	    if event == cv2.EVENT_LBUTTONDOWN:
	        drawing = True
	        points = [(x, y)]

	    elif event == cv2.EVENT_MOUSEMOVE:
	        if drawing == True:
	            cv2.line(img, points[-1], (x, y), (0, 0, 255), 2)
	            points.append((x, y))

	    elif event == cv2.EVENT_LBUTTONUP:
	        drawing = False
	        points.append((x, y))
	        contour = np.array(points)
	        contours.append(contour)
	        cv2.drawContours(img, [contour], 0, (0, 0, 0), 2)


def CropClasses(image):	
	global drawing, points, img, contours

	cv2.namedWindow('image')
	img = cv2.imread(image)
	# Bind the mouse event function to the window
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
	    mask = np.zeros(img.shape[:2], np.uint8)
	    cv2.drawContours(mask, [contour], 0, 255, -1)
	    result = cv2.bitwise_and(img, img, mask=mask)
	    x, y, w, h = cv2.boundingRect(contour)
	    cropped_image = result[y:y+h, x:x+w]
	    
	    # Create a mask with only the cropped area
	    mask = np.zeros(cropped_image.shape[:2], np.uint8)
	    cv2.drawContours(mask, [contour - contour.min(axis=0)], 0, 255, -1)
	    
	    # Apply the mask to the cropped image
	    masked_cropped_image = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)
	    
	    # Create binary mask with cropped region as white
	    binary_mask = np.zeros(img.shape[:2], np.uint8)
	    binary_mask[y:y+h, x:x+w] = mask
	    
	    # Save cropped image and binary mask
	    cv2.imwrite(f"output_image_{i+1}.png", masked_cropped_image)
	    cv2.imwrite(f"output_mask_{i+1}.png", binary_mask)

	# Close all windows
	cv2.destroyAllWindows()




###
#
#
# Aplicar filtro gaussiano a las imágenes
#
# Luego hacer segmentacion manual en las fotos de entrenamiento:
# 		- Loop para que se abran todas las imagenes de entrenamiento
#		- Loop para seleccionar el numero de clases y los recortes por cada clase
#		- Guardar los recortes de cada clase como imagenes RGB y blanco y negro, guardar las máscaras binarias
# 
# Utilizar las imágenes RGB para determinar la media, covarianza, varianza de los colores para cada clase.
#
# Utilizar las imágenes con filtro gaussiano para hacer la clasificación?
#
# 
#
#
###
