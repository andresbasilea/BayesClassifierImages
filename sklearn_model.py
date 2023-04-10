from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import cv2
######################
#######Platanos#######
######################
imagenP1 = Image.open('OutputFruits/Class_1_Contour_1_1.png')
imagenP2 = Image.open('OutputFruits/Class_1_Contour_2_1.png')
pixelesP1 = list(imagenP1.getdata())
pixelesP2 = list(imagenP2.getdata())
# Convierte la lista de tuplas en un DataFrame de pandas
columnas = ['R', 'G', 'B']
df1 = pd.DataFrame(pixelesP1, columns=columnas)
df1 = df1.loc[~(df1 == 0).all(axis=1)]
df2 = pd.DataFrame(pixelesP2, columns=columnas)
df2 = df2.loc[~(df2 == 0).all(axis=1)]
# Unir los dos DataFrames de forma vertical
df_platanos = pd.concat([df1, df2], ignore_index=True)
#Agrega una columna
df_platanos['objeto'] = 'platano'
#print(df_platanos)


######################
#######Huevos#######
######################
imagenH1 = Image.open('OutputFruits/Class_2_Contour_1_1.png')
imagenH2 = Image.open('OutputFruits/Class_2_Contour_2_1.png')
imagenH3 = Image.open('OutputFruits/Class_2_Contour_3_1.png')
pixelesH1 = list(imagenH1.getdata())
pixelesH2 = list(imagenH2.getdata())
pixelesH3 = list(imagenH3.getdata())
# Convierte la lista de tuplas en un DataFrame de pandas
columnas = ['R', 'G', 'B']
dfH1 = pd.DataFrame(pixelesH1, columns=columnas)
dfH1 = dfH1.loc[~(dfH1 == 0).all(axis=1)]
dfH2 = pd.DataFrame(pixelesH2, columns=columnas)
dfH2 = dfH2.loc[~(dfH2 == 0).all(axis=1)]
dfH3 = pd.DataFrame(pixelesH3, columns=columnas)
dfH3 = dfH3.loc[~(dfH3 == 0).all(axis=1)]
# Unir los dos DataFrames de forma vertical
df_huevos = pd.concat([dfH1, dfH2, dfH3 ], ignore_index=True)
#Agrega una columna
df_huevos['objeto'] = 'huevo'
#print(df_huevos)


######################
#######Chiless#######
######################
imagenC1 = Image.open('OutputFruits/Class_3_Contour_1_1.png')
imagenC2 = Image.open('OutputFruits/Class_3_Contour_2_1.png')
imagenC3 = Image.open('OutputFruits/Class_3_Contour_3_1.png')
pixelesC1 = list(imagenC1.getdata())
pixelesC2 = list(imagenC2.getdata())
pixelesC3 = list(imagenC3.getdata())
# Convierte la lista de tuplas en un DataFrame de pandas
columnas = ['R', 'G', 'B']
dfC1 = pd.DataFrame(pixelesC1, columns=columnas)
dfC1 = dfC1.loc[~(dfC1 == 0).all(axis=1)]
dfC2 = pd.DataFrame(pixelesC2, columns=columnas)
dfC2 = dfC2.loc[~(dfC2 == 0).all(axis=1)]
dfC3 = pd.DataFrame(pixelesC3, columns=columnas)
dfC3 = dfC3.loc[~(dfC3 == 0).all(axis=1)]
# Unir los dos DataFrames de forma vertical
df_chiles = pd.concat([dfC1, dfC2, dfC3 ], ignore_index=True)
#Agrega una columna
df_chiles['objeto'] = 'chile'
#print(df_chiles)

######################
#######Fondo#######
######################
imagenF1 = Image.open('OutputFruits/Class_4_Background_1.png')
pixelesF1 = list(imagenF1.getdata())
# Convierte la lista de tuplas en un DataFrame de pandas
columnas = ['R', 'G', 'B']
dfF1 = pd.DataFrame(pixelesF1, columns=columnas)
df_fondo = dfF1.loc[~(dfF1 == 0).all(axis=1)]
df_fondo.reset_index(drop=True, inplace=True)
#Agrega una columna
df_fondo['objeto'] = 'fondo'
#print(df_fondo)


#Unir todos los dataframes
df_entrenamiento = pd.concat([df_platanos, df_huevos, df_chiles, df_fondo ], ignore_index=True)
#print(df_entrenamiento)

########
#Medias#
########
# Filtrar las filas que corresponden a la clase 'platano'
platano_df = df_entrenamiento.loc[df_entrenamiento['objeto'] == 'platano']
rgb_df_P = platano_df.iloc[:, :-1]
mediaP = tuple(rgb_df_P.mean().tolist())
mediaP = tuple(int(valor) for valor in mediaP)
print(mediaP)

# Filtrar las filas que corresponden a la clase 'Huevo'
huevo_df = df_entrenamiento.loc[df_entrenamiento['objeto'] == 'huevo']
rgb_df_H = huevo_df.iloc[:, :-1]
mediaH = tuple(rgb_df_H.mean().tolist())
mediaH = tuple(int(valor) for valor in mediaH)
print(mediaH)

# Filtrar las filas que corresponden a la clase 'Chile'
chile_df = df_entrenamiento.loc[df_entrenamiento['objeto'] == 'chile']
rgb_df_C = chile_df.iloc[:, :-1]
mediaC = tuple(rgb_df_C.mean().tolist())
mediaC = tuple(int(valor) for valor in mediaC)
print(mediaC)

# Filtrar las filas que corresponden a la clase 'fondo'
fondo_df = df_entrenamiento.loc[df_entrenamiento['objeto'] == 'fondo']
rgb_df_F = fondo_df.iloc[:, :-1]
mediaF = tuple(rgb_df_F.mean().tolist())
mediaF = tuple(int(valor) for valor in mediaF)
print(mediaF)

#Entrenamiento
X = df_entrenamiento.iloc[:, :-1]
y = df_entrenamiento.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Crear una instancia de la clase GaussianNB
clf = GaussianNB()
# Entrenar el modelo
clf.fit(X_train, y_train)


# Obtener las predicciones del conjunto de pruebas
y_pred = clf.predict(X_test)

precision = precision_score(y_test, y_pred, average='macro')

# calcular la sensibilidad
recall = recall_score(y_test, y_pred, average='macro')

# calcular la exactitud
accuracy = accuracy_score(y_test, y_pred)

# calcular el F1-score
f1 = f1_score(y_test, y_pred, average='macro')

print("Precisión:", precision)
print("Sensibilidad:", recall)
print("Exactitud:", accuracy)
print("F1-score:", f1)

###########################
# Clasificar un solo pixel#
###########################
# pixel = pd.DataFrame({'R': [212], 'G': [184], 'B': [54]})
# prediccion = clf.predict(pixel)


img = cv2.imread('Prueba1_fruits.jpg')
color_list = []
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        # Obtener los valores RGB del píxel actual
        b, g, r = img[i,j]
        
        # Usar el modelo clasificador para predecir la clase del píxel actual
        pixel = pd.DataFrame({'R': [r], 'G': [g], 'B': [b]})
        clase_predicha = clf.predict(pixel)
        
        # Agregar una tupla con el color correspondiente a la lista de tuplas
        if clase_predicha == 'platano':
            color_list.append(mediaP)  # amarillo
        elif clase_predicha == 'chile':
            color_list.append(mediaC)  # verde
        elif clase_predicha == 'huevo':
            color_list.append(mediaH)  # blanco
        else:
            color_list.append(mediaF)  # rojo

# Crear una nueva imagen con los colores clasificados
#nueva_img = np.array(color_list).reshape(img.shape)
print("Ya con la lista")
# Mostrar la imagen clasificada
imagen = Image.new('RGB', (img.shape[0], img.shape[1]))
imagen.putdata(color_list)
imagen.show()
imagen.save('imagen_nueva_sklean.jpg')