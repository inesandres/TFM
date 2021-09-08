# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 18:04:45 2021

@author: Inés Andrés
"""
from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC



# #############################################################################
# Descargue los datos, si aún no están en el disco, y cárguelos como matrices numpy

lfw_people = fetch_lfw_people(min_faces_per_person=80)




# introspección de las matrices de imágenes para encontrar las formas (para trazar)
print("Size of data set: \n") 
samples, h, w = lfw_people.images.shape

# para el aprendizaje automático usamos los 2 datos directamente (como píxel relativo
# información de posiciones es ignorada por este modelo)
X = lfw_people.data
features = X.shape[1]
faces = X.shape[0]

# la etiqueta a predecir es el id de la persona
y = lfw_people.target
target_names = lfw_people.target_names
classes = target_names.shape[0]

print("Total dataset size:")
print("samples: %d" % samples)
print("features: %d" % features)
print("classes: %d" % classes)


# #############################################################################
# Dividir en un conjunto de entrenamiento y un conjunto de prueba usando un pliegue k estratificado

# dividido en un conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0)



# #############################################################################
# Calcule un PCA (caras propias) en el conjunto de datos faciales (tratado como sin etiqueta
# conjunto de datos): extracción de características / reducción de dimensionalidad sin supervisión
# PCA computation 

    
n_components =500
print("Computing PCA.....")
t0 = time()
pca = PCA(n_components=n_components, whiten=True)
pca.fit(X_train)      #standardizing the training data to mean =0 & variance =1 
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("Time taken to compute PCA: %f sec" % (time() - t0))


#MLP classification
from sklearn.neural_network import MLPClassifier

t0 = time()
print("Fitting the classifier to the training set")
clf = MLPClassifier(hidden_layer_sizes=(1024,), batch_size=256, verbose=False, 
                    early_stopping=True).fit(X_train_pca, y_train)
 
print("Time taken to train MLP: %f sec" % (time() - t0))
time0 = time()
y_pred = clf.predict(X_test_pca)

print("Time taken by MLP to predict: %f sec" % (time() - t0))
print("classfication report:")
print(classification_report(y_test, y_pred, target_names=target_names))
print("confusion matrix:")
print(confusion_matrix(y_test, y_pred, labels=range(classes)))




def titles(y_pred, y_test, target_names):
    for i in range(y_pred.shape[0]):
        pred_name = target_names[y_pred[i]].split(' ')[-1]
        true_name = target_names[y_test[i]].split(' ')[-1]
        yield 'predicted: {0}\ntrue: {1}'.format(pred_name, true_name)

def plot_images(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
       plt.subplot(n_row, n_col, i + 1)
       plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
       plt.title(titles[i], size=12)
       plt.xticks(())
       plt.yticks(())
        
prediction_titles = list(titles(y_pred, y_test, target_names))
plot_images(X_test, prediction_titles, h, w)

'''
# # Plotting eigen faces
height,width = lfw_people.images.shape
eigenfaces = pca.components_.reshape((n_components, height, width))



eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_images(eigenfaces, eigenface_titles, h, w)

plt.show()

'''
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
print("Accuracy: ", score)


 
