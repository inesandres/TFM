# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 20:59:14 2021

@author: Inés Andrés
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import (KNeighborsClassifier,
                               NeighborhoodComponentsAnalysis)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_lfw_people
from sklearn.neural_network import MLPClassifier

print(__doc__)

lfw_people = fetch_lfw_people(min_faces_per_person=100)


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



n_neighbors = 10
random_state = 0

# Load Digits dataset
X, y = datasets.load_digits(return_X_y=True)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y,
                     random_state=random_state)

dim = len(X[0])
n_classes = len(np.unique(y))

# Reduce dimension to 2 with PCA
pca = make_pipeline(StandardScaler(),
                    PCA(n_components=2, random_state=random_state))

# Reduce dimension to 2 with LinearDiscriminantAnalysis
lda = make_pipeline(StandardScaler(),
                    LinearDiscriminantAnalysis(n_components=2))

# Reduce dimension to 2 with NeighborhoodComponentAnalysis
nca = make_pipeline(StandardScaler(),
                    NeighborhoodComponentsAnalysis(n_components=2,
                                                   random_state=random_state))

# Use a nearest neighbor classifier to evaluate the methods
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
#clf = MLPClassifier(hidden_layer_sizes=(1024,), batch_size=256, verbose=False, 
#                    early_stopping=True)

# Make a list of the methods to be compared
dim_reduction_methods = [('PCA', pca), ('LDA', lda), ('NCA', nca)]

# plt.figure()
for i, (name, model) in enumerate(dim_reduction_methods):
    plt.figure()
    # plt.subplot(1, 3, i + 1, aspect=1)

    # Fit the method's model
    model.fit(X_train, y_train)

    # Fit a nearest neighbor classifier on the embedded training set
    knn.fit(model.transform(X_train), y_train)
    #clf.fit(model.tranform(X_train, y_train))

    # Compute the nearest neighbor accuracy on the embedded test set
    acc_knn = knn.score(model.transform(X_test), y_test)
    #acc_clf = clf.score(model.transform(X_test), y_test)

    # Embed the data set in 2 dimensions using the fitted model
    X_embedded = model.transform(X)

    # Plot the projected points and show the evaluation score
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, s=30, cmap='Set1')
    plt.title("{}, KNN (k={})\nTest accuracy = {:.2f}".format(name,
                                                              n_neighbors,
                                                              acc_knn))
#    plt.title("{}, MLP (k={})\nTest accuracy = {:.2f}".format(name,                                                            
 #                                                             acc_knn))
    
plt.show()