# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 19:02:29 2021

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




print(__doc__)

# Mostrar registros de progreso en stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


# #############################################################################
# Descargue los datos, si aún no están en el disco, y cárguelos como matrices numpy

lfw_people = fetch_lfw_people(min_faces_per_person=50, resize=0.4)

# introspección de las matrices de imágenes para encontrar las formas (para trazar)
n_samples, h, w = lfw_people.images.shape

# para el aprendizaje automático usamos los 2 datos directamente (como píxel relativo
# información de posiciones es ignorada por este modelo)
X = lfw_people.data
n_features = X.shape[1]

# la etiqueta a predecir es el id de la persona
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


# #############################################################################
# Dividir en un conjunto de entrenamiento y un conjunto de prueba usando un pliegue k estratificado

# dividido en un conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.2, random_state=42)


# #############################################################################
# Calcule un PCA (caras propias) en el conjunto de datos faciales (tratado como sin etiqueta
# conjunto de datos): extracción de características / reducción de dimensionalidad sin supervisión
n_components = 500
print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))

'''
plt.figure(figsize=(8,6))
plt.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')
'''


# #############################################################################
# Entrenar un modelo de clasificación SVM

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(
    SVC(kernel='rbf', class_weight='balanced'), param_grid
)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)


# #############################################################################
# Evaluación cuantitativa de la calidad del modelo en el conjunto de prueba

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))



# #############################################################################
# Evaluación cualitativa de las predicciones usando matplotlib

# trazar el resultado de la predicción en una parte del conjunto de prueba
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
       plt.subplot(n_row, n_col, i + 1)
       plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
       plt.title(titles[i], size=12)
       plt.xticks(())
       plt.yticks(())
        
        
prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)


# trazar la galería de las caras propias más significativas
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()


from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
print("Accuracy: ", score)
