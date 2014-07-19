"""
@created_at 2014-07-15
@author Exequiel Fuentes <efulet@gmail.com>
@author Brian Keith <briankeithn@gmail.com>
"""

# Se recomienda seguir los siguientes estandares:
#   1. Para codificacion: PEP 8 - Style Guide for Python Code (http://legacy.python.org/dev/peps/pep-0008/)
#   2. Para documentacion: PEP 257 - Docstring Conventions (http://legacy.python.org/dev/peps/pep-0257/)

import os
import traceback
import sys

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.naive_bayes import GaussianNB

from lib import *


def check_version():
    """Python v2.7 es requerida por el curso, entonces verificamos la version"""
    if sys.version_info[:2] != (2, 7):
        raise Exception("Parece que python v2.7 no esta instalado en el sistema")

def db_path():
    """Retorna el path de las base de datos"""
    pathfile = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(pathfile, "db")


def pca_lda(x_train, x_test, y_train, components):
    # Construir subespacio PCA con el valor optimo obtenido.
    pca = PCA(n_components=components)
    # --- Construyo PCA a partir del conjunto de entrenamiento
    pca.fit(x_train)
    pca_train = pca.transform(x_train)
    # --- Proyecto al subespacio PCA con los datos de prueba
    pca_test = pca.transform(x_test)
    # Construir el subespacio LDA
    clf = LDA()
    # --- Construyo el subespacio LDA a partir del conjunto de entrenamiento proyectado PCA
    clf.fit(pca_train, y_train)
    lda_train = clf.transform(pca_train)
    # --- Proyecto al subespacio LDA los datos de prueba proyectados del subespacio PCA
    lda_test = clf.transform(pca_test)
    return lda_train, lda_test


def find_optimal_dimension(x_train, x_test, y_train, y_test, dimensions):
    r = 0  # Classification rate
    k = 1  # Number of components
    for n_components in xrange(1, dimensions + 1):
        # Entrenar PCA+LDA con la cantidad de componentes dada.
        lda_train, lda_test = pca_lda(x_train, x_test, y_train, n_components)
        # Clasificar Bayes
        gnb = GaussianNB()
        gnb.fit(lda_train, y_train)
        r_i = gnb.score(lda_test, y_test)
        if r_i > r:
            r = r_i
            k = n_components
    return k


if __name__ == "__main__":
    try:
        # Verificar version de python
        check_version()
        
        # Cargar los datos
        datos_diabetes_path = os.path.join(db_path(), "datos_diabetes.npz")
    
        d = np.load(datos_diabetes_path)
        data = d['data']
        labels = d['labels']
    
        dimensions = data.shape[1]
    
        # Preparar los datos para validacion
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=0)
    
        # Se en-cuentra la dimension optima de PCA.
        k_opt = find_optimal_dimension(x_train, x_test, y_train, y_test, dimensions)
        print "Dimension Optima:", k_opt
        # Se entrena el clasificador PCA + LDA con la dimension optima.
        lda_train, lda_test = pca_lda(x_train, x_test, y_train, k_opt)
    
        # Clasificar Bayes
        gnb = GaussianNB()
        gnb.fit(lda_train, y_train)
        y_pred = gnb.predict(lda_test)
        y_prob = gnb.predict_proba(lda_test)
        
        # Se grafica la informacion.
        graph = Graph(lda_train, y_train)
        graph.frequencies_histogram()
        graph.probability_density_functions()
        graph.conditional_probability(lda_test, y_prob)
        
        print("**************")
        print("sklearn_Bayes:")
        print("Number of mislabeled points : %d" % (y_test != y_pred).sum())
        print("Accuracy: ", gnb.score(lda_test, y_test))
        print("**************")
        
        # Implementacion propia del clasificador.
        fknb = FKNaiveBayesClassifier()
        fknb.fit(lda_train, y_train)
        y_pred_FK = fknb.predict(lda_test)
        #p_positive, mu_positive, mu_negative, variance = naive_bayes_train(lda_train, y_train)
        #y_pred_FK = naive_bayes_classifier(lda_test, p_positive, mu_positive, mu_negative, variance)
        print("FK_Bayes")
        print("Number of mislabeled points : %d" % (y_test != y_pred_FK).sum())
        print("Accuracy: ", fknb.score(lda_test, y_test))
        print("**************")

        #print zip(y_pred, y_pred_FK)
        # Esto es para verificar que las predicciones son iguales, deberia entregar una lista vacia.
        print("...probando igualdad...")
        prueba = lda_test[[int(i) for i in y_pred] != y_pred]
        # Se verifica si la lista esta vacia.
        if not prueba:
            print "Son iguales los dos metodos!"
        else:
            print "No son iguales. :("
    except Exception, err:
        print traceback.format_exc()
    finally:
        sys.exit()

