"""
@created_at 2014-07-15
@author Exequiel Fuentes <efulet@gmail.com>
@author Brian Keith <briankeithn@gmail.com>
"""

# Se recomienda seguir los siguientes estandares:
#   1. Para codificacion: PEP 8 - Style Guide for Python Code (http://legacy.python.org/dev/peps/pep-0008/)
#   2. Para documentacion: PEP 257 - Docstring Conventions (http://legacy.python.org/dev/peps/pep-0257/)

import os
import math

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.lda import LDA
import pylab
from sklearn.naive_bayes import GaussianNB


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


def graph_information(lda_train):
    # Grafico de frecuencias del conjunto LDA por clase
    # http://matplotlib.org/examples/pylab_examples/histogram_demo_extended.html
    lda_data_positive = lda_train[y_train == 1]
    lda_data_negative = lda_train[y_train == 0]
    # --- Frecuencias
    pylab.figure()
    n, bins, patches = pylab.hist([lda_data_positive, lda_data_negative], 30, histtype='bar', color=['red', 'blue'],
                                  label=['positivo', 'negativo'])
    pylab.legend()
    pylab.xlabel('Caracteristicas LDA')
    pylab.ylabel('Frecuencia')
    pylab.title('Histograma de frecuencias LDA: Conjunto de entrenamiento')
    # --- Probabilidades
    pylab.figure()
    #print bins
    #print n
    #prob_positive = n[0]/np.sum(n[0])
    #prob_negative = n[1]/np.sum(n[1])
    #pylab.bar(left=bins[:-1], height=prob_positive, width=0.1, bottom=None, hold=None, color='red', alpha=0.5)
    #pylab.bar(left=bins[:-1], height=prob_negative, width=0.1, bottom=None, hold=None, color='blue', alpha=0.5)
    # add a 'best fit' line
    mu_positive = np.mean(lda_data_positive)
    sigma_positive = np.std(lda_data_positive)
    y_positive = pylab.mlab.normpdf(bins, mu_positive, sigma_positive)
    #Otra normal...
    mu_negative = np.mean(lda_data_negative)
    sigma_negative = np.std(lda_data_negative)
    y_negative = pylab.mlab.normpdf(bins, mu_negative, sigma_negative)
    #Clases...
    pylab.plot(bins, y_positive, 'r--')
    pylab.plot(bins, y_negative, 'b--')
    #Etiquetas...
    pylab.xlabel('LDA')
    pylab.ylabel('$P(LDA|DIABETES)$')
    pylab.show()


def naive_bayes_train(lda_train, y_train):
    #todo: Hacer esto en una clase. (Validar que el input no este vacio?)
    # Formulas obtenidas de: http://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf

    n = len(lda_train)
    # Se separan los elementos positivos de los negativos.
    lda_data_positive = lda_train[y_train == 1]
    lda_data_negative = lda_train[y_train == 0]

    # Se estima la probabilidad a priori (p_negative se obtendria con el complemento)
    p_positive = float(len(lda_data_positive)) / len(lda_train)

    # Se estima la distribucion que siguen, se asume que es Gaussiana.
    mu_positive = np.mean(lda_data_positive)
    mu_negative = np.mean(lda_data_negative)
    var_negative = 0
    var_positive = 0
    for i in xrange(0, len(lda_data_negative)):
        var_negative += (lda_data_negative[i] - mu_negative) * (lda_data_negative[i] - mu_negative).T
    for i in xrange(0, len(lda_data_positive)):
        var_positive += (lda_data_positive[i] - mu_positive) * (lda_data_positive[i] - mu_positive).T
    covar_matrix = (var_positive + var_negative) / (n - 2)
    #print math.sqrt(covar_matrix)
    #sigma_positive = np.std(lda_data_positive)
    #sigma_negative = np.std(lda_data_negative)
    #print sigma_positive, sigma_negative
    # Cuando se pase a una clase estos serian mejor como atributos, y este seria el constructor...
    return p_positive, mu_positive, mu_negative, covar_matrix


def naive_bayes_classifier(lda_train, p_positive, mu_positive, mu_negative, variance):
    #todo: Hacer esto en una clase. (Validar que el input no este vacio?)
    n = len(lda_train)
    y_predicted = [None] * n
    delta_mu = mu_positive - mu_negative
    sum_mu = mu_positive + mu_negative
    for i in range(0, n):
        lhs = lda_train[i].T * 1 / variance * delta_mu
        rhs = 0.5 * sum_mu.T * 1 / variance * delta_mu - math.log(p_positive / (1 - p_positive))
        criteria = int(lhs > rhs)
        y_predicted[i] = criteria
    return y_predicted


if __name__ == "__main__":
    # Cargar los datos
    datos_diabetes_path = os.path.join(db_path(), "datos_diabetes.npz")

    d = np.load(datos_diabetes_path)
    data = d['data']
    labels = d['labels']

    dimensions = data.shape[1]

    # Preparar los datos para validacion
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=0)

    # Se encuentra la dimension optima de PCA.
    k_opt = find_optimal_dimension(x_train, x_test, y_train, y_test, dimensions)

    # Se entrena el clasificador PCA + LDA con la dimension optima.
    lda_train, lda_test = pca_lda(x_train, x_test, y_train, k_opt)

    # Se grafica la informacion.
    # graph_information(lda_train)

    # Clasificar Bayes
    gnb = GaussianNB()
    gnb.fit(lda_train, y_train)
    y_pred = gnb.predict(lda_test)
    print("**************")
    print("sklearn_Bayes:")
    print("Number of mislabeled points : %d" % (y_test != y_pred).sum())
    print("Accuracy: ", gnb.score(lda_test, y_test))
    print("**************")

    # Implementacion propia del clasificador.
    p_positive, mu_positive, mu_negative, variance = naive_bayes_train(lda_train, y_train)
    y_pred_FK = naive_bayes_classifier(lda_test, p_positive, mu_positive, mu_negative, variance)
    print("FK_Bayes")
    mislabeled_points = (y_test != y_pred_FK).sum()
    print("Number of mislabeled points : %d" % mislabeled_points)
    print("Accuracy: ", 1 - float(mislabeled_points) / len(lda_test))
    print("**************")

    # Esto es para verificar que las predicciones son iguales, deberia entregar una lista vacia.
    print("...probando igualdad...")
    prueba = lda_test[y_pred_FK != y_pred]
    # Se verifica si la lista esta vacia.
    if not prueba:
        print "Son iguales los dos metodos!"
    else:
        print "No son iguales. :("