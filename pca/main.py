"""
@created_at 2014-07-15
@author Exequiel Fuentes <efulet@gmail.com>
@author Brian Keith <briankeithn@gmail.com>
"""

# Se recomienda seguir los siguientes estandares:
#   1. Para codificacion: PEP 8 - Style Guide for Python Code (http://legacy.python.org/dev/peps/pep-0008/)
#   2. Para documentacion: PEP 257 - Docstring Conventions (http://legacy.python.org/dev/peps/pep-0257/)

import os
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
    mu = np.mean(lda_data_positive)
    sigma = np.std(lda_data_positive)
    y_positive = pylab.mlab.normpdf(bins, mu, sigma)
    #Otra normal...
    mu = np.mean(lda_data_negative)
    sigma = np.std(lda_data_negative)
    y_negative = pylab.mlab.normpdf(bins, mu, sigma)
    #Clases...
    pylab.plot(bins, y_positive, 'r--')
    pylab.plot(bins, y_negative, 'b--')
    #Etiquetas...
    pylab.xlabel('LDA')
    pylab.ylabel('$P(LDA|DIABETES)$')
    pylab.show()


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
    graph_information(lda_train)

    # Clasificar Bayes
    gnb = GaussianNB()
    gnb.fit(lda_train, y_train)
    y_pred = gnb.predict(lda_test)

    print("Number of mislabeled points : %d" % (y_test != y_pred).sum())
    print("Accuracy: ", gnb.score(lda_test, y_test))

