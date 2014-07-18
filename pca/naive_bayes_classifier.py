"""
@created_at 2014-07-17
@author Exequiel Fuentes <efulet@gmail.com>
@author Brian Keith <briankeithn@gmail.com>

Basado en el trabajo de Juan Bekios-Calfa <juan.bekios@ucn.cl>
"""

# Se recomienda seguir los siguientes estandares:
#   1. Para codificacion: PEP 8 - Style Guide for Python Code (http://legacy.python.org/dev/peps/pep-0008/)
#   2. Para documentacion: PEP 257 - Docstring Conventions (http://legacy.python.org/dev/peps/pep-0257/)

import math

import numpy as np

from naive_bayes_classifier_exception import NaiveBayesClassifierException


class NaiveBayesClassifier:
    """Esta clase abstracta define los metodos que deben ser implementados por un
    clasificador bayesiano binario.
    """

    def fit(self, training_set, training_set_classes):
        """Este metodo entrena el clasificador bayesiano.
            :param training_set: Conjunto de valores de ejemplos de entrenamiento.
            :param training_set_classes: Conjunto de clases a las que pertenecen los ejemplos de entrenamiento.
        """
        raise NotImplementedError

    def predict(self, testing_set):
        """Este metodo utiliza el clasificador ya entrenado para clasificar un conjunto de datos de prueba.
            :param testing_set: Conjunto de valores de prueba.
            :returns Conjunto de valores de clases asociados a cada elemento del conjunto de prueba.
        """
        raise NotImplementedError

    def score(self, testing_set, testing_set_classes):
        """Este metodo calcula la precision que tiene el clasificador sobre un conjunto de datos de prueba.
            :param testing_set: Conjunto de valores de prueba.
            :param training_set_classes: Conjunto de clases a las que pertenecen los datos de prueba.
            :returns La precision como un valor entre 0 y 1.
        """
        raise NotImplementedError


class FKNaiveBayesClassifier(NaiveBayesClassifier):
    """Esta clase implementa los metodos de una clasificador bayesiano.
    """

    def __init__(self):
        """Crea una instancia de la clase FKNaiveBayesClassifier.
        """
        self._covariance_matrix = None
        self._p_positive = None
        self._p_negative = None
        self._mu_positive = None
        self._mu_negative = None

    def fit(self, training_set, training_set_classes):
        n = len(training_set)

        # Se separan los elementos positivos de los negativos.
        lda_data_positive = training_set[training_set_classes == 1]
        lda_data_negative = training_set[training_set_classes == 0]

        # Se estima la probabilidad a priori (p_negative se obtendria con el complemento)
        p_positive = float(len(lda_data_positive)) / len(training_set)

        # Se estima la distribucion que siguen, se asume que es Gaussiana.
        mu_positive = np.mean(lda_data_positive)
        mu_negative = np.mean(lda_data_negative)
        var_negative = 0
        var_positive = 0

        # Se calcula la matriz de covarianzas...
        for i in xrange(0, len(lda_data_negative)):
            var_negative += (lda_data_negative[i] - mu_negative) * (lda_data_negative[i] - mu_negative).T
        for i in xrange(0, len(lda_data_positive)):
            var_positive += (lda_data_positive[i] - mu_positive) * (lda_data_positive[i] - mu_positive).T
        covar_matrix = (var_positive + var_negative) / (n - 2)

        # Se asigna los valores a los atributos.
        self._covariance_matrix = covar_matrix
        self._p_positive = p_positive
        self._p_negative = 1 - p_positive
        self._mu_positive = mu_positive
        self._mu_negative = mu_negative

    def predict(self, testing_set):
        # Se verifica que el clasificador haya sido entrenado antes.
        if self._covariance_matrix is None:
            raise NaiveBayesClassifierException("Debe entrenar el clasificador primero.")

        # Se inicializan las variables requeridas por el clasificador.
        n = len(testing_set)
        y_predicted = [None] * n
        delta_mu = self._mu_positive - self._mu_negative
        sum_mu = self._mu_positive + self._mu_negative

        # Por cada elemento en el conjunto de pruebas se calcula el discriminante correspondiente.
        for i in range(0, n):
            # Se calcula el lado izquierdo de la comparacion.
            lhs = testing_set[i].T * 1 / self._covariance_matrix * delta_mu
            # Se calcula el lado derecho de la comparacion.
            rhs = 0.5 * sum_mu.T * 1 / self._covariance_matrix * delta_mu - math.log(
                self._p_positive / self._p_negative)
            # Se marca con un 1 si es que lhs > rhs y se marca con un 0 en caso contrario.
            criteria = int(lhs > rhs)
            y_predicted[i] = criteria

        # Se retorna el conjunto de predicciones para cada caso de prueba.
        return y_predicted

    def score(self, testing_set, testing_set_classes):
        # Se verificar que el clasificador haya sido entrenado.
        if self._covariance_matrix is None:
            raise NaiveBayesClassifierException("Debe entrenar el clasificador primero.")

        # Se calcula la precision.
        testing_pred = self.predict(testing_set)
        mislabeled_points = (testing_pred != testing_set_classes).sum()
        score = 1 - float(mislabeled_points) / len(testing_set)

        #Se retorna el valor calculado.
        return score