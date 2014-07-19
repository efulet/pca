"""
@created_at 2014-07-17
@author Exequiel Fuentes <efulet@gmail.com>
@author Brian Keith <briankeithn@gmail.com>

Basado en el trabajo de Juan Bekios-Calfa <juan.bekios@ucn.cl>
"""

# Se recomienda seguir los siguientes estandares:
#   1. Para codificacion: PEP 8 - Style Guide for Python Code (http://legacy.python.org/dev/peps/pep-0008/)
#   2. Para documentacion: PEP 257 - Docstring Conventions (http://legacy.python.org/dev/peps/pep-0257/)

import numpy as np


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
        self._var_positive = None
        self._var_negative = None
        self._p_positive = None
        self._p_negative = None
        self._mu_positive = None
        self._mu_negative = None

    def fit(self, training_set, training_set_classes):
        # Se separan los elementos positivos de los negativos.
        lda_data_positive = training_set[training_set_classes == 1]
        lda_data_negative = training_set[training_set_classes == 0]

        # Se estiman las medias.
        self._mu_positive = np.mean(lda_data_positive)
        self._mu_negative = np.mean(lda_data_negative)

        # Se estiman las varianzas...
        self._var_positive = np.var(lda_data_positive)
        self._var_negative = np.var(lda_data_negative)

        # Se estima la probabilidad a priori (p_negative se obtendria con el complemento)
        self._p_positive = float(len(lda_data_positive)) / len(training_set)
        self._p_negative = 1 - self._p_positive

    def predict(self, testing_set):
        # Se inicializan las variables requeridas por el clasificador.
        n = len(testing_set)
        y_predicted = [None] * n

        # Clase positiva...
        log_p_positive = np.log(self._p_positive)
        pdf_positive = - 0.5 * np.sum(np.log(np.pi * self._var_positive))
        pdf_positive -= 0.5 * np.sum(((testing_set - self._mu_positive) ** 2) /
                                     self._var_positive, 1)
        positive_discriminant = log_p_positive + pdf_positive

        # Clase negativa...
        log_p_negative = np.log(self._p_negative)
        pdf_negative = - 0.5 * np.sum(np.log(np.pi * self._var_negative))
        pdf_negative -= 0.5 * np.sum(((testing_set - self._mu_negative) ** 2) /
                                     self._var_negative, 1)
        negative_discriminant = log_p_negative + pdf_negative

        # Se retorna el conjunto de predicciones para cada caso de prueba.
        return [int(i) for i in positive_discriminant > negative_discriminant]

    def score(self, testing_set, testing_set_classes):
        # Se calcula la precision.
        testing_pred = self.predict(testing_set)
        mislabeled_points = (testing_pred != testing_set_classes).sum()
        score = 1 - float(mislabeled_points) / len(testing_set)

        #Se retorna el valor calculado.
        return score

