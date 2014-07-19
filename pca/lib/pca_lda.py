"""
@created_at 2014-07-19
@author Exequiel Fuentes <efulet@gmail.com>
@author Brian Keith <briankeithn@gmail.com>
"""

# Se recomienda seguir los siguientes estandares:
#   1. Para codificacion: PEP 8 - Style Guide for Python Code (http://legacy.python.org/dev/peps/pep-0008/)
#   2. Para documentacion: PEP 257 - Docstring Conventions (http://legacy.python.org/dev/peps/pep-0257/)

import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.naive_bayes import GaussianNB

from pca_lda_exception import FKSkLearnException

class FKSkLearn():
    """Esta clase encapsula los metodos de la libreria sklearn para calcular 
    PCA+LDA.
    """

    def __init__(self, db_path):
        """Crea una instancia de la clase PCA_LDA.
        
        :param db_path: La ruta a la base de datos.
        """
        d = np.load(db_path)
        self.data = d['data']
        self.labels = d['labels']
        
        self._dimensions = self.data.shape[1]
        
        self._x_train, self._x_test, self._y_train, self._y_test = (None,)*4
        
        self._lda_train, self._lda_test = (None,)*2
        
        self._y_pred, self._y_prob = (None,)*2
        
        self._gnb = None
        
        self._test_size = 0.3
        self._random_state = 0
    
    def fk_get_lda_train(self):
        """Retorna lda_train
        """
        return self._lda_train
    
    def fk_get_lda_test(self):
        """Retorna lda_test
        """
        return self._lda_test
    
    def fk_get_y_train(self):
        """Retorna y_train
        """
        return self._y_train
    
    def fk_get_y_test(self):
        """Retorna y_test
        """
        return self._y_test
    
    def fk_get_y_pred(self):
        """Retorna y_pred
        """
        return self._y_pred
    
    def fk_get_y_prob(self):
        """Retorna y_prob
        """
        return self._y_prob
    
    def fk_train_test_split(self):
        """Encapsula el metodo train_test_split
        """
        self._x_train, self._x_test, self._y_train, self._y_test = train_test_split(self.data, self.labels, test_size=self._test_size, random_state=self._random_state)
    
    def fk_pca_lda(self):
        """Construye el subespacio PCA y LDA encontrando la dimension optima de PCA
        """
        k_opt = self._find_optimal_dimension()
        print "Dimension Optima:", k_opt
        #k_opt = 6
        
        self._lda_train, self._lda_test = self._pca_lda(k_opt)
    
    def fk_bayes_classifier(self):
        """Contruye el clasificar Bayes
        """
        if self._lda_train is None:
            raise FKSkLearnException("lda_train no fue calculada.")
        
        if self._lda_test is None:
            raise FKSkLearnException("lda_test no fue calculada.")
        
        if self._y_train is None:
            raise FKSkLearnException("y_train no fue calculada.")
        
        self._gnb = GaussianNB()
        self._gnb.fit(self._lda_train, self._y_train)
        
        self._y_pred = self._gnb.predict(self._lda_test)
        self._y_prob = self._gnb.predict_proba(self._lda_train)
    
    def fk_score(self):
        """Encapsula el metodo score
        """
        if self._gnb is None:
            raise FKSkLearnException("Objecto GaussianNB es nulo.")
        
        if self._lda_test is None:
            raise FKSkLearnException("lda_test no fue calculada.")
        
        if self._y_test is None:
            raise FKSkLearnException("y_test no fue calculada.")
        
        return self._gnb.score(self._lda_test, self._y_test)
    
    def _pca_lda(self, components):
        """Construye el subespacio PCA y LDA.
        
        :param components: Valor entero para calcular la dimension
        """
        if components is None or not isinstance(components, int):
            raise FKSkLearnException("components no puede ser nulo o diferente de entero.")
        
        if self._x_train is None:
            raise FKSkLearnException("x_train no fue calculada.")
        
        if self._x_test is None:
            raise FKSkLearnException("x_test no fue calculada.")
        
        if self._y_train is None:
            raise FKSkLearnException("y_train no fue calculada.")
        
        # Construir subespacio PCA con el valor optimo obtenido.
        pca = PCA(n_components=components)
        
        # --- Construyo PCA a partir del conjunto de entrenamiento
        pca.fit(self._x_train)
        pca_train = pca.transform(self._x_train)
        
        # --- Proyecto al subespacio PCA con los datos de prueba
        pca_test = pca.transform(self._x_test)
        
        # Construir el subespacio LDA
        clf = LDA()
        
        # --- Construyo el subespacio LDA a partir del conjunto de entrenamiento proyectado PCA
        clf.fit(pca_train, self._y_train)
        lda_train = clf.transform(pca_train)
        
        # --- Proyecto al subespacio LDA los datos de prueba proyectados del subespacio PCA
        lda_test = clf.transform(pca_test)
        
        return lda_train, lda_test
    
    def _find_optimal_dimension(self):
        """Encuentra la dimension optima. Basado en el algoritmo creado 
        por Juan Bekios-Calfa <juan.bekios@ucn.cl>
        """
        if self._y_train is None:
            raise FKSkLearnException("y_train no fue calculada.")
        
        if self._y_test is None:
            raise FKSkLearnException("y_test no fue calculada.")
        
        r = 0  # Classification rate
        k = 1  # Number of components
        
        for n_components in xrange(1, self._dimensions + 1):
            # Entrenar PCA+LDA con la cantidad de componentes dada.
            lda_train, lda_test = self._pca_lda(n_components)
            
            # Clasificar Bayes
            gnb = GaussianNB()
            gnb.fit(lda_train, self._y_train)
            r_i = gnb.score(lda_test, self._y_test)
            
            if r_i > r:
                r = r_i
                k = n_components
        
        return k

