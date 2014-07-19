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

from lib import *


def check_version():
    """Python v2.7 es requerida por el curso, entonces verificamos la version"""
    if sys.version_info[:2] != (2, 7):
        raise Exception("Parece que python v2.7 no esta instalado en el sistema")

def db_path():
    """Retorna el path de las base de datos"""
    pathfile = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(pathfile, "db")


if __name__ == "__main__":
    try:
        # Verificar version de python
        check_version()
        
        # Cargar los datos
        my_pca_lda = FKSkLearn(os.path.join(db_path(), "datos_diabetes.npz"))
        
        # Preparar los datos para validacion
        my_pca_lda.fk_train_test_split()
        
        # Se entrena el clasificador PCA + LDA con la dimension optima.
        my_pca_lda.fk_pca_lda()
        
        # Contruye el clasificar Bayes usando la libreria sklearn
        my_pca_lda.fk_bayes_classifier()
        
        print("**************")
        print("sklearn_Bayes:")
        print("Number of mislabeled points : %d" % (my_pca_lda.fk_get_y_test() != my_pca_lda.fk_get_y_pred()).sum())
        print("Accuracy: ", my_pca_lda.fk_score())
        print("**************")
        
        # Implementacion propia del clasificador.
        fknb = FKNaiveBayesClassifier()
        fknb.fit(my_pca_lda.fk_get_lda_train(), my_pca_lda.fk_get_y_train())
        y_pred_FK = fknb.predict(my_pca_lda.fk_get_lda_test())
        
        print("FK_Bayes")
        print("Number of mislabeled points : %d" % (my_pca_lda.fk_get_y_test() != y_pred_FK).sum())
        print("Accuracy: ", fknb.score(my_pca_lda.fk_get_lda_test(), my_pca_lda.fk_get_y_test()))
        print("**************")
        
        # Esto es para verificar que las predicciones son iguales, deberia entregar una lista vacia.
        print("...probando igualdad...")
        y_pred_SK = [int(i) for i in my_pca_lda.fk_get_y_pred()]
        #print y_pred_SK
        #print y_pred_FK
        
        # Se verifica si la lista esta vacia.
        if y_pred_SK == y_pred_FK:
            print "Son iguales los dos metodos!"
        else:
            print "No son iguales. :("
        
        # Se grafica la informacion.
        graph = Graph(my_pca_lda.fk_get_lda_train(), my_pca_lda.fk_get_y_train())
        graph.frequencies_histogram()
        graph.probability_density_functions()
        graph.conditional_probability(my_pca_lda.fk_get_lda_train(), my_pca_lda.fk_get_y_prob())
        graph.show_graphs()
        
    except Exception, err:
        print traceback.format_exc()
    finally:
        sys.exit()

