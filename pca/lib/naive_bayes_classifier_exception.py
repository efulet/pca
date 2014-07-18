"""
@created_at 2014-07-17
@author Exequiel Fuentes <efulet@gmail.com>
@author Brian Keith <briankeithn@gmail.com>
"""

# Se recomienda seguir los siguientes estandares:
#   1. Para codificacion: PEP 8 - Style Guide for Python Code (http://legacy.python.org/dev/peps/pep-0008/)
#   2. Para documentacion: PEP 257 - Docstring Conventions (http://legacy.python.org/dev/peps/pep-0257/)


class NaiveBayesClassifierException(Exception):
    """NaiveBayesClassifierException maneja las excepciones para la clase NaiveBayesClassifier.
    
    Como usar esta clase:
      raise NaiveBayesClassifierException("Arreglo fuera de los limites")
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

