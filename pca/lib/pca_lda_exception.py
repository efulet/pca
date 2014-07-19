"""
@created_at 2014-07-19
@author Exequiel Fuentes <efulet@gmail.com>
@author Brian Keith <briankeithn@gmail.com>
"""

# Se recomienda seguir los siguientes estandares:
#   1. Para codificacion: PEP 8 - Style Guide for Python Code (http://legacy.python.org/dev/peps/pep-0008/)
#   2. Para documentacion: PEP 257 - Docstring Conventions (http://legacy.python.org/dev/peps/pep-0257/)


class FKSkLearnException(Exception):
    """FKSkLearnException maneja las excepciones para la clase FKSkLearn.
    
    Como usar esta clase:
      raise FKSkLearnException("Arreglo fuera de los limites")
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

