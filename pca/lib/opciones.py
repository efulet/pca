"""
@created_at 2014-07-15
@author Exequiel Fuentes <efulet@gmail.com>
@author Brian Keith <briankeithn@gmail.com>
"""

# Se recomienda seguir los siguientes estandares:
#   1. Para codificacion: PEP 8 - Style Guide for Python Code (http://legacy.python.org/dev/peps/pep-0008/)
#   2. Para documentacion: PEP 257 - Docstring Conventions (http://legacy.python.org/dev/peps/pep-0257/)

from argparse import ArgumentParser


class Opciones:
    """Esta clase ayuda a manejar los argumentos que pueden ser pasados por la 
    linea de comandos.
    
    Por ejemplo, escriba:
    $> ./bin/pca.sh --help
    """
    
    def __init__(self):
        self.parser = ArgumentParser(usage='/bin/pca.sh [--help]')
        self._init_parser()
    
    def _init_parser(self):
        raise NotImplementedError
        
    def parse(self, args=None):
        return self.parser.parse_args(args)
