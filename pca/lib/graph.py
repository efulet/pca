"""
@created_at 2014-07-17
@author Exequiel Fuentes <efulet@gmail.com>
@author Brian Keith <briankeithn@gmail.com>

Basado en el trabajo de Juan Bekios-Calfa <juan.bekios@ucn.cl>
"""

# Se recomienda seguir los siguientes estandares:
#   1. Para codificacion: PEP 8 - Style Guide for Python Code (http://legacy.python.org/dev/peps/pep-0008/)
#   2. Para documentacion: PEP 257 - Docstring Conventions (http://legacy.python.org/dev/peps/pep-0257/)

import pylab
import numpy as np

from graph_exception import GraphException


class Graph:
    """Esta clase implementa los graficos solicitados en el taller
    para los resultados del clasificador.
    """

    def __init__(self, original_data, data_classes):
        """Este constructor carga los datos requeridis por la clase Graph.
            :param original_data: Conjunto de valores a graficar.
            :param data_classes: Conjunto de valores de clases asociados a cada elemento del conjunto a graficar.
        """
        # Se cargan los datos originales y ademas los pertenecientes a cada clase.
        self._data = original_data
        self._data_positive = original_data[data_classes == 1]
        self._data_negative = original_data[data_classes == 0]
        
        # Se inicializan en vacio las variables requeridas para el funcionamiento del grafico de probabilidades.
        self._n, self._bins, self._patches = (None,)*3

    def frequencies_histogram(self):
        """Este metodo construye un histograma de frecuencias en base los datos cargados.
        """
        # Grafico de frecuencias del conjunto LDA por clase
        # http://matplotlib.org/examples/pylab_examples/histogram_demo_extended.html
        pylab.figure()
        self._n, self._bins, self._patches = pylab.hist([self._data_positive, self._data_negative], 30, histtype='bar',
                                                        color=['red', 'blue'],
                                                        label=['$\mathit{positivo}$', '$\mathit{negativo}$'])
        pylab.legend()
        pylab.xlabel('Caracteristicas LDA')
        pylab.ylabel('Frecuencia')
        pylab.title('Histograma de frecuencias LDA: Conjunto de entrenamiento')

    def probability_density_functions(self):
        """Este metodo construye el grafico de las funciones de densidad de probabilidad
        en base los datos cargados.
        """
        if self._bins is None:
            raise GraphException("Debe calcular el histograma primero.")

        # --- Probabilidades
        pylab.figure()
        #print bins
        #print n
        #prob_positive = n[0]/np.sum(n[0])
        #prob_negative = n[1]/np.sum(n[1])
        #pylab.bar(left=bins[:-1], height=prob_positive, width=0.1, bottom=None, hold=None, color='red', alpha=0.5)
        #pylab.bar(left=bins[:-1], height=prob_negative, width=0.1, bottom=None, hold=None, color='blue', alpha=0.5)
        # add a 'best fit' line
        mu_positive = np.mean(self._data_positive)
        sigma_positive = np.std(self._data_positive)
        y_positive = pylab.mlab.normpdf(self._bins, mu_positive, sigma_positive)
        #Otra normal...
        mu_negative = np.mean(self._data_negative)
        sigma_negative = np.std(self._data_negative)
        y_negative = pylab.mlab.normpdf(self._bins, mu_negative, sigma_negative)
        #Clases...
        pylab.plot(self._bins, y_positive, 'r--', label='$\mathcal{P}(LDA|D^+)$')
        pylab.plot(self._bins, y_negative, 'b--', label='$\mathcal{P}(LDA|D^-)$')
        #Etiquetas...
        pylab.xlabel('$LDA$')
        pylab.ylabel('$P(LDA|DIABETES)$')
        pylab.legend()
    
    def conditional_probability(self, x, y):
        """Este metodo construye el grafico de las funciones de probabilidad usando Bayes.
        """
        # --- Probabilidades
        pylab.figure()
        
        #Clases...
        pylab.scatter(x, y[:,0], color='blue', label='$\mathcal{P}(D^-|LDA)$')
        pylab.scatter(x, y[:,1], color='red', label='$\mathcal{P}(D^+|LDA)$')
        
        #Etiquetas...
        pylab.xlabel('$LDA$')
        pylab.ylabel('$P(DIABETES|LDA)$')
        pylab.legend()
        pylab.grid()
    
    def show_graphs(self):
        """Despliega los graficos
        """
        pylab.show()

