Segundo Taller de Sistemas Inteligentes
======================================

Problema
--------
Se desea construir un sistema que sea capaz de diagnosticar la presencia de 
diabetes en un conjunto de pacientes (para el desarrollo de clases se utilizara
un conjunto de pruebas de scikit-learn).


Desarrollo
----------
El trabajo consiste en construir un modelo matematico capaz de predecir si un 
paciente tiene la enfermedad de diabetes o no. Para esto se utilizara un conjunto 
de datos que seran utilizados para la construccion y validacion del modelo. Los 
pasos para el desarrollo del experimento son los siguientes:

  1. Carga de los datos
  2. Seleccion de datos para el entrenamiento del modelo y pruebas
  3. Construccion del subespacio PCA
  4. Construccion del subespacio LDA sobre el espacio PCA
  5. Construir un clasificador utilizando del teorema de Bayes


Se pide
-------
1. Construir un grafico de frecuencias seleccionando diferentes dimensiones PCA
2. Construir un grafico de densidad de probabilidades, ajustada a una normal, 
para las diferentes dimensiones PCA
3. Construir un grafico P(DIABETES='Positivo'|LDA) y P(DIABETES='Negativo'|LDA) 
por cada una de las dimensiones PCA y sin considerar proyecciones LDA
4. Implementar un clasificador de Bayes utilizando la materia vista en clases 
y explicarlo paso a paso. Ademas comparar los resultados con el implementado 
por sklearn.


Como ejecutar el programa
-------------------------
Este programa puede ejecutarse de la siguiente manera:

  $> ./bin/pca.sh

  O

  $> python main.py
