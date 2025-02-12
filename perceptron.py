# import math
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator
from sklearn.model_selection import train_test_split
# from sklearn.manifold import TSNE
# from sklearn.utils import shuffle
# from mpl_toolkits.mplot3d import Axes3D
# import pandas as pd
# from scipy import stats
# from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPClassifier
import numpy as np
import os
import csv

def print_sep():
    print("=================================================================")

# función para importar las series temporales
# se añade la serie temporal al vector de entrenamiento
# se añade al vector de etiquetas un 0 si la serie temporal
# no contiene ningún error ó un 1 si contiene algún error
def import_time_series(path,error):
    files = os.listdir(path)
    for f in files:
        with open(path + '/' + f, newline='',encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                X.append(row)
                y.append(error)  

# función para crear, entrenar y evaluar un perceptrón
def perceptron(layers):
    mlp = MLPClassifier(solver="lbfgs",max_iter=50000,random_state=42,hidden_layer_sizes=layers)
    mlp.fit(X_train, y_train)

    print("Rendimiento en entenamiento: {:.2f}".format(mlp.score(X_train, y_train)))
    print("Rendimiento en el conjunto de prueba: {:.2f}".format(mlp.score(X_test, y_test)))
    print("Rendimiento en el conjunto total: {:.2f}".format(mlp.score(X, y)))
    print('') 

sensors = ['S1', 'S3', 'S4']

for s in sensors:
    X = []
    y = []

    # importación de series temporales del sensor dado
    import_time_series(path='Series temporales/DNI/' + s + '/series/validas',error=0)
    import_time_series(path='Series temporales/DNI/' + s + '/series/no validas',error=1)

    print_sep()
    print("Sensor " + s)
    ind_del = [] # índices de los elementos a elimar
    for i in range(len(X)):
        if ('' in X[i]): # una serie temporal se elimina si un valor se encuentra vacío
            ind_del.append(i)

    len_antes = len(X)
    print("numero de series validas:", sum([v == 0 for v in y]))
    print("numero de series no validas:", sum([v == 1 for v in y]))

    X = np.delete(X,ind_del,0)
    y = np.delete(y,ind_del)
    X = np.array(X).astype(float) # conversión del conjunto de entrenamiento a numpy array
    y = np.array(y)

    len_despues = len(X)
    print(len_antes-len_despues, "instancias eliminadas") # número de instancias eliminadas
    print("numero de series validas:", sum([v == 0 for v in y]))
    print("numero de series no validas:", sum([v == 1 for v in y]))
    print('')

    print("Forma X:", X.shape)
    print("Forma y:", y.shape)
    print('')

    # división del conjunto de datos en entrenamiento y test
    # esta función baraja el conjunto de datos por defecto
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2345)

    perceptron((10,10))
