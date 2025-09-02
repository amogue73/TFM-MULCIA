# Copyright 2025 Alejandro Moreno Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from torch.utils.data import Dataset
import os
import csv
from sklearn.model_selection import train_test_split


PATH_D = "Series temporales/DNI/"
NUM_S = 3
SENSORS = ["S1", "S3", "S4"]

def get_series():
    """Devuelve la lista total de días válidos e inválidos para cada
       sensor.

    Returns:
        list: una matriz, donde:
            dim. 0: número de sensor
            dim. 1: válidos/inválidos
    """
    series = []
    for i in range(NUM_S):
        validos = sorted(os.listdir(PATH_D + SENSORS[i] + "/series/validas"))
        invalidos = sorted(os.listdir(PATH_D + SENSORS[i] + "/series/no validas"))
        series.append([validos,invalidos])
    return series

def search_day(year,day,series):
    """Indica qué sensores presentan una serie válida para el día indicado.

    Args:
        year (int): el año del día
        day (int): el número de día
        series (list): matriz del tipo
            que devuelve get_series()

    Returns:
        param (list): lista de 3 elementos. Cada elemento indica si la serie es
            válida para el sensor de esa posición. 0: válido, 1: inválido
            ejemplo: [1,0,0] Inválido para S1, válido para S3 y S4.

    """
    name = str(year) + '-' + str(day)
    param = []
    for i in range(NUM_S):
        for j in range(2):
            if name in series[i][j]:
                param.append(float(j))
    return param


def reshape_series(series):
    """ Devuelve el dataset en un formato diferente.

    Args:
        series (list): dataset en formato devuelto por get_series()

    Returns:
        series_reshaped (list): Lista total de días donde cada elemento es 
            un día seguido por la lista devuelta por search_day() para ese
            día.
    """
    series_reshaped = []
    for y in range(2013,2020):
        for d in range(1,366):
            labels = search_day(y,d,series)
            if len(labels) == NUM_S:
                series_reshaped.append([str(y)+'-'+str(d),labels])

    return series_reshaped


def get_points(path):
    """ Devuelve la lista de puntos para el archivo de una serie temporal.

    Args:
        path (str): Ruta del archivo de la serie temporal

    Returns:
        y (list): Lista de puntos de la serie temporal
        
    """
    with open(path, newline='',encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if not '' in row:
                y = [float(x) for x in row]
            else:
                y = []
                for x in row:
                    y.append(float('2000')) if x == '' else y.append(float(x))
    return y

def get_weights(train_series,coef=1):
    """Calcula los pesos necesarios para la función de pérdida
    de binary cross entropy

    Estos pesos son iguales al número de ejemplos negativos
    por cada positivo. Entendiendo como negativo: válido y
    positivo: inválido.

    Args:
        train_series: El dataset de entrenamiento, en formato
            igual al devuelto por reshape_series()
        coef: Coeficiente por el que se multiplican los pesos.
            Este parámetro es útil para modificar el balance
            precission-recall logrado por el modelo. Si coef>1,
            aumenta el recall. Si coef<1, aumenta la precission.

    Returns:
        lista de pesos, tiene tres elemenentos. Cada peso se
            calcula para cada sensor.
    """
    labels = np.zeros((len(train_series),NUM_S),dtype=float)
    for i in range(labels.shape[0]):
        labels[i] = train_series[i][1]

    labels = torch.tensor(labels)
    # Cuenta positivos y negativos por clase
    num_pos = labels.sum(dim=0)
    num_neg = labels.shape[0] - num_pos

    # Calcula el pos_weight
    pos_weight = num_neg / num_pos
    pos_weight = pos_weight
    return coef*pos_weight

class dni(Dataset):
    """Dataset de pytorch de las series temporales de radiación directa

    Attributes:
        root_dir (str): La ruta base donde se encuentran las series temporales
        files (list): lista de los nombres de los archivos de las series
            temporales. Debe estar en el mismo formato que el devuelto por
            reshape_series()
    """
    def __init__(self, root_dir, files, device):
        self.root_dir = root_dir
        self.files = files
        self.device = device

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        # series = []
        # items = []
        # for i in range(self.memory):
        #     ind = index - i
        #     if ind >= 0:
        #         items.append(self.files[index])
        #     else:
        #         items.append(None)

        series = []
        item = self.files[index]

        for i in range(NUM_S):
            # if item != None:
            #     path = self.root_dir + SENSORS[i] + "/series" 
            #     path += ("/validas" if item[1][i] == 0 else "/no validas")
            #     path += "/" + item[0]
            #     series.append(get_points(path))
            # else:
            #     series.append([0.0]*144)
            path = self.root_dir + SENSORS[i] + "/series" 
            path += ("/validas" if item[1][i] == 0 else "/no validas")
            path += "/" + item[0]
            series.append(get_points(path))

        return (torch.tensor(series,device=self.device), torch.tensor(item[1],device=self.device))
    

def label_to_categories(label):          
    return int(''.join(map(str, [int(label[i]) for i in range(len(label))])), 2)


def get_dataset(test_prop, val_prop, sequential, coef_weights=1, random_state=None, device="cpu"):

    series = reshape_series(get_series())
    #categories = [label_to_categories(series[i][1]) for i in range(len(series))]

    # Primero dividimos en entrenamiento y resto
    val_relative_size = val_prop / (test_prop + val_prop)
    
    if not sequential:
        # División estratificada para mantener distribución de intenciones
        train_series, temp = train_test_split(
            series,
            test_size=(test_prop + val_prop),
            random_state=random_state,
            stratify=[label_to_categories(series[i][1]) for i in range(len(series))]
        )

        val_series, test_series = train_test_split(
            temp,
            test_size=(1 - val_relative_size),
            random_state=random_state,
            stratify=[label_to_categories(temp[i][1]) for i in range(len(temp))]
        )
    
    else:
        train_series = series[:round((1-test_prop)*len(series))]
        test_series = series[round((1-test_prop)*len(series)):]
        val_series = None

    train_ds = dni(PATH_D, train_series, device)
    test_ds = dni(PATH_D, test_series, device)
    val_ds = dni(PATH_D, val_series, device)
    weights = get_weights(train_series,coef_weights)

    cat_train = np.array([label_to_categories(train_series[i][1]) for i in range(len(train_series))])
    cat_val = np.array([label_to_categories(val_series[i][1]) for i in range(len(val_series))])
    cat_test = np.array([label_to_categories(test_series[i][1]) for i in range(len(test_series))])

    print("Categorías entrenamiento:")
    values, counts = np.unique(cat_train,return_counts=True)
    print(values)
    print(counts)

    print("Categorías validación:")
    values, counts = np.unique(cat_val,return_counts=True)
    print(values)
    print(counts)

    print("Categorías test:")
    values, counts = np.unique(cat_test,return_counts=True)
    print(values)
    print(counts)


    return train_ds, test_ds, val_ds, weights