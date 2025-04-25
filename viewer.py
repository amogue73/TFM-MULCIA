#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import numpy as np
import os
import csv

PATH_D = "Series temporales/DNI/"
PATH_G = "Series temporales/GHI/"

SENSORS = ["S1","S3","S4"]

def get_key(x):
    fields = x.split("-")
    y = fields[0]
    d = int(fields[1])
    if d < 10:
        d = "00" + str(d)
    elif d < 100:
        d = "0" + str(d)
    else:
        d = str(d)

    return y+d


def make_path(tipo,sensor,valida,dia):
    if (tipo.lower() == "dni"):
        if (valida):
            path = PATH_D + sensor + "/series/validas/" + dia
        else:
            path = PATH_D + sensor + "/series/no validas/" + dia
    elif (tipo.lower() == "ghi"):
        if (valida):
            path = PATH_G + sensor + "/series/validas/" + dia
        else:
            path = PATH_G + sensor + "/series/no validas/" + dia     
    else:
        path = -1
    return path

def make_path_error(tipo,sensor,dia):
    if (tipo.lower() == "dni"):
        path = PATH_D + sensor + "/error/" + dia
    elif (tipo.lower() == "ghi"):
        path = PATH_G + sensor + "/error/" + dia 
    else:
        path = -1
    return path

def get_points(path):
    with open(path, newline='',encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if not '' in row:
                y = [float(x) for x in row]
            else:
                y = []
                for x in row:
                    y.append(2000) if x == '' else y.append(float(x))
    return y

def main(sensor):
    series_validas = {}
    series_invalidas = {}
    # dias = [str(2017) + "-" + str(d) for d in range(150,366)]
    # dias += [str(y) + "-" + str(d) for y in range(2018,2020) for d in range(1,366)]
    dias = [str(y) + "-" + str(d) for y in range(2013,2020) for d in range(1,366)]
    i = 0
    for d in dias:


        path_d_valida = make_path("dni", sensor, True, d)
        path_d_invalida = make_path("dni", sensor, False, d)
        path_g_valida = make_path("ghi", sensor, True, d)
        path_g_invalida = make_path("ghi", sensor, False, d)

        if os.path.isfile(path_d_valida):
            continue
            message = []
            y_d = []
            y_g = []
            valida_d = []
            valida_g = []

            for i in range(3):
                valida_d.append(os.path.isfile(make_path("dni",SENSORS[i],True,d)))
                valida_g.append(os.path.isfile(make_path("ghi",SENSORS[i],True,d)))
                message.append("VÁLIDA" if valida_d[i] else "INVÁLIDA")
                y_d.append(get_points(make_path("dni",SENSORS[i],valida_d[i],d)))
                y_g.append(get_points(make_path("ghi",SENSORS[i],valida_g[i],d)))
        elif os.path.isfile(path_d_invalida):
            message = []
            y_d = []
            y_g = []
            valida_d = []
            valida_g = []

            for i in range(3):
                valida_d.append(os.path.isfile(make_path("dni",SENSORS[i],True,d)))
                valida_g.append(os.path.isfile(make_path("ghi",SENSORS[i],True,d)))
                message.append("VÁLIDA" if valida_d[i] else "INVÁLIDA")
                y_d.append(get_points(make_path("dni",SENSORS[i],valida_d[i],d)))
                y_g.append(get_points(make_path("ghi",SENSORS[i],valida_g[i],d)))

            y_d = np.array(y_d)
        else:
            continue
        font = {'family' : 'monospace',
                'weight' : 'normal',
                'size'   : 10}
        matplotlib.rc('font', **font)
        x = [i for i in range(0,144)]
        #fig, (ax1, ax2, ax3) = plt.subplots(1,3,sharey=True )
        fig, ax = plt.subplots(1,3,sharey=True,figsize=(14,5))
        #ax1 = fig.add_subplot()

        for i in range(3):

            if (valida_d[i]):
                ax[i].title.set_color('g')
            else:
                ax[i].title.set_color('r')
            ax[i].title.set_text(d + " " + message[i] + " " + SENSORS[i])
            ax[i].set_ylim([0, 1200])
            ax[i].plot(x, y_d[i], label = "directa", linewidth=2,color="tab:orange", zorder=3)
            ax[i].scatter(x, y_d[i], s=10, color="tab:orange", zorder=2)
            ax[i].plot(x, y_g[i], label = "global", linewidth=2, color="tab:blue", zorder=1)
            ax[i].scatter(x, y_g[i], s=10, color="tab:blue", zorder=0)
            ax[i].grid()
            ax[i].legend(loc="best")

        fig.tight_layout()

        fm = plt.get_current_fig_manager()
        fm.window.geometry(f"+100+200")

        #fig.canvas.draw_idle()
        #plt.subplots_adjust(left=0, right=5, top=5, bottom=0)
        plt.show()
        i+=1

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Debes pasar un argumento: el sensor a usar")
        exit(-1)

    main(sys.argv[1])