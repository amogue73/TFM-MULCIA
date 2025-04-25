from matplotlib import rc
import matplotlib.pyplot as plt
import os
import csv

PATH_D = "Series temporales/DNI/"
PATH_G = "Series temporales/GHI/"

SENSORS = ['S1', 'S3', 'S4']

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

for s in [SENSORS[2]]:
    name_files = ['falsos' + ' ' + j + ' ' + s for j in ['invalidos']]

    print(name_files)

    for name in name_files:
        path_d_valida = make_path("dni", s, True, name)
        path_d_invalida = make_path("dni", s, False, name)

        print ('Archivo {}'.format(name))
        with open('outliers/list/' + name, "r") as file:
            lista = [line.strip().split()[0] for line in file]

        for l in lista:
            message = []
            y_d = []
            y_g = []
            valida_d = []
            valida_g = []

            for i in range(3):
                valida_d.append(os.path.isfile(make_path("dni",SENSORS[i],True,l)))
                valida_g.append(os.path.isfile(make_path("ghi",SENSORS[i],True,l)))
                message.append("VÁLIDA" if valida_d[i] else "INVÁLIDA")
                y_d.append(get_points(make_path("dni",SENSORS[i],valida_d[i],l)))
                y_g.append(get_points(make_path("ghi",SENSORS[i],valida_g[i],l)))

            font = {'family' : 'monospace',
                    'weight' : 'normal',
                    'size'   : 10}
            rc('font', **font)
            x = [i for i in range(0,144)]
            fig, ax = plt.subplots(1,3,sharey=True,figsize=(14,5))

            for i in range(3):

                if (valida_d[i]):
                    ax[i].title.set_color('g')
                else:
                    ax[i].title.set_color('r')
                ax[i].title.set_text(l + " " + message[i] + " " + SENSORS[i])
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

            #plt.savefig("outliers/imgs/{}/{}".format(name,l), dpi=300, bbox_inches="tight")
            plt.show()
