# ----- initialistion des modules -----#
import pandas as pd
import numpy
from tkinter import Tk
from tkinter import messagebox
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import requests
import datetime
from numpy import *
from matplotlib.pyplot import *
import colorama
from colorama import Fore
import os
from pystyle import Add, Center, Anime, Colors, Colorate, Write, System
from multiprocessing import Process
import math



# ----- initialistion des modules -----#

# ----- initialistion des couleurs du modules pystyle -----#
class bcolors:
    OK = '\033[92m'  # GREEN
    WARNING = '\033[93m'  # YELLOW
    FAIL = '\033[91m'  # RED
    RESET = '\033[0m'  # RESET COLOR
    PURPLE = '\033[35m'  # PURPLE


w = Fore.WHITE
b = Fore.BLACK
g = Fore.LIGHTGREEN_EX
y = Fore.LIGHTYELLOW_EX
m = Fore.LIGHTMAGENTA_EX
c = Fore.LIGHTCYAN_EX
lr = Fore.LIGHTRED_EX
lb = Fore.LIGHTBLUE_EX
# ----- initialistion des couleurs du modules pystyle -----#

# ----- initialistion des temps de recherches -----#
date = datetime.datetime.now()
my_lock = threading.RLock()
end = str(pd.Timestamp.today() + pd.DateOffset(5))[0:10]
start_5m = str(pd.Timestamp.today() + pd.DateOffset(-15))[0:10]
start_15m = str(pd.Timestamp.today() + pd.DateOffset(-15))[0:10]
start_30m = str(pd.Timestamp.today() + pd.DateOffset(-15))[0:10]
start_1h = str(pd.Timestamp.today() + pd.DateOffset(-15))[0:10]
start_6h = str(pd.Timestamp.today() + pd.DateOffset(-20))[0:10]
start_1d = str(pd.Timestamp.today() + pd.DateOffset(-50))[0:10]
start_1week = str(pd.Timestamp.today() + pd.DateOffset(-120))[0:10]
start_1month = str(pd.Timestamp.today() + pd.DateOffset(-240))[0:10]
# ----- initialistion des temps de recherches -----#

# ----- initialistion de l'API key et ticker -----#
api_key = '1KsqKOh1pTAJyWZx6Qm9pvnaNcpKVh_8'
ticker = 'WLFC'
tiker_live = 'WLFC'
indic = False


# ----- initialistion de l'API key et ticker -----#

# ----- fonction pour trouver les point intersection de la ligne de coup et de la Courbe -----#
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('les courbes ne se coupent pas')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


# ----- fonction pour trouver les point intersection de la ligne de coup et de la Courbe -----#



def sma(data, window):
    sma = data.rolling(window=window).mean()
    return sma




def bb(data, sma, window):
    std = data.rolling(window=window).std()
    upper_bb = sma + std * 2
    lower_bb = sma - std * 2
    return upper_bb, lower_bb




def createMACD(df):
    df['e26'] = pd.Series.ewm(df['c'], span=26).mean()
    df['e12'] = pd.Series.ewm(df['c'], span=12).mean()
    df['MACD'] = df['e12'] - df['e26']
    df['e9'] = pd.Series.ewm(df['MACD'], span=9).mean()
    df['HIST'] = df['MACD'] - df['e9']


def rsi(df, periods=14, ema=True):
    close_delta = df['c'].diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)

    if ema == True:
        # Use exponential moving average
        ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
        ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    else:
        # Use simple moving average
        ma_up = up.rolling(window=periods, adjust=False).mean()
        ma_down = down.rolling(window=periods, adjust=False).mean()

    rsi = ma_up / ma_down
    rsi = 100 - (100 / (1 + rsi))
    return rsi



def Finder_IETE(time1, time_name1, start1):
    #global ticker
    # while True:

    with my_lock:
        api_url_livePrice = f'http://api.polygon.io/v2/last/trade/{ticker}?apiKey={api_key}'
        data = requests.get(api_url_livePrice).json()
        df_livePrice = pd.DataFrame(data)
        # api_url_OHLC = f'http://api.polygon.io/v2/aggs/ticker/{ticker}/range/15/minute/2022-07-01/2022-07-15?adjusted=true&sort=asc&limit=30000&apiKey={api_key}'
        api_url_OHLC = f'http://api.polygon.io/v2/aggs/ticker/{ticker}/range/{time1}/{time_name1}/{start1}/{end}?adjusted=true&limit=50000&apiKey={api_key}'

        data = requests.get(api_url_OHLC).json()
        df = pd.DataFrame(data['results'])
        la_place_de_p = 0

        for k in range(0, len(df_livePrice.index)):
            if df_livePrice.index[k] == 'p':
                la_place_de_p = k
        livePrice = df_livePrice['results'][la_place_de_p]
    dernligne = len(df['c']) - 1
    df.drop([dernligne], axis=0, inplace=True)

    # df = df.drop(columns=['o', 'h', 'l', 'v', 'vw', 'n'])
    # df = df.append({'o': NAN, 'h': NAN, 'l': NAN, 'v': NAN, 'vw': NAN, 'n': NAN, 'c': livePrice, 't': NAN}, ignore_index=True)
    df_new_line = pd.DataFrame([[NAN, NAN, NAN, NAN, NAN, NAN, livePrice, NAN]],
                               columns=['o', 'h', 'l', 'v', 'vw', 'n', 'c', 't'])
    df = pd.concat([df, df_new_line], ignore_index=True)
    df_data_date = []
    df_data_price = []
    for list_df in range(len(df)):
        df_data_date.append(df['t'].iloc[list_df])
        df_data_price.append(df['c'].iloc[list_df])
    data_date = pd.DataFrame(df_data_date, columns=['Date'])
    data_price = pd.DataFrame(df_data_price, columns=['Price'])
    df_wise_index = pd.concat([data_date, data_price], axis=1)

    place_liveprice = (len(df) - 1)

    for data in range(len(df_wise_index)):

        try:

            if df_wise_index['Price'].iloc[data] == df_wise_index['Price'].iloc[data + 1]:
                df = df.drop(df_wise_index['Date'].iloc[data + 1])
        except:

            # print('ok')
            aaa = 0

    # ----- creation des local(min/max) -----#
    local_max = argrelextrema(df['c'].values, np.greater, order=1, mode='clip')[0]
    local_min = argrelextrema(df['c'].values, np.less, order=1, mode='clip')[0]
    local_max1 = argrelextrema(df['c'].values, np.greater, order=1, mode='clip')[0]
    local_min1 = argrelextrema(df['c'].values, np.less, order=1, mode='clip')[0]

    local_max2 = argrelextrema(df['c'].values, np.greater, order=1, mode='clip')[0]
    local_min2 = argrelextrema(df['c'].values, np.less, order=1, mode='clip')[0]

    # ----- creation des local(min/max) -----#

    # ----- suppresion des points morts de la courbe -----#
    test_min = []
    test_max = []

    # if local_min[0] > local_max[0]:
    #        local_max = local_max[1:]
    #        print('On a supprimer le premier point')
    #
    q = 0
    p = 0

    len1 = len(local_min)
    len2 = len(local_max)
    while p < len1 - 5 or p < len2 - 5:
        if local_min[p + 1] < local_max[p]:
            test_min.append(local_min[p])
            local_min = np.delete(local_min, p)

            p = p - 1
        if local_max[p + 1] < local_min[p + 1]:
            test_max.append(local_max[p])
            local_max = np.delete(local_max, p)

            p = p - 1
        p = p + 1

        len1 = len(local_min)
        len2 = len(local_max)

    highs = df.iloc[local_max, :]
    lows = df.iloc[local_min, :]
    highs1 = df.iloc[test_max, :]
    lows1 = df.iloc[test_min, :]

    decalage = 0

    # ----- suppresion des points morts de la courbe -----#
    A = float(df['c'].iloc[local_max[-3]])
    B = float(df['c'].iloc[local_min[-3]])
    C = float(df['c'].iloc[local_max[-2]])
    D = float(df['c'].iloc[local_min[-2]])
    E = float(df['c'].iloc[local_max[-1]])
    F = float(df['c'].iloc[local_min[-1]])
    G = float(livePrice)

    if C > E:
        differ = (C - E)
        pas = (local_max[-1] - local_max[-2])
        suite = differ / pas
    if C < E:
        differ = (E - C)
        pas = (local_max[-1] - local_max[-2])
        suite = differ / pas

    print(f'--- Mode recherche {ticker}', time1, time_name1, ' ---', flush=True)

    data_A = []
    data_B = []
    data_C = []
    data_D = []
    data_E = []
    data_F = []
    data_G = []

    rouge = []
    vert = []
    bleu = []

    rouge.append(local_max[-3])
    rouge.append(local_min[-3])
    rouge.append(local_max[-2])
    rouge.append(local_min[-2])
    rouge.append(local_max[-1])
    rouge.append(local_min[-1])
    rouge.append(place_liveprice)

    vert.append(local_max[-3])
    vert.append(local_max[-2])
    vert.append(local_max[-1])
    vert.append(place_liveprice)

    i = 0
    for i in range(local_max[-4] - 1, len(df)):
        bleu.append(i)

    mirande2 = df.iloc[vert, :]
    mirande = df.iloc[rouge, :]
    mirande3 = df.iloc[bleu, :]

    if E > C:
        mirande2['c'].values[0] = mirande2['c'].values[1] - ((suite * (local_max[-2] - local_max[-3])))
        mirande2['c'].values[3] = mirande2['c'].values[2] + ((suite * (place_liveprice - local_max[-1])))
    if E < C:
        mirande2['c'].values[0] = mirande2['c'].values[1] + ((suite * (local_max[-2] - local_max[-3])))
        mirande2['c'].values[3] = mirande2['c'].values[2] - ((suite * (place_liveprice - local_max[-1])))
    if E == C:
        mirande2['c'].values[0] = df['c'].values[local_max[-2]]
        mirande2['c'].values[3] = df['c'].values[local_max[-1]]

    vert1 = {'c': vert}
    vert2 = pd.DataFrame(data=vert1)
    rouge1 = {'c': rouge}
    rouge2 = pd.DataFrame(data=rouge1)
    bleu1 = {'c': bleu}
    bleu2 = pd.DataFrame(data=bleu1)
    # --- premier droite ---#
    AI = [local_max[-3], mirande2['c'].iloc[0]]
    BI = [local_max[-2], mirande2['c'].iloc[1]]

    # --- deuxieme droite ---#
    CI = [local_max[-3], A]
    DI = [local_min[-3], B]
    # I = line_intersection((AI, BI), (CI, DI))

    # ----------------------------------------------------------------------------#
    # ----------------------------------------------------------------------------#

    AJ = [local_max[-1], mirande2['c'].iloc[2]]
    BJ = [place_liveprice, mirande2['c'].iloc[3]]

    # --- deuxieme droite ---#
    CJ = [place_liveprice, G]
    DJ = [local_min[-1], F]
    # J = line_intersection((AJ, BJ), (CJ, DJ))

    # ----- verification qu'il n'y est pas de point mort dans la figure -----#
    pop = 0
    verif = 0

    for pop in range(0, len(test_min)):
        if test_min[pop] > local_max[-3] and test_min[pop] < place_liveprice:
            verif = verif + 1
    pop = 0
    for pop in range(0, len(test_max)):
        if test_max[pop] > local_max[-3] and test_max[pop] < place_liveprice:
            verif = verif + 1
    # ----- verification qu'il n'y est pas de point mort dans la figure -----
    ordre = False
    if local_max[-3] < local_min[-3] < local_max[-2] < local_min[-2] < local_max[-1] < local_min[-1]:
        ordre = True

    mini_pourcent = False
    if ((((C + E) / 2) - D) * 100) / D >= 2.8:
        mini_pourcent = True

    if (C - B) < (C - D) and (C - B) < (E - D) and (E - F) < (E - D) and (E - F) < (
            C - D) and B > D and F > D and B < C and F < E and A >= mirande2['c'].iloc[
        0] and verif == 0 and ordre == True and mini_pourcent == True:
        try:
            J = line_intersection((AJ, BJ), (CJ, DJ))
            I = line_intersection((AI, BI), (CI, DI))
            accept = True
        except:
            accept = False
        if accept == True:
            moyenne_epaule1 = ((I[1] - B) + (C - B)) / 2
            moyenne_epaule2 = ((E - F) + (J[1] - F)) / 2
            moyenne_tete = ((C - D) + (E - D)) / 2

            tuche = 0
            noo = 0
            place_pc = 0
            point_max = J[0] + ((J[0] - I[0]))
            point_max = int(round(point_max, 0))
        if I[1] > B and J[
            1] > F and moyenne_epaule1 <= moyenne_tete / 2 and moyenne_epaule2 <= moyenne_tete / 2 and moyenne_epaule1 >= moyenne_tete / 4 and moyenne_epaule2 >= moyenne_tete / 4 and accept == True and \
                df['c'].values[-2] <= J[1] + (moyenne_tete) / 4 and df['c'].values[-2] >= J[1] and df['c'].values[-1] <= \
                J[1] + (moyenne_tete) / 4 and df['c'].values[-1] >= J[1]:

            df['sma_20'] = sma(df['c'], 20)
            df.tail()

            df['upper_bb'], df['lower_bb'] = bb(df['c'], df['sma_20'], 20)
            df.tail()

            createMACD(df)
            df['rsi'] = rsi(df)

            plus_grand = round((J[1] + (moyenne_tete) / 2), 5)
            plus_petit = round(G, 5)
            pourcent_chercher = ((plus_grand - plus_petit) / plus_petit)*100
            pourcent_chercher = round(pourcent_chercher, 3)
            fig = plt.figure(figsize=(10, 7))
            #fig.patch.set_facecolor('#131722')
            #ax = plt.gca()
            #ax.set_facecolor('#131722')
            plt.plot([], [], ' ')

            plt.title(f'IETE : {tiker_live} | {time1} {time_name1} | {pourcent_chercher}%', fontweight="bold", color='black')

            mirande3['c'].plot(color=['blue'], label='Clotures')
            if indic == True:
                df['upper_bb'].iloc[(local_max[-3]-10):(place_liveprice) + 10].plot(label='Haut Band', linestyle='--', linewidth=1, color='red')
                df['sma_20'].iloc[(local_max[-3]-10):(place_liveprice) + 10].plot(label='Ema 20', linestyle='-', linewidth=1.2, color='grey')
                df['lower_bb'].iloc[(local_max[-3]-10):(place_liveprice) + 10].plot(label='Bas Band', linestyle='--', linewidth=1, color='green')
            mirande3['h'].plot(color='orange', alpha=0.3, label='highs')
            # mirande['c'].plot(color=['#FF0000'])
            mirande2['c'].plot(color=['green'], linestyle='--', label='Ligne de coup')
            plt.axhline(y=J[1] + moyenne_tete, linestyle='--', alpha=0.3, color='red', label='100% objectif')
            plt.axhline(y=J[1] + (((moyenne_tete) / 2) + ((moyenne_tete) / 4)), linestyle='--', alpha=0.3,
                        color='black', label='75% objectif')
            plt.axhline(y=J[1] + (moyenne_tete) / 2, linestyle='--', alpha=0.3, color='orange', label='50% objectif')
            plt.axhline(y=J[1] + (moyenne_tete) / 4, linestyle='--', alpha=0.3, color='black', label='25% objectif')
            plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.1)
            taille_diviser = (local_max[-1] - local_max[-2]) / (local_min[-2] - local_max[-2])
            # point_max = J[0]+((J[0] - I[0])/taille_diviser)
            point_max = J[0] + ((J[0] - I[0]))
            point_max = int(round(point_max, 0))
            # plt.scatter(point_max, df['c'].values[int(round(point_max, 0))], color='red',label='Max temps realisation')
            plt.legend()
            plt.text(local_max[-3], A, "A", ha='left', style='normal', size=10.5, color='red', wrap=True)
            plt.text(J[0], J[1] + (moyenne_tete) / 2, f"{round((J[1] + (moyenne_tete) / 2), 5)}", ha='left', style='normal', size=10.5, color='red', wrap=True)
            plt.text(local_min[-3], B, "B", ha='left', style='normal', size=10.5, color='red', wrap=True)
            plt.text(local_max[-2], C, "C", ha='left', style='normal', size=10.5, color='red', wrap=True)
            plt.text(local_min[-2], D, "D", ha='left', style='normal', size=10.5, color='red', wrap=True)
            plt.text(local_max[-1], E, "E", ha='left', style='normal', size=10.5, color='red', wrap=True)
            plt.text(local_min[-1], F, f"F  {round(F, 5)}", ha='left', style='normal', size=10.5, color='red', wrap=True)
            plt.text(place_liveprice, G, f"G  {round(G, 5)}", ha='left', style='normal', size=10.5, color='red', wrap=True)
            plt.text(I[0], I[1], "I", ha='left', style='normal', size=10.5, color='#00FF36', wrap=True)
            # test_valeur = df['c'].iloc[round(J[0]) + 1]
            # plt.text(round(J[0]), df['c'].iloc[round(J[0])], f"J+1 {test_valeur}", ha='left',style='normal', size=10.5, color='#00FF36', wrap=True)
            plt.scatter(len(df['c']) - 1, df['c'].values[-1], color='blue', label='liveprice')
            plt.scatter(len(df['c']) - 2, df['c'].values[-2], color='orange', label='cloture')
            plt.show()
            # -----------------------lire et connaitre nom de image et enregistrer image--------------------------#
            #file = open('/home/mat/Bureau/logi3_direct/compteur_images.txt', 'r')
            #compteur_nombre_image = int(file.read())
            #file.close()
            #file = open('/home/mat/Bureau/logi3_direct/compteur_images.txt', 'w')
            #compteur_nombre_image = compteur_nombre_image + 1
            #file.write(f'{compteur_nombre_image}')
            #file.close()
            #plt.savefig(f'/home/mat/Bureau/recupinfo/figure_{compteur_nombre_image}_r1.png')
            # -----------------------lire et connaitre nom de image et enregistrer image--------------------------#
            ###############################################################################################################################################

            fig = plt.figure(figsize=(10, 7))
            plt.subplot(2, 1, 1)
            # fig.patch.set_facecolor('#17abde')
            plt.plot([], [], ' ', label="e")

            plt.title(f'IETE : {tiker_live} | {time1} {time_name1} | {pourcent_chercher}%', fontweight="bold", color='black')
            plt.bar(df['v'][(local_max[-3]):(place_liveprice) + 1].index,
                    df['v'].values[(local_max[-3]):(place_liveprice) + 1])
            plt.legend(['Volumes'])

            plt.text(local_max[-3], df['v'][(local_max[-3])], "A", ha='left', style='normal', size=10.5, color='red',
                     wrap=True)
            plt.text(local_min[-3], df['v'][(local_min[-3])], "B", ha='left', style='normal', size=10.5, color='red',
                     wrap=True)
            plt.text(local_max[-2], df['v'][(local_max[-2])], "C", ha='left', style='normal', size=10.5, color='red',
                     wrap=True)
            plt.text(local_min[-2], df['v'][(local_min[-2])], "D", ha='left', style='normal', size=10.5, color='red',
                     wrap=True)
            plt.text(local_max[-1], df['v'][(local_max[-1])], "E", ha='left', style='normal', size=10.5, color='red',
                     wrap=True)
            plt.text(local_min[-1], df['v'][(local_min[-1])], "F", ha='left', style='normal', size=10.5, color='red',
                     wrap=True)
            plt.text(place_liveprice, df['v'][place_liveprice], "G", ha='left', style='normal', size=10.5, color='red',
                     wrap=True)

            plt.subplot(2, 1, 2)
            df['rsi'].iloc[(local_min[-3] - 3):(local_max[-1] + 10)].plot(color=['purple'], alpha=0.6)
            plt.axhline(y=30, alpha=0.3, color='black')
            plt.axhline(y=70, alpha=0.3, color='black')
            plt.axhline(y=50, linestyle='--', alpha=0.3, color='grey')
            plt.legend(['Rsi'])

            plt.text(local_max[-3], df['rsi'].iloc[local_max[-3]], "A", ha='left', style='normal', size=10.5,
                     color='blue',
                     wrap=True)
            plt.text(local_min[-3], df['rsi'].iloc[local_min[-3]], "B", ha='left', style='normal', size=10.5,
                     color='blue',
                     wrap=True)
            plt.text(local_max[-2], df['rsi'].iloc[local_max[-2]], "C", ha='left', style='normal', size=10.5,
                     color='blue',
                     wrap=True)
            plt.text(local_min[-2], df['rsi'].iloc[local_min[-2]], "D", ha='left', style='normal', size=10.5,
                     color='blue',
                     wrap=True)
            plt.text(local_max[-1], df['rsi'].iloc[local_max[-1]], "E", ha='left', style='normal', size=10.5,
                     color='blue',
                     wrap=True)
            plt.text(local_min[-1], df['rsi'].iloc[local_min[-1]], "F", ha='left', style='normal', size=10.5,
                     color='blue',
                     wrap=True)
            plt.text(place_liveprice, df['rsi'].iloc[place_liveprice], "G", ha='left', style='normal', size=10.5,
                     color='blue',
                     wrap=True)
            #plt.text(I[0], I[1], "I", ha='left', style='normal', size=10.5, color='blue', wrap=True)
            #plt.text(J[0], J[1], "J", ha='left', style='normal', size=10.5, color='blue', wrap=True)
            plt.show()
            #plt.savefig(f'images/figure_{compteur_nombre_image}_v1.png')
            ###############################################################################################################################################


            fig = plt.figure(figsize=(10, 7))

            #fig.patch.set_facecolor('#17abde')


            plt.plot([], [], ' ')

            #plt.subplot(3, 1, 1)
            plt.title(f'IETE : {tiker_live} | {time1} {time_name1} | {pourcent_chercher}%', fontweight="bold", color='black')
            mirande3['c'].plot(color=['blue'], alpha=0.3, label ='Clotures')
            mirande2['c'].plot(color=['green'], alpha=0.3, linestyle='--', label ='Ligne de coup')



            width = .4
            width2 = .05

            # define up and down prices
            up = mirande3[mirande3.c >= mirande3.o]
            down = mirande3[mirande3.c < mirande3.o]

            # define colors to use
            col1 = 'green'
            col2 = 'red'

            # plot up prices9
            plt.bar(up.index, up.c - up.o, width, bottom=up.o, color=col1, label ='Bougies Japonnaises')
            plt.bar(up.index, up.h - up.c, width2, bottom=up.c, color=col1)
            plt.bar(up.index, up.l - up.o, width2, bottom=up.o, color=col1)

            # plot down prices
            plt.bar(down.index, down.c - down.o, width, bottom=down.o, color=col2)
            plt.bar(down.index, down.h - down.o, width2, bottom=down.o, color=col2)
            plt.bar(down.index, down.l - down.c, width2, bottom=down.c, color=col2)
            plt.text(local_max[-3], A, "A", ha='left', style='normal', size=10.5, color='blue',
                    wrap=True)
            plt.text(local_min[-3], B, "B", ha='left', style='normal', size=10.5, color='blue',
                    wrap=True)
            plt.text(local_max[-2], C, "C", ha='left', style='normal', size=10.5, color='blue',
                    wrap=True)
            plt.text(local_min[-2], D, "D", ha='left', style='normal', size=10.5, color='blue',
                    wrap=True)
            plt.text(local_max[-1], E, "E", ha='left', style='normal', size=10.5, color='blue',
                    wrap=True)
            plt.text(local_min[-1], F, "F", ha='left', style='normal', size=10.5, color='blue',
                    wrap=True)
            plt.text(place_liveprice, G, "G", ha='left', style='normal', size=10.5, color='blue',
                    wrap=True)
            plt.text(I[0], I[1], "I", ha='left', style='normal', size=10.5, color='blue', wrap=True)
            plt.text(J[0], J[1], "J", ha='left', style='normal', size=10.5, color='blue', wrap=True)


            plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.1)
            plt.legend()
            plt.show()
            ###############################################################################################################################################

            fig = plt.figure(figsize=(10, 7))
            #fig.patch.set_facecolor('#17abde')
            plt.plot([], [], ' ')

            plt.subplot(2, 1, 1)
            plt.title(f'IETE : {tiker_live} | {time1} {time_name1} | {pourcent_chercher}%', fontweight="bold", color='black')
            plt.bar(df['HIST'][(local_max[-3]-1):(place_liveprice) + 5].index,df['HIST'].values[(local_max[-3]-1):(place_liveprice) + 5], color='purple', alpha=0.6)
            df['MACD'].iloc[(local_max[-3] - 1):(place_liveprice + 5)].plot(color=['blue'], alpha=0.6)
            df['e9'].iloc[(local_max[-3] - 1):(place_liveprice + 5)].plot(color=['red'], alpha=0.6)

            plt.text(local_max[-3], df['HIST'].iloc[(local_max[-3])], "A", ha='left', style='normal', size=10.5, color='blue', wrap=True)
            plt.text(local_min[-3], df['HIST'].iloc[(local_min[-3])], "B", ha='left', style='normal', size=10.5, color='blue', wrap=True)
            plt.text(local_max[-2], df['HIST'].iloc[(local_max[-2])], "C", ha='left', style='normal', size=10.5, color='blue', wrap=True)
            plt.text(local_min[-2], df['HIST'].iloc[(local_min[-2])], "D", ha='left', style='normal', size=10.5, color='blue', wrap=True)
            plt.text(local_max[-1], df['HIST'].iloc[(local_max[-1])], "E", ha='left', style='normal', size=10.5, color='blue', wrap=True)
            plt.text(local_min[-1], df['HIST'].iloc[(local_min[-1])], "F", ha='left', style='normal', size=10.5, color='blue', wrap=True)
            plt.text(place_liveprice, df['HIST'].iloc[place_liveprice], "G", ha='left', style='normal', size=10.5, color='blue', wrap=True)
            #plt.text(I[0], I[1], "I", ha='left', style='normal', size=10.5, color='#00FF36', wrap=True)
            #plt.text(J[0], J[1], "J", ha='left', style='normal', size=10.5, color='#00FF36', wrap=True)




            plt.legend(['Macd','Signal','histogramme'])
            a2 = plt.subplot(2, 1, 2)
            a2.axis('off')
            plt.axis([0, 10, 0, 10])
            temps_formation = int((J[0] - I[0]) * time1)
            #heure_debut = ((df['t'].iloc[local_max[-3]]) / 1000)  # ATTENTION ENLEVER L'AJOUT DES 6H SUR LES AUTRES ORDINATEURS
            #heure_fin = ((df['t'].iloc[place_liveprice]) / 1000)  # ATTENTION ENLEVER L'AJOUT DES 6H SUR LES AUTRES ORDINATEURS
            #temps_debut = datetime.datetime.fromtimestamp(heure_debut)
            #temps_fin = datetime.datetime.fromtimestamp(heure_fin)

            plt.text(0, 8, " ▶ LA FIGURE DEMARRE : ", ha='left', style='normal', size=9.5, color='black',
                    wrap=True, alpha=1)
            plt.text(0, 6.5, " ▶ LA FIGURE TERMINE : ", ha='left', style='normal', size=9.5, color='black',
                    wrap=True, alpha=1)
            plt.text(0, 5, f" ▶ LE  POURCENTAGE GAIN EST : {pourcent_chercher}%", ha='left', style='normal', size=9.5, color='black', wrap=True,
                    alpha=1)
            plt.text(0, 3.5, f" ▶ L'INCLINAISON DE LDC EST : {degres}°", ha='left', style='normal', size=9.5, color='black', wrap=True,
                    alpha=1)
            plt.text(5, 8, " ▶ LA TENDANCE PRECEDENTE EST : {tendance}", ha='left', style='normal', size=9.5,
                    color='black', wrap=True, alpha=1)
            plt.text(5, 6.5,
                    f" ▶ LE RSI DE F: {int(df['rsi'].iloc[local_min[-1]])}  I  LE RSI DE G:  {int(df['rsi'].iloc[place_liveprice])}  I  LE RSI DE J:  {int(df['rsi'].iloc[int(round(J[0], 0))])}",
                    ha='left', style='normal',
                    size=9.5, color='black', alpha=1)
            plt.text(5, 5, f" ▶ LA FIGURE S'EST FORMÉE EN : {temps_formation} {time_name1}", ha='left',
                    style='normal', size=9.5,
                    color='black', wrap=True, alpha=1)
            plt.text(5, 3.5, f" ▶ LE POURCENTAGE OBJECTIF EST : ", ha='left',
                    style='normal', size=9.5,
                    color='black', wrap=True, alpha=1)
            plt.show()



            multiplicateur = 0
            if time_name1 == 'minute':
                multiplicateur = 60

            if time_name1 == 'hour':
                multiplicateur = 3600

            if time_name1 == 'day':
                multiplicateur = 86400

            temps_attente = time1 * multiplicateur
            time.sleep(temps_attente)
            data_A.append(A)
            data_B.append(B)
            data_C.append(C)
            data_D.append(D)
            data_E.append(E)
            data_F.append(F)
            data_F.append(G)
            data_A_ = pd.DataFrame(data_A, columns=['A'])
            data_B_ = pd.DataFrame(data_B, columns=['B'])
            data_C_ = pd.DataFrame(data_C, columns=['C'])
            data_D_ = pd.DataFrame(data_D, columns=['D'])
            data_E_ = pd.DataFrame(data_E, columns=['E'])
            data_F_ = pd.DataFrame(data_E, columns=['F'])
            data_G_ = pd.DataFrame(data_E, columns=['G'])
            df_IETE = pd.concat([data_A_, data_B_, data_C_, data_D_, data_E_, data_F_, data_G_], axis=1)
    print('----------------------------------------------------------------------', flush=True)
    time.sleep(0.5)


minute = "minute"
heure = "hour"
jour = "day"

#print(' ')
#Write.Print(" CHOISSISEZ VOTRE TITRE : ", Colors.purple, interval=0.000)
#print(' ')
#print(' ')
#time.sleep(0.5)
#in0 = input(' >>\x1B[1m ')
#
#print(' ')
#Write.Print(" CHOISSISEZ VOTRE TIME1 : ", Colors.purple, interval=0.000)
#print(' ')
#print(' ')
#time.sleep(0.5)
#in1 = input(' >>\x1B[1m ')
#
#
#print(' ')
#Write.Print(" CHOISSISEZ VOTRE TIME NAME1 : ", Colors.purple, interval=0.000)
#print(' ')
#print(' ')
#time.sleep(0.5)
#in2 = input(' >>\x1B[1m ')
#
#print(' ')
#Write.Print(" CHOISSISEZ VOTRE START : ", Colors.purple, interval=0.000)
#print(' ')
#print(' ')
#time.sleep(0.5)
#in3 = input(' >>\x1B[1m ')


th1 = Process(target=Finder_IETE, args=(6,heure,start_6h))

th1.start()

th1.join()














