import cv2 as cv
import numpy as np
import copy, itertools
from transitions import Machine

# Definimos os landmarks facias da mascara que compoñen distintos elementos facias.
BORDES_CARA = [ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]
BEIZOS = [ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
BEIZOS_SUP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
BEIZOS_INF = [ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 
OLLO_ESQ = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
OLLO_DER = [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
# Facemos o mesmo para as mans. Son tuplas que permiten saber de que a que landmark debuxar as liñas dos dedos.
LANDMARKS_MANS = [(2,3),(3,4),(5,6),(6,7),(7,8),(9,10),(10,11),(11,12),(13,14),(14,15),(15,16), (17,18),(18,19),(19,20), (0,1),(1,2),(2,5), (5,9),(9,13),(13,17),(17,0)]
# Definimos algunhas cores para empregar ao engadir elementos ao frame
BLACK = (0,0,0)
WHITE = (255,255,255)
BLUE = (255,0,0)
RED = (0,0,255)
CYAN = (255,255,0)
YELLOW =(0,255,255)
MAGENTA = (255,0,255)
GRAY = (128,128,128)
GREEN = (0,255,0)
PURPLE = (128,0,128)
ORANGE = (0,165,255)
PINK = (147,20,255)

# Clase para a maquina de estados, onde indicamos tan so os estados que a compoñen e, neste caso, o estado clase como inicial
class FSM(object):
    states = ['clase', 'pausa', 'asercion', 'fin', 'pregunta']
    def __init__(self):
        self.machine = Machine(model=self, states=FSM.states, initial='clase')

# Funcion para engadir o texto cos FPS do video ao frame 
def engade_texto(imaxe, texto, fonte=cv.FONT_HERSHEY_COMPLEX, escalaFonte=1.0, posTexto=(20,50), grosorTexto=1,corTexto=(0,255,0), corFondo=(0,0,0), pad_x=3, pad_y=3, opacidadeFondo=0.9):
    (t_w, t_h), _= cv.getTextSize(texto, fonte, escalaFonte, grosorTexto)
    x, y = posTexto
    superposicion = imaxe.copy()
    cv.rectangle(superposicion, (x-pad_x, y+pad_y), (x+t_w+pad_x, y-t_h-pad_y), corFondo,-1)
    imaxe_nova = cv.addWeighted(superposicion, opacidadeFondo, imaxe, 1 - opacidadeFondo, 0)
    cv.putText(imaxe_nova, texto, posTexto,fonte, escalaFonte, corTexto,grosorTexto )
    imaxe = imaxe_nova
    return imaxe

# Funcions para o detector de mans:
def preprocesar_landmark(landmarks):
    def normalize_(n):
        return n / max_value
    tmp_landmark_list = copy.deepcopy(landmarks)
    # Convirte a coordenadas relativas
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(tmp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        tmp_landmark_list[index][0] = tmp_landmark_list[index][0] - base_x
        tmp_landmark_list[index][1] = tmp_landmark_list[index][1] - base_y
    # Convirte nunha lista unidimensional
    tmp_landmark_list = list(itertools.chain.from_iterable(tmp_landmark_list))
    # Normalizacion
    max_value = max(list(map(abs, tmp_landmark_list)))
    tmp_landmark_list = list(map(normalize_, tmp_landmark_list))
    return tmp_landmark_list

def calc_lista_landmarks(imaxe, landmarks):
    anchura, altura = imaxe.shape[1], imaxe.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * anchura), anchura - 1)
        landmark_y = min(int(landmark.y * altura), altura - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def calc_rectangulo_contedor(imaxe, landmarks):
    anchura, altura = imaxe.shape[1], imaxe.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * anchura), anchura - 1)
        landmark_y = min(int(landmark.y * altura), altura - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

# Funcion de debuxo para os puntos de detección das mans:
def debuxa_landmarks(imaxe, landmark_point, rectangulo):
    if len(landmark_point) > 0:
        for par in LANDMARKS_MANS:
            a, b = par
            cv.line(imaxe, tuple(landmark_point[a]), tuple(landmark_point[b]), (0, 0, 0), 6)
            cv.line(imaxe, tuple(landmark_point[a]), tuple(landmark_point[b]), (255, 255, 255), 2)
    for index, landmark in enumerate(landmark_point):
        if index in [0,1,2,3,5,6,7,9,10,11,13,14,15,17,18,19]: n=5
        elif index in [8,4,12,16,20]: n=8
        cv.circle(imaxe, (landmark[0], landmark[1]), n, (255, 255, 255), -1)
        cv.circle(imaxe, (landmark[0], landmark[1]), n, (0, 0, 0), 1)
    # Rectangulo contedor
    cv.rectangle(imaxe, (rectangulo[0], rectangulo[1]), (rectangulo[2], rectangulo[3]), (0, 0, 0), 1)
    return imaxe

# Funcion para obter as duas imaxes recortadas con so os ollos:
def recorte_ollos(imaxe, coords_ollo_der, coords_ollo_esq, draw=False):
    imaxe_gris = cv.cvtColor(imaxe, cv.COLOR_BGR2GRAY)
    mascara = np.zeros(imaxe_gris.shape, dtype=np.uint8)
    cv.fillPoly(mascara, [np.array(coords_ollo_der, dtype=np.int32)], 255)
    cv.fillPoly(mascara, [np.array(coords_ollo_esq, dtype=np.int32)], 255)
    imaxe_gris = cv.cvtColor(imaxe, cv.COLOR_BGR2GRAY)
    ollos = cv.bitwise_and(imaxe_gris, imaxe_gris, mask=mascara)
    if draw:
        cv.imshow('Ollos recortados', ollos)
    # Calculanse as minimas e maximas coordenadas en x e y para cada ollo
    # DEREITO
    r_max_x = (max(coords_ollo_der, key=lambda item: item[0]))[0]
    r_min_x = (min(coords_ollo_der, key=lambda item: item[0]))[0]
    r_max_y = (max(coords_ollo_der, key=lambda item: item[1]))[1]
    r_min_y = (min(coords_ollo_der, key=lambda item: item[1]))[1]
    # ESQUERDO
    l_max_x = (max(coords_ollo_esq, key=lambda item: item[0]))[0]
    l_min_x = (min(coords_ollo_esq, key=lambda item: item[0]))[0]
    l_max_y = (max(coords_ollo_esq, key=lambda item: item[1]))[1]
    l_min_y = (min(coords_ollo_esq, key=lambda item: item[1]))[1]

    # Recortes
    recorte_dereito = ollos[r_min_y: r_max_y, r_min_x: r_max_x]
    recorte_esquerdo = ollos[l_min_y: l_max_y, l_min_x: l_max_x]
    return recorte_dereito, recorte_esquerdo
