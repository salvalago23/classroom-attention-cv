#!/usr/bin/env python
# encoding: utf-8

import time
import mediapipe as mp
import threading
import math
from model import KeyPointClassifier
from imports import *

# Inicializamos o detector de mans e os xestos que pode identificar
keypoint_classifier = KeyPointClassifier()
keypoint_classifier_labels = ['OPEN', 'CLOSE', 'POINTER', 'OK']
# Funcion para clasificar o xesto da man identificada
def classify_landmark_list(landmarks):
    global keypoint_classifier
    landmarks_list_preprocesada = preprocesar_landmark(landmarks)
    return keypoint_classifier(landmarks_list_preprocesada)
# Definicions para o detector da mascara de landmarks
mp_landmarks = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
# Definicions para o detector de centroides de rasgos faciais
mp_centroides = mp.solutions.face_detection
OLLO1 = mp_centroides.FaceKeyPoint.RIGHT_EYE
OLLO2 = mp_centroides.FaceKeyPoint.LEFT_EYE
NARIZ = mp_centroides.FaceKeyPoint.NOSE_TIP
BOCA = mp_centroides.FaceKeyPoint.MOUTH_CENTER
# Variables globais booleanas para controlas a maquina de estados dende o fio da camara
global mirada_centrada, ollos_pechados, man_aberta, fin, porcentaxes
porcentaxes = (0,0)
mirada_centrada = True
ollos_pechados = False
man_aberta = False
fin = False

# Este fio encargase de ir lendo o video e procesandoo cos detectores, cambiando os flags que controlan a maquina de estados en funcion dos resultados que nos dean
def threadcamera():
    global mirada_centrada, ollos_pechados, man_aberta, fin, porcentaxes
    ollo_dereito_aberto = True
    ollo_esquerdo_aberto = True
    dim = (250,100)

    while True:
        camara = cv.VideoCapture(0)
        frame_counter = 0
        start_time_fps = time.time()
        ollos_pechados_counter = 0

        # Inicializamos os tres detectores
        face_detection = mp_centroides.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        face_mesh = mp_landmarks.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)#, max_num_faces=1)
        mans = mp.solutions.hands.Hands(static_image_mode=False)#, max_num_hands=1)

        while camara.isOpened():
            frame_counter += 1
            exito, frame = camara.read()
            # Se a maquina de estados alcanza o estado fin ou non hay exito ao ler o fram da camara, rompese o bucle e pechase a camara
            if (not exito) or fin:
                break
            # Procesamento do frame:
            _height, _width = frame.shape[:2]
            # Mediapipe emprega o formato RGB, pero opencv o BGR!!
            # Convertemos o frame de un a outro antes de pasarllo aos 3 detectores 
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            result_mans = mans.process(frame_rgb)
            result_face_mesh = face_mesh.process(frame_rgb)
            result_face_detection = face_detection.process(frame_rgb)

            # Detector das mans. E capaz de localizar landmarks da man e diferenciar distintas poses delas. Desta maneira permitenos detectar se o alumno ten
            # a man levantada (tanto se a ten aberta como se esta so co indice levantado). Se se detecta algun deses xestos, ponse a True o booleano man_aberta.
            # Se non se detecta ningunha man na imaxe pasase ao else para poñe man_aberta a False.
            if result_mans.multi_hand_landmarks is not None:
                man_aberta = False
                for hand_landmarks, _ in zip(result_mans.multi_hand_landmarks, result_mans.multi_handedness):
                    # Calculanse un rectangulo contedor cas dimensións máximas en X e Y dos landmarks da man detectados e o seu centro xeometrico
                    rectangulo_contedor = calc_rectangulo_contedor(frame, hand_landmarks)
                    centro = (int((rectangulo_contedor[0] + rectangulo_contedor[2]) / 2), int((rectangulo_contedor[1] + rectangulo_contedor[3]) / 2))
                    # Obtense a lista cas landmarks da man detectada
                    landmark_list = calc_lista_landmarks(frame, hand_landmarks)
                    # Identificase o xesto ao que corresponde a pose das landmarks detectadas calculadas
                    hand_sign_id = classify_landmark_list(landmark_list)
                    # Engadense o rectangulo contedor e mais os landmarks ao frame
                    frame = debuxa_landmarks(frame, landmark_list, rectangulo_contedor)

                    # Se a man esta aberta, tomanse as coordenadas x e y coma as do centro do rectangulo contedor, se estamos sinalando co dedo
                    # indice, x e y estaran na punta do dedo, e se a man esta pechada ou facendo outro xesto poñense como None
                    if hand_sign_id == 0: # MAN ABERTA
                        x = centro[0]
                        y = centro[1]
                        # Ao final comentamos esta liña para non considerar o xesto ca man aberta, porque por exemplo esto poderia dar falsos positivos ao rascar a cara
                        # ou se temos a cabeza apoiada na man (ainda que eso tamen poderia ser un sintoma de estar perdendo a atencion a clase)
                        #man_aberta = True
                    elif hand_sign_id == 2: # DEDO INDICE APUNTANDO
                        x = landmark_list[8][0]
                        y = landmark_list[8][1]
                        man_aberta = True
                    else:
                        x = None
                        y = None
                    # Se existen x e y debuxase un circulo vermello nesas coordenadas
                    if x is not None and y is not None: cv.circle(frame, (x, y), 5, RED, -1)
            # Se o detector falla ou directamente se non se detecta ningunha man no frame
            else:
                man_aberta = False
                x = None
                y = None

            # Detector de landmarks. Busca caras e se unha procede a aplicarlle unha mascara con landmarks predefinidos. De esta maneira, permitenos atopar a area
            # exacta dos ollos e recortala. Con estes recortes, ao aplicarlle un limiar para binarizalos, poderemos calcular a porcentaxe de "ollos pechados" do alumno.
            # Se a porcentaxe é menor do 5%, ponse a True o booleano ollos_pechados.
            # Se non se detecta ningunha cara, poñense directamente a True o flag ollos_pechados e a False mirada_centrada.
            if result_face_mesh.multi_face_landmarks:
                # Crease unha copia do frame e debúxase sobre a cara detectada a máscara de landmarks. Mostrase o resultado nunha ventá independente.
                frame_copy = frame.copy()
                for face_landmarks in result_face_mesh.multi_face_landmarks:
                    mp_drawing.draw_landmarks(frame_copy, face_landmarks, mp_landmarks.FACEMESH_TESSELATION, 
                                              mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1), mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1))
                    cv.imshow("Mascara de landmarks", frame_copy)

                # Obteñense as coordenadas no frame de todas as landmarks da máscara. As que corresponden cos beizos e os dous ollos engadense como circulos de cores ao frame.
                mesh_coords = [(int(point.x*_width), int(point.y*_height)) for point in result_face_mesh.multi_face_landmarks[0].landmark]
                [cv.circle(frame, mesh_coords[p], 1, GREEN , -1, cv.LINE_AA) for p in BEIZOS]
                [cv.circle(frame, mesh_coords[p], 1, MAGENTA ,- 1, cv.LINE_AA) for p in OLLO_DER]
                [cv.circle(frame, mesh_coords[p], 1, MAGENTA , -1, cv.LINE_AA) for p in OLLO_ESQ]
                # Identificase a area de cada un dos ollos empregando cadanseus landmarks, e devolvense como frames individuais que so conteñen un ollo cada un
                recorte_ollo_der, recorte_ollo_esq = recorte_ollos(frame, [mesh_coords[p] for p in OLLO_DER], [mesh_coords[p] for p in OLLO_ESQ], True)
                # Se ambalos dous teñen un tamaño maior que cero (i.e. existen)
                if recorte_ollo_der.size > 0 and recorte_ollo_esq.size > 0:
                    # Procesamos os frames cos recortes cunha gaussiana, un suavizado e posteriormente un limiar para obter dous frames binarios
                    _, ollo_binario_esquerdo = cv.threshold(cv.medianBlur(cv.GaussianBlur(recorte_ollo_esq, (9,9), 0), 3), 99, 255, cv.THRESH_BINARY)
                    _, ollo_binario_dereito = cv.threshold(cv.medianBlur(cv.GaussianBlur(recorte_ollo_der, (9,9), 0), 3), 99, 255, cv.THRESH_BINARY)
                    # A continuacion, comprobarase se os ollos estan abertos. Percorrense todos os pixeles de cada imaxe binaria e vaise contando cantos son brancos.
                    # Se a proporción de pixeles brancos respecto dos totais e mais que un 5%, considerase o ollo como aberto (cada un individualmente).
                    der_count = 0
                    esq_count = 0
                    # Primeiro comprobase o ollo dereito
                    for fila in ollo_binario_dereito:
                        for pixel in fila:
                            if pixel == 255:
                                der_count += 1
                    if der_count > 0.05*len(ollo_binario_dereito)*len(ollo_binario_dereito[0]): #and (der_count < 0.9*len(ollo_binario_dereito)*len(ollo_binario_dereito[0])):
                        ollo_dereito_aberto = True
                    else: ollo_dereito_aberto = False
                    # Agora comprobase o ollo esquerdo
                    for fila in ollo_binario_esquerdo:
                        for pixel in fila:
                            if pixel == 255: 
                                esq_count += 1
                    if esq_count > 0.05*len(ollo_binario_esquerdo)*len(ollo_binario_esquerdo[0]): #and (esq_count < 0.9*len(ollo_binario_esquerdo)*len(ollo_binario_esquerdo[0])):
                        ollo_esquerdo_aberto = True
                    else: ollo_esquerdo_aberto = False

                    porcentaxes =(esq_count/(len(ollo_binario_esquerdo)*len(ollo_binario_esquerdo[0])), der_count/(len(ollo_binario_dereito)*len(ollo_binario_dereito[0])))

                    # Se algun dos dous ollos esta pechado, sumase 1 ao contador de ollos pechados.
                    if not ollo_dereito_aberto or not ollo_esquerdo_aberto:
                        ollos_pechados_counter += 1
                    # Se os dous ollos estan abertos, reiniciase o contador
                    else:
                        ollos_pechados_counter = 0
                        ollos_pechados = False
                    # Mentres se manteñan os ollos pechados, incrementarase o contador. Cando este chega a 10, ponse a True o booleano ollos_pechados.
                    if ollos_pechados_counter > 10: ollos_pechados = True

                    # Mostranse as imaxes dos recortes e as binarias de cada ollo. Faise un resize para facelo mais comodo ao mostralo, xa que as ventas son moi pequenas
                    # porque teñen o mesmo tamaño que os ollos no frame orixinal, ainda que non seria necesario.
                    cv.imshow('Ollo dereito recortado', cv.resize(recorte_ollo_der, dim, interpolation = cv.INTER_AREA))
                    cv.imshow('Ollo dereito bin', cv.resize(ollo_binario_dereito, dim, interpolation = cv.INTER_AREA))
                    cv.imshow('Ollo esquerdo recortado', cv.resize(recorte_ollo_esq, dim, interpolation = cv.INTER_AREA))
                    cv.imshow('Ollo esquerdo bin', cv.resize(ollo_binario_esquerdo, dim, interpolation = cv.INTER_AREA))
                    #cv.imshow('Ollo dereito recortado', recorte_ollo_der)
                    #cv.imshow('Ollo dereito bin', ollo_binario_dereito)
                    #cv.imshow('Ollo esquerdo recortado', recorte_ollo_esq)
                    #cv.imshow('Ollo esquerdo bin', ollo_binario_esquerdo)

                # Se algunha das duas imaxes dos ollos non ten polo menos tamaño = 1
                else: # tamaño dos ollos recortados == 0
                    ollos_pechados = True
                    print('Os ollos non foron detectados correctamente')

                # Detector de centroides. Calcula os centroides dos rasgos faciais mais importantes, e mide o angulo entre a liña que une os dous ollos coa
                # que une o punto medio entre os ollos co nariz. Se o angulo e menor que 65 grados, considerase que non se esta a mirar cara a pantalla,
                # polo que se pon a False o flag mirada_centrada
                if result_face_detection.detections:
                    _nariz = mp_centroides.get_key_point(result_face_detection.detections[0], NARIZ)
                    _ollo1 = mp_centroides.get_key_point(result_face_detection.detections[0], OLLO1)
                    _ollo2 = mp_centroides.get_key_point(result_face_detection.detections[0], OLLO2)
                    #_boca = mp_centroides.get_key_point(results2.detections[0], BOCA)

                    # Achar o punto intermedio entre os ollos
                    x_nariz = int(_nariz.x * _width)
                    y_nariz = int(_nariz.y * _height)
                    x_ollo1 = int(_ollo1.x * _width)
                    y_ollo1 = int(_ollo1.y * _height)
                    x_ollo2 = int(_ollo2.x * _width)
                    y_ollo2 = int(_ollo2.y * _height)
                    x_ollo_medio = (x_ollo1 + x_ollo2) // 2
                    y_ollo_medio = (y_ollo1 + y_ollo2) // 2
                    # Debuxamos un pequeno circulo en cada centroide
                    cv.circle(frame, (x_ollo_medio, y_ollo_medio), 2, RED , -1)
                    cv.circle(frame, (x_nariz, y_nariz), 2, GREEN , -1)
                    cv.circle(frame, (x_ollo1, y_ollo1), 2, ORANGE , -1)
                    cv.circle(frame, (x_ollo2, y_ollo2), 2, YELLOW , -1)
                    #cv.circle(frame, (int(_boca.x * _width), int(_boca.y * _height)), 2, RED , -1)

                    # Calculase a recta que une os dous ollos a traves da sua pendente
                    if x_ollo2 - x_ollo1 != 0: m = (y_ollo2 - y_ollo1) / (x_ollo2 - x_ollo1)
                    else: m = 0
                    b = y_ollo1 - m * x_ollo1
                    # Calculase a recta que une o nariz co punto medio dos ollos
                    if x_ollo_medio - x_nariz != 0: m2 = (y_ollo_medio - y_nariz) / (x_ollo_medio - x_nariz)
                    else: m2 = 0
                    b2 = y_nariz - m2 * x_nariz
                    # Calculase o angulo entre as rectas
                    if 1 + m * m2 != 0: angulo = math.atan((m - m2) / (1 + m * m2))
                    else: angulo = 0
                    angulo = abs(angulo * 180 / math.pi)
                    # Se o angulo e maior que 65 grados, considerase que a persona esta mirando cara diante
                    if angulo > 65: mirada_centrada = True
                    else: mirada_centrada = False
                    # Engadense as rectas ao frame
                    cv.line(frame, (0, int(b)), (int(_width), int(m * _width + b)), BLUE, 1)
                    cv.line(frame, (0, int(b2)), (int(_width), int(m2 * _width + b2)), RED, 1)

                else:
                    mirada_centrada = False
                    #print('Erro no segundo detector')       
            else:
                mirada_centrada = False
                ollos_pechados = True
                #print('Erro no primeiro detector')

            # Calculamos os fps do video e engadimos a info ao curruncho superior esquerdo do frame
            end_time = time.time() - start_time_fps
            fps = frame_counter/end_time
            frame = engade_texto(frame, f'FPS: {round(fps, 1)}')
            # Mostramos o frame
            cv.imshow("Clase de vision artificial", frame)
            key = cv.waitKey(1)
            if key == ord('q') or key == ord('Q'):
                break

        # Se se chega ao estado fin ou falla a lectura dos frames, pechanse as ventas, liberase a camara e rompese o bucle principal deste fio
        if fin == True or exito == False:
            cv.destroyAllWindows()
            camara.release()
            break

# Este fio conten a implementacion da maquina de estados que simula un profesor que permite a interaccion co video mediante as variables globais booleanas definidas ao 
# comezo do programa.
def threadfsm():
    global mirada_centrada, ollos_pechados, man_aberta, fin, porcentaxes
    m = FSM() # Inicializamos a maquina de estados
    cont_aserc = 0 # Contador de paciencia do profesor
    # Comeza a clase:
    while True:
        # ESTADO 0 (INICIO): 'clase'
        # Aqui poderianse ir consumindo as liñas dun .txt cas leccions. Poderiamos ter un menu que permitese escoller entre distintas leccions antes de comezar.
        if m.state == 'clase':
            fromClase = True
            cont_aserc = 0
            print("Blah blah blah blah blah...", porcentaxes)
            time.sleep(5)
            # TRANSICIONS:
            # Cando o profesor detecta que o alumno ten os ollos pechados ou esta mirando cara a outro lado en lugar de atender a clase (ou se os detectores dan erros por 
            # exemplo por estar o alumno mal situado con respecto da camara ou pola iluminacion), pasase ao estado pausa
            if ollos_pechados == True or mirada_centrada == False:
                m.state = 'pausa'
            # Se se detecta a unha man aberta ou co indice levantado, transicionase ao estado preguntar para atender a dubida do alumno                
            elif man_aberta == True:
                m.state = 'pregunta'

        # ESTADO 1: 'pausa'
        # A este estado chegarase dende o de clase se se detectou que o alumno deixou de prestar atencion, para darlle un tempo a retomala antes de facer un comentario asertivo
        # dende o estado asercion. Se retoma atencion antes de iso, retornase ao estado clase.
        # Tamen existe a opcion de chegar a este estado dende o de asercion mentres o profe teña paciencia, para darlle de novo un tempiño a ver se retoma a atencion antes de
        # ter que volver a asercion.
        if m.state == 'pausa':
            if fromClase == True:
                print("Parece que non estas prestando atencion....", porcentaxes)
                fromClase = False
            else:
                print("Segues sen prestar atencion??....", porcentaxes)
            time.sleep(7)
            # TRANSICIONS:
            # Mentres o alumno este cos ollos pechados ou sen mirar cara adiante, pasarase ao estado asercion
            if ollos_pechados == True or mirada_centrada == False:
                m.state = 'asercion'
            # Se recuperou a atencion, retomase a clase
            else:
                m.state = 'clase'

        # ESTADO 2: 'asercion'
        # Cada vez que se chega a este estado incrementase en 1 o contador de paciencia. En funcion do valor deste, dirase un dialogo asertivo con distintos niveis de 
        # agresividade ca fin de recuperar a atencion do alumno.
        if m.state == 'asercion':
            cont_aserc += 1
            print(porcentaxes)
            if cont_aserc < 3:
                print("Oes, poderias facerme caso?")
            elif cont_aserc < 5:
                print("Se non me atendes, non poderemos continuar coa clase")
            elif cont_aserc < 7:
                print("ESTASEME REMATANDO A PACIENCIA. DEIXA O MOBIL.")
            time.sleep(5)
            # TRANSICIONS:
            # Cando o contador de paciencia chega a un límite de 7, pasase directamente ao estado fin sen importar o estado de ningunha outra variable
            if cont_aserc >= 7:
                m.state = 'fin'
            # Se o profe ainda ten paciencia, e o alumno volve a mirar ao frente e ademais ten os ollos pechados, regresase ao estado clase para continuar ca leccion
            elif mirada_centrada == True and ollos_pechados == False:
                m.state = 'clase'
            # Se o alumno segue sen prestar atención, a máquina volve ao estado pausa, para darlle un tempo ao alumno antes de volver a chamarlle a atencion 
            else:
                m.state = 'pausa'

        # ESTADO 3: 'pregunta'
        # Cando se chega a este estado, o profesor queda a escoita da pregunta do alumno, antes de retomar a clase. No noso caso non lle demos mais importancia que a propia
        # da deteccion da man levantada, pero o seu seria incorporar un novo estado no que o profesor respondera a pregunta ou, polo menos, se apuntase dalgunha forma para
        # que fose respondida mais tarde polo profesor de verdade.
        if m.state == 'pregunta':
            print("Semella que tes unha dúbida, adiante, estoute a escoitar...")
            time.sleep(8)
            m.state = 'clase'

        # ESTADO 4 (FINAL): 'fin'
        # Ao rematarselle a paciencia ao profe chegase a este estado. Ao poñer a True a variable global fin facemos que se termine o fio da camara (pechando ben as ventas, 
        # a camara e o bucle). Despois, rompemos o bucle desta maquina de estados, o que polo seu lado termina este fio, poñendo fin a execucion de todo o programa.
        if m.state == 'fin':
            print("Mellor deixámolo para outro momento... ¡¡Adeus!!")
            fin = True
            break

# MAIN
if __name__ == '__main__':
    # Creanse e lanzanse dous fios, o primeiro tera como target a funcion threadcamera e o segundo a funcion threadfsm
    threads = []
    t1 = threading.Thread(target=threadcamera)
    t1.daemon = True
    threads.append(t1)
    t1.start()
    t2 = threading.Thread(target=threadfsm)
    t2.daemon = True
    threads.append(t2)
    t2.start()

    for t in threads:
        t.join()
