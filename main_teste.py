from keypoint_test.model import KeyPointClassifier
from keypoint_test.model import PointHistoryClassifier
import cv2
import mediapipe as mp
import csv
import os
import string
import time
import numpy as np
import joblib
import copy
import itertools
from collections import deque, Counter

# Inicializa os classificadores personalizados
keypoint_classifier = KeyPointClassifier()
point_history_classifier = PointHistoryClassifier()

# Carrega os rótulos dos classificadores
with open('keypoint_test/model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
with open('keypoint_test/model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
    point_history_classifier_labels = [row[0] for row in csv.reader(f)]

# Configurações do MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Configurações para histórico de pontos
history_length = 16
point_history = deque(maxlen=history_length)
finger_gesture_history = deque(maxlen=history_length)

# Pasta para salvar os dados
output_dir = "dados_libras"
os.makedirs(output_dir, exist_ok=True)

# Modos de operação
MODO_NORMAL = 0
MODO_TREINAMENTO = 1
MODO_ESCRITA = 2
MODO_ESCRITA_STOP = 3
modo_atual = MODO_NORMAL

# Variáveis para armazenar a frase
frase_atual = []
palavra_atual = []
ultima_letra = None
letra_atual = None
ultimo_tempo_letra = 0
COOLDOWN_LETRAS = 0.5

# Captura de vídeo
cap = cv2.VideoCapture(0)

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    
    for landmark in landmarks.landmark:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    
    # Converte para coordenadas relativas
    base_x, base_y = temp_landmark_list[0]
    for index, landmark_point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    
    # Converte para lista unidimensional
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    
    # Normalização
    max_value = max(list(map(abs, temp_landmark_list)))
    temp_landmark_list = [n/max_value for n in temp_landmark_list]
    
    return temp_landmark_list

def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]
    temp_point_history = copy.deepcopy(point_history)
    
    # Converte para coordenadas relativas
    base_x, base_y = temp_point_history[0]
    for index, point in enumerate(temp_point_history):
        temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height
    
    # Converte para lista unidimensional
    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))
    
    return temp_point_history

def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == MODO_TREINAMENTO and (0 <= number <= 26):
        letra = chr(ord('a') + number)
        csv_path = os.path.join(output_dir, f"{letra}.csv")
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Processamento da imagem
    image = cv2.flip(frame, 1)  # Espelha a imagem
    debug_image = copy.deepcopy(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detecção de mãos
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    
    # Exibe o modo atual na tela
    if modo_atual == MODO_TREINAMENTO:
        modo_texto = "MODO: TREINAMENTO"
        cor_modo = (255, 0, 0)  # Vermelho
    elif modo_atual == MODO_ESCRITA:
        modo_texto = "MODO: ESCRITA"
        cor_modo = (0, 255, 0)  # Verde
    elif modo_atual == MODO_ESCRITA_STOP:
        modo_texto = "MODO: ESCRITA (STOP)"
        cor_modo = (0, 255, 255)  # Amarelo
    else:
        modo_texto = "MODO: NORMAL"
        cor_modo = (0, 0, 255)  # Azul
    
    cv2.putText(debug_image, modo_texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, cor_modo, 2)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Calcula os landmarks
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)
            
            # Pré-processamento para classificação
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            
            # Classificação do gesto da mão
            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
            
            # Atualiza o histórico de pontos se for gesto de apontar (ID 2)
            if hand_sign_id == 2:  # Point gesture
                point_history.append(landmark_list[8])  # Ponto do dedo indicador
            else:
                point_history.append([0, 0])
            
            # Classificação do gesto do dedo (movimento)
            finger_gesture_id = 0
            pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)
            if len(pre_processed_point_history_list) == (history_length * 2):
                finger_gesture_id = point_history_classifier(pre_processed_point_history_list)
            
            # Atualiza histórico de gestos
            finger_gesture_history.append(finger_gesture_id)
            most_common_fg_id = Counter(finger_gesture_history).most_common(1)
            
            # Obtém os rótulos das classificações
            hand_sign_label = keypoint_classifier_labels[hand_sign_id]
            finger_gesture_label = point_history_classifier_labels[most_common_fg_id[0][0]] if most_common_fg_id else ""
            
            # Desenha os landmarks e informações
            mp.solutions.drawing_utils.draw_landmarks(
                debug_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)
            
            cv2.putText(debug_image, f"Sinal: {hand_sign_label}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(debug_image, f"Movimento: {finger_gesture_label}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Nos modos de escrita, usa a classificação para formar palavras
            if modo_atual in [MODO_NORMAL, MODO_ESCRITA]:
                letra_atual = hand_sign_label  # Usa o rótulo do classificador como letra
    
    # Exibe a palavra e frase atual nos modos escrita
    if modo_atual in [MODO_ESCRITA, MODO_ESCRITA_STOP]:
        cv2.putText(debug_image, f"Palavra: {' '.join(palavra_atual)}", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(debug_image, f"Frase: {' '.join(frase_atual)}", (10, 190), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Exibe imagem
    cv2.imshow("Hand Gesture Recognition", debug_image)

    # Teclas de controle
    key = cv2.waitKey(1) & 0xFF
    
    # Teclas para mudança de modo
    if key == 27:  # ESC
        break
    elif key == ord('0'):
        modo_atual = MODO_TREINAMENTO
        print("[Modo] Entrou no modo de treinamento")
    elif key == ord('1'):
        modo_atual = MODO_NORMAL
        frase_atual = []
        palavra_atual = []
        print("[Modo] Voltou ao modo normal")
    elif key == ord('2'):
        modo_atual = MODO_ESCRITA
        ultimo_tempo_letra = 0
        print("[Modo] Entrou no modo escrita")
    elif key == ord('3'):
        modo_atual = MODO_ESCRITA_STOP
        print("[Modo] Entrou no modo escrita (stop) - Captura pausada")
    
    # Teclas para controle da frase (apenas no modo escrita)
    tempo_atual = time.time()
    
    if modo_atual == MODO_ESCRITA:
        if key == 32:  # Espaço - finaliza palavra atual
            if palavra_atual:
                frase_atual.append(''.join(palavra_atual))
                palavra_atual = []
                ultimo_tempo_letra = tempo_atual
                print(f"Palavra adicionada: {' '.join(frase_atual)}")
        
        elif key == 13:  # Enter - finaliza frase
            if palavra_atual:
                frase_atual.append(''.join(palavra_atual))
                palavra_atual = []
            print("\n--- FRASE FINALIZADA ---")
            print(' '.join(frase_atual))
            print("-----------------------\n")
            frase_atual = []
            ultimo_tempo_letra = tempo_atual
        
        # Verifica o cooldown antes de adicionar nova letra
        elif (letra_atual and letra_atual != ultima_letra and 
              (tempo_atual - ultimo_tempo_letra) >= COOLDOWN_LETRAS):
            palavra_atual.append(letra_atual)
            ultima_letra = letra_atual
            ultimo_tempo_letra = tempo_atual
            print(f"Letra adicionada: {letra_atual} | Palavra atual: {''.join(palavra_atual)}")

cap.release()
cv2.destroyAllWindows()