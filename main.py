import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import drawing_utils
import csv
import os
import string

import numpy as np
import joblib

modelo = joblib.load("modelo_libras.pkl")

# Caminho do modelo
model_path = os.path.join(".", "hand_landmarker", "hand_landmarker.task")

# Configuração do modelo
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,  # Apenas uma mão por vez para o treinamento
    running_mode=vision.RunningMode.IMAGE
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

# Pasta para salvar os dados
output_dir = "dados_libras"
os.makedirs(output_dir, exist_ok=True)

# Modo inicial
modo_treinamento = False

# Captura de vídeo
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Converte imagem para RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    # Detecta mãos
    result = hand_landmarker.detect(mp_image)

    # Volta pra BGR para exibir
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Exibe o modo atual na tela
    modo_texto = "MODO: TREINAMENTO" if modo_treinamento else "MODO: NORMAL"
    cv2.putText(image_bgr, modo_texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Desenha os landmarks
    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            # Criar lista de landmarks
            landmark_list = landmark_pb2.NormalizedLandmarkList()
            for landmark in hand_landmarks:
                landmark_proto = landmark_pb2.NormalizedLandmark(
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z
                )
                landmark_list.landmark.append(landmark_proto)

            drawing_utils.draw_landmarks(
                image_bgr,
                landmark_list,
                mp.solutions.hands.HAND_CONNECTIONS
            )

            # Se estiver em modo de treinamento e pressionar uma letra, salve os dados
            key = cv2.waitKey(5) & 0xFF
            if modo_treinamento and chr(key).lower() in string.ascii_lowercase:
                letra = chr(key).lower()
                dados = []
                for lm in hand_landmarks:
                    dados.extend([lm.x, lm.y, lm.z])
                dados.append(letra)

                # Salvar no arquivo da letra
                arquivo_csv = os.path.join(output_dir, f"{letra}.csv")
                with open(arquivo_csv, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(dados)

                print(f"[✓] Dados salvos para a letra '{letra}'")
            if not modo_treinamento:
                dados = []
                for lm in hand_landmarks:
                    dados.extend([lm.x, lm.y, lm.z])

                if len(dados) == 63:  # 21 pontos * 3
                    dados_np = np.array(dados).reshape(1, -1)
                    letra_predita = modelo.predict(dados_np)[0]

                    # Mostra a letra na tela
                    cv2.putText(image_bgr, f"Letra: {letra_predita}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Exibe imagem
    cv2.imshow("Hands", image_bgr)

    # Teclas fora do modo de treino (0 e 1)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('0'):
        modo_treinamento = True
        print("[Modo] Entrou no modo de treinamento")
    elif key == ord('1'):
        modo_treinamento = False
        print("[Modo] Voltou ao modo normal")

cap.release()
cv2.destroyAllWindows()
