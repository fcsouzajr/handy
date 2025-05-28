import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import drawing_utils
import csv
import os
import string
import time
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

# Modos de operação
MODO_NORMAL = 0
MODO_TREINAMENTO = 1
MODO_ESCRITA = 2
MODO_ESCRITA_STOP = 3  # Novo modo
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
    
    cv2.putText(image_bgr, modo_texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, cor_modo, 2)

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
            if modo_atual == MODO_TREINAMENTO and chr(key).lower() in string.ascii_lowercase:
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
            
            # Detecta letras apenas nos modos normal e escrita
            if modo_atual in [MODO_NORMAL, MODO_ESCRITA]:
                dados = []
                for lm in hand_landmarks:
                    dados.extend([lm.x, lm.y, lm.z])

                if len(dados) == 63:  # 21 pontos * 3
                    dados_np = np.array(dados).reshape(1, -1)
                    letra_predita = modelo.predict(dados_np)[0]
                    letra_atual = letra_predita

                    # Mostra a letra na tela
                    cv2.putText(image_bgr, f"Letra: {letra_predita}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Exibe a palavra e frase atual nos modos escrita
    if modo_atual in [MODO_ESCRITA, MODO_ESCRITA_STOP]:
        cv2.putText(image_bgr, f"Palavra: {' '.join(palavra_atual)}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(image_bgr, f"Frase: {' '.join(frase_atual)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Exibe imagem
    cv2.imshow("Hands", image_bgr)

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
        # Limpa as variáveis de escrita ao sair do modo
        frase_atual = []
        palavra_atual = []
        print("[Modo] Voltou ao modo normal")
    elif key == ord('2'):
        modo_atual = MODO_ESCRITA
        # Mantém as variáveis ao entrar no modo escrita
        ultimo_tempo_letra = 0  # Reseta o cooldown ao entrar no modo
        print("[Modo] Entrou no modo escrita")
    elif key == ord('3'):
        modo_atual = MODO_ESCRITA_STOP
        print("[Modo] Entrou no modo escrita (stop) - Captura pausada")
    
    # Teclas para controle da frase (apenas no modo escrita)
    tempo_atual = time.time()  # Obtém o tempo atual
    
    if modo_atual == MODO_ESCRITA:
        if key == 32:  # Espaço - finaliza palavra atual
            if palavra_atual:
                frase_atual.append(''.join(palavra_atual))
                palavra_atual = []
                ultimo_tempo_letra = tempo_atual  # Reseta o cooldown ao adicionar espaço
                print(f"Palavra adicionada: {' '.join(frase_atual)}")
        
        elif key == 13:  # Enter - finaliza frase
            if palavra_atual:
                frase_atual.append(''.join(palavra_atual))
                palavra_atual = []
            print("\n--- FRASE FINALIZADA ---")
            print(' '.join(frase_atual))
            print("-----------------------\n")
            frase_atual = []
            ultimo_tempo_letra = tempo_atual  # Reseta o cooldown ao finalizar frase
        
        # Verifica o cooldown antes de adicionar nova letra
        elif (letra_atual and letra_atual != ultima_letra and 
              (tempo_atual - ultimo_tempo_letra) >= COOLDOWN_LETRAS):
            palavra_atual.append(letra_atual)
            ultima_letra = letra_atual
            ultimo_tempo_letra = tempo_atual  # Atualiza o tempo da última letra
            print(f"Letra adicionada: {letra_atual} | Palavra atual: {''.join(palavra_atual)}")

cap.release()
cv2.destroyAllWindows()