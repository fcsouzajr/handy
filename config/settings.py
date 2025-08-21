import os

# Constantes de modos
MODO_NORMAL = 0
MODO_TREINAMENTO = 1
MODO_ESCRITA = 2
MODO_ESCRITA_STOP = 3

# Configurações
HISTORY_LENGTH = 16
COOLDOWN_LETRAS = 0.5
OUTPUT_DIR = "keypoint_test/model/keypoint_classifier"

# Configurações MediaPipe
MEDIAPIPE_CONFIG = {
    'static_image_mode': False,
    'max_num_hands': 1,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5
}