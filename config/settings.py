import os

# Constantes de modos
MODO_NORMAL = 0
MODO_TREINAMENTO = 1
MODO_ESCRITA = 2
MODO_ESCRITA_STOP = 3
MODO_MENU = 4

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

# Classe para gerenciar configurações persistentes
class AppConfig:
    def __init__(self):
        self.tts_enabled = True
    
    def toggle_tts(self):
        self.tts_enabled = not self.tts_enabled
        return self.tts_enabled
    
    def get_tts_status(self):
        return "ON" if self.tts_enabled else "OFF"

# Instância global de configuração
app_config = AppConfig()