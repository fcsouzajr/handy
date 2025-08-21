from .base_mode import BaseMode
from config.settings import MODO_ESCRITA_STOP, MODO_ESCRITA, MODO_NORMAL
import cv2

class StopMode(BaseMode):
    def __init__(self, app):
        super().__init__(app)
        self.mode_id = MODO_ESCRITA_STOP
    
    def handle_key(self, key):
        """Processa teclas pressionadas no modo stop"""
        if key == ord('2'):  # Volta para modo escrita normal
            return MODO_ESCRITA
        elif key == ord('1'):  # Volta para modo normal
            # Limpa estado de escrita
            self.app.frase_atual = []
            self.app.palavra_atual = []
            return MODO_NORMAL
        return None
    
    def update_display(self, debug_image):
        """Atualiza display com informações do modo stop"""
        modo_texto = "MODO: ESCRITA (STOP)"
        cor_modo = (0, 255, 255)  # Amarelo
        
        cv2.putText(debug_image, modo_texto, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, cor_modo, 2)
        
        # Exibe palavra e frase atual (apenas para visualização, não captura novas)
        cv2.putText(debug_image, f"Palavra: {' '.join(self.app.palavra_atual)}", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(debug_image, f"Frase: {' '.join(self.app.frase_atual)}", (10, 190), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Mensagem informativa
        cv2.putText(debug_image, "Captura pausada - Pressione '2' para continuar", (10, 230), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def process_gesture(self, hand_sign_label, finger_gesture_label):
        """No modo stop, ignora todos os gestos - captura pausada"""
        pass