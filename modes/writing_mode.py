import cv2
from .base_mode import BaseMode
from config.settings import MODO_ESCRITA, MODO_ESCRITA_STOP, COOLDOWN_LETRAS
import time

class WritingMode(BaseMode):
    def __init__(self, app):
        super().__init__(app)
        self.mode_id = MODO_ESCRITA
        self.stop_mode = MODO_ESCRITA_STOP
        self.ultima_letra = None
        self.ultimo_tempo_letra = 0
    
    def handle_key(self, key):
        tempo_atual = time.time()
        
        if key == 32:  # Espaço
            self._finalizar_palavra()
        elif key == 13:  # Enter
            self._finalizar_frase()
        elif key == ord('3'):
            return self.stop_mode
        
        return None
    
    def _finalizar_palavra(self):
        if self.app.palavra_atual:
            self.app.frase_atual.append(''.join(self.app.palavra_atual))
            self.app.palavra_atual = []
            self.ultimo_tempo_letra = time.time()
    
    def _finalizar_frase(self):
        self._finalizar_palavra()
        if self.app.frase_atual:
            self.app.tts_engine.speak_word_list(self.app.frase_atual)
            self.app.frase_atual = []
            self.ultimo_tempo_letra = time.time()
    
    def update_display(self, debug_image):
        modo_texto = "MODO: ESCRITA" if self.mode_id == MODO_ESCRITA else "MODO: ESCRITA (STOP)"
        cor_modo = (0, 255, 0) if self.mode_id == MODO_ESCRITA else (0, 255, 255)
        
        cv2.putText(debug_image, modo_texto, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, cor_modo, 2)
        
        # Exibe palavra e frase atual
        cv2.putText(debug_image, f"Palavra: {' '.join(self.app.palavra_atual)}", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(debug_image, f"Frase: {' '.join(self.app.frase_atual)}", (10, 190), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    def process_gesture(self, hand_sign_label, finger_gesture_label):
        if self.mode_id == MODO_ESCRITA_STOP:
            return
        
        tempo_atual = time.time()
        if (hand_sign_label and hand_sign_label != self.ultima_letra and 
            (tempo_atual - self.ultimo_tempo_letra) >= COOLDOWN_LETRAS):
            
            if hand_sign_label.lower() != "stop":
                self.app.palavra_atual.append(hand_sign_label)
            
            self.ultima_letra = hand_sign_label
            self.ultimo_tempo_letra = tempo_atual