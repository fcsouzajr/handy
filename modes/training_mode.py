import cv2
from .base_mode import BaseMode
from config.settings import MODO_TREINAMENTO
from utils.csv_handler import logging_csv

class TrainingMode(BaseMode):
    def __init__(self, app):
        super().__init__(app)
        self.mode_id = MODO_TREINAMENTO
    
    def handle_key(self, key):
        if key in range(ord('a'), ord('z') + 1):
            letra = chr(key).lower()
            number = ord(letra) - ord('a')
            logging_csv(number, self.app.output_dir, 
                       self.app.pre_processed_landmark_list)
        return None
    
    def update_display(self, debug_image):
        cv2.putText(debug_image, "MODO: TREINAMENTO", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    def process_gesture(self, hand_sign_label, finger_gesture_label):
        pass