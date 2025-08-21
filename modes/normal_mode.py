import cv2
from .base_mode import BaseMode
from config.settings import MODO_NORMAL

class NormalMode(BaseMode):
    def __init__(self, app):
        super().__init__(app)
        self.mode_id = MODO_NORMAL
    
    def handle_key(self, key):
        if key == ord('1'):
            return self.mode_id  # Permanece no mesmo modo
        return None
    
    def update_display(self, debug_image):
        cv2.putText(debug_image, "MODO: NORMAL", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    def process_gesture(self, hand_sign_label, finger_gesture_label):
        # No modo normal, apenas monitora os gestos
        pass