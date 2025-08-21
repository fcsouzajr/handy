from abc import ABC, abstractmethod

class BaseMode(ABC):
    def __init__(self, app):
        self.app = app
    
    @abstractmethod
    def handle_key(self, key):
        """Processa teclas pressionadas"""
        pass
    
    @abstractmethod
    def update_display(self, debug_image):
        """Atualiza display com informações do modo"""
        pass
    
    @abstractmethod
    def process_gesture(self, hand_sign_label, finger_gesture_label):
        """Processa gestos reconhecidos"""
        pass