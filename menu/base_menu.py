from abc import ABC, abstractmethod
import cv2

class BaseMenu(ABC):
    def __init__(self, app):
        self.app = app
        self.options = []
        self.selected_index = 0
    
    @abstractmethod
    def handle_key(self, key):
        """Processa teclas pressionadas no menu"""
        pass
    
    @abstractmethod
    def render(self, image):
        """Renderiza o menu na imagem"""
        pass
    
    def navigate_up(self):
        """Navega para opção acima"""
        if self.options:
            self.selected_index = (self.selected_index - 1) % len(self.options)
    
    def navigate_down(self):
        """Navega para opção abaixo"""
        if self.options:
            self.selected_index = (self.selected_index + 1) % len(self.options)
    
    def get_selected_option(self):
        """Retorna a opção selecionada"""
        if self.options and 0 <= self.selected_index < len(self.options):
            return self.options[self.selected_index]
        return None