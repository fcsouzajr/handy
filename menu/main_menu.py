import cv2
from .base_menu import BaseMenu
from config.settings import MODO_NORMAL
from utils.text_utils import put_unicode_text, put_unicode_text_centered

class MainMenu(BaseMenu):
    def __init__(self, app):
        super().__init__(app)
        self.options = ["Configurações", "Sair do Menu"]
        self.selected_index = 0
    
    def handle_key(self, key):
        """Processa teclas no menu principal"""
        if key == 27:  # ESC - sai do menu
            return MODO_NORMAL
        elif key == 13:  # Enter - seleciona opção
            selected = self.get_selected_option()
            if selected == "Configurações":
                return "SETTINGS_MENU"
            elif selected == "Sair do Menu":
                return MODO_NORMAL
        elif key == ord('w'): # Mover para cima
            self.navigate_up()
        elif key == ord('s'): # Mover para baixo
            self.navigate_down()
        
        return None
    
    def render(self, image):
        """Renderiza o menu principal"""
        height, width = image.shape[:2]
        
        # Fundo semi-transparente
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Título do menu
        image = put_unicode_text_centered(image, "MENU PRINCIPAL", 50, 
                                        font_size=30, color=(255, 255, 255))
        
        # Opções do menu
        for i, option in enumerate(self.options):
            color = (0, 255, 0) if i == self.selected_index else (255, 255, 255)
            y_pos = 100 + i * 40
            
            prefix = "> " if i == self.selected_index else "  "
            full_text = prefix + option
            
            image = put_unicode_text_centered(image, full_text, y_pos, 
                                            font_size=24, color=color)
        
        # Instruções
        cv2.putText(image, "W/S: Navegar  Enter: Selecionar  ESC: Sair", 
                   (width//2 - 180, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return image