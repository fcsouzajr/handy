import cv2
from .base_menu import BaseMenu
from config.settings import MODO_NORMAL

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
        elif key == ord('w') or key == 82:  # Seta para cima
            self.navigate_up()
        elif key == ord('s') or key == 84:  # Seta para baixo
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
        cv2.putText(image, "MENU PRINCIPAL", (width//2 - 100, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Opções do menu
        for i, option in enumerate(self.options):
            color = (0, 255, 0) if i == self.selected_index else (255, 255, 255)
            y_pos = 100 + i * 40
            cv2.putText(image, f"{'>' if i == self.selected_index else ' '} {option}", 
                       (width//2 - 80, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Instruções
        cv2.putText(image, "Setas: Navegar  Enter: Selecionar  ESC: Sair", 
                   (width//2 - 180, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return image