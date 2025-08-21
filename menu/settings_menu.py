import cv2
from .base_menu import BaseMenu
from config.settings import app_config
from utils.text_utils import put_unicode_text, put_unicode_text_centered

class SettingsMenu(BaseMenu):
    def __init__(self, app):
        super().__init__(app)
        self.update_options()
        self.selected_index = 0

    def update_options(self):
        """Atualiza as opções do menu com o estado atual"""
        self.options = [f"TTS: {app_config.get_tts_status()}", "Voltar"]
    
    
    def handle_key(self, key):
        """Processa teclas no menu de configurações"""
        if key == 27:  # ESC - volta ao menu principal
            return "MAIN_MENU"
        elif key == 13:  # Enter - seleciona opção
            selected = self.get_selected_option()
            if "TTS:" in selected:
                # Alterna estado do TTS
                app_config.toggle_tts()
                self.update_options()  # Atualiza a exibição
                self.app.tts_engine.update_from_config()  # Atualiza o motor TTS
                print(f"TTS {'ativado' if app_config.tts_enabled else 'desativado'}")
            elif selected == "Voltar":
                return "MAIN_MENU"
        elif key == ord('w'): # Mover para cima
            self.navigate_up()
        elif key == ord('s'): # Mover para baixo
            self.navigate_down()
        
        return None
    
    def render(self, image):
        """Renderiza o menu de configurações"""
        # Atualiza as opções antes de renderizar para garantir que estejam atualizadas
        self.update_options()
        
        height, width = image.shape[:2]
        
        # Fundo semi-transparente
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Título do menu
        image = put_unicode_text_centered(image, "CONFIGURAÇÕES", 50, 
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
        cv2.putText(image, "W/S: Navegar  Enter: Alternar/Selecionar  ESC: Voltar", 
                   (width//2 - 200, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return image