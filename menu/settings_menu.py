import cv2
from .base_menu import BaseMenu
from config.settings import app_config

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
        elif key == ord('w') or key == 82:  # Seta para cima
            self.navigate_up()
        elif key == ord('s') or key == 84:  # Seta para baixo
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
        cv2.putText(image, "CONFIGURAÇÕES", (width//2 - 100, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Opções do menu
        for i, option in enumerate(self.options):
            color = (0, 255, 0) if i == self.selected_index else (255, 255, 255)
            y_pos = 100 + i * 40
            cv2.putText(image, f"{'>' if i == self.selected_index else ' '} {option}", 
                       (width//2 - 80, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Instruções
        cv2.putText(image, "Setas: Navegar  Enter: Alternar/Selecionar  ESC: Voltar", 
                   (width//2 - 200, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return image