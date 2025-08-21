from modes.base_mode import BaseMode
from config.settings import MODO_MENU, MODO_NORMAL
from menu.main_menu import MainMenu
from menu.settings_menu import SettingsMenu

class MenuMode(BaseMode):
    def __init__(self, app):
        super().__init__(app)
        self.mode_id = MODO_MENU
        self.current_menu = "MAIN_MENU"
        self.menus = {
            "MAIN_MENU": MainMenu(app),
            "SETTINGS_MENU": SettingsMenu(app)
        }
    
    def handle_key(self, key):
        """Processa teclas no modo menu"""
        result = self.menus[self.current_menu].handle_key(key)
        
        if result == MODO_NORMAL:
            return MODO_NORMAL
        elif result in ["MAIN_MENU", "SETTINGS_MENU"]:
            self.current_menu = result
            self.menus[self.current_menu].selected_index = 0
        
        return None
    
    def update_display(self, debug_image):
        """Atualiza display com o menu atual"""
        debug_image = self.menus[self.current_menu].render(debug_image)
        return debug_image
    
    def process_gesture(self, hand_sign_label, finger_gesture_label):
        """No modo menu, ignora gestos"""
        pass