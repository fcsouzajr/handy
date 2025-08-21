import pyttsx3
from config.settings import app_config

class TTSEngine:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.enabled = app_config.tts_enabled
    
    def speak_text(self, text):
        """Fala o texto usando TTS se estiver habilitado"""
        if self.enabled and text:
            self.engine.say(text)
            self.engine.runAndWait()
    
    def speak_word_list(self, word_list):
        """Fala uma lista de palavras se TTS estiver habilitado"""
        if self.enabled and word_list:
            text = ' '.join(word_list)
            self.speak_text(text)
    
    def toggle_enabled(self):
        """Alterna o estado do TTS"""
        self.enabled = not self.enabled
        return self.enabled
    
    def update_from_config(self):
        """Atualiza o estado do TTS baseado na configuração"""
        self.enabled = app_config.tts_enabled