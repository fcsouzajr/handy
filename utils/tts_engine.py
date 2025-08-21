import pyttsx3

class TTSEngine:
    def __init__(self):
        self.engine = pyttsx3.init()
    
    def speak_text(self, text):
        """Fala o texto usando TTS"""
        self.engine.say(text)
        self.engine.runAndWait()
    
    def speak_word_list(self, word_list):
        """Fala uma lista de palavras"""
        text = ' '.join(word_list)
        self.speak_text(text)