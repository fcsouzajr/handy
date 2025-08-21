import cv2
import mediapipe as mp
from collections import deque

from config.settings import *
from models.keypoint_classifier import KeyPointClassifier
from models.point_history_classifier import PointHistoryClassifier
from models.labels import load_labels
from utils.image_processing import *
from utils.gesture_processing import *
from utils.tts_engine import TTSEngine
from camera.camera_capture import CameraCapture

from modes.normal_mode import NormalMode
from modes.training_mode import TrainingMode
from modes.writing_mode import WritingMode
from modes.stop_mode import StopMode

class HandGestureApp:
    def __init__(self):
        # Inicializa componentes
        self.camera = CameraCapture()
        self.tts_engine = TTSEngine()
        
        # Inicializa modelos
        self.keypoint_classifier = KeyPointClassifier()
        self.point_history_classifier = PointHistoryClassifier()
        
        # Carrega rótulos
        self.keypoint_classifier_labels = load_labels(
            'keypoint_test/model/keypoint_classifier/keypoint_classifier_label.csv')
        self.point_history_classifier_labels = load_labels(
            'keypoint_test/model/point_history_classifier/point_history_classifier_label.csv')
        
        # Configura MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(**MEDIAPIPE_CONFIG)
        
        # Históricos
        self.point_history = deque(maxlen=HISTORY_LENGTH)
        self.finger_gesture_history = deque(maxlen=HISTORY_LENGTH)
        
        # Modos
        self.modes = {
            MODO_NORMAL: NormalMode(self),
            MODO_TREINAMENTO: TrainingMode(self),
            MODO_ESCRITA: WritingMode(self),
            MODO_ESCRITA_STOP: StopMode(self)
        }
        self.current_mode = MODO_NORMAL
        self.current_mode_instance = self.modes[self.current_mode]
        
        # Estado da aplicação
        self.frase_atual = []
        self.palavra_atual = []
        self.output_dir = OUTPUT_DIR
        self.pre_processed_landmark_list = None
    
    def run(self):
        if not self.camera.is_opened():
            print("Erro: Não foi possível abrir a câmera")
            return
        
        while True:
            ret, frame = self.camera.read_frame()
            if not ret:
                break
            
            # Processamento da imagem
            image = flip_image(frame)
            debug_image = copy.deepcopy(image)
            image_rgb = convert_to_rgb(image)
            
            # Detecção de mãos
            results = self.process_hands(image_rgb)
            
            # Atualiza display e processa gestos
            self.current_mode_instance.update_display(debug_image)
            if results.multi_hand_landmarks:
                self.process_hand_results(results, debug_image)
            
            # Exibe imagem
            cv2.imshow("Hand Gesture Recognition", debug_image)
            
            # Processa teclas
            if self.process_keys():
                break
        
        self.cleanup()
    
    def process_hands(self, image):
        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True
        return results
    
    def process_hand_results(self, results, debug_image):
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, 
                                            results.multi_handedness):
            # Calcula landmarks
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)
            
            # Pré-processamento
            self.pre_processed_landmark_list = pre_process_landmark(landmark_list)
            
            # Classificação
            hand_sign_id = self.keypoint_classifier(self.pre_processed_landmark_list)
            hand_sign_label = self.keypoint_classifier_labels[hand_sign_id]
            
            # Atualiza histórico de pontos
            self.update_point_history(hand_sign_id, landmark_list)
            
            # Classificação de movimento
            finger_gesture_id = self.classify_finger_gesture(debug_image)
            finger_gesture_label = self.point_history_classifier_labels[finger_gesture_id]
            
            # Desenha landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                debug_image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Exibe informações
            cv2.putText(debug_image, f"Sinal: {hand_sign_label}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(debug_image, f"Movimento: {finger_gesture_label}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Processa gesto no modo atual
            self.current_mode_instance.process_gesture(hand_sign_label, finger_gesture_label)
    
    def update_point_history(self, hand_sign_id, landmark_list):
        if hand_sign_id == 2:  # Point gesture
            self.point_history.append(landmark_list[8])
        else:
            self.point_history.append([0, 0])
    
    def classify_finger_gesture(self, debug_image):
        if len(self.point_history) == HISTORY_LENGTH:
            pre_processed_point_history = pre_process_point_history(
                debug_image, self.point_history)
            finger_gesture_id = self.point_history_classifier(pre_processed_point_history)
            self.finger_gesture_history.append(finger_gesture_id)
            return get_most_common_gesture(self.finger_gesture_history)
        return 0
    
    def process_keys(self):
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            return True
        
        # Muda modos
        if key == ord('0'):
            self.change_mode(MODO_TREINAMENTO)
        elif key == ord('1'):
            self.change_mode(MODO_NORMAL)
        elif key == ord('2'):
            self.change_mode(MODO_ESCRITA)
        elif key == ord('3') and self.current_mode == MODO_ESCRITA:
            self.change_mode(MODO_ESCRITA_STOP)
        
        # Processa teclas no modo atual
        new_mode = self.current_mode_instance.handle_key(key)
        if new_mode is not None:
            self.change_mode(new_mode)
        
        return False
    
    def change_mode(self, new_mode):
        if new_mode != self.current_mode:
            print(f"[Modo] Mudando para {new_mode}")
            self.current_mode = new_mode
            self.current_mode_instance = self.modes[new_mode]
    
    def cleanup(self):
        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = HandGestureApp()
    app.run()