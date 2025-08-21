import copy
import itertools
from collections import deque, Counter

def pre_process_point_history(image, point_history):
    """Pré-processa histórico de pontos"""
    image_width, image_height = image.shape[1], image.shape[0]
    temp_point_history = copy.deepcopy(point_history)
    
    # Converte para coordenadas relativas
    base_x, base_y = temp_point_history[0]
    for index, point in enumerate(temp_point_history):
        temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height
    
    # Converte para lista unidimensional
    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))
    
    return temp_point_history

def get_most_common_gesture(gesture_history):
    """Retorna o gesto mais comum do histórico"""
    if gesture_history:
        most_common = Counter(gesture_history).most_common(1)
        return most_common[0][0] if most_common else 0
    return 0