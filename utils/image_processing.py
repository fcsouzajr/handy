import cv2
import copy
import itertools
import numpy as np

def flip_image(image):
    """Espelha a imagem"""
    return cv2.flip(image, 1)

def convert_to_rgb(image):
    """Converte BGR para RGB"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def calc_landmark_list(image, landmarks):
    """Calcula lista de landmarks"""
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    
    for landmark in landmarks.landmark:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    
    return landmark_point

def pre_process_landmark(landmark_list):
    """Pré-processa landmarks para classificação"""
    temp_landmark_list = copy.deepcopy(landmark_list)
    
    # Converte para coordenadas relativas
    base_x, base_y = temp_landmark_list[0]
    for index, landmark_point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    
    # Converte para lista unidimensional
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    
    # Normalização
    max_value = max(list(map(abs, temp_landmark_list)))
    temp_landmark_list = [n/max_value for n in temp_landmark_list]
    
    return temp_landmark_list