import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def put_unicode_text(image, text, position, font_size=20, color=(255, 255, 255)):
    """
    Adiciona texto com caracteres Unicode/UTF-8 em uma imagem OpenCV
    """
    # Converte a imagem OpenCV para PIL
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    
    # Tenta carregar uma fonte que suporte Unicode
    try:
        # Tenta usar Arial que suporta caracteres portugueses
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            # Fallback para fonte padrão
            font = ImageFont.truetype("arialuni.ttf", font_size)
        except:
            # Fallback final
            font = ImageFont.load_default()
    
    # Desenha o texto
    draw.text(position, text, font=font, fill=color)
    
    # Converte de volta para OpenCV
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

def put_unicode_text_centered(image, text, y_position, font_size=20, color=(255, 255, 255)):
    """
    Adiciona texto centralizado horizontalmente com suporte a Unicode
    """
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("arialuni.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # Calcula a largura do texto para centralizar
    if hasattr(font, 'getsize'):
        text_width = font.getsize(text)[0]
    else:
        # Fallback para fontes que não têm getsize
        text_width = len(text) * font_size * 0.6
    
    width = image.shape[1]
    x_position = (width - text_width) // 2
    
    draw.text((x_position, y_position), text, font=font, fill=color)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)