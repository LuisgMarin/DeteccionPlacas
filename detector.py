import numpy as np
import cv2
import json
import numpy as np
import cv2
import easyocr
import re
from datetime import datetime
from utils import validar_formato_placa, verificar_pico_y_placa

reader = easyocr.Reader(['es'], gpu=False)

class PlacaDetector:
    def __init__(self):
        self.last_detection = None
        self.confidence_threshold = 0.45  # Umbral más bajo para detectar mejor caracteres individuales
        self.min_plate_width = 100  # Ancho mínimo de la placa en píxeles
        self.min_plate_height = 50  # Alto mínimo de la placa en píxeles
        
    def preprocess_image(self, image):
        """Preprocesa la imagen para mejorar la detección de placas amarillas"""
        # Convertir a HSV para mejor detección del color amarillo
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Rango de color amarillo típico de placas colombianas
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        
        # Crear máscara para color amarillo
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Aplicar operaciones morfológicas para limpiar la máscara
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Aplicar la máscara a la imagen original
        yellow_regions = cv2.bitwise_and(image, image, mask=mask)
        
        return yellow_regions, mask

    def extract_plate_text(self, text):
        """
        Extrae y formatea el texto de la placa en el formato colombiano (XXX ###)
        """
        # Eliminar espacios y caracteres no deseados
        text = ''.join(c for c in text if c.isalnum())
        text = text.upper()
        
        # Patrón para placas colombianas (3 letras seguidas de 3 números)
        match = re.match(r'^([A-Z]{3})(\d{3})$', text)
        if match:
            letters, numbers = match.groups()
            return f"{letters} {numbers}"
        return None

    def detect_plate(self, frame):
        """
        Detecta y extrae el texto de la placa en el frame
        """
        # Preprocesar imagen
        yellow_regions, mask = self.preprocess_image(frame)
        
        # Encontrar contornos en la máscara
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filtrar por tamaño mínimo
            if w < self.min_plate_width or h < self.min_plate_height:
                continue
                
            # Verificar ratio de aspecto típico de placas (2.0 - 2.5)
            aspect_ratio = float(w)/h
            if not (2.0 <= aspect_ratio <= 2.5):
                continue
            
            # Extraer región de la placa
            plate_region = frame[y:y+h, x:x+w]
            
            # Aplicar OCR
            results = reader.readtext(plate_region, 
                                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                                    batch_size=1,
                                    detail=1)
            
            for (bbox, text, confidence) in results:
                if confidence > self.confidence_threshold:
                    formatted_text = self.extract_plate_text(text)
                    if formatted_text:
                        # Dibujar rectángulo y texto
                        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                        cv2.putText(frame, formatted_text, (x, y-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                        
                        return {
                            'text': formatted_text,
                            'confidence': confidence,
                            'bbox': (x,y,w,h)
                        }
        
        return None
    
def gen_frames():
    cap = cv2.VideoCapture(0)
    detector = PlacaDetector()
    while True:
        success, frame = cap.read()
        if not success:
            break
        result = detector.detect_plate(frame)
        if result:
            pico_y_placa_info = verificar_pico_y_placa(result['text'], result['confidence'])
            with open('static/placa_info.json', 'w') as f:
                json.dump({
                    'placa_texto': result['text'],
                    'confianza': f"{result['confidence']:.2%}",
                    'verificacion': pico_y_placa_info['message'],
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }, f)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')