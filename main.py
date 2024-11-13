import numpy as np
from datetime import datetime
import json
import cv2
import easyocr
import time
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

# Inicializar EasyOCR
reader = easyocr.Reader(['en'])

def verificar_pico_y_placa(placa_texto):
    # Filtrar solo los dígitos del texto de la placa
    digitos = ''.join(filter(str.isdigit, placa_texto))
    if not digitos:
        return {
            'message': f"No se encuentra una placa válida para '{placa_texto}'.",
            'status': 'warning'
        }

    ultimo_digito = int(digitos[-1])
    dia_semana = datetime.today().weekday()
    # Lógica de pico y placa según el día de la semana y el último dígito de la placa
    if (dia_semana == 0 and ultimo_digito in [1, 2]) or \
       (dia_semana == 1 and ultimo_digito in [3, 4]) or \
       (dia_semana == 2 and ultimo_digito in [5, 6]) or \
       (dia_semana == 3 and ultimo_digito in [7, 8]) or \
       (dia_semana == 4 and ultimo_digito in [9, 0]):
        return {
            'message': f"La placa '{placa_texto}' tiene pico y placa hoy.",
            'status': 'warning'
        }
    else:
        return {
            'message': f"La placa '{placa_texto}' no tiene pico y placa hoy.",
            'status': 'success'
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/placa_info')
def placa_info():
    with open('static/placa_info.json') as f:
        data = json.load(f)
    return jsonify(data)

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Convertir la imagen a escala de grises
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detectar bordes para resaltar las placas
            edges = cv2.Canny(gray_frame, 30, 200)

            # Encontrar contornos
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

            plate_region = None

            # Buscar un contorno que se parezca a una placa
            for contour in contours:
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
                
                if len(approx) == 4:  # Buscar contornos con 4 lados
                    plate_region = approx
                    x, y, w, h = cv2.boundingRect(plate_region)
                    plate_image = gray_frame[y:y+h, x:x+w]

                    # Usar OCR para extraer el texto de la placa
                    result = reader.readtext(plate_image)

                    if result:
                        placa_texto = result[0][1]
                        pico_y_placa_info = verificar_pico_y_placa(placa_texto)
                        
                        # Agregar delay de 2 segundos
                        time.sleep(4)
                        
                        # Imprimir con la información adicional
                        print("Placa detectada:", placa_texto, "-", pico_y_placa_info['message'])

                    # Dibujar el contorno de la placa en la imagen
                    cv2.drawContours(frame, [plate_region], -1, (0, 255, 0), 3)
                    break

            # Mostrar la imagen con la placa detectada
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    app.run(debug=True)