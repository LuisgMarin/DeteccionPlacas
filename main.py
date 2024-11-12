from flask import Flask, render_template, Response, jsonify
import cv2
import easyocr
import numpy as np
from datetime import datetime
import json

app = Flask(__name__)

# Initialize EasyOCR
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
    restricciones = {
        0: [7, 8], # Lunes
        1: [9, 0], # Martes 
        2: [1, 2], # Miércoles 
        3: [3, 4], # Jueves 
        4: [5, 6], # Viernes 
    }

    # Verificar si es fin de semana
    if dia_semana >= 5:
        return {
            'message': f"La placa '{placa_texto}' no tiene pico y placa hoy (fin de semana).",
            'status': 'success'
        }

    if ultimo_digito in restricciones.get(dia_semana, []):
        return {
            'message': f"La placa '{placa_texto}' tiene pico y placa hoy.",
            'status': 'warning'
        }
    else:
        return {
            'message': f"La placa '{placa_texto}' no tiene pico y placa hoy.",
            'status': 'success'
        }

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Convertir a escala de grises y HSV
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Definir el rango de color amarillo
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])
            mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

            # Encontrar contornos
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                epsilon = 0.018 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    placa = gray_frame[y:y + h, x:x + w]
                    resultado = reader.readtext(placa)
                    if resultado:
                        placa_texto = resultado[0][-2]
                        result = verificar_pico_y_placa(placa_texto)
                        cv2.putText(frame, result['message'], (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        # Guardar la imagen de la placa detectada
                        cv2.imwrite('static/placa_detectada.jpg', placa)
                        # Guardar la información de OCR y verificación de pico y placa
                        with open('static/placa_info.json', 'w') as f:
                            json.dump({
                                'placa_texto': placa_texto,
                                'verificacion': result['message']
                            }, f)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

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

if __name__ == '__main__':
    app.run(debug=True)