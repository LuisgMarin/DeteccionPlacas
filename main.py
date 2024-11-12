# -*- coding: utf-8 -*-
import cv2
import easyocr
from flask import Flask, render_template, Response
from datetime import datetime

# Función para verificar si la placa tiene pico y placa
def verificar_pico_y_placa(placa_texto):
    # Filtrar solo los dígitos del texto de la placa
    digitos = ''.join(filter(str.isdigit, placa_texto))
    if not digitos:
        print("No se encuentra una placa válida.")
        return

    ultimo_digito = int(digitos[-1])
    dia_semana = datetime.today().weekday()
    restricciones = {
        0: [7, 8], # Lunes
        1: [9, 0], # Martes 
        2: [1, 2], # Miércoles 
        3: [3, 4], #Jueves 
        4: [5, 6], # Viernes 
}

    # Verificar si es fin de semana
    if dia_semana >= 5:
        print(f"La placa '{placa_texto}' no tiene pico y placa hoy (fin de semana).")
        return

    if ultimo_digito in restricciones.get(dia_semana, []):
        print(f"La placa '{placa_texto}' tiene pico y placa hoy.")
    else:
        print(f"La placa '{placa_texto}' no tiene pico y placa hoy.")      

# Inicializar EasyOCR
reader = easyocr.Reader(['en'])  # Configura el idioma según tu necesidad

# Iniciar la captura de video de la cámara
cap = cv2.VideoCapture(0)  # Cambia el índice si usas una cámara diferente

if not cap.isOpened():
    print("Error al abrir la cámara.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al acceder a la cámara.")
        break

    # Convertir la imagen a escala de grises
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar bordes para resaltar las placas
    edges = cv2.Canny(gray_frame, 30, 200)

    # Encontrar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for contour in contours:
        # Aproximar el contorno a un polígono
        epsilon = 0.018 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Si el polígono tiene 4 lados, podría ser una placa
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            placa = gray_frame[y:y + h, x:x + w]

            # Usar EasyOCR para leer el texto de la placa
            resultado = reader.readtext(placa)
            if resultado:
                placa_texto = resultado[0][-2]
                verificar_pico_y_placa(placa_texto)

    # Mostrar el frame con los contornos detectados
    cv2.imshow('Frame', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()