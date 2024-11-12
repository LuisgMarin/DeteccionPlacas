# Sistema de Detección de Pico y Placa

Este es un proyecto Flask que utiliza OpenCV y EasyOCR para detectar placas vehiculares en tiempo real y verificar si están sujetas a la restricción de pico y placa.

## Requisitos del Sistema

- Python 3.7 o superior
- Conda (recomendado)

## Configuración del Entorno Virtual

1. Crea un entorno virtual usando Conda:

    ```sh
    conda create -n pico-y-placa python=3.9
    ```

2. Activa el entorno virtual:

    ```sh
    conda activate pico-y-placa
    ```

3. Instala las dependencias necesarias:

    ```sh
    pip install flask opencv-python easyocr numpy
    ```

## Ejecución del Proyecto

1. Inicia la aplicación Flask:

    ```sh
    python main.py
    ```

2. Abre tu navegador web y ve a `http://127.0.0.1:5000` para ver la aplicación en funcionamiento.

## Estructura del Proyecto
