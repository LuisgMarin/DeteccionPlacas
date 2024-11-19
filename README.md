# Sistema de Detección de Pico y Placa

Este es un proyecto Flask que utiliza OpenCV y EasyOCR para detectar placas vehiculares en tiempo real y verificar si están sujetas a la restricción de pico y placa.

## Tecnologías Utilizadas
- Python
- OpenCV
- EasyOCR
- Flask
- HTML/CSS (Bootstrap)
- JavaScript

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
    pip install flask opencv-python easyocr numpy re datetime json
    ```

## Ejecución del Proyecto

1. Inicia la aplicación Flask:

    ```sh
    python main.py
    ```

2. Abre tu navegador web y ve a `http://127.0.0.1:5000` para ver la aplicación en funcionamiento.

## Estructura del Proyecto

- `main.py`: Contiene la lógica principal del servidor Flask y la detección de placas.
- `templates/index.html`: Página principal de la aplicación.
- `static/`: Directorio para archivos estáticos como CSS y JavaScript.

## Funcionalidades

### Detección de Placas
La clase `PlacaDetector` se encarga de detectar placas vehiculares en imágenes capturadas por la cámara. Utiliza OpenCV para preprocesar la imagen y EasyOCR para extraer el texto de la placa.

### Validación de Formato
La función `validar_formato_placa` valida si el texto extraído corresponde a un formato válido de placa colombiana.

### Verificación de Pico y Placa
La función `verificar_pico_y_placa` verifica si un vehículo tiene restricciones de "Pico y Placa" basado en el último dígito de su placa y el día de la semana.

### Generación de Frames de Video
La función `gen_frames` genera frames de video desde la cámara y detecta placas en tiempo real. Si se detecta una placa, se valida y se verifica si tiene pico y placa, y se guarda la información en un archivo JSON.


## Rutas Flask

- `/`: Renderiza la página principal.
- `/video_feed`: Proporciona el feed de video en tiempo real.
- `/placa_info`: Proporciona la información de la última placa detectada.

## Flujo de la Aplicación

### 1. Inicio de la Aplicación
- La aplicación se inicia ejecutando el archivo `main.py`.
- Flask levanta un servidor web y renderiza la página principal (`index.html`) cuando se accede a la ruta `/`.

### 2. Renderización de la Página Principal
- La página principal (`index.html`) se carga en el navegador.
- La página incluye una barra de navegación, una sección principal con el título y la descripción del sistema, un contenedor de video en tiempo real y una tarjeta de estado de detección.

### 3. Captura de Video en Tiempo Real
- La función `gen_frames()` en `main.py` utiliza OpenCV para capturar frames de video desde la cámara.
- Cada frame capturado se procesa para detectar placas vehiculares.

### 4. Detección de Placas
- La clase `PlacaDetector` en `main.py` se encarga de detectar placas en los frames capturados.
- La función `detect_plate()` preprocesa la imagen para resaltar las regiones amarillas.
- Se encuentran los contornos de las regiones amarillas y se aplica OCR (reconocimiento óptico de caracteres) para extraer el texto de la placa.
- Si se detecta una placa con suficiente confianza, se extrae y formatea el texto de la placa.

### 5. Validación y Verificación de Pico y Placa
- La función `verificar_pico_y_placa()` valida el formato de la placa y verifica si el vehículo tiene restricciones de "Pico y Placa" según el día de la semana y el último dígito de la placa.
- La información de la placa detectada, incluyendo el texto, la confianza y el resultado de la verificación, se guarda en un archivo JSON (`static/placa_info.json`).

### 6. Actualización del Estado en la Interfaz de Usuario
- El archivo `script.js` contiene una función `actualizarPlacaDetectada()` que se ejecuta cada segundo.
- Esta función realiza una solicitud a la ruta `/placa_info` para obtener la información de la última placa detectada.
- Si se detecta una placa, la información se muestra en la tarjeta de estado de detección en la página principal.

### 7. Rutas Flask
- `/`: Renderiza la página principal (`index.html`).
- `/video_feed`: Proporciona el feed de video en tiempo real utilizando la función `gen_frames()`.
- `/placa_info`: Proporciona la información de la última placa detectada leyendo el archivo JSON (`static/placa_info.json`).

