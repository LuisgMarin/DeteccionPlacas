# test_evaluator.py
import sys
import os
import cv2
import logging
import time
import json
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar directorios usando rutas absolutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
TEST_DIR = os.path.join(PROJECT_DIR, 'DeteccionPlacas\\test')
RESULTS_DIR = os.path.join(TEST_DIR, 'resultados')
DETECTED_DIR = os.path.join(RESULTS_DIR, 'detected_images')
IMG_DIR = os.path.join(PROJECT_DIR, 'DeteccionPlacas\img')

# Agregar directorio raíz al path
sys.path.append(PROJECT_DIR)

from detector import PlacaDetector
from evaluator import Evaluator

def setup_directories():
    """Crear estructura de directorios necesaria"""
    for directory in [RESULTS_DIR, DETECTED_DIR]:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directorio creado/verificado: {directory}")

def test_detector():
    try:
        logger.info("Iniciando prueba del detector...")
        setup_directories()
        
        detector = PlacaDetector()
        evaluator = Evaluator(detector)
        
        # Verificar imágenes de prueba
        if not os.path.exists(IMG_DIR):
            logger.error(f"Directorio de imágenes no encontrado: {IMG_DIR}")
            return
            
        test_images = [f for f in os.listdir(IMG_DIR) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not test_images:
            logger.error(f"No se encontraron imágenes en: {IMG_DIR}")
            return
            
        logger.info(f"Procesando {len(test_images)} imágenes...")
        
        for image_file in test_images:
            image_path = os.path.join(IMG_DIR, image_file)
            frame = cv2.imread(image_path)
            
            if frame is None:
                logger.error(f"No se pudo leer: {image_file}")
                continue
                
            logger.info(f"Procesando: {image_file}")
            resultado = evaluator.evaluar_deteccion_realtime(frame)
            
            if resultado:
                # Guardar imagen procesada
                output_path = os.path.join(DETECTED_DIR, f'detected_{image_file}')
                cv2.imwrite(output_path, frame)
                
                logger.info(f"Placa: {resultado['placa']} | "
                          f"Confianza: {resultado['confianza']} | "
                          f"Tiempo: {resultado['tiempo_ms']}")
            
            time.sleep(0.5)  # Breve pausa entre imágenes
        
        # Guardar métricas finales
        results_file = os.path.join(RESULTS_DIR, 'resultados_evaluacion.json')
        evaluator.guardar_resultados(results_file)
        logger.info(f"Resultados guardados en: {results_file}")
        
    except Exception as e:
        logger.error(f"Error durante la prueba: {str(e)}")
        raise
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_detector()