# test_evaluator.py
import sys
import os
import cv2
import json
import logging
import time
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
from visualizador import Visualizador

# Ground truth para las imágenes
GROUND_TRUTH = {
    "1.jpg": "NCK 182",
    "2.jpg": "PLO 388",
    "3.jpg": "FDP 761",
    "4.jpg": "GNK 495",
    "5.jpg": "LAC 290",
    "6.jpg": "RII 282",
    "7.jpg": "MRQ 523",
    "8.jpg": "WNA 756",
    "9.jpg": "FRA 098",
    "10.jpg": "AUR 242",
    "11.jpg": "RZK 115",
    "12.jpg": "JWF 579",
    "13.jpg": "JAW 589",
    "14.jpg": "IBR 014",
    "15.jpg": "RCD 549",
    "16.jpg": "CIZ 826",
    "17.jpg": "CBR 070",
    "18.jpg": "KDO 042",
    "19.jpg": "RKW 326",
    "20.jpg": "RJV 247",
    "21.jpg": "DFV 275",
    "22.jpg": "GGU 267",
    "23.jpg": "BTN 068",
    "24.jpg": "GXL 041",
    "25.jpg": "NAG 598",
    "26.jpg": "FBU 803",
    "27.jpg": "NCK 182"
}

def setup_directories():
    """Crear estructura de directorios necesaria"""
    for directory in [RESULTS_DIR, DETECTED_DIR]:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directorio creado/verificado: {directory}")

def test_detector():
    """Función principal de prueba del detector"""
    try:
        resultados_detallados = []
        imagenes_no_detectadas = []
        
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
                
            ground_truth = GROUND_TRUTH.get(image_file)
            logger.info(f"Procesando: {image_file}")
            logger.info(f"Ground truth: {ground_truth}")
            
            resultado = evaluator.evaluar_deteccion_realtime(frame, ground_truth)
            
            if resultado:
                output_path = os.path.join(DETECTED_DIR, f'detected_{image_file}')
                cv2.imwrite(output_path, frame)
                
                resultados_detallados.append({
                    'imagen': image_file,
                    'placa_detectada': resultado['placa'],
                    'placa_esperada': ground_truth,
                    'coincide': resultado['placa'] == ground_truth,
                    'confianza': resultado['confianza'],
                    'tiempo_ms': resultado['tiempo_ms']
                })
                
                logger.info(f"Resultado: {resultado['placa']} | "
                          f"Esperado: {ground_truth} | "
                          f"Confianza: {resultado['confianza']}")
            else:
                imagenes_no_detectadas.append({
                    'imagen': image_file,
                    'placa_esperada': ground_truth,
                    'detectada': False
                })
                logger.warning(f"No se detectó placa en: {image_file}")
        
        # Guardar resultados
        results_file = os.path.join(RESULTS_DIR, 'resultados_evaluacion.json')
        resultados = {
            'metricas_finales': evaluator.get_current_metrics(),
            'resultados_detallados': resultados_detallados,
            'imagenes_no_detectadas': imagenes_no_detectadas,
            'timestamp': datetime.now().isoformat(),
            'total_imagenes': len(test_images),
            'total_detecciones': len(resultados_detallados),
            'total_no_detectadas': len(imagenes_no_detectadas)
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(resultados, f, indent=4)
        
        logger.info(f"Resultados guardados en: {results_file}")

        # Generar gráficas
        try:
            viz = Visualizador()
            grafica_path = viz.generar_graficas_generales(results_file)
            logger.info(f"Gráficas generadas en: {grafica_path}")
        except Exception as viz_error:
            logger.error(f"Error al generar gráficas: {str(viz_error)}")
            
    except Exception as e:
        logger.error(f"Error durante la prueba: {str(e)}")
        raise
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_detector()