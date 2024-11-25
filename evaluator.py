import json
import logging
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import cv2

class Evaluator:
    def __init__(self, detector):
        self.detector = detector
        self.reset_stats()
        self.setup_logging()
        
    def setup_logging(self):
        """Configura el sistema de logging"""
        logging.basicConfig(
            filename=f'evaluacion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        
    def reset_stats(self):
        """Reinicia todas las estadísticas"""
        self.total_frames = 0
        self.total_detections = 0
        self.correct_detections = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.confidence_scores = []
        self.detection_times = []
        self.results_history = []

    def evaluar_deteccion_realtime(self, frame, ground_truth=None):
        """Evalúa la detección en tiempo real con métricas detalladas"""
        start_time = datetime.now()
        
        # Realizar detección
        result = self.detector.detect_plate(frame)
        
        # Calcular tiempo de detección
        detection_time = (datetime.now() - start_time).total_seconds() * 1000  # en ms
        self.detection_times.append(detection_time)
        
        self.total_frames += 1
        
        if result:
            self.total_detections += 1
            self.confidence_scores.append(result['confidence'])
            
            detection_result = {
                'placa': result['text'],
                'confianza': result['confidence'],
                'tiempo_deteccion': detection_time,
                'timestamp': datetime.now().isoformat()
            }
            
            if ground_truth:
                detection_result['ground_truth'] = ground_truth
                if result['text'] == ground_truth:
                    self.correct_detections += 1
                    detection_result['match'] = True
                else:
                    self.false_positives += 1
                    detection_result['match'] = False
            
            self.results_history.append(detection_result)
            logging.info(f"Detección: {json.dumps(detection_result)}")
            
            return {
                'placa': result['text'],
                'confianza': f"{result['confidence']:.2%}",
                'tiempo_ms': f"{detection_time:.1f}ms",
                'metricas': self.get_current_metrics()
            }
        else:
            if ground_truth:
                self.false_negatives += 1
            return None

    def get_current_metrics(self):
        """Obtiene métricas actuales del detector"""
        metrics = {
            'precision': self.get_precision(),
            'recall': self.get_recall(),
            'f1_score': self.get_f1_score(),
            'tiempo_promedio': self.get_average_detection_time(),
            'confianza_promedio': self.get_confidence_average(),
            'tasa_deteccion': self.get_detection_rate()
        }
        return metrics

    def get_precision(self):
        """Calcula la precisión"""
        if self.total_detections == 0:
            return 0
        return self.correct_detections / self.total_detections

    def get_recall(self):
        """Calcula el recall"""
        total_expected = self.correct_detections + self.false_negatives
        if total_expected == 0:
            return 0
        return self.correct_detections / total_expected

    def get_f1_score(self):
        """Calcula el F1-Score"""
        precision = self.get_precision()
        recall = self.get_recall()
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)

    def get_average_detection_time(self):
        """Calcula el tiempo promedio de detección"""
        if not self.detection_times:
            return 0
        return np.mean(self.detection_times)

    def get_confidence_average(self):
        """Calcula la confianza promedio"""
        if not self.confidence_scores:
            return 0
        return np.mean(self.confidence_scores)

    def get_detection_rate(self):
        """Calcula la tasa de detección"""
        if self.total_frames == 0:
            return 0
        return self.total_detections / self.total_frames

    def guardar_resultados(self, filename='resultados_evaluacion.json'):
        """Guarda los resultados de la evaluación"""
        resultados = {
            'metricas_finales': self.get_current_metrics(),
            'historial_detecciones': self.results_history,
            'timestamp': datetime.now().isoformat(),
            'total_frames': self.total_frames,
            'total_detecciones': self.total_detections
        }
        
        with open(filename, 'w') as f:
            json.dump(resultados, f, indent=4)
        
        logging.info(f"Resultados guardados en {filename}")