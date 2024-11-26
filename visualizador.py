# visualizador.py
import matplotlib.pyplot as plt
import json
import os

class Visualizador:
    def __init__(self):
        # Configuración básica de matplotlib
        plt.style.use('default')
        
    def generar_graficas_generales(self, results_file):
        """Genera gráficas de métricas generales del algoritmo"""
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        metricas = data['metricas_finales']
        total_imgs = data['total_imagenes']
        detectadas = data['total_detecciones']
        no_detectadas = data['total_no_detectadas']
        
        # Crear figura con subplots
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Métricas principales
        plt.subplot(2, 2, 1)
        metricas_principales = ['precision', 'recall', 'f1_score', 'tasa_deteccion']
        valores = [metricas[m] * 100 for m in metricas_principales]
        bars = plt.bar(metricas_principales, valores, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'])
        plt.title('Métricas Principales del Detector')
        plt.ylabel('Porcentaje (%)')
        plt.ylim(0, 100)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # 2. Distribución de detecciones
        plt.subplot(2, 2, 2)
        plt.pie([detectadas, no_detectadas], 
                labels=[f'Detectadas\n{detectadas}', f'No Detectadas\n{no_detectadas}'],
                colors=['#2ecc71', '#e74c3c'],
                autopct='%1.1f%%',
                startangle=90)
        plt.title('Distribución de Detecciones')
        
        # 3. Distribución de confianza
        plt.subplot(2, 2, 3)
        confianzas = [float(r['confianza'].strip('%')) for r in data['resultados_detallados']]
        plt.hist(confianzas, bins=10, color='#3498db', alpha=0.7)
        plt.axvline(metricas['confianza_promedio']*100, color='red', linestyle='dashed',
                   label=f'Promedio: {metricas["confianza_promedio"]*100:.1f}%')
        plt.title('Distribución de Confianza')
        plt.xlabel('Confianza (%)')
        plt.ylabel('Cantidad de Detecciones')
        plt.legend()
        
        # 4. Tiempo promedio
        plt.subplot(2, 2, 4)
        tiempos = [float(r['tiempo_ms'].strip('ms')) for r in data['resultados_detallados']]
        plt.boxplot(tiempos, patch_artist=True, 
                   boxprops=dict(facecolor='#3498db', color='black'),
                   medianprops=dict(color='red'))
        plt.title('Análisis de Tiempos de Detección')
        plt.ylabel('Tiempo (ms)')
        
        # Ajustar y guardar
        plt.tight_layout()
        output_path = os.path.join(os.path.dirname(results_file), 'metricas_generales.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path