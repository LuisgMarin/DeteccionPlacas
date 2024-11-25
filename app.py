from flask import Flask, render_template, Response, jsonify
from detector import PlacaDetector, gen_frames
import json

app = Flask(__name__)

@app.route('/')
def index():
    """
    Renderiza la página principal.
    """
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """
    Proporciona el feed de video en tiempo real.
    """
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/placa_info')
def placa_info():
    """
    Proporciona la información de la última placa detectada.
    """
    try:
        with open('static/placa_info.json') as f:
            data = json.load(f)
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({'error': 'No se ha detectado ninguna placa'})

if __name__ == '__main__':
    app.run(debug=True)