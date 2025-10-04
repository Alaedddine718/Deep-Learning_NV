from __future__ import annotations

# --- IMPORTS Y FIX DE PATH ---
import os, sys
# Asegura que Python vea la carpeta raíz del proyecto (para importar utils/*)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

from utils.preprocessing import base64_to_image, preprocess_digit
from utils.training_utils import BASE_MODEL_PATH

# --- APP / SOCKET.IO ---
app = Flask(__name__, static_folder='.', template_folder='.')
app.config['SECRET_KEY'] = 'dev-secret'
socketio = SocketIO(app, cors_allowed_origins='*')

APP_DIR = os.path.dirname(__file__)
T = lambda name: open(os.path.join(APP_DIR, name), 'r', encoding='utf-8').read()

# --- SERVIR ESTÁTICOS DE /app ---
@app.route('/app/<path:path>')
def serve_app_static(path):
    return send_from_directory(APP_DIR, path)

# --- CARGA PEREZOSA DEL MODELO ---
_model = None
def get_model():
    """Carga y cachea el modelo .keras una sola vez."""
    global _model
    if _model is None:
        if not os.path.exists(BASE_MODEL_PATH):
            raise FileNotFoundError(
                "No existe models/mnist_compiled_model.keras. "
                "Entrena y guarda el modelo primero desde el notebook."
            )
        _model = load_model(BASE_MODEL_PATH)
    return _model

# --- PÁGINAS ---
@app.route('/')
def index():
    return T('index.html')

@app.route('/login')
def login_page():
    return T('login.html')

@app.route('/register')
def register_page():
    return T('register.html')

@app.route('/navbar')
def navbar_partial():
    return T('navbar.html')

# --- SALUD (para diagnosticar) ---
@app.route('/health')
def health():
    try:
        m = get_model()
        return jsonify(status="ok", model_path=BASE_MODEL_PATH,
                       layers=len(getattr(m, 'layers', [])))
    except Exception as e:
        return jsonify(status="error", error=str(e)), 500

# --- PREDICCIÓN ---
@app.route('/predict', methods=['POST'])
def predict():
    # Soporte para archivo subido o dataURL base64
    if 'image' in request.files:
        img = Image.open(request.files['image']).convert('L')
    else:
        payload = request.get_json(silent=True) or {}
        data_url = request.form.get('image_base64') or payload.get('image_base64')
        if not data_url:
            return jsonify({'error': 'No image provided'}), 400
        img = base64_to_image(data_url)

    x = preprocess_digit(img)  # (1, 784)
    model = get_model()
    probs = model.predict(x, verbose=0)[0]  # (10,)
    pred = int(np.argmax(probs))
    return jsonify({'prediction': pred, 'probs': [float(p) for p in probs]})

# --- SUBIDA SIMPLE (GUARDA EN app/uploads) ---
@app.route('/upload', methods=['POST'])
def upload():
    f = request.files.get('image')
    if not f:
        return jsonify({'error': 'no file'}), 400
    save_dir = os.path.join(APP_DIR, 'uploads')
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f.filename)
    f.save(path)
    return jsonify({'ok': True, 'path': f'uploads/{f.filename}'})

# --- SOCKET.IO DEMO ---
@socketio.on('stroke')
def on_stroke(data):
    emit('stroke_ack', {'points': data.get('points', [])}, broadcast=False)

# --- MAIN ---
if __name__ == '__main__':
    # Forzar host local para evitar firewall en redes públicas
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='127.0.0.1', port=port, debug=True)


