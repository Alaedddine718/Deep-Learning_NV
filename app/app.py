from __future__ import annotations
import os, io, base64
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO
from tensorflow.keras.models import load_model

# --- Rutas base ---
APP_DIR  = os.path.dirname(__file__)                 # .../Deep-Learning_NV/app
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
MODEL_PATH = os.path.abspath(os.path.join(ROOT_DIR, "models", "mnist_compiled_model.keras"))

app = Flask(__name__, static_folder=APP_DIR, template_folder=APP_DIR)
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Cache del modelo ---
_model = None
def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No existe el modelo en: {MODEL_PATH}\n"
                "Entrena en Colab y coloca mnist_compiled_model.keras en /models."
            )
        _model = load_model(MODEL_PATH)
    return _model

# --- Servir la web desde /app ---
@app.route("/")
def index():
    # Sirve el archivo app/index.html
    return send_from_directory(APP_DIR, "index.html")

@app.route("/app/<path:path>")
def serve_app_assets(path):
    # Sirve cualquier recurso estático dentro de /app (css, js, imágenes…)
    return send_from_directory(APP_DIR, path)

# --- Predicción ---
@app.route("/predict", methods=["POST"])
def predict():
    """
    Espera JSON: {"image": "data:image/png;base64,AAAA..."}
    Devuelve: {"prediction": int, "probs": [10 floats]}
    """
    payload = request.get_json(silent=True) or {}
    data_url = payload.get("image")
    if not data_url or "," not in data_url:
        return jsonify({"error": "Falta image base64 (dataURL)"}), 400

    # Decodificar base64 y preprocesar a 28x28, escala [0,1] y shape (1,784)
    b64 = data_url.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("L")
    img = img.resize((28, 28))
    x = np.array(img, dtype="float32").reshape(1, 784) / 255.0

    # Predecir
    model = get_model()
    probs = model.predict(x, verbose=0)[0]  # (10,)
    pred = int(np.argmax(probs))
    return jsonify({"prediction": pred, "probs": [float(p) for p in probs]})

# --- Salud ---
@app.route("/ping")
def ping():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port, debug=True)



