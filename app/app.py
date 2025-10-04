# app/app.py
import os, io, base64
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO

# Usar tf-keras compatible con TF 2.19 (ya instalado)
import tf_keras as keras

# ⬇ OJO: aquí va _file_ (dos guiones bajos), NO file
APP_DIR  = os.path.dirname(_file_)
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
MODEL_PATH = os.path.abspath(os.path.join(ROOT_DIR, "models", "mnist_compiled_model.keras"))

app = Flask(_name_, static_folder=APP_DIR, template_folder=APP_DIR)
app.config["SECRET_KEY"] = "dev-secret"
socketio = SocketIO(app, cors_allowed_origins="*")

_model = None
def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No existe el modelo en: {MODEL_PATH}\n"
                "Entrena/guarda en Colab y coloca mnist_compiled_model.keras en /models."
            )
        # Carga “legacy” para evitar el error de batch_shape en InputLayer
        _model = keras.models.load_model(
            MODEL_PATH,
            safe_mode=False,
            compile=False
        )
    return _model

@app.route("/")
def index():
    return send_from_directory(APP_DIR, "index.html")

@app.route("/app/<path:path>")
def serve_app_assets(path):
    return send_from_directory(APP_DIR, path)

@app.route("/ping")
def ping():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    """
    Espera JSON: {"image": "data:image/png;base64,AAAA..."}
    Devuelve:    {"prediction": int, "probs": [10 floats]}
    """
    payload = request.get_json(silent=True) or {}
    data_url = payload.get("image")
    if not data_url or "," not in data_url:
        return jsonify({"error": "Falta image base64 (dataURL)"}), 400

    # Decodificar imagen
    try:
        b64 = data_url.split(",", 1)[1]
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("L")
    except Exception as e:
        return jsonify({"error": f"Imagen inválida: {e}"}), 400

    # Preprocesar a (1,784)
    img = img.resize((28, 28))
    x = np.array(img, dtype="float32").reshape(1, 784) / 255.0

    # Predecir
    model = get_model()
    probs = model.predict(x, verbose=0)[0]
    pred = int(np.argmax(probs))
    return jsonify({"prediction": pred, "probs": [float(p) for p in probs]})

if _name_ == "_main_":
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port, debug=True)



