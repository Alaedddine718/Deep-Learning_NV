# app/app.py
# -*- coding: utf-8 -*-
import os, io, sys, base64, traceback
print(">> [BOOT] Iniciando app.py... (Python", sys.version, ")")

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO

# --- Rutas base ---
APP_DIR  = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
MODEL_PATH = os.path.abspath(os.path.join(ROOT_DIR, "models", "mnist_compiled_model.keras"))
print(">> [BOOT] APP_DIR:", APP_DIR)
print(">> [BOOT] ROOT_DIR:", ROOT_DIR)
print(">> [BOOT] MODEL_PATH:", MODEL_PATH, "exists?", os.path.exists(MODEL_PATH))

# --- Flask + SocketIO ---
app = Flask(__name__, static_folder=APP_DIR, template_folder=APP_DIR)
app.config["SECRET_KEY"] = "dev-secret"
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Carga robusta del modelo ---
class InferenceWrapper:
    """
    Envuelve cualquier forma de modelo (tf.keras / tf_keras / SavedModel signatures)
    para ofrecer un método .predict(x) que devuelva un np.array de probabilidades (1,10).
    """
    def __init__(self, kind, obj):
        self.kind = kind
        self.obj = obj
        self._setup()

    def _setup(self):
        if self.kind in ("tf.keras", "tf_keras"):
            # Keras: ya es un modelo con .predict
            self.predict_fn = lambda x: self.obj.predict(x, verbose=0)
        elif self.kind == "saved_model":
            import tensorflow as tf
            # Obtenemos la signature por defecto y descubrimos el nombre de entrada/salida.
            func = self.obj.signatures.get("serving_default")
            if func is None:
                raise RuntimeError("SavedModel sin 'serving_default'.")

            # Descubrir la(s) clave(s) de entrada
            _, input_spec = func.structured_input_signature
            input_keys = list(input_spec.keys())
            if not input_keys:
                raise RuntimeError("No se encontraron entradas en SavedModel.")
            self.input_key = input_keys[0]

            # Descubrir la/s salida/s
            out_spec = func.structured_outputs
            out_keys = list(out_spec.keys())
            if not out_keys:
                raise RuntimeError("No se encontraron salidas en SavedModel.")
            self.output_key = out_keys[0]
            self.func = func
            self.tf = tf

            def _predict(x_np):
                x_tf = self.tf.convert_to_tensor(x_np)
                out = self.func(**{self.input_key: x_tf})
                y = out[self.output_key].numpy()
                return y
            self.predict_fn = _predict
        else:
            raise ValueError("Tipo de modelo desconocido")

    def predict(self, x):
        return self.predict_fn(x)

def try_load_model():
    """
    Intenta cargar el modelo en este orden:
    1) tf.keras.models.load_model
    2) tf_keras.models.load_model
    3) tf.saved_model.load  (y crea wrapper de signature)
    Devuelve un InferenceWrapper.
    """
    # 1) tf.keras
    try:
        import tensorflow as tf
        tfk = tf.keras
        print(">> [LOAD] Probando tf.keras.load_model ... (TF:", tf.__version__, ")")
        mdl = tfk.models.load_model(MODEL_PATH, compile=False)
        print(">> [LOAD] Cargado con tf.keras ✅")
        return InferenceWrapper("tf.keras", mdl)
    except Exception as e:
        print(">> [LOAD] tf.keras.load_model falló:", repr(e))

    # 2) tf_keras
    try:
        import tf_keras as k3
        print(">> [LOAD] Probando tf_keras.load_model ... (tf_keras:", getattr(k3, '__version__', 'unknown'), ")")
        mdl = k3.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
        print(">> [LOAD] Cargado con tf_keras ✅")
        return InferenceWrapper("tf_keras", mdl)
    except Exception as e:
        print(">> [LOAD] tf_keras.load_model falló:", repr(e))

    # 3) SavedModel signatures
    try:
        import tensorflow as tf
        print(">> [LOAD] Probando tf.saved_model.load (firmas) ...")
        sm = tf.saved_model.load(MODEL_PATH)
        print(">> [LOAD] Cargado como SavedModel ✅")
        return InferenceWrapper("saved_model", sm)
    except Exception as e:
        print(">> [LOAD] tf.saved_model.load falló:", repr(e))

    raise RuntimeError(
        "No se pudo cargar el modelo .keras con ninguno de los métodos. "
        "Asegúrate de haberlo guardado en Colab con TensorFlow/Keras compatibles."
    )

_model = None
def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No existe el modelo en: {MODEL_PATH}\n"
                "Entrena/guarda en Colab y coloca mnist_compiled_model.keras en /models."
            )
        _model = try_load_model()
    return _model

# --- Rutas HTTP ---
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
    payload = request.get_json(silent=True) or {}
    data_url = payload.get("image")
    if not data_url or "," not in data_url:
        return jsonify({"error": "Falta image base64 (dataURL) en JSON {'image': 'data:image/png;base64,...'}"}), 400

    try:
        b64 = data_url.split(",", 1)[1]
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("L")
    except Exception as e:
        return jsonify({"error": f"Imagen inválida: {e}"}), 400

    # Preprocesado igual que en el entrenamiento (28x28, escala 0-1, flatten)
    img = img.resize((28, 28))
    x = np.array(img, dtype="float32").reshape(1, 784) / 255.0

    model = get_model()
    probs = model.predict(x)[0]
    pred = int(np.argmax(probs))
    return jsonify({"prediction": pred, "probs": [float(p) for p in probs]})

# --- Main ---
if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", 5000))
        print(f">> [RUN] Lanzando servidor en http://127.0.0.1:{port}")
        socketio.run(app, host="0.0.0.0", port=port, debug=True)
    except Exception:
        print(">> [FATAL] Excepción al iniciar el servidor:")
        traceback.print_exc()
        sys.exit(1)





