# app/app.py
# -----------------------------------------------------------
# Backend Flask para clasificar dígitos dibujados (MNIST)
# Versión: CNN (entrada 28x28x1) con TensorFlow 2.19 + tf.keras
# Preprocesado robusto: RGBA->L, inversión auto, binarizado,
# recorte a bbox, centrado, margen, dilatación y resize a 28x28.
# -----------------------------------------------------------

from __future__ import annotations
import os
import io
import base64
import sys
import numpy as np
from PIL import Image, ImageOps, ImageFilter

from flask import Flask, jsonify, request, send_from_directory

# ====== RUTAS DE PROYECTO ======
APP_DIR  = os.path.dirname(__file__)                   # .../app
ROOT_DIR = os.path.dirname(APP_DIR)                    # raíz del repo
MODEL_PATH = os.path.join(ROOT_DIR, "models", "mnist_compiled_model.keras")

# >>> AÑADIDO: asegurar que el root del repo está en sys.path
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ====== IMPORT DE FIREBASE (paquete local /firebase) ======
from firebase.firebase_utils import save_prediction_to_firestore

# ====== FLASK ======
app = Flask(__name__, static_folder=APP_DIR, static_url_path="/app")

@app.route("/")
def index():
    return send_from_directory(APP_DIR, "index.html")

# ====== TENSORFLOW / KERAS ======
import tensorflow as tf
keras = tf.keras

# Cache del modelo
_MODEL_CACHE = {"obj": None}

def get_model():
    if _MODEL_CACHE["obj"] is not None:
        return _MODEL_CACHE["obj"]
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"No existe el modelo en: {MODEL_PATH}\n"
            f"Asegúrate de tener mnist_compiled_model.keras en /models."
        )
    print(f">> [LOAD] Cargando modelo desde {MODEL_PATH} (TF: {tf.__version__}) ...")
    model = keras.models.load_model(MODEL_PATH)
    _MODEL_CACHE["obj"] = model
    print(">> [LOAD] Modelo cargado ✅")
    return model

# ====== PREPROCESADO ROBUSTO (CNN 28x28x1) ======
def _to_grayscale(img: Image.Image) -> Image.Image:
    """Convierte a L, componiendo alfa sobre fondo blanco si viene RGBA."""
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")
    if img.mode != "L":
        img = img.convert("L")
    return img

def _maybe_invert(img: Image.Image) -> Image.Image:
    """Invierte si el fondo parece claro (MNIST espera fondo negro)."""
    mean_v = np.array(img, dtype=np.float32).mean()  # 0=negro, 255=blanco
    if mean_v > 127:
        img = ImageOps.invert(img)
    return img

def _binarize(img: Image.Image, thresh: int = 50) -> Image.Image:
    """Binariza con umbral sencillo para ganar contraste."""
    return img.point(lambda p: 255 if p > thresh else 0)

def _crop_to_bbox(img: Image.Image, margin_ratio: float = 0.15) -> Image.Image:
    """Recorta al bounding box del dígito y añade un margen proporcional."""
    arr = np.array(img)
    ys, xs = np.where(arr > 0)  # píxeles blancos (dígito)
    if len(xs) == 0 or len(ys) == 0:
        return Image.new("L", (28, 28), 0)  # nada dibujado

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    crop = img.crop((x_min, y_min, x_max + 1, y_max + 1))

    w, h = crop.size
    side = max(w, h)
    sq = Image.new("L", (side, side), 0)
    sq.paste(crop, ((side - w) // 2, (side - h) // 2))

    margin = int(side * margin_ratio)
    out = ImageOps.expand(sq, border=margin, fill=0)
    return out

def _thicken(img: Image.Image, size: int = 3) -> Image.Image:
    """Engrosa ligeramente el trazo con una dilatación pequeña."""
    try:
        return img.filter(ImageFilter.MaxFilter(size=size))
    except Exception:
        return img

def preprocess_png_base64(data_url: str) -> np.ndarray:
    """
    Convierte PNG base64 del canvas a tensor (1, 28, 28, 1) en [0,1],
    centrado, con trazo legible y fondo negro.
    """
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    img_bytes = base64.b64decode(data_url)
    img = Image.open(io.BytesIO(img_bytes))

    img = _to_grayscale(img)
    img = _maybe_invert(img)
    img = _binarize(img, thresh=50)
    img = _crop_to_bbox(img, margin_ratio=0.15)
    img = _thicken(img, size=3)
    img = img.resize((28, 28), Image.Resampling.BILINEAR)

    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    return arr

# ====== API ======
@app.route("/predict", methods=["POST"])
def predict():
    """
    Espera JSON: { "image": "data:image/png;base64,...." }
    Devuelve predicción y vector de probabilidades.
    Además, guarda el resultado en Firebase.
    """
    data = request.get_json(force=True, silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "Falta el campo 'image' en el cuerpo de la petición."}), 400

    x = preprocess_png_base64(data["image"])
    model = get_model()

    probs = model.predict(x, verbose=0)[0].astype(float)  # (10,)
    pred  = int(np.argmax(probs))

    # ---- Guardar en Firestore (imagen en [0,1] de 28x28) ----
    try:
        x_img01 = x.squeeze().astype("float32")  # (28,28)
        save_prediction_to_firestore(pred, probs.tolist(), x_img01, model_name="mnist_cnn")
    except Exception as e:
        print(f">> [FB] Error al guardar: {e}")

    return jsonify({
        "prediccion": pred,
        "prediction": pred,
        "probabilidades": probs.tolist(),
        "probs": probs.tolist()
    })

# ====== ARRANQUE ======
if __name__ == "__main__":
    print(f">> [BOOT] Iniciando app.py... (Python {os.sys.version.split()[0]})")
    print(f">> [BOOT] APP_DIR: {APP_DIR}")
    print(f">> [BOOT] ROOT_DIR: {ROOT_DIR}")
    print(f">> [BOOT] MODEL_PATH: {MODEL_PATH} exists? {os.path.isfile(MODEL_PATH)}")
    print(">> [RUN] Lanzando servidor en http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)








