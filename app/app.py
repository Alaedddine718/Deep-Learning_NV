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
import numpy as np
from PIL import Image, ImageOps, ImageFilter

from flask import Flask, jsonify, request, send_from_directory

# ====== RUTAS DE PROYECTO ======
APP_DIR  = os.path.dirname(__file__)                   # .../app
ROOT_DIR = os.path.dirname(APP_DIR)                    # raíz del repo
MODEL_PATH = os.path.join(ROOT_DIR, "models", "mnist_compiled_model.keras")

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
    # Promedio global (0=negro, 255=blanco)
    mean_v = np.array(img, dtype=np.float32).mean()
    # Si es muy claro, invertimos para que fondo sea negro y dígito blanco
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
        # Nada dibujado: devolvemos un 28x28 vacío
        return Image.new("L", (28, 28), 0)

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    crop = img.crop((x_min, y_min, x_max + 1, y_max + 1))

    # Hacemos el recorte cuadrado: rellenamos el lado corto
    w, h = crop.size
    side = max(w, h)
    sq = Image.new("L", (side, side), 0)
    sq.paste(crop, ((side - w) // 2, (side - h) // 2))

    # Margen alrededor (para que no quede pegado al borde)
    margin = int(side * margin_ratio)
    out = ImageOps.expand(sq, border=margin, fill=0)
    return out

def _thicken(img: Image.Image, size: int = 3) -> Image.Image:
    """Engrosa ligeramente el trazo con una dilatación pequeña."""
    # MaxFilter “ensancha” trazos blancos (255) sobre fondo negro (0)
    try:
        return img.filter(ImageFilter.MaxFilter(size=size))
    except Exception:
        return img

def preprocess_png_base64(data_url: str) -> np.ndarray:
    """
    Convierte PNG base64 del canvas a tensor (1, 28, 28, 1) en [0,1],
    centrado, con trazo legible y fondo negro.
    """
    # Quitar encabezado "data:image/png;base64,"
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    img_bytes = base64.b64decode(data_url)
    img = Image.open(io.BytesIO(img_bytes))

    # 1) Escala de grises, comp. alfa
    img = _to_grayscale(img)

    # 2) Inversión automática (si fondo claro)
    img = _maybe_invert(img)

    # 3) Binarizar para quitar grises y ruido
    img = _binarize(img, thresh=50)

    # 4) Recortar a bbox + cuadrar + margen
    img = _crop_to_bbox(img, margin_ratio=0.15)

    # 5) Engrosar trazo (suave)
    img = _thicken(img, size=3)

    # 6) Redimensionar a 28x28 (bilinear mantiene forma sin “dientes”)
    img = img.resize((28, 28), Image.Resampling.BILINEAR)

    # 7) A numpy y normalizar [0,1] (blanco=1, fondo=0)
    arr = np.array(img).astype("float32") / 255.0

    # 8) Expandir dims -> (1,28,28,1)
    arr = np.expand_dims(arr, axis=-1)
    arr = np.expand_dims(arr, axis=0)

    return arr

# ====== API ======
@app.route("/predict", methods=["POST"])
def predict():
    """
    Espera JSON: { "image": "data:image/png;base64,...." }
    Devuelve predicción y vector de probabilidades.
    """
    data = request.get_json(force=True, silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "Falta el campo 'image' en el cuerpo de la petición."}), 400

    x = preprocess_png_base64(data["image"])
    model = get_model()

    probs = model.predict(x, verbose=0)[0].astype(float)  # (10,)
    pred  = int(np.argmax(probs))

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






