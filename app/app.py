# app/app.py
# -----------------------------------------------------------
# Backend Flask para clasificar dígitos dibujados (MNIST)
# Versión: CNN (entrada 28x28x1) con TensorFlow 2.19 + tf.keras
# Preprocesado robusto: RGBA->L, inversión auto, binarizado,
# recorte a bbox, centrado, margen, dilatación y resize a 28x28.
# Además: registro de resultados en Firebase (Firestore).
# -----------------------------------------------------------

from __future__ import annotations
import os
import io
import base64
from datetime import datetime
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
    mean_v = np.array(img, dtype=np.float32).mean()
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
        return Image.new("L", (28, 28), 0)

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    crop = img.crop((x_min, y_min, x_max + 1, y_max + 1))

    # Cuadrar
    w, h = crop.size
    side = max(w, h)
    sq = Image.new("L", (side, side), 0)
    sq.paste(crop, ((side - w) // 2, (side - h) // 2))

    # Margen
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
    arr = np.expand_dims(arr, axis=-1)  # (28,28,1)
    arr = np.expand_dims(arr, axis=0)   # (1,28,28,1)
    return arr

# ====== FIREBASE (Firestore) ======
# Guardamos cada predicción en la colección "predictions".
FIREBASE_ENABLED = False
FB_DB = None
FIREBASE_KEY_PATH = os.path.join(ROOT_DIR, "firebase", "serviceAccountKey.json")

def _init_firebase_if_possible():
    """Inicializa Firestore si existe la clave y está instalado firebase-admin."""
    global FIREBASE_ENABLED, FB_DB
    if FIREBASE_ENABLED or FB_DB is not None:
        return
    try:
        if os.path.isfile(FIREBASE_KEY_PATH):
            import firebase_admin
            from firebase_admin import credentials, firestore
            if not firebase_admin._apps:
                cred = credentials.Certificate(FIREBASE_KEY_PATH)
                firebase_admin.initialize_app(cred)
            FB_DB = firestore.client()
            FIREBASE_ENABLED = True
            print(">> [FB] Firestore inicializado ✅")
        else:
            print(">> [FB] Clave no encontrada. Se ejecuta sin Firebase.")
    except Exception as e:
        print(f">> [FB] Desactivado ({type(e).__name__}): {e}")
        FIREBASE_ENABLED = False
        FB_DB = None

def _tensor_to_png_data_url(x_012: np.ndarray) -> str:
    """Convierte (1,28,28,1) en 'data:image/png;base64,...' (miniatura pequeña)."""
    img = Image.fromarray((x_012.squeeze() * 255).astype("uint8"), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

def _log_prediction_to_firestore(x_012: np.ndarray, pred: int, probs: np.ndarray):
    """Inserta un documento en la colección 'predictions' (si Firebase está activo)."""
    if not FIREBASE_ENABLED or FB_DB is None:
        return {"saved": False, "reason": "firebase_disabled"}
    try:
        doc = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "client_ip": request.remote_addr,
            "pred": int(pred),
            "probs": [float(p) for p in probs.tolist()],
            # Miniatura del 28x28 para depurar (muy pequeña, segura para Firestore)
            "thumb_28x28_png": _tensor_to_png_data_url(x_012),
        }
        FB_DB.collection("predictions").add(doc)
        return {"saved": True}
    except Exception as e:
        print(f">> [FB] Error al guardar: {e}")
        return {"saved": False, "reason": str(e)}

# ====== API ======
@app.route("/predict", methods=["POST"])
def predict():
    """
    Espera JSON: { "image": "data:image/png;base64,...." }
    Devuelve predicción y vector de probabilidades, y registra en Firestore si está disponible.
    """
    data = request.get_json(force=True, silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "Falta el campo 'image' en el cuerpo de la petición."}), 400

    # Preproceso
    x = preprocess_png_base64(data["image"])

    # Modelo
    model = get_model()
    probs = model.predict(x, verbose=0)[0].astype(float)  # (10,)
    pred  = int(np.argmax(probs))

    # Firebase (no bloquea si no está configurado)
    _init_firebase_if_possible()
    fb_status = _log_prediction_to_firestore(x, pred, probs)

    return jsonify({
        "prediccion": pred,
        "prediction": pred,
        "probabilidades": probs.tolist(),
        "probs": probs.tolist(),
        "firebase": fb_status
    })

# ====== ARRANQUE ======
if __name__ == "__main__":
    print(f">> [BOOT] Iniciando app.py... (Python {os.sys.version.split()[0]})")
    print(f">> [BOOT] APP_DIR: {APP_DIR}")
    print(f">> [BOOT] ROOT_DIR: {ROOT_DIR}")
    print(f">> [BOOT] MODEL_PATH: {MODEL_PATH} exists? {os.path.isfile(MODEL_PATH)}")
    print(">> [RUN] Lanzando servidor en http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)







