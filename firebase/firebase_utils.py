# firebase/firebase_utils.py
from __future__ import annotations
import os
import io
import base64
from datetime import datetime

import numpy as np
from PIL import Image

import firebase_admin
from firebase_admin import credentials, firestore

# ----- Rutas -----
THIS_DIR = os.path.dirname(__file__)                          # .../firebase
ROOT_DIR = os.path.dirname(THIS_DIR)                          # raíz del repo
SA_PATH  = os.path.join(THIS_DIR, "serviceAccountKey.json")   # clave

# ----- Inicializar Firebase Admin una sola vez -----
if not firebase_admin._apps:
    if not os.path.isfile(SA_PATH):
        raise FileNotFoundError(
            f"No encuentro serviceAccountKey.json en: {SA_PATH}\n"
            f"Colócalo dentro de /firebase."
        )
    cred = credentials.Certificate(SA_PATH)
    firebase_admin.initialize_app(cred)
    print(">> [FB] Firestore inicializado ✅")

_db = firestore.client()

def _img_np_to_png_b64(x_012: np.ndarray) -> str:
    """
    x_012: (28,28) o (28,28,1) en rango [0,1] -> PNG base64 (sin header data:)
    """
    if x_012.ndim == 3 and x_012.shape[-1] == 1:
        x_012 = x_012.squeeze(-1)
    x_255 = (np.clip(x_012, 0, 1) * 255).astype("uint8")
    im = Image.fromarray(x_255, mode="L")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def save_prediction_to_firestore(
    pred: int,
    probs: list[float],
    image_np01: np.ndarray,
    model_name: str = "mnist_cnn"
) -> None:
    """
    Guarda un documento en la colección 'predictions':
      - prediction (int)
      - probs (list[float])
      - image_png_b64 (string)
      - model (string)
      - client_time (ISO)
      - ts (server timestamp)
    """
    try:
        img_b64 = _img_np_to_png_b64(image_np01)
        doc = {
            "prediction": int(pred),
            "probs": [float(p) for p in probs],
            "image_png_b64": img_b64,
            "model": model_name,
            "client_time": datetime.utcnow().isoformat() + "Z",
            "ts": firestore.SERVER_TIMESTAMP,
        }
        _db.collection("predictions").add(doc)
        print(">> [FB] Guardado en Firestore ✅")
    except Exception as e:
        print(f">> [FB] Error al guardar: {e}")


