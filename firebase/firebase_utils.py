# firebase/firebase_utils.py
import firebase_admin
from firebase_admin import credentials, firestore
import os

# Ruta absoluta al archivo de credenciales
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SERVICE_ACCOUNT_PATH = os.path.join(ROOT_DIR, "firebase", "serviceAccountKey.json")

# Inicializar Firebase (solo una vez)
if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred)
    print(">> [FB] Firestore inicializado ✅")

# Cliente Firestore
db = firestore.client()

def save_prediction_to_firestore(pred, probs, img_base64, ip="unknown"):
    """Guarda los resultados de una predicción en Firebase Firestore."""
    doc = {
        "pred": pred,
        "probs": probs,
        "thumb_28x28_png": img_base64[:200] + "...",  # acortado
        "created_at": firestore.SERVER_TIMESTAMP,
        "client_ip": ip,
    }
    db.collection("predictions").add(doc)
    print(">> [FB] Guardado en Firestore ✅")
