from __future__ import annotations
import os
from typing import Dict, Any

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
except Exception:
    firebase_admin = None
    credentials = None
    firestore = None

_app = None
_db = None

def init_firebase(path_key: str = "firebase/serviceAccountKey.json"):
    global _app, _db
    if firebase_admin is None:
        print("[firebase] Paquete no disponible, saltando.")
        return None
    if _app:
        return _db
    if not os.path.exists(path_key):
        print(f"[firebase] Clave no encontrada en {path_key}, saltando.")
        return None
    cred = credentials.Certificate(path_key)
    _app = firebase_admin.initialize_app(cred)
    _db = firestore.client()
    print("[firebase] inicializado.")
    return _db

def save_result(user_id: str, payload: Dict[str, Any], collection: str = "results"):
    if _db is None:
        print("[firebase] DB no inicializada, omitiendo save.")
        return
    _db.collection(collection).add({"user_id": user_id, **payload})
