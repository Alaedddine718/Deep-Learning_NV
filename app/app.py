import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from PIL import Image
import io
import base64

app = Flask(__name__)
socketio = SocketIO(app)

# Ruta absoluta del modelo
BASE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "mnist_compiled_model.keras")
BASE_MODEL_PATH = os.path.abspath(BASE_MODEL_PATH)

_model = None

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(BASE_MODEL_PATH):
            raise FileNotFoundError(
                f"No existe {BASE_MODEL_PATH}. Entrena y guarda el modelo primero desde el notebook."
            )
        _model = load_model(BASE_MODEL_PATH)
    return _model

@app.route("/")
def index():
    # Renderiza la página principal
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    model = get_model()

    # Recibir imagen en base64 desde el frontend
    data = request.json["image"]
    image_bytes = base64.b64decode(data.split(",")[1])
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = image.resize((28, 28))
    img_array = np.array(image).reshape(1, 784).astype("float32") / 255.0

    preds = model.predict(img_array)
    pred_class = np.argmax(preds, axis=1)[0]
    pred_prob = float(np.max(preds))

    return jsonify({
        "prediction": int(pred_class),
        "probability": pred_prob
    })

# Endpoint simple para verificar que el servidor funciona
@app.route("/ping")
def ping():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    socketio.run(app, debug=True)



