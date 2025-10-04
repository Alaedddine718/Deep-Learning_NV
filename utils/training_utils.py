from __future__ import annotations
import os, argparse
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from .interpreter import compile_model

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
BASE_MODEL_PATH = os.path.join(MODELS_DIR, "mnist_compiled_model.keras")  # formato .keras

def train_base(arch: str = "Dense(256,relu)->Dense(128,relu)->Dense(10,softmax)",
               epochs: int = 5, batch_size: int = 128):
    (trainX, trainY), (testX, testY) = mnist.load_data()
    trainX = trainX.reshape((trainX.shape[0], -1)).astype("float32") / 255.0
    testX  = testX.reshape((testX.shape[0], -1)).astype("float32") / 255.0
    ytr = to_categorical(trainY, 10)
    yte = to_categorical(testY, 10)
    model = compile_model(arch, input_dim=784)
    model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(trainX, ytr, validation_data=(testX, yte),
              epochs=epochs, batch_size=batch_size, verbose=2)
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(BASE_MODEL_PATH)
    return BASE_MODEL_PATH

def incremental_update(user_id: str, X: np.ndarray, y: np.ndarray,
                       epochs: int = 1, batch_size: int = 64):
    """Continúa entrenando el modelo base con nuevos ejemplos (X: (N,784), y: [0..9])."""
    if not os.path.exists(BASE_MODEL_PATH):
        raise FileNotFoundError("Modelo base no encontrado, entrena primero.")
    from tensorflow.keras.utils import to_categorical
    model = load_model(BASE_MODEL_PATH)
    y_cat = to_categorical(y, 10)
    model.fit(X, y_cat, epochs=epochs, batch_size=batch_size, verbose=0)

    user_dir = os.path.join(MODELS_DIR, "user_models")
    os.makedirs(user_dir, exist_ok=True)
    out_path = os.path.join(user_dir, f"{user_id}.keras")
    model.save(out_path)
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-mnist", action="store_true")
    parser.add_argument("--arch", type=str, default="Dense(256,relu)->Dense(128,relu)->Dense(10,softmax)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    a = parser.parse_args()
    if a.train_mnist:
        p = train_base(arch=a.arch, epochs=a.epochs, batch_size=a.batch_size)
        print("Modelo guardado en:", p)
