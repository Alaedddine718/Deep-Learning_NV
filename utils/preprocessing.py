from __future__ import annotations
import io, base64, numpy as np
from PIL import Image, ImageOps

def load_image_to_array(file_or_bytes) -> Image.Image:
    if isinstance(file_or_bytes, (bytes, bytearray)):
        return Image.open(io.BytesIO(file_or_bytes)).convert("L")
    return Image.open(file_or_bytes).convert("L")

def base64_to_image(b64: str) -> Image.Image:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    data = base64.b64decode(b64)
    return load_image_to_array(data)

def preprocess_digit(img: Image.Image, size=(28,28)) -> np.ndarray:
    img = ImageOps.invert(img)  # invierte si viene blanco sobre negro
    img = img.resize(size, Image.LANCZOS)
    arr = np.array(img).astype("float32")/255.0
    arr = arr.reshape(1, -1)  # (1, 784)
    return arr
