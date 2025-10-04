from __future__ import annotations
import re
from typing import Optional
import tensorflow as tf

_ACT = {"relu":"relu", "sigmoid":"sigmoid", "tanh":"tanh", "softmax":"softmax", "linear":"linear"}
_PAT = {
    "input": re.compile(r"^Input\((\d+)\)$", re.IGNORECASE),
    "dense": re.compile(r"^Dense\((\d+)\s*,\s*([A-Za-z]+)\)$", re.IGNORECASE)
}

def _parse(tok: str):
    tok = tok.strip()
    m = _PAT["input"].match(tok)
    if m:
        return ("Input", (int(m.group(1)),))
    m = _PAT["dense"].match(tok)
    if m:
        units = int(m.group(1)); act = m.group(2).lower()
        if act not in _ACT: raise ValueError(f"Activación no soportada: {act}")
        return ("Dense", (units, act))
    raise ValueError(f"Token no reconocido: {tok}")

def compile_model(architecture: str, input_dim: Optional[int] = None) -> tf.keras.Model:
    if not architecture or not isinstance(architecture, str):
        raise ValueError("architecture debe ser una cadena no vacía")
    tokens = [t for t in (s.strip() for s in architecture.split('->')) if t]
    layers = [_parse(t) for t in tokens]
    explicit_in = next((args[0] for (name,args) in layers if name=="Input"), None)
    final_in = explicit_in if explicit_in is not None else input_dim
    if final_in is None:
        raise ValueError("Especifica Input(dim) o input_dim=...")

    model = tf.keras.Sequential(name="compiled_from_text")
    first = True
    for name, args in layers:
        if name == "Input":
            continue
        if name == "Dense":
            units, act = args
            if first:
                model.add(tf.keras.layers.Dense(units, activation=_ACT[act], input_shape=(final_in,)))
                first = False
            else:
                model.add(tf.keras.layers.Dense(units, activation=_ACT[act]))
    return model
