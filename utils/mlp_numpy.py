from __future__ import annotations
import numpy as np
from typing import Callable, List

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

ACT = {"sigmoid": sigmoid, "relu": relu}

class Layer:
    def __init__(self, n_in: int, n_out: int, activation: str = "relu", seed: int = 42):
        rng = np.random.default_rng(seed)
        std = np.sqrt(2.0 / (n_in + n_out))
        self.W = rng.normal(0.0, std, size=(n_in, n_out))
        self.b = np.zeros((n_out,), dtype=float)
        self.act: Callable[[np.ndarray], np.ndarray] = ACT[activation]

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.act(x @ self.W + self.b)

class MLP:
    def __init__(self, sizes: List[int], activations: List[str]):
        assert len(sizes) >= 2 and len(activations) == len(sizes) - 1
        self.layers = [Layer(sizes[i], sizes[i+1], activations[i]) for i in range(len(sizes)-1)]

    def predict(self, x: np.ndarray) -> np.ndarray:
        out = x
        for lyr in self.layers:
            out = lyr.forward(out)
        return out
