from __future__ import annotations
import numpy as np
from typing import Callable

def rk4_step(f: Callable[[float, np.ndarray], np.ndarray], y: np.ndarray, t: float, h: float) -> np.ndarray:
    k1 = f(t, y)
    k2 = f(t + 0.5*h, y + 0.5*h*k1)
    k3 = f(t + 0.5*h, y + 0.5*h*k2)
    k4 = f(t + h,     y + h*k3)
    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
