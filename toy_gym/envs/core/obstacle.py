import numpy as np

class Obstacle():
    def __init__(self, name: str, shape: str, dimension:np.ndarray, position, is_fixed):
        self._name = name
        self._shape = shape
        self._dimension = dimension
        self._position = position
        self._is_fixed = is_fixed