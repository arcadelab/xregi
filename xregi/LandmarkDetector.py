import numpy as np
from abc import ABC


class LandmarkDetector(ABC):
    def __init__(self, image):
        self.image = image

    def detect(self):
        pass
