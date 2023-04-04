import numpy as np
from utils import *
from abc import ABC, abstractmethod

class RegistrationSolver(ABC):

    @abstractmethod()
    def solve(self) -> np.ndarray:
        '''
        Solve registration problem, return 3D coordinates of landmarks

        Args:
        -------
        self.image: np.ndarray
        self.landmarks_2D: dict[str, np.ndarray]
        self.landmarks_3D: dict[str, np.ndarray]

        Returns:
        -------
        landmarks_3D: np.ndarray
        '''
        pass   



class XregSlover(RegistrationSolver):
    def __init__(self, image:np.ndarray, landmarks_2D:dict, landmarks_3D:dict):
        self.image = image
        self.landmarks_2D = landmarks_2D
        self.landmarks_3D = landmarks_3D
    
