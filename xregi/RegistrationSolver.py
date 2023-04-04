import numpy as np

class RegistrationSolver:
    def __init__(self, image:np.ndarray, landmarks_2D:dict, landmarks_3D:dict):
        self.image = image
        self.landmarks_2D = landmarks_2D
        self.landmarks_3D = landmarks_3D
    
