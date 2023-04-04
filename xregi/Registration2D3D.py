import numpy as np

class Registration2D3D:
    def __init__(self, image: np.ndarray, ct_path: str, landmarks_3d: dict, extrinsic: np.ndarray):
        self.image = image
        self.ct_path = ct_path
        self.landmarks_3d = landmarks_3d
        self.extrinsic = extrinsic