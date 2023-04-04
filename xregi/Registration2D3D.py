import numpy as np
from utils import *
import pandas as pd

class Registration2D3D:
    def __init__(self, image: np.ndarray, ct_path: str, landmarks_3d: dict, intrinsic: np.ndarray):
        self.image = image
        self.ct_path = ct_path
        self.landmarks_3d = landmarks_3d
        self.intrinsic = intrinsic

    @classmethod
    def load(cls, image_path_load: str,ct_path_load: str,landmarks_3d_path: str, intrinsic_load: np.ndarray):
        image_load = read_xray_dicom(image_path_load)
        landmarks_3d=get_3d_landmarks(landmarks_3d_path, folder_type = "fcsv",label_idx=11)
        #intrinsic load from dicom?
        return cls(image_load,ct_path_load, landmarks_3d,intrinsic_load)
