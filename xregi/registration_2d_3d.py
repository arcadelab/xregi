import numpy as np
from utils import *
from typing import Type, Dict, List
import pandas as pd
from landmark_detector import SynthexDetector, LandmarkDetector
from registration_solver import XregSolver, RegistrationSolver
from syn_args import default_args



class Registration2D3D:
    # Define detector and solver types
    # Could be changed to other types in the future
    registration_solver_type: Type[RegistrationSolver] = XregSolver
    landmark_detector_type: Type[LandmarkDetector] = SynthexDetector

    def __init__(
        self,
        image: np.ndarray,
        ct_path: str,
        landmarks_3d: Dict[str, List[float, float, float]],
        intrinsic: np.ndarray,
    ):
        """
        Initialize Registration2D3D class

        Args:
        -------
            image: np.ndarray, x-ray image in the shape of (# of image, width, height)
            ct_path: str, path to the CT scan file
            landmarks_3d: dict[str, list[float]], 3d landmarks in the shape of (landmark name, [x, y, z])
            intrinsic: np.ndarray, intrinsic parameters of the x-ray imaging system

        """
        self.image = image
        self.ct_path = ct_path
        self.landmarks_3d = landmarks_3d
        self.intrinsic = intrinsic

    def run(self):
        """
        run the registration process and return the 3d coordinates of landmarks

        """
        args = default_args()
        landmark_detector = self.landmark_detector_type.load(args.xray_path,args.label_path, args.output_path, "01")
        landmarks_2d = landmark_detector.run()
        registration_solver = self.registration_solver_type(
            self.image, landmarks_2d, self.landmarks_3d
        )  # TODO: intrinsics?
        return registration_solver.solve()

    @classmethod
    def load(
        cls,
        image_path_load: str,
        ct_path_load: str,
        landmarks_3d_path: str,
        intrinsic_param: np.ndarray,
    ):
        """
        Initialize Registration2D3D class by loading image, ct, landmarks, intrinsic parameters from files with given paths

        Args:
        -------
            image_path_load: str, path to the x-ray image folder which contains multiple dicom files
            ct_path_load: str, path to the CT scan file
            landmarks_3d_path: str, path to the 3d landmarks file, currently only support fcsv file
            intrinsic_param: np.ndarray, intrinsic parameters of the x-ray imaging system

        """
        image_load = read_xray_dicom(image_path_load)
        landmarks_3d = get_3d_landmarks(
            landmarks_3d_path, folder_type="fcsv", label_idx=11
        )
        # intrinsic load from dicom?
        return cls(image_load, ct_path_load, landmarks_3d, intrinsic_param)


if __name__ == "__main__":
   pass