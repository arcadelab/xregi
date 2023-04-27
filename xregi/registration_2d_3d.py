import numpy as np
from utils import *
from typing import Type, Dict, List
import pandas as pd
from landmark_detector import SynthexDetector, LandmarkDetector
from registration_solver import XregSolver, RegistrationSolver
from args import  xreg_args, cam_params
import json


class Registration2D3D:
    # Define detector and solver types
    # Could be changed to other types in the future
    registration_solver_type: Type[RegistrationSolver] = XregSolver
    landmark_detector_type: Type[LandmarkDetector] = SynthexDetector

    def __init__(
        self,
        image: np.ndarray,
        ct_path: str,
        landmarks_3d: Dict[str, List[float]],
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

        landmark_detector = self.landmark_detector_type.load()
        landmark_detector.run()

        path, cam_params = xreg_args()
        registration_solver = self.registration_solver_type(
            path["image_path_load"],
            path["ct_path_load"],
            path["ct_segmentation_path"],
            path["landmarks_2d_path"],
            path["landmarks_3d_path"],
            cam_params,
        )

        return registration_solver.solve("run_reg")

    @classmethod
    def load(
        cls
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
        ##read json file
        with open("config/config.json") as f:
            data = json.load(f)

        image_load, scale = preprocess_dicom(data["xray_path"], img_size=360)
        landmarks_3d = get_3d_landmarks(
            data['landmarks_3d_path'], folder_type="fcsv", label_idx=11
        )
        # intrinsic params are hardcoded for now
        intrinsic_param = scale * cam_params()["intrinsic"]
        intrinsic_param[-1] = 1

        print(intrinsic_param)

        return cls(image_load, data["ct_path"], landmarks_3d, intrinsic_param)


if __name__ == "__main__":
    path, camera_params = xreg_args()
    reg2d3d = Registration2D3D.load(
        "data/xray",
        "data/ct/ct.dcm",
        "data/landmarks/landmarks.fcsv",
        camera_params["intrinsic"],
    )
