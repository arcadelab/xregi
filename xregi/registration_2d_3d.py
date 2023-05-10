import numpy as np
import os
from typing import Type, Dict, List, Optional
import pandas as pd

from .args import cam_param
from .utils import *
from .landmark_detector import SynthexDetector, LandmarkDetector
from .registration_solver import XregSolver, RegistrationSolver
from . import config


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
        cam_paras: Dict[str, np.ndarray],
        path_container: Optional[Dict[str, str]],
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
        self.cam_params = cam_paras
        self.path = path_container

    def run(self):
        """
        run the registration process and return the 3d coordinates of landmarks

        """

        landmark_detector = self.landmark_detector_type.load()
        landmark_detector.run()

        registration_solver = self.registration_solver_type.load(
            self.path["image_path"],
            self.path["ct_path"],
            self.path["CT_segmentation_path"],
            self.path["out"],
            self.path["landmarks_3d_path"],
            self.cam_params,
        )

        registration_solver.solve("run_reg")
        registration_solver.solve("run_viz")

    @classmethod
    def load(cls):
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
        path = config.load_json(os.path.abspath(config.__file__))

        image_path = newestfile(path["xray_path"])
        path["image_path"] = image_path

        # with open("config/config.json", "w") as file:
        #     json.dump(path, file)

        cam_params = cam_param()
        print("image_path", image_path)
        if cam_params["img_type"] == "DICOM":
            resized_img, image_load, scale = preprocess_dicom(image_path, 360)

        elif cam_params["img_type"] == "PNG":
            resized_img, image_load, scale = read_xray_png(image_path, 360)

        else:
            raise ValueError("Image type not supported")
        cam_params["scale"] = scale
        landmarks_3d = get_3d_landmarks(path["landmarks_3d_path"], "fcsv", 11)
        # intrinsic params are hardcoded for now
        # intrinsic_param = scale * cam_params()["intrinsic"]
        # intrinsic_param[-1] = 1

        # print(intrinsic_param)

        return cls(image_load, path["ct_path"], landmarks_3d, cam_params, path)


if __name__ == "__main__":
    # path, camera_params = xreg_args()
    # reg2d3d = Registration2D3D.load(
    #     "data/xray",
    #     "data/ct/ct.dcm",
    #     "data/landmarks/landmarks.fcsv",
    #     camera_params["intrinsic"],
    # )
    pass
