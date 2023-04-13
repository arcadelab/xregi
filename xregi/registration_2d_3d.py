import numpy as np
from utils import *
from typing import Type, Dict, List
import pandas as pd
from landmark_detector import SynthexDetector, LandmarkDetector
from registration_solver import XregSolver, RegistrationSolver
import argparse


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

        landmark_detector = self.landmark_detector_type(self.image)
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
    image_path_load = ""
    ct_path_load = ""
    landmarks_3d_path = ""
    intrinsic_load = ""
    reg = Registration2D3D.load(
        image_path_load, ct_path_load, landmarks_3d_path, intrinsic_load
    )
    # select detector
    path = {
        "image": "data/xray",
        "label": "data/real_label.h5",
        "output": "data",
        "pats": "01",
    }
    args = argparse.Namespace()
    args.nets = "data/yy_checkpoint_net_20.pt"
    args.input_data_file_path = "data/synthex_input.h5"
    args.input_label_file_path = "data/synthex_label_input.h5"
    args.output_data_file_path = "data/output.h5"
    args.rand = True
    args.pats = "01"
    args.no_gpu = True
    args.times = ""
    reg.select_detetctor("SyntheX", path, args)
    # run SyntheX
    args2 = argparse.Namespace()
    args2.heat_file_path = reg.syn.output_data_file_path
    args2.heats_group_path = "nn-heats"
    args2.out = "data/own_data.csv"
    args2.out = os.path.join(reg.syn.current_path, args2.out)
    args2.pat = "01"
    args2.use_seg = "nn-segs"
    args2.rand = True
    args2.hm_lvl = True
    args2.ds_factor = 4
    args2.no_hdr = True
    args2.use_seg = ""
    args2.threshold = 2
    reg.run_synthex(args2)
