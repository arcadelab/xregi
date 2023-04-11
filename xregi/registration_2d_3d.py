import numpy as np
from utils import *
from typing import Type, Dict, List
import pandas as pd
from landmark_detector import SynthexDetector, LandmarkDetector
from registration_solver import XregSolver, RegistrationSolver
import argparse


class Registration2D3D:
    # Define detector and solver types
    registration_solver_type: Type[RegistrationSolver] = XregSolver
    landmark_detector_type: Type[LandmarkDetector] = SynthexDetector

    # def __init__(
    #     self,
    #     image: np.ndarray,
    #     ct_path: str,
    #     landmarks_3d: Dict[str, List[float]],
    #     intrinsic: np.ndarray,
    # ):
    #     self.image = image
    #     self.ct_path = ct_path
    #     self.landmarks_3d = landmarks_3d
    #     self.intrinsic = intrinsic

    # def select_detector(self, detector: str, path: dict, args):
    #     self.path = path
    #     if detector == "synthex":
    #         self.syn = SynthexDetector.load(
    #             path["image"], path["label"], path["output"], path["pats"]
    #         )
    #         self.syn.load_data(args)

    #     elif detector == "else":
    #         pass  # TODO

    # def run_synthex(self, image_path):
    #     pass

    # def run_xreg(self):
    #     pass

    def run(self):
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
        intrinsic_load: np.ndarray,
    ):
        image_load = read_xray_dicom(image_path_load)
        landmarks_3d = get_3d_landmarks(
            landmarks_3d_path, folder_type="fcsv", label_idx=11
        )
        # intrinsic load from dicom?
        return cls(image_load, ct_path_load, landmarks_3d, intrinsic_load)


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
