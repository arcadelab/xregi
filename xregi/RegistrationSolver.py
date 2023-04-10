import numpy as np
import pandas as pd
from SyntheX.utils import *
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
    def __init__(self, image: np.ndarray, landmarks_2D: dict, landmarks_3D: dict):
        self.image = image
        self.landmark = Landmark()
        self.landmarks_2D = landmarks_2D
        self.landmarks_3D = landmarks_3D

    def load(self, image_path_load: str, ct_path_load: str, landmarks_2d_path: str, landmarks_3d_path: str):
        image_load = read_xray_dicom(image_path_load)
        landmarks_3d = get_3d_landmarks(
            landmarks_3d_path, folder_type="fcsv", label_idx=11)
        landmarks_2d = get_2d_landmarks(landmarks_2d_path)

        return cls(image_load, ct_path_load, landmarks_3d)

    def solve(self, runOptions) -> np.ndarray:
        '''Call the executable file
        Params:
        -------
        runOptions: str
            'run_reg' or 'run_viz' , 
            'run_reg' is used to run the registration
            'run_viz' is used to visualize the registration result

        Returns:
        --------
            None 

        '''

        if runOptions == 'run_reg':
            print("run_reg is running ...")

            result = subprocess.run(["bin/xreg-hip-surg-pelvis-single-view-regi-2d-3d",
                                    "data/pelvis.nii.gz",
                                     "data/pelvis_regi_2d_3d_lands_wo_id.fcsv",
                                     "data/example1_1_pd_003.h5",
                                     "result/regi_pose_example1_1_pd_003_proj0.h5",
                                     "result/regi_debug_example1_1_pd_003_proj0_w_seg.h5",
                                     "-s",
                                     "data/pelvis_seg.nii.gz"], stdout=subprocess.PIPE)

            # Print the output of the executable file
            print(result.stdout.decode())

        elif runOptions == 'run_viz':
            result = subprocess.run(["bin/xreg-regi2d3d-replay",
                                    "result/regi_debug_example1_1_pd_003_proj0_w_seg.h5",
                                     "--video-fps",
                                     "10",
                                     "--proj-ds",
                                     "0.5"], stdout=subprocess.PIPE)
            print(result.stdout.decode())

    def get_2d_landmarks(self, landmarks_2d_path: str) -> dict:
        '''Get 2D landmarks from the csv file
        Params:
        -------
        landmarks_2d_path: str
            Path to the csv file

        Returns:
        --------
        landmarks_2d: dict[str, np.ndarray]
            A dictionary of 2D landmarks
        '''
        landmarks_2d = {}
        data_frame = pd.read_csv(landmarks_2d_path)
        landmar_value = data_frame.drop(
            columns=['pat', 'proj', 'time'], axis=1)

        return landmarks_2d
