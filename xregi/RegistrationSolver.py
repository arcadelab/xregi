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
    def __init__(self, image: np.ndarray, landmarks_2D: dict, landmarks_3D: dict):
        self.image = image
        self.landmarks_2D = landmarks_2D
        self.landmarks_3D = landmarks_3D

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
