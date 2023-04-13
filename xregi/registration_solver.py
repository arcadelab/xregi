import numpy as np
import pandas as pd
import subprocess
from typing import Dict, List, Optional
from utils import *
from landmark_container import LandmarkContainer
from abc import ABC, abstractmethod


class RegistrationSolver(ABC):
    """
    Abstract class for registration solver

    Args:
    -------
        image (np.ndarray): x-ray image 3d array with (# of image, height, width) shape
        landmarks_2D (dict[str, list]): dictionary (landmark name, [x, y])
        ct_path (str): string contains the path to the CT scan file
        landmarks_3D (dict[str, list]): dictionary (landmark name, [x, y, z])
        cam_param (dict[str, np.ndarray]): camera intrinsic and extrinsic parameters

    Returns:
    -------
        np.ndarray: 4x4 transformation matrix
    """

    def __init__(
        self,
        image: np.ndarray,
        landmarks_2D: Dict[str, List[float]],
        ct_path: str,
        landmarks_3D: Dict[str, List[float]],
        cam_param: Dict[str, np.ndarray],
    ):
        self._image = image
        self.landmarks_2D = landmarks_2D
        self.ct_path = ct_path
        self.landmarks_3D = landmarks_3D
        self.cam_param = cam_param

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def solve(self) -> np.ndarray:
        """
        Solve registration problem, return 3D coordinates of landmarks

        Returns:
        -------
            np.ndarray: 4x4 transformation matrix
        """
        pass


class XregSolver(RegistrationSolver):
    """
    Solve 2d 3d registration problem using xreg
    This class is for single-view registration, which only takes one x-ray image as input
    """

    def __init__(
        self,
        image: np.ndarray,
        landmarks_2D: Dict[str, List[float]],
        ct_path: str,
        landmarks_3D: Dict[str, List[float]],
        cam_param: Dict[str, np.ndarray],
        path: Optional[Dict[str, str]],
    ):
        """
        Initialize XregSolver class

        Args:
        -------
            image (np.ndarray): x-ray image 3d array with (# of image, height, width) shape
            landmarks_2D (dict[str, list]): dictionary (landmark name, [x, y])
            ct_path (str): string contains the path to the CT scan file
            landmarks_3D (dict[str, list]): dictionary (landmark name, [x, y, z])
            cam_param (dict[str, np.ndarray]): camera intrinsic and extrinsic parameters
            path (dict[str, str]): dictionary contains the path to the xreg input and output files

        """
        super().__init__(
            image, landmarks_2D, ct_path, landmarks_3D, cam_param
        )  # not load the image here

        current_path = os.path.abspath(os.path.dirname(__file__))

        self.path = path if path is not None else {}
        path["current_path"] = current_path
        path["ct_path"] = os.path.join(current_path, self.ct_path)

    @classmethod
    def load(
        cls,
        image_path_load: str,
        ct_path_load: str,
        landmarks_2d_path: str,
        landmarks_3d_path: str,
        cam_param: Dict[str, np.ndarray] = None,
    ):
        """
        Load the image and landmarks from local file with given path

        Args:
        -------
            image_path_load (str): path to the x-ray image
            ct_path_load (str): path to the CT scan
            landmarks_2d_path (str): path to the 2d landmarks
            landmarks_3d_path (str): path to the 3d landmarks
            cam_param (dict[str, np.ndarray]): camera intrinsic and extrinsic parameters

        Returns:
        -------
            XregSolver: XregSolver object
        """

        image_load = read_xray_dicom(image_path_load)

        landmarks_2d = cls.get_2d_landmarks(landmarks_2d_path)

        return cls(
            image_load,
            landmarks_2d,
            ct_path_load,
            None,
            None,
            {"landmark_3d_path": landmarks_3d_path},
        )

    @property
    def image(self) -> np.ndarray:
        """
        Get the image for registration solver
        """
        return self._image

    @image.setter
    def image(self, new_image: np.ndarray):
        """
        Set the image for registration solver
        This allows the solver to change the image without re-initializing the class
        """
        self._image = new_image

    def generate_h5(self):
        """
        Generate the h5 file for xreg
        the h5 file contains x-ray image and 2d landmarks
        """

        h5_file = h5py.File(self.path, "w")
        h5_file.create_dataset("num_projs", data=1, dtype="u8")
        h5_file.create_group("proj-000")

        with h5py.File("data/example1_1_pd_003.h5", "r") as h5_template:
            for key in h5_template["proj-000"].keys():
                # print(h5_template['proj-000'][key].values())
                h5_file["proj-000"].create_group(key)
                for dataset in h5_template["proj-000"][key].keys():
                    # print(dataset)

                    if dataset == "pixels":
                        h5_file["proj-000"][key].create_dataset(
                            dataset,
                            data=self.image,
                            dtype=h5_template["proj-000"][key][dataset].dtype,
                        )
                    else:
                        h5_file["proj-000"][key].create_dataset(
                            dataset,
                            data=h5_template["proj-000"][key][dataset][...],
                            dtype=h5_template["proj-000"][key][dataset].dtype,
                        )

        h5_file["proj-000"]["cam"]["num-cols"][...] = self.image.shape[1]
        h5_file["proj-000"]["cam"]["num-rows"][...] = self.image.shape[0]

        h5_template.close()

        # write the 2d landmarks to the HDF5 file
        lm_names_synthex = [
            "FH-l",
            "FH-r",
            "GSN-l",
            "GSN-r",
            "IOF-l",
            "IOF-r",
            "MOF-l",
            "MOF-r",
            "SPS-l",
            "SPS-r",
            "IPS-l",
            "IPS-r",
            "ASIS-l",
            "ASIS-r",
        ]  # this is the order of the landmarks in the SyntheX dataset

        for lms in h5_file["proj-000"]["landmarks"].keys():
            landmark_2d = self.landmark.get_landmark(mode="xreg")

            h5_file["proj-000"]["landmarks"][lms][...] = np.reshape(
                np.asarray(landmark_2d[lms], (2, 1))
            )
            # print(np.asarray(landmarks_2d.iloc[lm_idx].values))
            # h5_file['proj-000']['landmarks'][lms] = 0.0

    def solve(self, runOptions: str) -> np.ndarray:
        """Call the executable file
        Args:
        -------
        runOptions: str
                    'run_reg' or 'run_viz' ,
                    'run_reg' is used to run the registration
                    'run_viz' is used to visualize the registration result

        Returns:
        --------
            None

        """
        xreg_path = {}
        xreg_path["solver_path"] = os.path.join(
            self.path["current_path"], "bin/xreg-hip-surg-pelvis-single-view-regi-2d-3d"
        )
        xreg_path["ct_path"] = self.path["ct_path"]
        xreg_path["ct_segmentation_path"] = os.path.join(
            self.path["current_path"], "data/pelvis_seg.nii.gz"
        )
        xreg_path["3d_landmarks_path"] = os.path.join(
            self.path["current_path"], "data/pelvis_regi_2d_3d_lands_wo_id.fcsv"
        )
        xreg_path["xray_path"] = os.path.join(
            self.path["current_path"], "data/example1_1_pd_003.h5"
        )
        xreg_path["result_path"] = os.path.join(
            self.path["current_path"], "data/xreg_result_pose.h5"
        )
        xreg_path["debug_path"] = os.path.join(
            self.path["current_path"], "data/xreg_debug_log.h5"
        )

        if runOptions == "run_reg":
            print("run_reg is running ...")

            result = subprocess.run(
                [
                    xreg_path["solver_path"],
                    xreg_path["ct_path"],
                    xreg_path["3d_landmarks_path"],
                    xreg_path["xray_path"],
                    xreg_path["result_path"],
                    xreg_path["debug_path"],
                    "-s",  # option to use the segmentation to mask out the irrelevant part of the CT
                    xreg_path["ct_segmentation_path"],
                ],
                stdout=subprocess.PIPE,
            )

            # Print the output of the executable file
            print(result.stdout.decode())

        elif runOptions == "run_viz":
            result = subprocess.run(
                [
                    "bin/xreg-regi2d3d-replay",
                    "result/regi_debug_example1_1_pd_003_proj0_w_seg.h5",
                    "--video-fps",
                    "10",
                    "--proj-ds",
                    "0.5",
                ],
                stdout=subprocess.PIPE,
            )
            print(result.stdout.decode())

    def get_2d_landmarks(landmarks_path: str) -> dict:
        """Get 2D landmarks from the csv file
        Params:
        -------
        landmarks_2d_path: str
            Path to the csv file

        Returns:
        --------
        landmarks_2d: dict[str, np.ndarray]
            A dictionary of 2D landmarks
        """
        # This is the synthex format and order for landmarks
        land_name = [
            "FH-l",
            "FH-r",
            "GSN-l",
            "GSN-r",
            "IOF-l",
            "IOF-r",
            "MOF-l",
            "MOF-r",
            "SPS-l",
            "SPS-r",
            "IPS-l",
            "IPS-r",
            "ASIS-l",
            "ASIS-r",
        ]

        landmarks_2d = {}
        data_frame = pd.read_csv(landmarks_path)
        data_frame = pd.DataFrame.drop(
            data_frame, columns=["pat", "proj", "time", "land"], axis=1
        )

        data_frame["land-name"] = land_name
        print(data_frame["land-name"][0])

        for i in range(len(data_frame)):
            landmarks_2d[data_frame["land-name"][i]] = [
                data_frame["row"][i],
                data_frame["col"][i],
            ]

        return landmarks_2d


if __name__ == "__main__":
    # image = np.ones((3,3))

    # lm_2d = {}
    # lm_3d = {}

    # lm_2d['sps_l'] = [1, 2]
    # lm_2d['sps_r'] = [2, 3]
    # lm_3d['gsn_l'] = [3, 4]
    # lm_3d['gsn_r'] = [4, 5]

    # xreg = XregSlover(image,lm_2d,lm_3d)
    # x = xreg.get_2d_landmarks("data/own_data.csv")
    # print(x.values())

    reg_solver = XregSolver.load(
        image_path_load="data/x_ray1.dcm",
        ct_path_load="data/pelvis.nii.gz",
        landmarks_2d_path="data/own_data.csv",
        landmarks_3d_path="data/pelvis_regi_2d_3d_lands_wo_id.fcsv",
    )

    print(reg_solver.path)

    # x = {}
    # x['sps_l'] = [1, 2]
    # x['sps_r'] = [2, 3]
    # x['gsn_l'] = [3, 4]
    # x['gsn_r'] = [4, 5]
    # print()

    # lm = LandmarkContainer.load('2d', list(
    #     x.values()), list(x.keys()))
