import numpy as np
import pandas as pd
import subprocess
from typing import Dict, List, Optional
from abc import ABC, abstractmethod

from .args import *
from .utils import *
from .landmark_container import LandmarkContainer
from . import vis_utils as vis_utils


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
        landmarks_2D: Dict[str, np.ndarray],
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
        landmarks_2D: Dict[str, np.ndarray],
        ct_path: str,
        landmarks_3D: Dict[str, List[float]],
        cam_params: Dict[str, np.ndarray],
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
            image, landmarks_2D, ct_path, landmarks_3D, cam_params
        )  # not load the image here
        self.cam_params = cam_params
        ############################################
        ########### set default path ###############
        ############################################
        current_path = os.path.abspath(os.path.dirname(__file__))
        self.path = path if path is not None else {}
        self.path["current_path"] = current_path
        self.path["ct_path"] = os.path.join(current_path, self.ct_path)
        self.path["xray_path"] = os.path.join(current_path, "data/xreg_input.h5")
        self.path["solver_path"] = os.path.join(
            current_path,
            "bin/xreg-hip-surg-pelvis-single-view-regi-2d-3d",
        )
        self.path["verifier_path"] = os.path.join(
            current_path, "bin/xreg-regi2d3d-replay"
        )
        self.path["h5_path_template"] = os.path.join(
            current_path, "data/example1_1_pd_003.h5"
        )
        self.path["result_path"] = os.path.join(
            current_path, "data/xreg_result_pose.h5"
        )
        self.path["debug_path"] = os.path.join(
            current_path, "data/xreg_result_debug.h5"
        )

        # print(self.path)

        self.generate_h5()

    @classmethod
    def load(
        cls,
        image_path_load: str,
        ct_path_load: str,
        ct_segmentation_path: str,
        landmarks_2d_path: str,
        landmarks_3d_path: str,
        cam_params: Dict[str, np.ndarray] = None,
    ):
        """
        Load the image and landmarks from local file with given path

        Args:
        -------
            image_path_load (str): path to the x-ray image
            ct_path_load (str): path to the CT scan
            landmarks_2d_path (str): path to the 2d landmarks
            landmarks_3d_path (str): path to the 3d landmarks
            cam_param (dict[str, np.ndarray]): camera intrinsic and extrinsic parameters, and the image type

        Returns:
        -------
            XregSolver: XregSolver object
        """

        # load the image with specified type
        if cam_params["img_type"] == "DICOM":
            resized_img, image_load, scale = preprocess_dicom(image_path_load, 360)

        elif cam_params["img_type"] == "PNG":
            resized_img, image_load, scale = read_xray_png(image_path_load, 360)

        else:
            raise ValueError("Image type not supported")
        cam_params["scale"] = scale
        # print("Image loaded", image_load)
        landmarks_2d = cls.get_2d_landmarks(landmarks_2d_path)

        land_names = list(landmarks_2d.keys())
        land_vals = np.array([landmarks_2d[name] for name in land_names])
        land_vals[:, [0, -1]] = land_vals[:, [-1, 0]]

        keypoint_img = vis_utils.draw_keypoints(resized_img, land_vals, land_names)

        cv2.imwrite("keypoint_img.png", keypoint_img)

        if ct_segmentation_path is None:
            raise ValueError(
                "CT segmentation path is not provided, please refer to [total segmentator](https://github.com/wasserth/TotalSegmentator)"
            )
        return cls(
            image_load,
            landmarks_2d,
            ct_path_load,
            None,
            cam_params,
            {
                "landmark_3d_path": landmarks_3d_path,
                "ct_segmentation_path": ct_segmentation_path,
            },
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

        # print("iamge", self.image.shape)

        h5_file = h5py.File(self.path["xray_path"], "w")
        h5_file.create_dataset("num-projs", data=1, dtype="u8")
        h5_file.create_group("proj-000")

        with h5py.File(self.path["h5_path_template"], "r") as h5_template:
            for key in h5_template["proj-000"].keys():
                h5_file["proj-000"].create_group(key)

                if key != "landmarks":
                    # print(key)
                    for dataset in h5_template["proj-000"][key].keys():
                        # print(dataset)

                        if dataset == "pixels":
                            h5_file["proj-000"][key].create_dataset(
                                dataset,
                                data=self.image,
                                dtype=h5_template["proj-000"][key][dataset].dtype,
                            )
                        elif dataset == "intrinsic":
                            # intrinsic_param = (
                            #     cam_params["intrinsic"] / cam_params["scale"]
                            # )
                            # intrinsic_param[-1, -1] = 1
                            print("here is cam", self.cam_params)
                            h5_file["proj-000"][key].create_dataset(
                                dataset,
                                data=self.cam_params["intrinsic"],
                                dtype=h5_template["proj-000"][key][dataset].dtype,
                            )

                        elif dataset == "cam-coord-frame-type":
                            h5_file["proj-000"][key].create_dataset(
                                dataset,
                                data="origin-at-focal-pt-det-pos-z".encode("utf-8"),
                                dtype=h5_template["proj-000"][key][dataset].dtype,
                            )

                        else:
                            h5_file["proj-000"][key].create_dataset(
                                dataset,
                                data=h5_template["proj-000"][key][dataset][...],
                                dtype=h5_template["proj-000"][key][dataset].dtype,
                            )

                else:
                    pass  # skip the landmarks group and create it later
        h5_template.close()

        h5_file["proj-000"]["cam"]["num-cols"][...] = self.image.shape[1]
        h5_file["proj-000"]["cam"]["num-rows"][...] = self.image.shape[0]

        # write the 2d landmarks to the HDF5 file
        landmark = LandmarkContainer("2d", self.landmarks_2D)
        # print(landmark.landmark_name)
        landmark_2d = landmark.get_landmark()
        print(landmark_2d)
        print(self.landmarks_2D)

        # print(landmark_2d.keys())
        for lms in landmark_2d.keys():
            # print(lms)
            landmark_values = self.cam_param["scale"] * np.reshape(
                np.asarray([landmark_2d[lms][1], landmark_2d[lms][0]]), (2, 1)
            )
            print(lms)
            print(landmark_values)
            h5_file["proj-000"]["landmarks"].create_dataset(
                name=lms, data=landmark_values, dtype="f4"
            )

            # print(np.asarray(landmarks_2d.iloc[lm_idx].values))
            # h5_file['proj-000']['landmarks'][lms] = 0.0
        h5_file.flush()
        h5_file.close()

    def draw_landmarks(self, image: np.ndarray, landmarks: dict) -> np.ndarray:
        """
        Draw the landmarks on the image and save it

        Args:
        ------
            image: np.ndarray
                image to draw the landmarks on
            landmarks: dict
                dictionary of landmarks to draw on the image

        Returns:
        --------
            image: np.ndarray
                image with landmarks drawn on it
        """
        image = vis_utils.draw_keypoints(
            image,
            np.array([landmarks[name] for name in landmarks.keys()]),
            landmarks.keys(),
        )
        return image

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

        # calling the executable file for registration
        if runOptions == "run_reg":
            print("run_reg is running ...")

            result = subprocess.run(
                [
                    self.path["solver_path"],
                    self.path["ct_path"],
                    self.path["landmark_3d_path"],
                    self.path["xray_path"],
                    # "data/example1_1_pd_003.h5",
                    self.path["result_path"],
                    self.path["debug_path"],
                    "-s",  # option to use the segmentation to mask out the irrelevant part of the CT
                    self.path["ct_segmentation_path"],
                ],
                stdout=subprocess.PIPE,
            )

            # Print the output of the executable file
            print(result.stdout.decode())

            # extract the projection matrix from the resulting h5 file
            with h5py.File(self.path["result_path"], "r") as f:
                tp = f["TransformGroup/0/TransformParameters"]
                print("The projection matrix is: \n", tp[...])
                tfp = f["TransformGroup/0/TransformFixedParameters"]
                print(tfp[...])

            f.close()

            # return tp[...]

        # calling the executable file for visualization
        elif runOptions == "run_viz":
            print("run_viz is running ...")
            result = subprocess.run(
                [
                    self.path["verifier_path"],
                    self.path["debug_path"],
                    "--video-fps",
                    "10",
                    "--proj-ds",
                    "0.5",
                ],
                stdout=subprocess.PIPE,
            )
            print(result.stdout.decode())

        else:
            RuntimeError(
                "runOptions not supported \n runOptions: 'run_reg' or 'run_viz' "
            )

    def get_2d_landmarks(landmarks_path: str) -> dict:
        """Get 2D landmarks from the csv file
        Params:
        -------
            landmarks_2d_path (str): Path to the csv file

        Returns:
        --------
            landmarks_2d (dict[str, np.ndarray]): A dictionary of 2D landmarks
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
        ]  # this is the naming convention for the 2D landmarks in synthex generated csv file
        # land_name = [
        #     "FH-r",
        #     "FH-l",
        #     "GSN-r",
        #     "GSN-l",
        #     "IOF-r",
        #     "IOF-l",
        #     "MOF-r",
        #     "MOF-l",
        #     "SPS-r",
        #     "SPS-l",
        #     "IPS-r",
        #     "IPS-l",
        #     "ASIS-r",
        #     "ASIS-l",
        # ]

        landmarks_2d = {}
        data_frame = pd.read_csv(landmarks_path)
        print(landmarks_path)
        print(data_frame)
        data_frame = pd.DataFrame.drop(
            data_frame, columns=["pat", "proj", "time", "land"], axis=1
        )

        # add the column for the landmark names
        data_frame["land-name"] = land_name

        # remove the rows with -1 values
        mask = (data_frame == -1).any(axis=1)
        data_frame = data_frame.drop(index=data_frame[mask].index)
        data_frame = data_frame.reset_index(drop=True)

        # store the landmarks in a dictionary
        # print(data_frame)
        for i in range(len(data_frame)):
            landmarks_2d[data_frame["land-name"][i]] = [
                data_frame["row"][i],
                data_frame["col"][i],
            ]

        # print(landmarks_2d)
        return landmarks_2d


if __name__ == "__main__":
    pass
    # import os

    # folder_path = "/home/jeremy/Documents/xregi-dev"
    # mode = 0o777  # permission bits in octal

    # # iterate over all files in the folder
    # for file_name in os.listdir(folder_path):
    #     # construct the full file path
    #     file_path = os.path.join(folder_path, file_name)
    #     # change the permission bits of the file
    #     os.chmod(file_path, mode)

    # path = xreg_args()
    # cam_params = cam_param()
    # reg_solver = XregSolver.load(
    #     image_path_load=path["image_path_load"],
    #     ct_path_load=path["ct_path_load"],
    #     ct_segmentation_path=path["ct_segmentation_path"],
    #     landmarks_2d_path=path["landmarks_2d_path"],
    #     landmarks_3d_path=path["landmarks_3d_path"],
    #     cam_params=cam_params,
    # )

    # reg_solver.solve("run_reg")
    # reg_solver.solve("run_viz")

    # x = {}
    # x['sps_l'] = [1, 2]
    # x['sps_r'] = [2, 3]
    # x['gsn_l'] = [3, 4]
    # x['gsn_r'] = [4, 5]
    # print()

    # lm = LandmarkContainer.load('2d', list(
    #     x.values()), list(x.keys()))
