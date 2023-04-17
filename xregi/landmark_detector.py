import numpy as np
from utils import *
from abc import ABC, abstractmethod
import SyntheX.class_ensemble as class_ensemble
from SyntheX.est_land_csv import est_land_csv
from typing import List, Dict, Optional
import argparse
from xregi.args import default_args


class LandmarkDetector(ABC):
    """
    Abstract class for landmark detection

    Args:
    -------
        image (np.ndarray): x-ray image in the shape of (# of image, height, width)

    Returns:
    -------
        create csv file with 2D coordinates of landmarks

    """

    def __init__(self, image: np.ndarray):
        self.image = image

    @property
    @abstractmethod
    def landmarks(self) -> List[str]:
        """
        Landmarks names are defined in the child class
        """
        pass

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, image):
        self._image = image

    @abstractmethod
    def load_data(self):
        """
        load data using specific method
        """
        pass

    @abstractmethod
    def detect(self) -> Dict[str, Optional[np.ndarray]]:
        """
        Detect landmarks in xray image, return 2D coordinates of landmarks

        Args:
            self.image: np.ndarray
            self.landmarks: dict[str, np.ndarray]

        Returns:
            landmarks_2D: np.ndarray
        """
        pass


class SynthexDetector(LandmarkDetector):
    """
    Synthex landmark detector

    Args:
    -------
        image(np.ndarray): x-ray image in the shape of (# of image, height, width)
        landmarks(dict[str, np.ndarray]): 3d landmarks in the shape of (landmark name, [x, y, z])

    Returns:
    -------
        create csv file with 2D coordinates of landmarks


    """

    def __init__(
        self,
        image: np.ndarray,
        landmarks: Dict[str, np.ndarray],
        args: Optional[argparse.Namespace],
    ):
        self.args = default_args() if args is None else args
        super().__init__(image)

    @property
    def landmarks(self) -> List[str]:
        self.landmarks = [
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
        return self.landmarks

    @property
    def image_path(self):
        return self._image_path

    @image_path.setter
    def image_path(self, image_path):
        print("image_path: ", image_path)
        dicom2h5(
            image_path, self.args.input_data_file_path, self.args.input_label_file_path
        )
        self._image_path = image_path
        return self._image_path

    def load_data(self):
        """
        load network and data
        Args:
        -------
            self.args: args from syn_args.py

        """
        self.output_data_file_path = self.args.output_data_file_path
        self.ensemble_seg = class_ensemble.Ensemble(self.args)
        self.ensemble_seg.loadnet()

    def savedata(self):
        """
        save data
        Args:
        -------
            self.args: args from syn_args.py

        """
        self.ensemble_seg.savedata(
            self.args.input_data_file_path, self.args.input_label_file_path
        )

    def detect(self):
        """
        detect landmarks
        Args:
        -------
            self.args: args from syn_args.py

        """
        est_land_csv(self.args)

    def run(self):
        """
        run landmark detection
        Args:
        -------
            self.args: args from syn_args.py

        """
        self.load_data()
        self.savedata()
        self.detect()

    @classmethod
    def load(cls, xray_folder_path, label_path, output_path, pats):
        """
        load data from x-ray image and label

        Args:
        -------
            xray_folder_path(str): path to x-ray image
            label_path(str): path to label
            output_path(str): path to output
            pats(str): patient id

        Returns:
        -------
            SynthexDetector: Synthex landmark detector

        """
        dicom2h5(xray_folder_path, label_path, output_path)
        f = h5py.File(os.path.join(output_path, "synthex_input.h5"), "r")
        image = f[pats]["projs"]
        args = default_args()

        return cls(image, None, args)


if __name__ == "__main__":
    # syn = SynthexDetector.load(r"data\xray", r"data\real_label.h5", "data", "01")
    # syn.load_data()
    # syn.savedata()
    # syn.detect()
    # def get_2d_landmarks(landmarks_path: str) -> dict:
    #     """Get 2D landmarks from the csv file
    #     Params:
    #     -------
    #     landmarks_2d_path: str
    #         Path to the csv file

    #     Returns:
    #     --------
    #     landmarks_2d: dict[str, np.ndarray]
    #         A dictionary of 2D landmarks
    #     """
    #     # This is the synthex format and order for landmarks
    #     land_name = [
    #         "FH-l",
    #         "FH-r",
    #         "GSN-l",
    #         "GSN-r",
    #         "IOF-l",
    #         "IOF-r",
    #         "MOF-l",
    #         "MOF-r",
    #         "SPS-l",
    #         "SPS-r",
    #         "IPS-l",
    #         "IPS-r",
    #         "ASIS-l",
    #         "ASIS-r",
    #     ]

    #     landmarks_2d = {}
    #     data_frame = pd.read_csv(landmarks_path)
    #     data_frame = pd.DataFrame.drop(
    #         data_frame, columns=["pat", "proj", "time", "land"], axis=1
    #     )

    #     data_frame["land-name"] = land_name
    #     print(data_frame["land-name"][0])

    #     for i in range(len(data_frame)):
    #         landmarks_2d[data_frame["land-name"][i]] = [
    #             data_frame["row"][i],
    #             data_frame["col"][i],
    #         ]

    #     return landmarks_2d

    # print(get_2d_landmarks(r"data\own_data.csv"))

    pass
