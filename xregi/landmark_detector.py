import numpy as np
from utils import *
from abc import ABC, abstractmethod

import SyntheX.class_ensemble as class_ensemble
from SyntheX.est_land_csv import est_land_csv
from typing import List, Dict, Optional
import argparse
from syn_args import default_args


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

    @abstractmethod
    @property
    def landmarks(self) -> List[str]:
        """
        Landmarks names are defined in the child class
        """
        pass

    @abstractmethod
    def load_data(self):
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

    def __init__(self, image: np.ndarray, landmarks: Dict[str, np.ndarray]):
        super.__init__(image)
        self.args = default_args()

    def reload_image(self, image_folder_path, label_path, output_path):
        """
        reload image

        Args:
        -------
            image_folder_path(str): path to x-ray image
            label_path(str): path to label
            output_path(str): path to output


        """
        dicom2h5(image_folder_path, label_path, output_path)

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
        self.ensemble_seg.savedata(self.args.input_data_file_path, self.args.input_label_file_path)

    def detect(self):
        """
        detect landmarks
        Args:
        -------
            self.args: args from syn_args.py
            
        """
        est_land_csv(self.args)

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
        output_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), output_path
        )
        f = h5py.File(os.path.join(output_path, "synthex_input.h5"), "r")
        image = f[pats]["projs"]

        return cls(image, None)


if __name__ == "__main__":
    syn = SynthexDetector.load("data/xray", "data/real_label.h5", "data", "01")
    syn.load_data()
    syn.savedata()
    syn.detect()
