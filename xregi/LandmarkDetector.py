import numpy as np
from utils import *
from abc import ABC, abstractmethod
from SyntheX.class_ensemble import ensemble
from SyntheX.est_land_csv import est_land_csv
import argparse

class LandmarkDetector(ABC):
    '''
    Abstract class for landmark detection
    '''

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def detect(self) -> np.ndarray:
        '''
        Detect landmarks in xray image, return 2D coordinates of landmarks

        Args:
        -------
        self.image: np.ndarray
        self.landmarks: dict[str, np.ndarray]

        Returns:
        -------
        landmarks_2D: np.ndarray
        '''
        pass


class SynthexDetector(LandmarkDetector):
    def __init__(self, image: np.ndarray, landmarks: dict):
        self.image = image
        self.landmarks = landmarks

    def load_network(self,args):
        self.ensemble_seg = ensemble(args)
        self.nets = self.ensemble_seg.load_nets()

    def load_data(self): # save test_ds
        self.ensemble_seg.save_data()

    def est_lands(self):
        test_ds_path = self.ensemble_seg.dst_data_file_path
        est_land_csv(test_ds_path)
        


    @classmethod
    def load(clc,xray_folder_path,label_path,output_path,pats):
        dicom2h5(xray_folder_path, label_path,output_path)

        f = h5py.File((os.path.join(output_path, "synthex_input.h5"), "r"))
        image = f[pats]["projs"]


        return clc(None,None)