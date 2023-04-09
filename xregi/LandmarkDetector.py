import numpy as np
from utils import *
from abc import ABC, abstractmethod
from SyntheX.class_ensemble import ensemble

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

    def load_data(self):
        self.ensemble_seg.save_data()

    def est_lands
    


    @classmethod
    def load(clc,xray_folder_path,label_path,output_path,landmarks,pats):
        dicom2h5(xray_folder_path, label_path,output_path)
        landmarks = ['FH-l', 'FH-r', 'GSN-l', 'GSN-r', 'IOF-l', 'IOF-r', 'MOF-l',
                 'MOF-r', 'SPS-l', 'SPS-r', 'IPS-l', 'IPS-r', 'ASIS-l', 'ASIS-r']
        f = h5py.File((os.path.join(output_path, "synthex_input.h5"), "r"))
        image = f[pats]["projs"]


        return clc(None,None)