import numpy as np
from utils import *
from abc import ABC, abstractmethod


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

    def load_network(self, nn_path: str):
        self.nets = []
        pass

    def load_data(self):
        self.image
        pass

    def detect(self) -> np.ndarray:
        seg_dataset_ensemble(test_ds, self.nets, f, dev=dev,
                             num_lands=num_lands, times=times, adv_loss=False)

        landmarks_2D = self.read_landmarks_h5(f)

        return landmarks_2D
