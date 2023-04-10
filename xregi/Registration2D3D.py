import numpy as np
from utils import *
import pandas as pd
from LandmarkDetector import SynthexDetector
import LandmarkDetector
import RegistrationSolver
import argparse

class Registration2D3D:
    def __init__(self, image: np.ndarray, ct_path: str, landmarks_3d: dict, intrinsic: np.ndarray):
        self.image = image
        self.ct_path = ct_path
        self.landmarks_3d = landmarks_3d
        self.intrinsic = intrinsic
    
    def select_detector(self,detector:str, path: dict,args):
        self.path = path
        if detector == 'synthex':
            self.syn = SynthexDetector.load(path['image'],path['label'],path['output'],path["pats"])
            self.syn.load_data(args)

        elif detector == 'else':
            pass # TODO
        


    def run_synthex(self,image_path):
        pass


    def run_xreg(self):
        pass

    @classmethod
    def load(cls, image_path_load: str, ct_path_load: str, landmarks_3d_path: str, intrinsic_load: np.ndarray):
        image_load = read_xray_dicom(image_path_load)
        landmarks_3d = get_3d_landmarks(
            landmarks_3d_path, folder_type="fcsv", label_idx=11)
        # intrinsic load from dicom?
        return cls(image_load, ct_path_load, landmarks_3d, intrinsic_load)
    
if __name__ == "__main__":
    reg = Registration2D3D.load()
    parser = argparse.ArgumentParser(description='Run ensemble segmentation and heatmap estimation for hip imaging application.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_data_file_path', help='Path to the datafile containing projections', type=str)
    parser.add_argument('input_label_file_path', help='Path to the datafile containing groundtruth segmentations and landmarks', type=str)
    parser.add_argument('output_data_file_path', help='Path to the output datafile containing segmentations', type=str)
    parser.add_argument('--nets', help='Paths to the networks used to perform segmentation - specify this after the positional arguments', type=str, nargs='+')
    parser.add_argument('--pats', help='comma delimited list of patient IDs used for testing', type=str)
    parser.add_argument('--no-gpu', help='Only use CPU - do not use GPU even if it is available', action='store_true')
    parser.add_argument('--times', help='Path to file storing runtimes for each image', type=str, default='')
    parser.add_argument('--rand', help='Run test on rand data', action='store_true')
    args = parser.parse_args()
    path ={}
    reg.select_detector('synthex',path,args)