import numpy as np
from utils import *
from abc import ABC, abstractmethod
# from SyntheX.class_ensemble import ensemble
# from SyntheX.est_land_csv import est_land_csv
import SyntheX.class_ensemble as class_ensemble
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

    def load_data(self, args):
        self.current_path = os.path.abspath(os.path.dirname(__file__))
        args.nets = os.path.join(self.current_path, args.nets)
        args.output_data_file_path = os.path.join(
            self.current_path, args.output_data_file_path)
        self.ensemble_seg = class_ensemble.ensemble(args)
        self.nets = self.ensemble_seg.loadnet()

    def savedata(self, input_data_file_path, input_label_file_path):
        input_data_file_path = os.path.join(
            self.current_path, input_data_file_path)
        input_label_file_path = os.path.join(
            self.current_path, input_label_file_path)
        self.ensemble_seg.savedata(
            input_data_file_path, input_label_file_path)

    def detect(self):
        test_ds_path = self.ensemble_seg.dst_data_file_path
        subprocess.run([    "python",
                    "SyntheX/est_land_csv.py",
                    test_ds_path, # input_data_file_path
                    "nn-heats",
                    "--use-seg","nn-segs",
                    "--pat", "1",  # patient ID
                    "--out", "data/own_data.csv"]) # output_data_file_path
        
        


    @classmethod
    def load(clc,xray_folder_path,label_path,output_path,pats):
        dicom2h5(xray_folder_path, label_path,output_path)

        output_path = os.path.join(os.path.abspath(
            os.path.dirname(__file__)), output_path)
        f = h5py.File(os.path.join(output_path, "synthex_input.h5"), "r")
        image = f[pats]["projs"]


        return clc(image,None)
    

if __name__ == "__main__":
    syn = SynthexDetector.load(
        "data/xray", "data/real_label.h5", "data", "01")
    args = argparse.Namespace()

    args.nets = "data/yy_checkpoint_net_20.pt"

    args.input_data_file_path = "data/synthex_input.h5"
    args.input_label_file_path = "data/synthex_label_input.h5"
    args.output_data_file_path = "data/output.csv"

    args.rand = True
    args.pats = "01"
    args.no_gpu = True
    args.times = ''

    syn.load_data(args)
    syn.savedata(args.input_data_file_path, args.input_label_file_path)
    syn.detect()

    # args.heat_file_path = "data/"
    # args.heats_group_path = "nn-heats"

    # args.out = "data/own_data.csv"

    # args.rand = False

    # args.hm_lvl = 0

    # args.ds_factor = 1

    # args.pat = "1"

    # no_csv_hdr = args.no_hdr

    # seg_ds_path = args.use_seg

    # threshold = args.threshold
    # syn.load_network()
