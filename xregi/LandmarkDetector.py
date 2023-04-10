import numpy as np
from utils import *
from abc import ABC, abstractmethod
# from SyntheX.class_ensemble import ensemble
# from SyntheX.est_land_csv import est_land_csv
import SyntheX
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

    def load_network(self, args):
        self.ensemble_seg = SyntheX.class_esemble.ensemble(args)
        self.nets = self.ensemble_seg.load_nets()

    def savedata(self, input_data_file_path, input_label_file_path):  # save test_ds
        self.ensemble_seg.save_data(
            input_data_file_path, input_label_file_path)

    def est_lands(self):
        test_ds_path = self.ensemble_seg.dst_data_file_path
        subprocess.run(["python",
                        "SyntheX/est_land_csv.py",
                        test_ds_path,  # input_data_file_path
                        "nn-heats",
                        "--use-seg", "nn-segs",
                        "--pat", "1",  # patient ID
                        "--out", "data/own_data.csv"])  # output_data_file_path

    @classmethod
    def load(clc, xray_folder_path, label_path, output_path, pats):

        dicom2h5(xray_folder_path, label_path, output_path)

        f = h5py.File((os.path.join(output_path, "synthex_input.h5"), "r"))
        image = f[pats]["projs"]

        return clc(image, None)


if __name__ == "__main__":
    syn = SynthexDetector.load("data/xray", "data/real_label.h5", "data", "1")
    args = argparse.Namespace()

    args.nets = "data/yy_checkpoint_net_20.pt"

    args.input_data_file_path = "data/synthex_input.h5"
    args.input_label_file_path = "data/synthex_label_input.h5"
    args.output_data_file_path = "data/output.csv"

    args.rand = True
    args.pats = "1"
    args.no_gpu = True
    args.times = ''

    syn.load_network(args)

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
    syn.load_network()
