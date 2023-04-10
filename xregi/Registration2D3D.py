import numpy as np
from utils import *
import pandas as pd
from LandmarkDetector import SynthexDetector
from RegistrationSolver import XregSlover
import argparse


class Registration2D3D:
    def __init__(self, image: np.ndarray, ct_path: str, landmarks_3d: dict, intrinsic: np.ndarray):
        self.image = image
        self.ct_path = ct_path
        self.landmarks_3d = landmarks_3d
        self.intrinsic = intrinsic

    def select_detector(self, detector: str, path: dict, args):
        '''
        Select the detector type and instantiate it


        '''
        self.path = path
        if detector == 'SyntheX':
            self.syn = SynthexDetector.load(
                path['image'], path['label'], path['output'], path["pats"])
            self.syn.load_data(args)
            self.syn.savedata(args.input_data_file_path,
                              args.input_label_file_path)

        elif detector == 'else':
            pass  # TODO

    def run_synthex(self, args2):
        self.syn.detect(args2)

    def run_xreg(self):
        self.xreg = XregSlover()
        self.xreg.solve(runOptions='run_reg')

    @classmethod
    def load(cls, image_path_load: str, ct_path_load: str, landmarks_3d_path: str, intrinsic_load: np.ndarray):
        image_load = read_xray_dicom(image_path_load)
        landmarks_3d = get_3d_landmarks(
            landmarks_3d_path, folder_type="fcsv", label_idx=11)
        # intrinsic load from dicom?
        return cls(image_load, ct_path_load, landmarks_3d, intrinsic_load)


if __name__ == "__main__":
    image_path_load = ""
    ct_path_load = ""
    landmarks_3d_path = ""
    intrinsic_load = ""
    reg = Registration2D3D.load(
        image_path_load, ct_path_load, landmarks_3d_path, intrinsic_load)
    # select detector
    path = {
        "image": "data/xray",
        "label": "data/real_label.h5",
        "output": "data",
        "pats": "01"
    }
    args = argparse.Namespace()
    args.nets = "data/yy_checkpoint_net_20.pt"
    args.input_data_file_path = "data/synthex_input.h5"
    args.input_label_file_path = "data/synthex_label_input.h5"
    args.output_data_file_path = "data/output.h5"
    args.rand = True
    args.pats = "01"
    args.no_gpu = True
    args.times = ''
    reg.select_detetctor('SyntheX', path, args)
    # run SyntheX
    args2 = argparse.Namespace()
    args2.heat_file_path = reg.syn.output_data_file_path
    args2.heats_group_path = "nn-heats"
    args2.out = "data/own_data.csv"
    args2.out = os.path.join(reg.syn.current_path, args2.out)
    args2.pat = "01"
    args2.use_seg = "nn-segs"
    args2.rand = True
    args2.hm_lvl = True
    args2.ds_factor = 4
    args2.no_hdr = True
    args2.use_seg = ''
    args2.threshold = 2
    reg.run_synthex(args2)
