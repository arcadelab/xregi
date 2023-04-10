import numpy as np
import pandas as pd
import subprocess
from utils import *
from LandmarkContainer import LandmarkContainer
from abc import ABC, abstractmethod


class RegistrationSolver(ABC):
    
    @abstractmethod
    def solve(self) -> np.ndarray:
        '''
        Solve registration problem, return 3D coordinates of landmarks

        Args:
        -------
        self.image: np.ndarray
        self.landmarks_2D: dict[str, np.ndarray]
        self.landmarks_3D: dict[str, np.ndarray]

        Returns:
        -------
        landmarks_3D: np.ndarray
        '''
        pass


class XregSlover(RegistrationSolver):
    '''
    Solve 2d 3d registration problem using xreg
    '''
    def __init__(self, image: np.ndarray, ct_path:str, landmarks_2D: dict, landmarks_3D: dict):
        self.image = image
        self.CT = ct_path # no need to load CT in the file, just use the path of it
        self.landmark = LandmarkContainer.load('2d', list(landmarks_2D.values()), list(landmarks_2D.keys()))
        self.landmarks_2D = landmarks_2D
        self.landmarks_3D = landmarks_3D

        current_path = os.path.abspath(os.path.dirname(__file__))
        self.temp_file_path = os.path.join(current_path,"data/xreg_input.h5")

    @classmethod
    def load(clc, image_path_load: str, ct_path_load: str, landmarks_2d_path: str, landmarks_3d_path: str):
        image_load = read_xray_dicom(image_path_load)
        # landmarks_3d = get_3d_landmarks(
        #     landmarks_3d_path, folder_type="fcsv", label_idx=11)

        landmarks_2d = clc.get_2d_landmarks(landmarks_2d_path)

        return clc(image_load, ct_path_load, landmarks_2d, None)

    def generate_h5(self):
        '''
        Generate the h5 file for xreg
        the h5 file contains x-ray image and 2d landmarks
        '''

        h5_file = h5py.File(self.temp_file_path, "w")
        h5_file.create_dataset('num_projs', data=1, dtype='u8')
        h5_file.create_group("proj-000")
        
        with h5py.File("data/example1_1_pd_003.h5", "r") as h5_template:
            for key in h5_template['proj-000'].keys():
                # print(h5_template['proj-000'][key].values())
                h5_file['proj-000'].create_group(key)
                for dataset in h5_template['proj-000'][key].keys():
                    # print(dataset)

                    if dataset == 'pixels':
                        h5_file['proj-000'][key].create_dataset(
                            dataset, data=self.image, dtype=h5_template['proj-000'][key][dataset].dtype)
                    else:
                        h5_file['proj-000'][key].create_dataset(dataset, data=h5_template['proj-000']
                                                                [key][dataset][...], dtype=h5_template['proj-000'][key][dataset].dtype)

        h5_file['proj-000']['cam']['num-cols'][...] = self.image.shape[1]
        h5_file['proj-000']['cam']['num-rows'][...] = self.image.shape[0]

        h5_template.close()

        # write the 2d landmarks to the HDF5 file
        lm_names_synthex = ['FH-l', 'FH-r', 'GSN-l', 'GSN-r', 'IOF-l', 'IOF-r', 'MOF-l', 'MOF-r', 'SPS-l', 'SPS-r',
                            'IPS-l', 'IPS-r', 'ASIS-l', 'ASIS-r']  # this is the order of the landmarks in the SyntheX dataset

        for lms in h5_file['proj-000']['landmarks'].keys():
            landmark_2d = self.landmark.get_landmark(mode='xreg')    

            h5_file['proj-000']['landmarks'][lms][...] = np.reshape(
                np.asarray(landmark_2d[lms],(2,1)) )
            # print(np.asarray(landmarks_2d.iloc[lm_idx].values))
            # h5_file['proj-000']['landmarks'][lms] = 0.0




    def solve(self, runOptions) -> np.ndarray:
        '''Call the executable file
        Args:
        -------
        runOptions: str
            'run_reg' or 'run_viz' , 
            'run_reg' is used to run the registration
            'run_viz' is used to visualize the registration result

        Returns:
        --------
            None 

        '''

        if runOptions == 'run_reg':
            print("run_reg is running ...")

            result = subprocess.run(["bin/xreg-hip-surg-pelvis-single-view-regi-2d-3d",
                                    "data/pelvis.nii.gz",
                                     "data/pelvis_regi_2d_3d_lands_wo_id.fcsv",
                                     "data/example1_1_pd_003.h5",
                                     "result/regi_pose_example1_1_pd_003_proj0.h5",
                                     "result/regi_debug_example1_1_pd_003_proj0_w_seg.h5",
                                     "-s",
                                     "data/pelvis_seg.nii.gz"], stdout=subprocess.PIPE)

            # Print the output of the executable file
            print(result.stdout.decode())

        elif runOptions == 'run_viz':
            result = subprocess.run(["bin/xreg-regi2d3d-replay",
                                    "result/regi_debug_example1_1_pd_003_proj0_w_seg.h5",
                                     "--video-fps",
                                     "10",
                                     "--proj-ds",
                                     "0.5"], stdout=subprocess.PIPE)
            print(result.stdout.decode())

    def get_2d_landmarks(landmarks_path: str) -> dict:
        '''Get 2D landmarks from the csv file
        Params:
        -------
        landmarks_2d_path: str
            Path to the csv file

        Returns:
        --------
        landmarks_2d: dict[str, np.ndarray]
            A dictionary of 2D landmarks
        '''
        # This is the synthex format and order for landmarks
        land_name = ['FH-l', 'FH-r', 'GSN-l', 'GSN-r', 'IOF-l', 'IOF-r', 'MOF-l', 'MOF-r', 'SPS-l', 'SPS-r', 'IPS-l', 'IPS-r', 'ASIS-l', 'ASIS-r']

        landmarks_2d = {}
        data_frame = pd.read_csv(landmarks_path)
        data_frame = pd.DataFrame.drop(data_frame,
            columns=['pat', 'proj', 'time','land'], axis=1)

        data_frame['land-name'] = land_name
        print(data_frame['land-name'][0])

        for i in range(len(data_frame)):
            landmarks_2d[data_frame['land-name'][i]] = [data_frame['row'][i],data_frame['col'][i]]

        return landmarks_2d
    



if __name__ == '__main__':
    # image = np.ones((3,3))

    # lm_2d = {}
    # lm_3d = {}

    # lm_2d['sps_l'] = [1, 2]
    # lm_2d['sps_r'] = [2, 3]
    # lm_3d['gsn_l'] = [3, 4]
    # lm_3d['gsn_r'] = [4, 5]


    # xreg = XregSlover(image,lm_2d,lm_3d)
    # x = xreg.get_2d_landmarks("data/own_data.csv")
    # print(x.values())

    reg_solver = XregSlover.load(image_path_load="data/x_ray1.dcm", ct_path_load="data/pelvis.nii.gz", landmarks_2d_path="data/own_data.csv",landmarks_3d_path=None)
    
    
    # x = {}
    # x['sps_l'] = [1, 2]
    # x['sps_r'] = [2, 3]
    # x['gsn_l'] = [3, 4]
    # x['gsn_r'] = [4, 5]
    # print()

    # lm = LandmarkContainer.load('2d', list(
    #     x.values()), list(x.keys()))
    