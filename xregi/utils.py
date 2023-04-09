import numpy as np
import h5py
import subprocess
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2


class LandmarkContainer():
    # this class is used to store the landmarks
    # the landmarks can be 2d or 3d

    def __init__(self, landmark: dict, name_format: str, type: str):
        self.name = self.regulate_landmark_label(landmark.keys(), name_format)

    @classmethod
    def load(cls, landmark_type: str, landmark_value: list, landmark_label: list, name_format: str):
        '''
        load the landmarks from a file with specified suffix
        the landmarks can be 2d or 3d

        Args:
        ------
        landmark_type: str, the type of the landmarks, e.g. '2d', '3d'
        landmark_value: list, the value of the landmarks
        landmark_label: list, the label of the landmarks

        Returns:
        --------
        class instance

        '''
        landmark = {}
        for i in range(landmark_label):
            landmark[landmark_label[i]] = landmark_value[i]

        if landmark_type == '2d':
            pass
        elif landmark_type == '3d':
            pass
        else:
            raise ValueError(
                "The type of the landmarks should be '2d' or '3d'")

        return cls(landmark, name_format, landmark_type)

    def regulate_landmark_label(name: list, name_format: str) -> list:
        '''
        rename the label name of the landmarks based on the source label template and the target label template

        Args:
        ------
        name: list, the name of the landmarks with certain format
        name_format: str, the format of the name, e.g. 'r_sps', 'l_sps', l stands for left, r stands for right, sps stands for sacroiliac point

        Returns:
        --------
        name: list
        '''
        if name_format[1] == '_':  # e.g. 'r_sps'
            anatomy_name = name.split("_")
            target_label_name = ''.join(
                anatomy_name[1:-2:-1]).upper() + '-' + ''.join(anatomy_name[0])
            print(target_label_name)

        elif name_format[1] == '-':  # e.g. 'r-sps'
            # anatomy_name = src_label_name.split("-")
            pass  # TODO

        elif name_format[-2] == '_':  # e.g. 'sps_r'
            pass  # TODO

        return target_label_name

    def get_value(mode: str) -> dict:
        '''
        get the value of the landmarks for a certain mode

        Args:
        ------
        mode: str,  the order and the format of the landmarks,
                    e.g. 'synthex' stands for the synthex ways of labeling the landmarks


        Returns:
        --------
        landmarks: dict,    a dictionary with keys: 'landmarks_name', 'landmarks_values'
        '''
        if mode == 'synthex':
            pass

        elif mode == 'xreg':
            pass

        elif mode == 'other':
            pass

        else:
            print('The mode is not supported yet')
        pass

    pass


def get_3d_landmarks(source_file_path: str, source_file_type: str, label_idx: int = 11) -> dict:
    '''
    get 3d landmarks from a file with specified suffix
    and return a numpy array

    Params:
    -------
    source_file_path: str
        path to the file
    source_file_type: str
        suffix of the file, could be 'fcsv', 'txt', 'csv'
    label_idx: int
        the index of the tag "label" in the fcsv file, 
        this can be found on the third line of the fcsv file
        it corresponds to the index of the landmarks name in the fcsv file

    Returns:
    --------
    a dictionary with keys: 'landmarks_name', 'landmarks_info'
    '''

    if source_file_type == 'fcsv':
        # read the fcsv file
        landmarks = {}  # a dictionary to store all the information of the landmarks
        header = []  # a list to store the header of the fcsv file
        with open(source_file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if line[0] == '#':
                header.append(line)
            else:
                # get the landmarks name
                landmarks_name = line.split(',')[label_idx]
                # get the landmarks info
                landmarks_param = line.split(',')[1:11]
                # landmarks_param = np.asarray(landmarks_param, dtype=np.float32)

                landmarks[landmarks_name] = landmarks_param
        # get the landmarks name
        landmarks['header'] = header

    elif source_file_type == 'txt':
        pass  # TODO
    elif source_file_type == 'csv':
        pass  # TODO

    return landmarks


def write_3d_landmarks_xreg(output_file_path: str, landmark_info: dict):
    '''
    write the 3d landmarks to a file with specified suffix
    '''
    output_fcsv_header = ''
    output_fcsv_value = ''

    for key in landmark_info.keys():
        if key == 'header':
            for header in landmark_info['header']:
                output_fcsv_header += header

        else:
            # if the landmark's name is in lower case, convert it to upper case
            label = regulate_landmark_label(key)

            # put the landmarks info into a string
            # ','.join() is used to convert a list to a string with ',' as the separator
            output_fcsv_value += ',' + \
                ','.join(landmark_info[key]) + ',' + label + ', , \n'

    with open(output_file_path, 'w') as f:
        f.write(output_fcsv_header)
        f.write(output_fcsv_value)

    f.close()


def regulate_landmark_label(src_label_name: str, src_label_template: str = 'r_sps', target_label_template: str = 'SPS-r') -> str:
    '''
    rename the label name of the landmarks based on the source label template and the target label template

    Params:
    -------
    src_label_name: str
        the name of the source label
    src_label_template: str
        the template of the source label, e.g. 'r_sps', 'l_sps', l stands for left, r stands for right, sps stands for sacroiliac point
    target_label_template: str
        the template of the target label, e.g. 'SPS-r', 'SPS-l'

    Returns:
    --------
    target_label_name: str
        the name of the target label after regulation
    '''
    if src_label_template[1] == '_':
        anatomy_name = src_label_name.split("_")
        target_label_name = ''.join(
            anatomy_name[1:-2:-1]).upper() + '-' + ''.join(anatomy_name[0])
        print(target_label_name)

    elif src_label_template[1] == '-':
        # anatomy_name = src_label_name.split("-")
        pass  # TODO

    return target_label_name


def run_xreg(runOptions: str):
    '''Call the executable file
    Params:
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


def readh5(h5_path: str):
    '''Read the h5 filex`
    Params:
    -------
    h5_path: str
        path to the h5 file

    Returns:
    --------
        None 

    '''
    with h5py.File(h5_path, 'r') as f:
        # List all groups
        print("Keys: %s" % f.keys())

        key_list = list(f.keys())

        print(key_list[1])
        sub_key = f['proj-000'].keys()
        print(sub_key)

        data = f['proj-000/landmarks/GSN-l']
        # data = f['num-projs']
        # data = [0,1]
        print(data[...])

        image = np.array(data)

    # plt.imshow(image,'gray')
    # plt.show()

    f.close()

    return image


def read_xray_dicom(path, to_32_bit=False, voi_lut=True, fix_monochrome=True):
    '''
    Read the dicom file and return the pixel_array as a numpy array

    Params:
    -------
    path: str
        path to the dicom file
        voi_lut: bool
        fix_monochrome: bool, if True, invert the image if PhotometricInterpretation is MONOCHROME1

    Returns:
    --------
    image: numpy array
        the pixel_array of the x-ray image
    '''
    dicom = pydicom.read_file(path)

    if voi_lut:
        image = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        image = dicom.pixel_array

    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        image = np.amax(image) - image

    if to_32_bit:
        image = image.astype(np.float32)

    return image


def read_2d_landmarks(landmarks_dir: str) -> pd.DataFrame:
    '''
    Read the 2D landmarks from the csv file

    Params:
    -------
    landmarks_dir: str
        path to the directory of the landmarks

    Returns:
    --------
    landmarks: DataFrame
        the 2D landmarks
    '''
    landmarks = pd.read_csv(landmarks_dir)
    landmarks = landmarks.drop(columns=['pat', 'proj', 'time'], axis=1)

    return landmarks


def generate_xreg_input(xray_dir: str, landmarks_dir: str, output_dir: str):
    '''
    Generate the input files for xreg

    Params:
    -------
    xray_dir: str
        path to the directory of the x-ray images
    landmarks_dir: str
        path to the directory of the landmarks
    output_dir: str
        path to the directory of the output HDF5 file

    Returns:
    --------
        None 

    '''
    # read the x-ray image
    xray_image = read_xray_dicom(xray_dir)

    # read the 2d landmarks
    landmarks_2d = read_2d_landmarks(landmarks_dir)

    # create the HDF5 file that contains the x-ray image and the 2d landmarks
    h5_file = h5py.File(output_dir, "w")
    h5_file.create_dataset('num_projs', data=1, dtype='u8')
    h5_file.create_group("proj-000")

    # write the x-ray image and to the HDF5 file
    with h5py.File("data/example1_1_pd_003.h5", "r") as h5_template:
        for key in h5_template['proj-000'].keys():
            # print(h5_template['proj-000'][key].values())
            h5_file['proj-000'].create_group(key)
            for dataset in h5_template['proj-000'][key].keys():
                # print(dataset)

                if dataset == 'pixels':
                    h5_file['proj-000'][key].create_dataset(
                        dataset, data=xray_image, dtype=h5_template['proj-000'][key][dataset].dtype)
                else:
                    h5_file['proj-000'][key].create_dataset(dataset, data=h5_template['proj-000']
                                                            [key][dataset][...], dtype=h5_template['proj-000'][key][dataset].dtype)

    h5_file['proj-000']['cam']['num-cols'][...] = xray_image.shape[1]
    h5_file['proj-000']['cam']['num-rows'][...] = xray_image.shape[0]

    h5_template.close()

    # write the 2d landmarks to the HDF5 file
    lm_names_synthex = ['FH-l', 'FH-r', 'GSN-l', 'GSN-r', 'IOF-l', 'IOF-r', 'MOF-l', 'MOF-r', 'SPS-l', 'SPS-r',
                        'IPS-l', 'IPS-r', 'ASIS-l', 'ASIS-r']  # this is the order of the landmarks in the SyntheX dataset

    for lms in h5_file['proj-000']['landmarks'].keys():
        lm_idx = lm_names_synthex.index(lms)

        h5_file['proj-000']['landmarks'][lms][...] = np.reshape(
            np.asarray(landmarks_2d.iloc[lm_idx].values)[1:], (2, 1))
        # print(np.asarray(landmarks_2d.iloc[lm_idx].values))
        # h5_file['proj-000']['landmarks'][lms] = 0.0


def read_ct_dicom(ct_path: str):
    pass

<<<<<<< HEAD
def dicom2h5(xray_folder_path:str, label_path:str,output_path:str):
=======

def dicom2h5(xray_path: str, h5_path: str, label_path: str):
    def read_xray(path, voi_lut=True, fix_monochrome=True):
        dicom = pydicom.read_file(path)
>>>>>>> 7b4b887d1df075bfba6abaa6ae2d3e39a7d39ebb


    # folder_path = "dicom_image"
    folder_path = xray_folder_path

    file_names = [f for f in os.listdir(
        folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    num_images = len(file_names)



    # Create an HDF5 file
    h5_file = h5py.File(os.path.join(output_path, "synthex_input.h5"), "w")
    h5_reallabel = h5py.File(os.path.join(output_path, "synthex_label_input.h5"), "w")

    # create group for synthex_input.h5
    grp = h5_file.create_group("01")
    # create group for synthex_label_input.h5
    label_grp = h5_reallabel.create_group("01")

    # create landnames
    names = h5_file.create_group("land-names")
    label_names = h5_reallabel.create_group("land-names")
    keys = [f'land-{i:02d}' for i in range(14)] + ['num-lands']
    landmarks = ['FH-l', 'FH-r', 'GSN-l', 'GSN-r', 'IOF-l', 'IOF-r', 'MOF-l',
                 'MOF-r', 'SPS-l', 'SPS-r', 'IPS-l', 'IPS-r', 'ASIS-l', 'ASIS-r', "14"]
    for i, key in enumerate(keys):
        if i < len(landmarks):
            dtype_str = h5py.special_dtype(vlen=str)
            dataset_names = names.create_dataset(keys[i], (), dtype=dtype_str)
            dataset_names[()] = landmarks[i].encode('utf-8')
            label_dataset_names = label_names.create_dataset(
                keys[i], (), dtype=dtype_str)
            label_dataset_names[()] = landmarks[i].encode('utf-8')

    # Create the dataset with the appropriate shape
    dataset_shape = (num_images, 360, 360)
    dataset = grp.create_dataset("projs", dataset_shape, dtype='f4')
    # Store all images in the dataset
    for i, file_name in enumerate(file_names):
        file_path = os.path.join(folder_path, file_name)
        image_data = read_xray_dicom(file_path)
        resized_image_data = cv2.resize(image_data, (360, 360), interpolation=cv2.INTER_LINEAR)  # Add this line
        dataset[i, :, :] = resized_image_data

    # currently unkown of camera paras, now just copy content from label_real.h5
    real_label = h5py.File(label_path, "r")
    # proj-paras part
    label_proj_paras = h5_reallabel.create_group("proj-params")
    label_proj_paras = real_label["proj-params"]  # copy group
    # gt-poses part
    label_grp_gtpose = label_grp.create_group("gt-poses")
    for i, image_file in enumerate(file_names):
        group_name = f"{i:03}"
        label_grp_gtpose_content = real_label["01"]["gt-poses"][group_name]
        gtpose_dataset = label_grp_gtpose.create_dataset(
            group_name, (4, 4), dtype='f4')
        gtpose_dataset[()] = label_grp_gtpose_content
    # lands part
    label_grp_lands = label_grp.create_dataset(
        "lands", (num_images, 2, 14), dtype='f4')
    label_grp_lands[()] = real_label["01"]["lands"][0:num_images]
    # segs part
    label_grp_segs = label_grp.create_dataset(
        "segs", (num_images, 360, 360), dtype="|u1")
    label_grp_segs[()] = real_label["01"]["segs"][0:num_images]

    # Close the HDF5 file to save changes
    h5_file.close()
    real_label.close()
    h5_reallabel.close()

if __name__ == '__main__':
    # source_file_path = 'data/case-100114/landmarks.fcsv'
    # source_file_type = 'fcsv'
    # lm_3d = get_3d_landmarks(source_file_path, source_file_type)
    # # print(lm_3d)
    # write_3d_landmarks_xreg('data/test.fcsv', lm_3d)

    # data = readh5('data/example1_1_pd_003.h5')
    # image = read_xray_dicom('data/x_ray1.dcm',to_32_bit=True)
    # print(image)
    # plt.imshow(image,'gray')
    # plt.show()

    # read_2d_landmarks('data/own_data.csv')
    # lm_names_synthex = ['FH-l', 'FH-r', 'GSN-l', 'GSN-r', 'IOF-l', 'IOF-r', 'MOF-l', 'MOF-r', 'SPS-l', 'SPS-r', 'IPS-l', 'IPS-r', 'ASIS-l', 'ASIS-r'] # this is the order of the landmarks in the SyntheX dataset
    # print(lm_names_synthex.index('GSN-l'))

    # generate_xreg_input('data/x_ray1.dcm', 'data/own_data.csv', 'data/test.h5')

    x = {}
    x['a'] = 1
    x['b'] = 2
    x['c'] = 3
    y = list(x.keys())  # convert dict_keys object to list
    # print the first element of the dict_keys object
    print(y[-2])
