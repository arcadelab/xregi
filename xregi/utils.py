import numpy as np
import h5py
import subprocess
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
import warnings


def newestfile(directory):
    """
    get the newest file in the folder
    """
    # get a list of all files in the directory
    files = os.listdir(directory)

    # sort the files by modification time (newest first)
    files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)

    # get the name of the newest file
    newest_file = files[0]

    # find absolute path
    newest_file = os.path.join(directory, newest_file)
    return newest_file


def get_3d_landmarks(
    source_file_path: str, source_file_type: str, label_idx: int = 11
) -> dict:
    """
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
    """

    if source_file_type == "fcsv":
        # read the fcsv file
        landmarks = {}  # a dictionary to store all the information of the landmarks
        header = []  # a list to store the header of the fcsv file
        with open(source_file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            if line[0] == "#":
                header.append(line)
            else:
                # get the landmarks name
                landmarks_name = line.split(",")[label_idx]
                # get the landmarks info
                landmarks_param = line.split(",")[1:11]
                # landmarks_param = np.asarray(landmarks_param, dtype=np.float32)

                landmarks[landmarks_name] = landmarks_param
        # get the landmarks name
        landmarks["header"] = header

    elif source_file_type == "txt":
        pass  # TODO
    elif source_file_type == "csv":
        pass  # TODO

    return landmarks


def write_3d_landmarks_xreg(output_file_path: str, landmark_info: dict):
    """
    write the 3d landmarks to a file with specified suffix
    """
    output_fcsv_header = ""
    output_fcsv_value = ""

    for key in landmark_info.keys():
        if key == "header":
            for header in landmark_info["header"]:
                output_fcsv_header += header

        else:
            # if the landmark's name is in lower case, convert it to upper case
            label = regulate_landmark_label(key)

            # put the landmarks info into a string
            # ','.join() is used to convert a list to a string with ',' as the separator
            output_fcsv_value += (
                "," + ",".join(landmark_info[key]) + "," + label + ", , \n"
            )

    with open(output_file_path, "w") as f:
        f.write(output_fcsv_header)
        f.write(output_fcsv_value)

    f.close()


def regulate_landmark_label(
    src_label_name: str,
    src_label_template: str = "r_sps",
    target_label_template: str = "SPS-r",
) -> str:
    """
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
    """
    if src_label_template[1] == "_":
        anatomy_name = src_label_name.split("_")
        target_label_name = (
            "".join(anatomy_name[1:-2:-1]).upper() + "-" + "".join(anatomy_name[0])
        )
        print(target_label_name)

    elif src_label_template[1] == "-":
        anatomy_name = src_label_name.split("-")
        target_label_name = (
            "".join(anatomy_name[1:-2:-1]).upper() + "-" + "".join(anatomy_name[0])
        )
        print(target_label_name)

    elif src_label_template[1] == " ":
        pass  # TODO

    return target_label_name


def readh5(h5_path: str):
    """Read the h5 filex`
    Params:
    -------
    h5_path: str
        path to the h5 file

    Returns:
    --------
        None

    """
    with h5py.File(h5_path, "r") as f:
        # List all groups
        # print("Keys: %s" % f.keys())

        # key_list = list(f.keys())

        # print(key_list[1])
        # tp = f["TransformGroup/0/TransformParameters"]
        # print(tp[...])
        # tfp = f["TransformGroup/0/TransformFixedParameters"]
        # print(tfp[...])

        # data = f["proj-000/landmarks/GSN-l"]
        # # data = f['num-projs']
        # # data = [0,1]
        # print(data[...])

        # image = np.array(data)
        print(f.keys())
        for key in f.keys():
            if key != "num-projs":
                print(key)
                for subkey in f[key].keys():
                    print(subkey)

                    if subkey == "lands":
                        print(f[key][subkey][...])

                    elif subkey == "gt-poses":
                        for subsubkey in f[key][subkey].keys():
                            print(subsubkey)
                            print(f[key][subkey][subsubkey][...])

                    elif subkey == "segs":
                        print(f[key][subkey][...].shape)
                    print("---------------------------------")
            else:
                print(f[key][...])

    # plt.imshow(image,'gray')
    # plt.show()

    f.close()

    return None


def read_xray_dicom(path, to_32_bit=False, voi_lut=True, fix_monochrome=True):
    """
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
    """
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


def read_xray_png(path, img_size, to_32_bit=False):
    """
    Read the png file and return the pixel_array as a numpy array

    Params:
    -------
    path: str
        path to the png file

    Returns:
    --------
    image: numpy array
        the pixel_array of the x-ray image
    """
    origin_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if to_32_bit:
        image = image.astype(np.float32)

    # read dicom image and limit the

    # check if image is square
    if origin_image.shape[0] == origin_image.shape[1]:
        crop_image = origin_image[
            0 : origin_image.shape[0],
            0 : origin_image.shape[0],
        ]
    else:
        warnings.warn("Image is not square, cropping image to square.")
        crop_image = origin_image[
            0 : min(origin_image.shape[0], origin_image.shape[1]),
            0 : min(origin_image.shape[0], origin_image.shape[1]),
        ]  # crop image to square

    # resize image to img_size
    resized_image = cv2.resize(
        crop_image, (img_size, img_size), interpolation=cv2.INTER_LINEAR
    )

    # calculate scale factor, to scale the intrinsic camera matrix
    scale = crop_image.shape[0] / img_size

    return resized_image, origin_image, scale


def read_2d_landmarks(landmarks_dir: str) -> pd.DataFrame:
    """
    Read the 2D landmarks from the csv file

    Params:
    -------
    landmarks_dir: str
        path to the directory of the landmarks

    Returns:
    --------
    landmarks: DataFrame
        the 2D landmarks
    """
    landmarks = pd.read_csv(landmarks_dir)
    landmarks = landmarks.drop(columns=["pat", "proj", "time"], axis=1)

    return landmarks


def read_ct_dicom(ct_path: str):
    pass


def dicom2h5(xray_folder_path: str, label_path: str, output_path: str):
    current_path = os.path.abspath(os.path.dirname(__file__))
    xray_folder_path = os.path.join(current_path, xray_folder_path)
    label_path = os.path.join(current_path, label_path)
    output_path = os.path.join(current_path, output_path)

    file_names = [newestfile(xray_folder_path)]
    print("***", file_names)
    num_images = 1

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
    keys = [f"land-{i:02d}" for i in range(14)] + ["num-lands"]
    landmarks = [
        "FH-l",
        "FH-r",
        "GSN-l",
        "GSN-r",
        "IOF-l",
        "IOF-r",
        "MOF-l",
        "MOF-r",
        "SPS-l",
        "SPS-r",
        "IPS-l",
        "IPS-r",
        "ASIS-l",
        "ASIS-r",
        "14",
    ]
    for i, key in enumerate(keys):
        if i < len(landmarks):
            dtype_str = h5py.special_dtype(vlen=str)
            dataset_names = names.create_dataset(keys[i], (), dtype=dtype_str)
            dataset_names[()] = landmarks[i].encode("utf-8")
            label_dataset_names = label_names.create_dataset(
                keys[i], (), dtype=dtype_str
            )
            label_dataset_names[()] = landmarks[i].encode("utf-8")

    # Store all images in the dataset
    for i, file_name in enumerate(file_names):
        file_path = os.path.join(xray_folder_path, file_name)

        img_shape = 360  # 360 is the default size of the image in synthex
        image_data, origin_image, scale = preprocess_dicom(file_path, img_shape)
        if i == 0:
            # Create the dataset with the appropriate shape
            dataset_shape = (num_images, img_shape, img_shape)
            dataset = grp.create_dataset("projs", dataset_shape, dtype="f4")

        print(image_data.shape)
        dataset[i, :, :] = image_data

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
        gtpose_dataset = label_grp_gtpose.create_dataset(group_name, (4, 4), dtype="f4")
        gtpose_dataset[()] = label_grp_gtpose_content

    # complete the rest part of label_grp
    # "lands" part
    label_grp_lands = label_grp.create_dataset(
        "lands",
        (num_images, 2, 14),
        data=real_label["01"]["lands"][0:num_images],
        dtype="f4",
    )

    # "segs" part
    label_grp_segs = label_grp.create_dataset(
        "segs",
        (num_images, img_shape, img_shape),
        data=np.zeros((num_images, img_shape, img_shape)),
        dtype="|u1",
    )

    # Close the HDF5 file to save changes
    h5_file.close()
    real_label.close()
    h5_reallabel.close()


def preprocess_dicom(dicom_path: str, img_size: int):
    """
    resize dicom image to a square image with size img_size

    Args:
    --------
        dicom_path (str): path to dicom file
        img_size (int): size of the output image

    Returns:
    --------
        resized_image (np.array): resized image
        scale (float): scale factor
    """
    # check if the img_size is provided correctly
    assert isinstance(img_size, int), "img_size should be an integer"
    assert img_size > 0, "img_size should be larger than 0"

    # read dicom image and limit the
    origin_image = read_xray_dicom(dicom_path, to_32_bit=True) / 200

    # check if image is square
    if origin_image.shape[0] == origin_image.shape[1]:
        crop_image = origin_image[
            0 : origin_image.shape[0],
            0 : origin_image.shape[0],
        ]
    else:
        warnings.warn("Image is not square, cropping image to square.")
        crop_image = origin_image[
            0 : min(origin_image.shape[0], origin_image.shape[1]),
            0 : min(origin_image.shape[0], origin_image.shape[1]),
        ]  # crop image to square

    # resize image to img_size
    resized_image = cv2.resize(
        crop_image, (img_size, img_size), interpolation=cv2.INTER_LINEAR
    )

    # calculate scale factor, to scale the intrinsic camera matrix
    scale = crop_image.shape[0] / img_size

    image_name = os.path.join(
        os.path.dirname(__file__),
        "data/png",
        os.path.basename(dicom_path).split(".")[0] + ".png",
    )  # save the image as png with same name as dicom file

    # print(image_name)
    cv2.imwrite(image_name, origin_image)
    return resized_image, origin_image, scale


def gen_synthex_h5(image_data: np.ndarray, label_path: str, output_path: str):
    """
    generate synthex h5 file from image data and label file
    """

    pass


if __name__ == "__main__":
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

    # x = 'sps_l'
    # dicom2h5("data/xray", "data/real_label.h5", "data")

    # readh5("data/example1_1_pd_003.h5")
    # readh5("data/real_label.h5")
    # x, y = preprocess_dicom("data/xray/x_ray1.dcm", 360)
    # pass
    print(newestfile("data/xray"))
