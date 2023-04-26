import argparse
import os


def synthex_args():
    """Default arguments for the landmark detector.

    Returns:
    -------
        args: argparse.Namespace
            Arguments for the landmark detector:
                xray_path: str, path to the x-ray image folder which contains multiple dicom files
                label_path: str, path to the label file
                output_path: str, path to the output folder
                input_data_file_path: str, path to the input data file
                input_label_file_path: str, path to the input label file
                output_data_file_path: str, path to the output data file
                output_label_file_path: str, path to the output label file
                output_landmark_file_path: str, path to the output landmark file
                output_landmark_csv_file_path: str, path to the output landmark csv file
                nets: str, path to the network file
                rand: bool, whether to use random data
                pats: str, patient id
                no_gpu: bool, whether to use gpu
                times: str, time
                heat_file_path: str, path to the heat file
                heats_group_path: str, path to the heat group in h5 file
                out: str, path to the output csv file
                pat: str, patient id
                use_seg: str, path to the segmentation group in h5 file
                hm_lvl: bool, whether to use heatmap level
                ds_factor: int, downsample factor
                no_hdr: bool, whether to use header
                threshold: int, threshold

    """
    current_path = os.path.abspath(os.path.dirname(__file__))
    args = argparse.Namespace()
    args.xray_path = "data/xray"
    args.nets = "data/yy_checkpoint_net_20.pt"
    args.nets = os.path.join(current_path, args.nets)
    args.xray_path = "data/xray"
    args.xray_path = os.path.join(current_path, args.xray_path)
    args.label_path = "data/real_label.h5"
    args.label_path = os.path.join(current_path, args.label_path)
    args.output_path = "data"
    args.output_path = os.path.join(current_path, args.output_path)
    args.input_data_file_path = "data/synthex_input.h5"
    args.input_data_file_path = os.path.join(current_path, args.input_data_file_path)
    args.input_label_file_path = "data/synthex_label_input.h5"
    args.input_label_file_path = os.path.join(current_path, args.input_label_file_path)
    args.output_data_file_path = "data/output.h5"
    args.output_data_file_path = os.path.join(current_path, args.output_data_file_path)

    args.rand = True
    args.pats = "01"
    args.no_gpu = True
    args.times = ""

    args.heat_file_path = "data/output.h5"
    args.heat_file_path = os.path.join(current_path, args.heat_file_path)
    args.heats_group_path = "nn-heats"
    args.out = "data/own_data.csv"
    args.out = os.path.join(current_path, args.out)
    args.pat = "01"
    args.use_seg = "nn-segs"
    # args.rand = True
    args.hm_lvl = True
    args.ds_factor = 4
    args.no_hdr = False
    args.threshold = 1
    return args


def xreg_args():
    """
    Default arguments for the landmark registration.

    Returns:
    -------
        args: argparse.Namespace
            Arguments for the landmark registration:
                image_path_load: str, path to the x-ray image
                ct_path: str, path to the ct image
                ct_segmentation_path: str, path to the ct segmentation
                landmarks_2d_path: str, path to the 2d landmarks
                landmarks_3d_path: str, path to the 3d landmarks

    """

    path = {}
    path["image_path_load"] = "data/x_ray1.dcm"

    path["ct_path_load"] = "data/pelvis.nii.gz"
    path["ct_segmentation_path"] = "data/pelvis_seg.nii.gz"
    path["landmarks_2d_path"] = "data/own_data.csv"
    path["landmarks_3d_path"] = "data/pelvis_regi_2d_3d_lands_wo_id.fcsv"
    return path


def cam_param():
    cam_params = {}
    cam_params["intrinsic"] = [-5257.73, 0, 767.5, 0, -5257.73, 767.5, 0, 0, 1]
    cam_params["img_type"] = "DICOM"

    return cam_params
