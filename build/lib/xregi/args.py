import argparse
import numpy as np
import os


# def xreg_args():
#     """
#     Default arguments for the landmark registration.

#     Returns:
#     -------
#         args: argparse.Namespace
#             Arguments for the landmark registration:
#                 image_path_load: str, path to the x-ray image
#                 ct_path: str, path to the ct image
#                 ct_segmentation_path: str, path to the ct segmentation
#                 landmarks_2d_path: str, path to the 2d landmarks
#                 landmarks_3d_path: str, path to the 3d landmarks

#     """

#     path = {}
#     path["image_path_load"] = "data/x_ray1.dcm"
#     path["ct_path_load"] = "data/pelvis.nii.gz"
#     path["ct_segmentation_path"] = "data/pelvis_seg.nii.gz"
#     path["landmarks_2d_path"] = "data/own_data.csv"
#     path["landmarks_3d_path"] = "data/pelvis_regi_2d_3d_lands_wo_id.fcsv"
#     return path


def cam_param():
    cam_params = {}
    cam_params["intrinsic"] = np.asarray(
        [[-5257.73, 0, 767.5], [0, -5257.73, 767.5], [0, 0, 1]]
    )
    cam_params["img_type"] = "PNG"

    return cam_params


if __name__ == "__main__":
    x = cam_param()
    y = x["intrinsic"] * 0.5
    y[-1, -1] = 1
    print(y)
