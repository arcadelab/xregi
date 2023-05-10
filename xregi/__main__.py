import argparse
import os
from .registration_2d_3d import Registration2D3D
from . import config


def main():
    # Create the parser

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xray_folder_path",
        type=str,
        required=True,
        help="path to the x-ray image folder which contains multiple dicom files",
    )
    parser.add_argument(
        "--ct_path",
        type=str,
        required=True,
        help="path to the CT scan file",
    )
    parser.add_argument(
        "--landmarks_3d_path",
        type=str,
        required=True,
        help="path to the 3d landmarks file, currently only support fcsv file",
    )
    parser.add_argument(
        "--CT_segmentation_path",
        type=str,
        required=True,
        help="path to the CT segmentation file, currently only support nrrd file",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="path to the checkpoint file",
    )
    # Parse the arguments provided by the user
    args = parser.parse_args()

    ## run config.py to set environment variables
    config.config_json(
        args.xray_folder_path,
        args.checkpoint_path,
        args.ct_path,
        args.landmarks_3d_path,
        args.CT_segmentation_path,
        os.path.abspath(config.__file__),
    )

    ## run registration_2d_3d.py
    registration = Registration2D3D.load()
    registration.run()


if __name__ == "__main__":
    main()
