import argparse
import os
from .registration_2d_3d import Registration2D3D
from . import config
import time


def main():
    # Create the parser

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xray",
        type=str,
        required=True,
        help="path to the x-ray image folder which contains multiple dicom files",
    )
    parser.add_argument(
        "--ct",
        type=str,
        required=True,
        help="path to the CT scan file",
    )
    parser.add_argument(
        "--landmarks_3d",
        type=str,
        required=True,
        help="path to the 3d landmarks file, currently only support fcsv file",
    )
    parser.add_argument(
        "--ct_seg",
        type=str,
        required=True,
        help="path to the CT segmentation file, currently only support nrrd file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="path to the checkpoint file",
    )
    # Parse the arguments provided by the user
    args = parser.parse_args()

    ## run config.py to set environment variables
    config.config_json(
        args.xray,
        args.checkpoint,
        args.ct,
        args.landmarks_3d,
        args.ct_seg,
        os.path.abspath(config.__file__),
    )

    ## run registration_2d_3d.py
    registration = Registration2D3D.load()
    registration.run()


if __name__ == "__main__":
    st = time.time()
    main()
    ft = time.time()
    print("time: ", ft - st)
