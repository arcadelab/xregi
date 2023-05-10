import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import argparse
import json

from .args import cam_param
from .utils import *
from .synthex.est_land_csv import est_land_csv
from .synthex import class_ensemble
from . import config


class LandmarkDetector(ABC):
    """
    Abstract class for landmark detection

    Args:
    -------
        image (np.ndarray): x-ray image in the shape of (# of image, height, width)

    Returns:
    -------
        create csv file with 2D coordinates of landmarks

    """

    def __init__(self, image: np.ndarray):
        self.image = image

    @property
    @abstractmethod
    def landmarks(self) -> List[str]:
        """
        Landmarks names are defined in the child class
        """
        pass

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, image):
        self._image = image

    @abstractmethod
    def load_data(self):
        """
        load data using specific method
        """
        pass

    @abstractmethod
    def detect(self) -> Dict[str, Optional[np.ndarray]]:
        """
        Detect landmarks in xray image, return 2D coordinates of landmarks

        Args:
            self.image: np.ndarray
            self.landmarks: dict[str, np.ndarray]

        Returns:
            landmarks_2D: np.ndarray
        """
        pass


class SynthexDetector(LandmarkDetector):
    """
    Synthex landmark detector

    Args:
    -------
        image(np.ndarray): x-ray image in the shape of (# of image, height, width)
        landmarks(dict[str, np.ndarray]): 3d landmarks in the shape of (landmark name, [x, y, z])

    Returns:
    -------
        create csv file with 2D coordinates of landmarks


    """

    def __init__(
        self,
        image: np.ndarray,
        landmarks: Dict[str, np.ndarray],
        args: Optional[argparse.Namespace],
    ):
        self.args = args if args is not None else {}
        super().__init__(image)

    @property
    def landmarks(self) -> List[str]:
        self.landmarks = [
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
        ]
        return self.landmarks

    @property
    def image_path(self):
        return self._image_path

    @image_path.setter
    def image_path(self, image_path):
        print("image_path: ", image_path)
        self.gen_h5(image_path, self.args["label_path"], self.args["output_path"])
        self._image_path = image_path
        return self._image_path

    def load_data(self):
        """
        load network and data
        Args:
        -------
            self.args: args from syn_args.py

        """
        self.output_data_file_path = self.args["output_data_file_path"]
        self.ensemble_seg = class_ensemble.Ensemble(self.args)
        self.ensemble_seg.loadnet()

    def savedata(self):
        """
        save data
        Args:
        -------
            self.args: args from syn_args.py

        """
        self.ensemble_seg.savedata(
            self.args["input_data_file_path"], self.args["input_label_file_path"]
        )

    def detect(self):
        """
        detect landmarks
        Args:
        -------
            self.args: args from syn_args.py

        """
        est_land_csv(self.args)

    def run(self):
        """
        run landmark detection
        Args:
        -------
            self.args: args from syn_args.py

        """
        self.load_data()
        self.savedata()
        self.detect()

    @classmethod
    def load(cls):
        """
        load data from x-ray image and label based on json file

        Returns:
        -------
            SynthexDetector: Synthex landmark detector

        """
        path = config.load_json(os.path.abspath(config.__file__))

        cls.gen_h5(path["xray_path"], path["label_path"], path["output_path"])
        f = h5py.File(os.path.join(path["output_path"], "synthex_input.h5"), "r")
        image = f[path["pats"]]["projs"]
        return cls(image, None, path)

    def gen_h5(xray_folder_path: str, label_path: str, output_path: str):
        cam_params = cam_param()
        current_path = os.path.abspath(os.path.dirname(__file__))
        xray_folder_path = os.path.join(current_path, xray_folder_path)
        label_path = os.path.join(current_path, label_path)
        output_path = os.path.join(current_path, output_path)

        file_names = [newestfile(xray_folder_path)]
        print("***", file_names)
        num_images = 1

        # Create an HDF5 file
        h5_file = h5py.File(os.path.join(output_path, "synthex_input.h5"), "w")
        h5_reallabel = h5py.File(
            os.path.join(output_path, "synthex_label_input.h5"), "w"
        )

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

            if cam_params["img_type"] == "DICOM":
                resized_img, image_load, scale = preprocess_dicom(file_path, 360)

            elif cam_params["img_type"] == "PNG":
                resized_img, image_load, scale = read_xray_png(file_path, 360)

            else:
                raise ValueError("Image type not supported")
            if i == 0:
                # Create the dataset with the appropriate shape
                dataset_shape = (num_images, img_shape, img_shape)
                dataset = grp.create_dataset("projs", dataset_shape, dtype="f4")

            print(resized_img.shape)
            dataset[i, :, :] = resized_img

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
                group_name, (4, 4), dtype="f4"
            )
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


if __name__ == "__main__":
    syn = SynthexDetector.load(r"data\xray", r"data\real_label.h5", "data", "01")
    syn.load_data()
    syn.savedata()
    syn.detect()

    def get_2d_landmarks(landmarks_path: str) -> dict:
        """Get 2D landmarks from the csv file
        Params:
        -------
        landmarks_2d_path: str
            Path to the csv file

        Returns:
        --------
        landmarks_2d: dict[str, np.ndarray]
            A dictionary of 2D landmarks
        """
        # This is the synthex format and order for landmarks
        land_name = [
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
        ]

        landmarks_2d = {}
        data_frame = pd.read_csv(landmarks_path)
        data_frame = pd.DataFrame.drop(
            data_frame, columns=["pat", "proj", "time", "land"], axis=1
        )

        data_frame["land-name"] = land_name
        print(data_frame["land-name"][0])

        for i in range(len(data_frame)):
            landmarks_2d[data_frame["land-name"][i]] = [
                data_frame["row"][i],
                data_frame["col"][i],
            ]

        return landmarks_2d

    print(get_2d_landmarks("data/own_data.csv"))
