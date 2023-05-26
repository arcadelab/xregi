import json
import os


# Read Existing JSON File
def config_json(
    xray_path,
    net_path,
    ct_path,
    landmarks_3d_path,
    CT_segmentation_path,
    config_file_path,
):
    current_path = os.path.abspath(os.path.dirname(__file__))

    json_path = os.path.join(os.path.dirname(config_file_path), "config/config.json")
    print("json_path", json_path)

    with open(json_path, "r") as f:
        data = json.load(f)
    f.close()

    ## add args into config.json

    data["xray_path"] = os.path.join(current_path, xray_path)
    data["nets"] = os.path.join(current_path, net_path)
    data["ct_path"] = os.path.join(current_path, ct_path)
    data["landmarks_3d_path"] = os.path.join(current_path, landmarks_3d_path)
    data["CT_segmentation_path"] = os.path.join(current_path, CT_segmentation_path)

    # add default args into config.json
    data["label_path"] = os.path.join(current_path, "data/real_label.h5")
    data["output_path"] = os.path.join(current_path, "data")
    data["input_data_file_path"] = os.path.join(current_path, "data/synthex_input.h5")
    data["input_label_file_path"] = os.path.join(
        current_path, "data/synthex_label_input.h5"
    )
    data["output_data_file_path"] = os.path.join(current_path, "data/output.h5")
    data["heat_file_path"] = os.path.join(current_path, "data/output.h5")
    data["out"] = os.path.join(current_path, "data/own_data.csv")

    # Create new JSON file
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    # Closing file
    f.close()


def load_json(config_file_path):
    json_path = os.path.join(os.path.dirname(config_file_path), "config/config.json")

    with open(json_path, "r") as f:
        path = json.load(f)

    return path


if __name__ == "__main__":
    config_json()
