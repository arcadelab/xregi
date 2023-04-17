import numpy as np
import h5py
import os


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
        print("Keys: %s" % f.keys())

        key_list = list(f.keys())

        print(key_list)
        sub_key = f["proj-000"]["landmarks"].keys()
        print(sub_key)

        # data = f["proj-000/landmarks/GSN-l"]
        # # data = f['num-projs']
        # # data = [0,1]
        # print(data[...])

        # image = np.array(data)

    # plt.imshow(image,'gray')
    # plt.show()

    f.close()

    return None


if __name__ == "__main__":
    # h5_path = "data/synthex_input.h5"
    current_path = os.path.abspath(os.path.dirname(__file__))
    h5_path = os.path.join(current_path, "../data/example1_1_pd_003.h5")

    readh5(h5_path)
