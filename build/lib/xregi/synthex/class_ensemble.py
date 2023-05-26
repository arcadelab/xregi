import argparse
from .dataset import *
from .util import *
from .TransUNet.transunet import VisionTransformer as ViT_seg
from .TransUNet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import os


class Ensemble:
    def __init__(self, args) -> None:
        self.args = args
        self.network_paths = self.args["nets"]

        self.dst_data_file_path = self.args["output_data_file_path"]

        self.rand = self.args["rand"]
        assert self.args["pats"] is not None
        self.test_pats = (
            [i for i in self.args["pats"].split(",")]
            if self.rand
            else [int(i) for i in self.args["pats"].split(",")]
        )
        assert len(self.test_pats) > 0
        self.torch_map_loc = None

        if self.args["no_gpu"]:
            self.dev = torch.device("cpu")
            self.torch_map_loc = "cpu"
        else:
            self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def loadnet(self) -> list:
        self.nets = []
        net_path = self.network_paths
        print(net_path)

        print("  loading state from disk for: {}".format(net_path))

        state = torch.load(net_path, map_location=self.torch_map_loc)

        print("  loading unet params from checkpoint state dict...")
        num_classes = state["num-classes"]
        unet_num_lvls = state["depth"]
        unet_init_feats_exp = state["init-feats-exp"]
        unet_batch_norm = state["batch-norm"]
        unet_padding = state["padding"]
        unet_no_max_pool = state["no-max-pool"]
        unet_use_res = state["unet-use-res"]
        unet_block_depth = state["unet-block-depth"]
        proj_unet_dim = state["pad-img-size"]
        batch_size = state["batch-size"]
        num_lands = state["num-lands"]

        print("             num. classes: {}".format(num_classes))
        print("                    depth: {}".format(unet_num_lvls))
        print("        init. feats. exp.: {}".format(unet_init_feats_exp))
        print("              batch norm.: {}".format(unet_batch_norm))
        print("         unet do pad img.: {}".format(unet_padding))
        print("              no max pool: {}".format(unet_no_max_pool))
        print("    reflect pad img. dim.: {}".format(proj_unet_dim))
        print("            unet use res.: {}".format(unet_use_res))
        print("         unet block depth: {}".format(unet_block_depth))
        print("               batch size: {}".format(batch_size))
        print("              num. lands.: {}".format(num_lands))

        print("    creating network")
        vit_name = "R50-ViT-B_16"
        vit_patches_size = 16
        img_size = proj_unet_dim
        config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
        config_vit.n_classes = num_classes
        config_vit.n_skip = 3
        if vit_name.find("R50") != -1:
            config_vit.patches.grid = (
                int(img_size / vit_patches_size),
                int(img_size / vit_patches_size),
            )
        net = ViT_seg(
            config_vit,
            img_size=img_size,
            num_classes=config_vit.n_classes,
            batch_norm=unet_batch_norm,
            padding=unet_padding,
            n_classes=num_classes,
            num_lands=num_lands,
        )

        net.load_state_dict(state["model-state-dict"])

        del state

        print("  moving network to device...")
        net.to(self.dev)

        self.nets.append(net)
        self.num_classes = num_classes
        self.proj_unet_dim = proj_unet_dim
        self.num_lands = num_lands

    def savedata(self, input_data_file_path, input_label_file_path):
        self.src_data_file_path = input_data_file_path
        self.src_label_file_path = input_label_file_path
        land_names = None
        if self.num_lands > 0:
            land_names = get_land_names_from_dataset(self.src_data_file_path)
            assert len(land_names) == self.num_lands

        print("initializing testing dataset")
        test_ds = (
            get_rand_dataset(
                self.src_data_file_path,
                self.src_label_file_path,
                self.test_pats,
                num_classes=self.num_classes,
                pad_img_dim=self.proj_unet_dim,
                no_seg=True,
            )
            if self.rand
            else get_dataset(
                self.src_data_file_path,
                self.src_label_file_path,
                self.test_pats,
                num_classes=self.num_classes,
                pad_img_dim=self.proj_unet_dim,
                no_seg=True,
                valid=True,
            )
        )

        print("Length of testing dataset: {}".format(len(test_ds)))

        print("opening destination file for writing")

        f = h5.File(self.dst_data_file_path, "w")

        # save off the landmark names
        if land_names:
            land_names_g = f.create_group("land-names")
            land_names_g["num-lands"] = self.num_lands

            for l in range(self.num_lands):
                land_names_g["land-{:02d}".format(l)] = land_names[l]

        times = []

        print("running network on projections")
        seg_dataset_ensemble(
            test_ds,
            self.nets,
            f,
            dev=self.dev,
            num_lands=self.num_lands,
            times=times,
            adv_loss=False,
        )

        print("closing file...")
        f.flush()
        f.close()

        if self.args["times"]:
            times_out = open(self.args["times"], "w")
            for t in times:
                times_out.write("{:.6f}\n".format(t))
            times_out.flush()
            times_out.close()


if __name__ == "__main__":
    pass
