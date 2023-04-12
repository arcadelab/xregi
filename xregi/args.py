import argparse

def default_args():
    args = argparse.Namespace()
    args.nets = "data/yy_checkpoint_net_20.pt"
    args.input_data_file_path = "data/synthex_input.h5"
    args.input_label_file_path = "data/synthex_label_input.h5"
    args.output_data_file_path = "data/output.h5"
    args.rand = True
    args.pats = "01"
    args.no_gpu = True
    args.times = ""

    args.heat_file_path = "data/output.h5"
    args.heats_group_path = "nn-heats"
    args.out = "data/own_data.csv"
    # args.pat = "01"
    args.use_seg = "nn-segs"
    # args.rand = True
    args.hm_lvl = True
    args.ds_factor = 4
    args.no_hdr = True
    args.threshold = 2
    return args
