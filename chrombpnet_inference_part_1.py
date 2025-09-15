import numpy as np
from sklearn.model_selection import train_test_split
from data_preprocess import *
from data_preprocess import get_interpolated_seq_label_pairs, get_dataloc, load_trained_model, one_hot_encode, simple_pad


import argparse
parser = argparse.ArgumentParser(description="Plot predictions from ChromBPNet.")
parser.add_argument("--enh", type=str, default="E25E10", help="Name of the dataset.")
parser.add_argument("--cov", type=int, default=100, help="Coverage of the dataset.")
args = parser.parse_args()

inference_coverage = args.cov
inference_enh = args.enh

AA = 1
BB = 0.45/0.95
lv = {"MM": -AA, "NM":-BB, "NP":BB, "PP":AA}


inference_bin_cells = get_bin_cells(inference_enh)

inference_enh_data_dict = get_interpolated_seq_label_pairs(coverage=inference_coverage, label_values=lv, enh_name=inference_enh, local_path=get_dataloc(inference_enh), BIN_CELLS=inference_bin_cells)
inference_enh_values = inference_enh_data_dict.values()
inference_enh_keys = inference_enh_data_dict.keys()
inference_enh_data = zip(inference_enh_keys, inference_enh_values)
inference_enh_data = list(inference_enh_data)

train_seqs, test_seqs = train_test_split(inference_enh_data, test_size=0.2, random_state=42)
train_seqs, val_seqs = train_test_split(train_seqs, test_size=0.2, random_state=42)
# for i, (seq, label) in enumerate(train_seqs):
#     print(f">{i}: {label:.3f}")
#     print(simple_pad(seq, 250).upper())

filename = "chrombpnet_experiments/interpolated_data/interpolated_{inference_enh}_{inference_coverage}_seqs.fa"

os.makedirs(os.path.dirname(filename.format(inference_enh=inference_enh, inference_coverage=inference_coverage)), exist_ok=True)

with open(filename.format(inference_enh=inference_enh, inference_coverage=inference_coverage), "w") as f:
    for i, (seq, label) in enumerate(train_seqs):
        text = f">{i}_{label:.3f}\n"
        text = text.replace(".", "p").replace("-", "m")
        f.write(text)
        f.write(simple_pad(seq, 250).upper() + "\n")

    # add a last line because the chrombpnet code has a bug that makes the prediction of the last sequence fail
    f.write(f">{i+1}last_\n")
    f.write(simple_pad(seq, 250).upper() + "\n")

# create a copy of this file named "custom_seqs.fa" in the chrombpnet folder
chrombpnet_folder = "./ChromBPNet/"
import os
import shutil
shutil.copy(filename.format(inference_enh=inference_enh, inference_coverage=inference_coverage), os.path.join(chrombpnet_folder, "custom_seqs.fa"))


# then, run the chrombpnet inference_custom_seqs.sh script
# after that, the results will be in the chrombpnet_experiments/predictions folder

# you can then run the part 2 of this code to get the plot the results and calculate the metrics