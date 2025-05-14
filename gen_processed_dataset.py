import os
import torch
import torch.nn as nn
import numpy as np  
import argparse

from data_preprocess import get_coverage_seq_label_pairs, preset_labels, EnhancerDataset, get_dataloc

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset', type=str, default="E25E10", help='Dataset name')
parser.add_argument('--coverage', type=int, default=5, help='Coverage')
parser.add_argument('--out_dir', type=str, default="", help='Output directory')
args = parser.parse_args()


DATASET = args.dataset
DATALOC = get_dataloc(DATASET)
COVERAGE = args.coverage
LABELS = preset_labels("all_v_all")
print(LABELS)

dataset = get_coverage_seq_label_pairs(enh_name = DATASET, local_path = DATALOC, labels=LABELS, coverage=COVERAGE)

X, y = dataset.keys(), dataset.values()
del dataset
print("Dataset loaded")
print("X shape: ", len(X))
print("y shape: ", len(y))

dataset = EnhancerDataset(X, y, LABELS)

seq_dict = "ACGT"

counts = [0, 0 ,0, 0]

# for seq, label in dataset:
#     # print("Seq: ", "".join([seq_dict[i] for i in np.argmax(seq, axis=0)]))
#     # print("Label: ", np.argmax(label, axis=0))
#     counts[np.argmax(label, axis=0)] += 1
# print("Counts: ", counts)


# # subsample
# indexes = np.random.choice(len(dataset), size=5000, replace=False)
# dataset = [dataset[i] for i in indexes]


c=0
datapath = os.path.join(args.out_dir, f"processed_data_{DATASET}_{COVERAGE}.fa")
with open(datapath, "w") as f:
    for seq, label in dataset:
        f.write(f">{c}-{np.argmax(label, axis=0)}\n")
        f.write("".join([seq_dict[i] for i in np.argmax(seq, axis=0)]) + "\n")

        # print(f">{c}:{np.argmax(label, axis=0)}\n")
        # print("".join([seq_dict[i] for i in np.argmax(seq, axis=0)]) + "\n")

        c+=1

print(f"Finished writing processed data in {DATASET}.fa")