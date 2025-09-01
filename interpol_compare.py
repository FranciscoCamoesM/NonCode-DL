import os
import numpy as np
from data_preprocess import *
import matplotlib.pyplot as plt
from data_preprocess import import_indiv_enhancers, get_dataloc, simple_pad_batch
from data_preprocess import one_hot_encode
from tqdm import tqdm

AA = 1
BB = 0.45/0.95


interpolated_seqs_1 = None
interpolated_seqs_2 = None

locus_name = "ank1"

enh_name_1 = "E25E10"
coverage_1 = 50
enh_name_2 = "E25E10R3"
coverage_2 = 50

bin_cells_1 = get_bin_cells(enh_name_1)
label_values1 = {"MM": -AA, "NM":-BB, "NP":BB, "PP":AA}
interpolated_seqs_1 = get_interpolated_seq_label_pairs(enh_name=enh_name_1, coverage=coverage_1, local_path=get_dataloc(enh_name_1), BIN_CELLS=bin_cells_1, label_values=label_values1)


bin_cells_2 = get_bin_cells(enh_name_2)
label_values2 = {"MM": -AA, "NM":-BB, "NP":BB, "PP":AA}
interpolated_seqs_2 = get_interpolated_seq_label_pairs(enh_name=enh_name_2, coverage=coverage_2, local_path=get_dataloc(enh_name_2), BIN_CELLS=bin_cells_2, label_values=label_values2)

inter_commons = {}
inter_2_keys = list(interpolated_seqs_2.keys())
for seq, label in interpolated_seqs_1.items():
    if seq in inter_2_keys:
        inter_commons[seq] = [label, interpolated_seqs_2[seq]]

data_1 = [x[0] for x in inter_commons.values()]
data_2 = [x[1] for x in inter_commons.values()]

# compute spearman correlation
from scipy.stats import spearmanr
correlation = spearmanr(data_1, data_2)

os.makedirs(f"interpol_maps/{locus_name}", exist_ok=True)


plt.figure(figsize=(10, 10))
plt.scatter(data_1, data_2, alpha=0.5)
plt.title(f"Interpolated {enh_name_1} vs {enh_name_2}", fontsize=22)
plt.xlabel(f"Interpolated {enh_name_1} values (coverage of {coverage_1})", fontsize=20)
plt.ylabel(f"Interpolated {enh_name_2} values (coverage of {coverage_2})", fontsize=20)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
# write the spearman correlation
plt.text(0.05, 0.95, f"Spearman correlation: {correlation.correlation:.2f}", transform=plt.gca().transAxes, fontsize=14)
plt.tight_layout()
# plt.savefig(f"/home/franciscom/NonCode/temp_figs/.final_interpolation_comparison.png")
plt.savefig(f"interpol_maps/{locus_name}/comparison_{enh_name_1}_{enh_name_2}_coverage_{coverage_1}_{coverage_2}.png")
plt.show()