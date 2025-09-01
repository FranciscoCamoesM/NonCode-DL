import os
import numpy as np
from data_preprocess import *
import matplotlib.pyplot as plt
from data_preprocess import import_indiv_enhancers, get_dataloc, simple_pad_batch
from data_preprocess import one_hot_encode
from tqdm import tqdm
from scipy.stats import spearmanr
import seaborn as sns

AA = 1
BB = 0.45/0.95


interpolated_seqs_1 = None
interpolated_seqs_2 = None

covs = [5, 10, 50, 100]


locus_name = "ank1"
enh_list = ["E22P1A3", "E25E10", "E25E102", "E25E10R3"]

# locus_name = "chr20"
# enh_list = ["E22P1D10", "E22P3B3", "E22P3B3R3"]

# locus_name = "chr7"
# enh_list = ["E22P3B4", "E22P1C1"]



results_frame = -1* np.ones((len(enh_list), len(enh_list), len(covs)))
for cov in covs:
    coverage_1 = cov
    coverage_2 = cov
    for i, enh_name_1 in enumerate(enh_list):
        for enh_name_2 in enh_list[i+1:]:

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

            if len(inter_commons) == 0:
                continue

            data_1 = [x[0] for x in inter_commons.values()]
            data_2 = [x[1] for x in inter_commons.values()]

            # compute spearman correlation
            correlation = spearmanr(data_1, data_2)
            results_frame[i, enh_list.index(enh_name_2), covs.index(cov)] = correlation.correlation
            results_frame[enh_list.index(enh_name_2), i, covs.index(cov)] = correlation.correlation

            os.makedirs(f"interpol_maps/{locus_name}", exist_ok=True)



            plt.figure(figsize=(10, 10))
            plt.scatter(data_1, data_2, alpha=0.5)
            plt.title(f"Coverage of {cov}", fontsize=22)
            plt.xlabel(f"Interpolated {enh_name_1} values", fontsize=20)
            plt.ylabel(f"Interpolated {enh_name_2} values", fontsize=20)
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            # write the spearman correlation
            plt.text(0.05, 0.95, f"Spearman correlation: {correlation.correlation:.2f}", transform=plt.gca().transAxes, fontsize=18)
            plt.tight_layout()
            # plt.savefig(f"/home/franciscom/NonCode/temp_figs/.final_interpolation_comparison.png")
            plt.savefig(f"interpol_maps/{locus_name}/comparison_{enh_name_1}_{enh_name_2}_coverage_{coverage_1}_{coverage_2}.png")
            plt.show()
            plt.close()

# replace -1s with NaN


fig, axs = plt.subplots(1, len(covs), figsize=(2+10*len(covs), 10), sharex=True)
for i, cov in enumerate(covs):
    sns.heatmap(results_frame[:, :, i], annot=True, fmt=".2f", ax=axs[i], cmap='viridis', cbar_kws={'label': 'Spearman Correlation'}, mask=(results_frame[:, :, i] == np.nan), vmin=0, vmax=1)
    axs[i].set_title(f"Coverage: {cov}", fontsize=16)
    axs[i].set_xlabel("Dataset", fontsize=14)
    axs[i].set_ylabel("Dataset", fontsize=14)
    axs[i].set_xticks(np.arange(len(enh_list)), labels=enh_list)
    axs[i].set_yticks(np.arange(len(enh_list)), labels=enh_list)

    for text in axs[i].texts:
        print(text.get_text())
        if text.get_text() == '-1.00':
            text.set_text('N/A')
plt.tight_layout()
plt.savefig(f"interpol_maps/{locus_name}/locus_comparison.png")
plt.show()

# plot average against coverages
mean_results = np.nanmean(results_frame, axis=(0,1))
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(covs, mean_results, '--o')
ax.set_title(f"Average Spearman Correlation vs Coverage", fontsize=16)
ax.set_xlabel("Coverage", fontsize=14)
ax.set_ylabel("Average Spearman Correlation", fontsize=14)
plt.tight_layout()
plt.savefig(f"interpol_maps/{locus_name}/locus_comparison_average_vs_coverage.png")
plt.show()
