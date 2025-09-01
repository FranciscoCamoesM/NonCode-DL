import os
import numpy as np
from data_preprocess import *
import matplotlib.pyplot as plt
from data_preprocess import import_indiv_enhancers, get_dataloc, simple_pad_batch
from data_preprocess import one_hot_encode
from tqdm import tqdm

# coverage_1 = 12
# coverage_2 = 100
# enh_name_1 = "E25B2"
# enh_name_2 = "E22P1A3"
# bin_cells_1 = np.array([14866, 63461, 30286, 3838])
# bin_cells_2 = np.array([23731, 69026, 57444, 15199])

# coverage_1 = 25
# coverage_2 = 25
# enh_name_1 = "E22P3B3R3"
# enh_name_2 = "E22P1D10"
# bin_cells_1 = np.array([12001,17423,10126,1234])
# bin_cells_2 = np.array([8867,26445,31755,6627])

# coverage_1 = 10
# coverage_2 = 10
# enh_name_1 = "E24C3"
# enh_name_2 = "E22P3F2"
# bin_cells_1 = np.array([1957,8165,7623,689])
# bin_cells_2 = np.array([11765,31087,21297,2339])

# coverage_1 = 100
# coverage_2 = 100
# enh_name_1 = "E25E102"
# enh_name_2 = "E25E10R3"
# bin_cells_1 = np.array([74226, 107443, 103524, 7799])
# bin_cells_2 = np.array([68519, 111643, 60318, 5687])

# coverage_1 = 100
# coverage_2 = 100
# enh_name_1 = "E22P1A3"
# enh_name_2 = "E25E10R3"
# bin_cells_1 = np.array([23731, 69026, 57444, 15199])
# bin_cells_2 = np.array([68519, 111643, 60318, 5687])


ENH_LIST = [[100, "E22P1A3", np.array([23731, 69026, 57444, 15199])],
                # [10, "E24A3", np.array([1, 1, 1, 1])],
                # [25, "E25E10", np.array([1, 1, 1, 1])],
                [100, "E25E102", np.array([74226,107443,103524,7799])],
                [100, "E25E10R3", np.array([68519,111643,60318,5687])],
                # [10, "E25B2", np.array([14866,63461,30286,3838])],
                # [10, "E25B3", np.array([9327,39125,25464,2188])]
                ]


# ENH_LIST = [[5, "E18C2", np.array([3135, 4108, 3866, 501])],
#                 [5, "E22P3F2", np.array([1957, 8165, 7623, 689])],
#                 [10, "E24C3", np.array([11765, 31087, 21297, 2339])]]

# ENH_LIST = [[25, "E22P1D10", np.array([8867, 26445, 31755, 6627])],
#                 [3, "E22P3B3", np.array([8931, 9735, 7685, 635])],
#                 [100, "E22P3B3R3", np.array([12001, 17423, 10126, 1234])]]


# ENH_LIST = [[25, "E22P1C1", np.array([2152, 6691, 4350, 258])],
#                 [50, "E22P3B4", np.array([5112, 24376, 19383, 2478])]]


# ENH_LIST = [[20, "E22P2F4", np.array([4919, 16352, 9135, 534])],
#                 [20, "E22P3G3", np.array([3644, 10704, 6332, 443])]]


# compile all
# ENH_LIST = [[100, "E22P1A3", np.array([23731, 69026, 57444, 15199])],
#                 [10, "E24A3", np.array([1, 1, 1, 1])],
#                 [25, "E25E10", np.array([1, 1, 1, 1])],
#                 [100, "E25E102", np.array([74226,107443,103524,7799])],
#                 [100, "E25E10R3", np.array([68519,111643,60318,5687])],
#                 [10, "E25B2", np.array([14866,63461,30286,3838])],
#                 [10, "E25B3", np.array([9327,39125,25464,2188])],
#                 [1, "E18C2", np.array([3135, 4108, 3866, 501])],
#                 [50, "E22P3F2", np.array([1957, 8165, 7623, 689])],
#                 [100, "E24C3", np.array([11765, 31087, 21297, 2339])],
#                 [25, "E22P1D10", np.array([8867, 26445, 31755, 6627])],
#                 [3, "E22P3B3", np.array([8931, 9735, 7685, 635])],
#                 [100, "E22P3B3R3", np.array([12001, 17423, 10126, 1234])],
#                 [25, "E22P1C1", np.array([2152, 6691, 4350, 258])],
#                 [50, "E22P3B4", np.array([5112, 24376, 19383, 2478])],
#                 [20, "E22P2F4", np.array([4919, 16352, 9135, 534])],
#                 [20, "E22P3G3", np.array([3644, 10704, 6332, 443])]]

print("Loading enhancer data...")

AA = 1
BB = 0.475

result_matrix = np.zeros((len(ENH_LIST), len(ENH_LIST)))

for i, enh_data in enumerate(ENH_LIST):
    coverage_1, enh_name_1, bin_cells_1 = enh_data

    label_values1 = {"MM": -AA, "NM":-BB, "NP":BB, "PP":AA}
    label_values3 = {"MM": -AA, "NM":-BB, "NP":BB, "PP":AA}

    interpolated_seqs_1 = get_interpolated_seq_label_pairs(enh_name=enh_name_1, coverage=coverage_1, local_path=get_dataloc(enh_name_1), BIN_CELLS=bin_cells_1, label_values=label_values1)
    non_weighted_interpolated_seqs_1 = get_interpolated_seq_label_pairs(coverage=coverage_1, enh_name=enh_name_1, local_path=get_dataloc(enh_name_1), label_values=label_values3)
    
    for j, enh_data2 in enumerate(ENH_LIST[i+1:]):

        coverage_2, enh_name_2, bin_cells_2 = enh_data2

        label_values2 = {"MM": -AA, "NM":-BB, "NP":BB, "PP":AA}
        label_values4 = {"MM": -AA, "NM":-BB, "NP":BB, "PP":AA}

        interpolated_seqs_2 = get_interpolated_seq_label_pairs(enh_name=enh_name_2, coverage=coverage_2, local_path=get_dataloc(enh_name_2), BIN_CELLS=bin_cells_2, label_values=label_values2)
        non_weighted_interpolated_seqs_2 = get_interpolated_seq_label_pairs(coverage=coverage_2, enh_name=enh_name_2, local_path=get_dataloc(enh_name_2), label_values=label_values4)

        inter_commons = {}
        inter_2_keys = list(interpolated_seqs_2.keys())
        for seq, label in interpolated_seqs_1.items():
            if seq in inter_2_keys:
                inter_commons[seq] = [label, interpolated_seqs_2[seq]]


        inter_commons_non_weighted = {}
        for seq, label in non_weighted_interpolated_seqs_1.items():
            if seq in non_weighted_interpolated_seqs_2:
                inter_commons_non_weighted[seq] = [label, non_weighted_interpolated_seqs_2[seq]]


        print(f"Number of sequences in {enh_name_1} interpolated dataset: {len(interpolated_seqs_1)}")
        print(f"Number of sequences in {enh_name_2} interpolated dataset: {len(interpolated_seqs_2)}")
        print(len(inter_commons), "common sequences found between the two interpolated datasets.")
        print(len(inter_commons_non_weighted), "common sequences found between the two non-weighted interpolated datasets.")


        if len(inter_commons_non_weighted) == 0 and len(inter_commons) == 0:
            print("No common sequences found between the two datasets.")
            continue


        # Plotting
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].scatter([x[0] for x in inter_commons.values()], [x[1] for x in inter_commons.values()], alpha=0.5, s=10)
        ax[1].scatter([x[0] for x in inter_commons_non_weighted.values()], [x[1] for x in inter_commons_non_weighted.values()], alpha=0.5, s=10)
        ax[0].set_title(f"Interpolated {enh_name_1} vs {enh_name_2} (weighted)\nR^2 = {np.corrcoef([x[0] for x in inter_commons.values()], [x[1] for x in inter_commons.values()])[0, 1]:.2f}")
        ax[1].set_title(f"Interpolated {enh_name_1} vs {enh_name_2} (non-weighted)\nR^2 = {np.corrcoef([x[0] for x in inter_commons_non_weighted.values()], [x[1] for x in inter_commons_non_weighted.values()])[0, 1]:.2f}")
        ax[0].set_xlabel(f"{enh_name_1} coverage {coverage_1}")
        ax[1].set_xlabel(f"{enh_name_1} coverage {coverage_1}")
        ax[0].set_ylabel(f"{enh_name_2} coverage {coverage_2}")
        ax[1].set_ylabel(f"{enh_name_2} coverage {coverage_2}")
        plt.tight_layout()
        plt.savefig(f"/home/franciscom/NonCode/temp_figs/.interpolation_comparisson.png")
        plt.show()
        plt.close()

        print(f"Interpolated {enh_name_1} vs {enh_name_2} (weighted)\nR^2 = {np.corrcoef([x[0] for x in inter_commons.values()], [x[1] for x in inter_commons.values()])[0, 1]:.2f}")
        print(f"Interpolated {enh_name_1} vs {enh_name_2} (non-weighted)\nR^2 = {np.corrcoef([x[0] for x in inter_commons_non_weighted.values()], [x[1] for x in inter_commons_non_weighted.values()])[0, 1]:.2f}")


        result_matrix[i, j+i+1] = np.corrcoef([x[0] for x in inter_commons.values()], [x[1] for x in inter_commons.values()])[0, 1]
        result_matrix[j+i+1, i] = np.corrcoef([x[0] for x in inter_commons_non_weighted.values()], [x[1] for x in inter_commons_non_weighted.values()])[0, 1]

    
plt.imshow(result_matrix, cmap='jet', interpolation='none', vmin=0, vmax=1)
plt.colorbar()
plt.xticks(ticks=np.arange(len(ENH_LIST)), labels=[f"{x[1]} ({x[0]})" for x in ENH_LIST], rotation=45, ha='right')
plt.yticks(ticks=np.arange(len(ENH_LIST)), labels=[f"{x[1]} ({x[0]})" for x in ENH_LIST])
#place the text in the middle of the matrix in black color
for i in range(len(ENH_LIST)):
    for j in range(len(ENH_LIST)):
        if i != j:
            plt.text(j, i, f"{result_matrix[i, j]:.2f}", ha='center', va='center', color='black', fontsize=8)
        else:
            plt.text(j, i, "1.00", ha='center', va='center', color='white', fontsize=8)
plt.title("Correlation between interpolated datasets")
plt.tight_layout()
plt.savefig(f"/home/franciscom/NonCode/temp_figs/.interpolation_comparisson_matrix.png")
plt.show()