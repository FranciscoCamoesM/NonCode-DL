from data_preprocess import get_bin_cells, get_interpolated_seq_label_pairs, get_dataloc
import numpy as np


coverages  = [1, 5, 10, 20, 50, 75, 100]
coverages = coverages[::-1]  # reverse the order for descending coverage
enh_names = ["E22P3B4", "E23H5", "E22P1C1", "E22P2F4", "E22P1D10", "E22P1B925", "E22P3B3", "E24C3", "E25E10", "E22P1A3", "E22P3B3R3", "E22P1F10", "E25E102", "E25E10R3"]



AA = 1
BB = .45/.95
lv =  {"MM": -AA, "NM": -BB, "NP":BB, "PP":AA}

for enh_name in enh_names:
    for coverage in coverages:

        bin_cells = get_bin_cells(enh_name)

        data = get_interpolated_seq_label_pairs(coverage=coverage, label_values=lv, enh_name=enh_name, local_path=get_dataloc(enh_name), BIN_CELLS=bin_cells)
        data = list(data.values())

        if len(data) >= 1000:
            print(f"With coverage of {coverage}, enhancer {enh_name} has {len(data)} interpolated values.")
            break


# # plot histogram
# import matplotlib.pyplot as plt

# plt.figure(figsize=(12, 6))
# plt.hist(data, bins=50, color='blue', alpha=0.7)
# plt.title(f"Histogram of Interpolated Values for {enh_name} at Coverage {coverage}")
# plt.xlabel("Interpolated Values")
# plt.ylabel("Frequency")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(f"temp_figs/.histogram_interpolated_values_{enh_name}_coverage_{coverage}.png")
# plt.show()
