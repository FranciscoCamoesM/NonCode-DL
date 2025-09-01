from data_preprocess import get_dataloc, get_interpolated_seq_label_pairs, get_coverage_seq_label_pairs
import numpy as np

enhancers_and_performances = [
                                ["E25B3", 0.46],
                                ["E22P3B4", 0.49],
                                ["E23G1", 0.5],
                                ["E24A3", 0.55],
                                ["E25B2", 0.595],
                                ["E24A3", 0.615],
                                ["E25B3", 0.67],
                                ["E22P3B3", 0.685],
                                ["E22P3B3R3", 0.685],
                                ["E22P3B4", 0.69],
                                ["E24C3", 0.7],
                                ["E22P3B3R3", 0.705],
                                ["E22P3G3", 0.715],
                                ["E22P3G3", 0.715],
                                ["E22P1E6", 0.715],
                                ["E22P3B3R2", 0.725],
                                ["E22P3G3", 0.725],
                                ["E25E102", 0.725],
                                ["E24C3", 0.73],
                                ["E25B2", 0.73],
                                ["E22P3B3", 0.745],
                                ["E22P3G3", 0.745],
                                ["E25E10R3", 0.745],
                                ["E22P3F2", 0.75],
                                ["E23H5", 0.765],
                                ["E22P1E6", 0.765],
                                ["E18C2", 0.77],
                                ["E22P1C1", 0.77],
                                ["E18C2", 0.775],
                                ["E22P1C1", 0.775],
                                ["E25E102", 0.775],
                                ["E22P1E6", 0.785],
                                ["E23H5", 0.79],
                                ["E22P1D10", 0.795],
                                ["E22P1D10", 0.8],
                                ["E22P2F4", 0.805],
                                ["E22P1A3", 0.81],
                                ["E22P3F2", 0.82],
                                ["E22P1F10", 0.825],
                                ["E22P1B925", 0.85],
                                ["E22P1A3", 0.855],
                                ["E22P2F4", 0.885],
                                ["E25E10", 0.89],
                                ["E25E10", 0.905],
                                ["E22P1B925", 0.99],
]


# keep only the best of each enhancer
enhancers_and_performances_dict = {}
for enh_name, performance in enhancers_and_performances:
    if enh_name not in enhancers_and_performances_dict or enhancers_and_performances_dict[enh_name] < performance:
        enhancers_and_performances_dict[enh_name] = performance

# Convert the dictionary back to a list of tuples
enhancers_and_performances = [(enh_name, performance) for enh_name, performance in enhancers_and_performances_dict.items()]



enhancers = [enh[0] for enh in enhancers_and_performances]
images_save_dir = "temp_figs"

enhancer_ratios = {}
for enh_name in enhancers:
    AA = 1
    BB = 0.2
    COVN = 3
    # dataset_from_interpolation_cov1 = get_interpolated_seq_label_pairs(enh_name=enh_name, coverage=1    , local_path=get_dataloc(enh_name), BIN_CELLS=np.array([1, 1, 1, 1]), label_values={"MM": -AA, "NM":-BB, "NP":BB, "PP":AA})
    # dataset_from_interpolation_covN = get_interpolated_seq_label_pairs(enh_name=enh_name, coverage=COVN , local_path=get_dataloc(enh_name), BIN_CELLS=np.array([1, 1, 1, 1]), label_values={"MM": -AA, "NM":-BB, "NP":BB, "PP":AA})

    data_coverage_cov1 = get_coverage_seq_label_pairs(enh_name=enh_name, coverage=1, local_path=get_dataloc(enh_name))
    data_coverage_covN = get_coverage_seq_label_pairs(enh_name=enh_name, coverage=COVN, local_path=get_dataloc(enh_name))

    # print(f"Enhancer Name: {enh_name}")
    # print(f"Data coverage coverage 1: {len(data_coverage_cov1)} sequences")
    # print(f"Data coverage coverage {COVN}: {len(data_coverage_covN)} sequences")
    enhancer_ratios[enh_name] = [len(data_coverage_covN), len(data_coverage_cov1)]



# plot the points with color of the performance
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
for enh_name, performance in enhancers_and_performances:
    ratio = enhancer_ratios[enh_name]
    plt.scatter(ratio[0], ratio[1], label=f"{enh_name} (Performance: {performance})", c=performance, cmap='gnuplot', s=100, vmin=0.5, vmax=1, alpha=0.5)
plt.title("Enhancer Interpolation Ratios")
plt.xlabel("Coverage Ratio (COVN)")
plt.ylabel("Coverage Ratio (COV1)")
# plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.colorbar(label='Performance')
plt.grid(True)
plt.savefig(f"{images_save_dir}/enhancer_interpolation_ratios.png", dpi=150)
print(f"Enhancer interpolation ratios plot saved as {images_save_dir}/enhancer_interpolation_ratios.png")

# plot COVN vs score
plt.figure(figsize=(10, 6))
for enh_name, performance in enhancers_and_performances:
    ratio = enhancer_ratios[enh_name]
    plt.scatter(ratio[0], performance, label=f"{enh_name} (COVN: {ratio[0]})", c=performance, cmap='gnuplot', s=100, vmin=0.5, vmax=1, alpha=0.5)
plt.title("Enhancer Interpolation COVN vs Performance")
plt.xlabel("Coverage Ratio (COVN)")
plt.ylabel("Performance")
# plt.legend()
plt.xscale('log')
plt.colorbar(label='Performance')
plt.grid(True)
plt.savefig(f"{images_save_dir}/enhancer_interpolation_covN_vs_performance.png", dpi=150)
print(f"Enhancer interpolation COVN vs performance plot saved as {images_save_dir}/enhancer_interpolation_covN_vs_performance.png")