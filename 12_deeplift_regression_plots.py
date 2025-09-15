import os
import matplotlib.pyplot as plt
from tangermeme.plot import plot_logo
import numpy as np
from data_preprocess import load_wt, simple_pad, one_hot_encode

output_dir = "explicability_final_reg/deeplift"

enh_to_plot = []

enh_to_plot.append(["E25E10", "2025-07-19_03-10-32"])
enh_to_plot.append(["E22P1A3", "2025-07-19_05-26-53"])
enh_to_plot.append(["E25E102", "2025-07-18_15-17-32"])
enh_to_plot.append(["E25E10R3", "2025-07-18_17-22-54"])

enh_to_plot.append(["E22P1D10", "2025-07-19_15-27-12"])
enh_to_plot.append(["E22P1F10", "2025-07-18_12-55-41"])
enh_to_plot.append(["E23H5", "2025-07-20_09-56-28"])


WT_seq = load_wt(enh_to_plot[0][0])
WT_seq = simple_pad(WT_seq)
WT_seq = one_hot_encode(WT_seq).T


values = []
for enh, model_id in enh_to_plot:
    try:
        atts = np.load(f"{output_dir}/{enh}_{model_id}_deeplift_shap.npy")
        fig, axs = plt.subplots(2, 1, figsize=(22, 8))
        plot_logo(atts, ax=axs[0])
        axs[0].set_xticks(np.arange(0, 251, 25))
        axs[0].set_xticklabels(np.arange(0, 251, 25) - 25, fontsize=14)
        axs[0].set_yticklabels(np.round(axs[0].get_yticks(), 2), fontsize=14)
        axs[0].set_xlabel("Position (bp)", fontsize=16)
        axs[0].set_ylabel("Importance score", fontsize=16)
        values.append(atts * WT_seq)
        plot_logo(values[-1], ax=axs[1])
        axs[1].set_xticks(np.arange(0, 251, 25))
        axs[1].set_xticklabels(np.arange(0, 251, 25) - 25, fontsize=14)
        axs[1].set_yticklabels(np.round(axs[1].get_yticks(), 2), fontsize=14)
        axs[1].set_xlabel("Position (bp)", fontsize=16)
        axs[1].set_ylabel("Importance score", fontsize=16)
        plt.savefig(f"{output_dir}/{enh}_{model_id}_deeplift_shap.png", dpi=200)
        plt.tight_layout()
        plt.close()
    except FileNotFoundError:
        print(f"File not found: {output_dir}/{enh}_{model_id}_deeplift_shap.npy")
        exit(1)


fig, axs = plt.subplots(1+len(values), 1, figsize=(22, len(values) * 8 + 12))

for i, (enh, model_id) in enumerate(enh_to_plot):
    plot_logo(values[i], ax=axs[i])
    axs[i].set_xticks(np.arange(0, 251, 25))
    axs[i].set_xticklabels(np.arange(0, 251, 25) - 25, fontsize=14)
    y_ticks = axs[i].get_yticks()
    y_ticks = np.round(y_ticks, 2)
    axs[i].set_yticklabels(y_ticks, fontsize=14)
    axs[i].set_xlabel("Position (bp)", fontsize=16)
    axs[i].set_title(f"{enh} {model_id}", fontsize=20)

max_values = np.max(values, axis=(1, 2))
average_values = np.mean(values/ max_values[:, None, None], axis=0)
plot_logo(average_values, ax=axs[-1])
axs[-1].set_title("Average", fontsize=20)
axs[-1].set_xticks(np.arange(0, 251, 25))
axs[-1].set_xticklabels(np.arange(0, 251, 25) - 25, fontsize=14)
axs[-1].set_xlabel("Position (bp)", fontsize=16)
axs[-1].set_yticklabels(np.round(axs[-1].get_yticks(), 2), fontsize=14)


plt.tight_layout()
plt.savefig("explicability_final_reg/deeplift_plot.png", dpi=200)

print("All done!")