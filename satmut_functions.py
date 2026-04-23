import os
import numpy as np
from data_preprocess import one_hot_encode, load_wt, simple_pad
import torch
import matplotlib.pyplot as plt
import tangermeme.plot as tp
import pandas as pd

def saturation_mutagenesis(model, enh_name, device, BATCH_SIZE=256, WT_SEQ=None):
    if WT_SEQ is None:
        WT_SEQ = load_wt(enh_name.strip())
        WT_SEQ = simple_pad(WT_SEQ, 250, jitter=0)

    OH_WT = one_hot_encode(WT_SEQ).T  # Shape (sequence_length, 4)
    saturation_mut_data = [OH_WT]  # Start with the wild-type sequence
    for i in range(len(WT_SEQ)):
        for nuc in ["A", "C", "G", "T"]:
            if WT_SEQ[i] != nuc:
                saturation_mut_data.append(one_hot_encode(WT_SEQ[:i] + nuc + WT_SEQ[i+1:]).T)
            else:
                saturation_mut_data.append(one_hot_encode("N" + WT_SEQ[:i] + WT_SEQ[i+1:]).T)
        saturation_mut_data.append(one_hot_encode( WT_SEQ[:i] + WT_SEQ[i+1:] + "N").T)

    saturation_mut_data = np.array(saturation_mut_data)

    data_loader = torch.utils.data.DataLoader(dataset=saturation_mut_data,
                                                batch_size=BATCH_SIZE,
                                                shuffle=False)

    sat_mut = []
    model.eval()
    model.to(device)

    with torch.no_grad():
        for images in data_loader:
            images = images.to(device).float()
            output = model(images)
            sat_mut.extend(output.cpu().detach().numpy())

    sat_mut = np.array(sat_mut)

    WT_value= sat_mut[0]
    sat_mut = sat_mut[1:]  # Remove the wild-type sequence output

    sat_mut = sat_mut.reshape(len(WT_SEQ), 5).T - WT_value  # Reshape to (4, sequence_length) and center on WT
    deletions_right = sat_mut[4, :] * OH_WT # Deletion effects at the right end
    sat_mut = sat_mut[:4, :]  # Remove the deletion effects at the right end

    deletions_left = sat_mut * OH_WT         # Deletion effects at the left end
    sat_mut = sat_mut * (1 - OH_WT)  # Substiution effects (set to 0 where there are deletions)

    deletion_effects = (deletions_left + deletions_right) / 2   # Average the left and right deletion effects

    return sat_mut, deletion_effects, WT_value

def save_sat_mut(sat_mut, deletion_effects, WT_value, enh_name, WT_SEQ, images_save_dir):
    file_save_loc = os.path.join(images_save_dir, f"{enh_name}_saturation_mutagenesis.tsv")
    data = []
    for i in range(len(WT_SEQ)):
        position = i-24
        WT_nucleotide = WT_SEQ[i]
        substitution_effects = sat_mut[:, i]
        deletion_effect = deletion_effects[:, i].sum()  # Total deletion effect at this position
        data.append([position, WT_nucleotide] + list(substitution_effects) + [deletion_effect])
    df = pd.DataFrame(data, columns=["Position", "WT_Nucleotide", "Substitution_Effect_A", "Substitution_Effect_C", "Substitution_Effect_G", "Substitution_Effect_T", "Deletion_Effect"])
    df.to_csv(file_save_loc, index=False, sep="\t")
    print(f"Saturation mutation effects saved as {file_save_loc}")



def plot_saturation_mutagenesis_logo(sat_mut, deletion_effects, WT_value, enh_name, images_save_dir, model_id, WT_SEQ=None, DPI=150):
    if WT_SEQ is None:
        WT_SEQ = load_wt(enh_name.strip())
        WT_SEQ = simple_pad(WT_SEQ, 250, jitter=0)

    plot_save_loc = os.path.join(images_save_dir, f"{enh_name}_{model_id}_saturation_mutagenesis.png")
    if not os.path.exists(images_save_dir):
        os.makedirs(images_save_dir)

    fig, ax = plt.subplots(3, 1, figsize=(16, 10))
    ax[0].set_xticks(np.arange(10)*25, np.arange(10)*25-24, fontsize=14)
    tp.plot_logo(sat_mut, ax[0])
    ax[0].set_title(f"Saturation Mutation Logo for {enh_name} (Centered on WT)", fontsize=18)
    ax[0].set_xlabel("Position", fontsize=16)
    ax[0].set_ylabel("Substitution Effect", fontsize=16)
    ax[0].set_yticklabels(np.round(ax[0].get_yticks(), 2), fontsize=14)
    ax[2].plot(np.arange(6)*50, np.arange(6)*0, color='red', linestyle='--', linewidth=1)
    ax[2].plot(np.sum(deletion_effects * one_hot_encode(WT_SEQ).T, axis=0), color='black', linewidth=2)
    ax[2].set_title(f"Deletion Effects for {enh_name}", fontsize=18)
    ax[2].set_xlabel("Position", fontsize=16)
    ax[2].set_ylabel("Deletion Effect", fontsize=16)
    ax[2].grid(True)
    ax[2].set_xticks(np.arange(10)*25, np.arange(10)*25-24, fontsize=14)
    ax[2].set_xlim(0, 250)
    ax[2].set_yticklabels(np.round(ax[2].get_yticks(), 2), fontsize=14)
    tp.plot_logo(deletion_effects, ax[1])
    ax[1].set_title(f"Deletion Effects for {enh_name}", fontsize=18)
    ax[1].set_xlabel("Position", fontsize=16)
    ax[1].set_ylabel("Deletion Effect", fontsize=16)
    ax[1].set_xticks(np.arange(10)*25, np.arange(10)*25-24, fontsize=14)
    ax[1].set_yticklabels(np.round(ax[1].get_yticks(), 2), fontsize=14)
    plt.tight_layout()
    plt.savefig(plot_save_loc, dpi=DPI)
    print(f"Saturation mutation logo saved as {plot_save_loc}")
    plt.close()

def plot_deeplift(shap_values, fig_save_loc, WT_enhancer=None, enh_name=None, DPI=150):
    if WT_enhancer is None:
        if enh_name is None:
            raise ValueError("Either WT_enhancer or enh_name must be provided")
        else:
            WT_enhancer = load_wt(enh_name.strip())
            WT_enhancer = simple_pad(WT_enhancer)
            WT_enhancer = one_hot_encode(WT_enhancer).T

    WT_shap = shap_values * WT_enhancer  # Get the SHAP values for the wild-type enhancer

    fig, ax = plt.subplots(2, 1, figsize=(16, 6))
    tp.plot_logo(shap_values, ax[0])
    ax[0].set_title(f"DeepLIFT SHAP Values for {enh_name}", fontsize=18)
    ax[0].set_xlabel("Position", fontsize=16)
    ax[0].set_ylabel("SHAP Value", fontsize=16)
    ax[0].set_xticks(np.arange(10)*25, np.arange(10)*25-24, fontsize=14)
    ax[0].set_yticklabels(np.round(ax[0].get_yticks(), 2), fontsize=14)
    tp.plot_logo(WT_shap, ax[1])
    ax[1].set_title(f"WT Enhancer Sequence for {enh_name}", fontsize=18)
    ax[1].set_xlabel("Position", fontsize=16)
    ax[1].set_ylabel("SHAP Value", fontsize=16)
    ax[1].set_xticks(np.arange(10)*25, np.arange(10)*25-24, fontsize=14)
    ax[1].set_yticklabels(np.round(ax[1].get_yticks(), 2), fontsize=14)
    plt.tight_layout()
    plt.savefig(fig_save_loc, dpi=DPI)
    print(f"DeepLIFT SHAP logo saved as {fig_save_loc}")
    plt.close()
