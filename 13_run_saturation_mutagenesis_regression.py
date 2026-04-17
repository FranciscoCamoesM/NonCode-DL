import os
import torch
import numpy as np
import pandas as pd
import tangermeme.plot as tp
import matplotlib.pyplot as plt
from data_preprocess import one_hot_encode, load_wt, simple_pad, preset_labels, load_trained_model, pad_samples
from models import SignalPredictor1D
import argparse



def saturation_mutagenesis(model, enh_name, device, BATCH_SIZE=256, WT_SEQ=None):
    if WT_SEQ is None:
        WT_SEQ = load_wt(enh_name.strip())
        WT_SEQ = simple_pad(WT_SEQ, 250, jitter=JITTER)

    saturation_mut_data = [one_hot_encode(WT_SEQ).T]  # Start with the wild-type sequence
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
    deletions_right = sat_mut[4, :] * one_hot_encode(WT_SEQ).T # Deletion effects at the right end
    sat_mut = sat_mut[:4, :]  # Remove the deletion effects at the right end

    deletions_left = sat_mut * one_hot_encode(WT_SEQ).T         # Deletion effects at the left end
    sat_mut = sat_mut * (1 - one_hot_encode(WT_SEQ).T)  # Substiution effects (set to 0 where there are deletions)

    deletion_effects = (deletions_left + deletions_right) / 2   # Average the left and right deletion effects

    return sat_mut, deletion_effects, WT_value



def plot_saturation_mutagenesis_logo(sat_mut, deletion_effects, WT_value, enh_name, WT_SEQ, images_save_dir, DPI=150):

    plot_save_loc = os.path.join(images_save_dir, f"{enh_name}_saturation_mutagenesis_logo.png")
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

def save_sat_mut(sat_mut, deletion_effects, WT_value, enh, WT_SEQ, images_save_dir):
    file_save_loc = os.path.join(images_save_dir, f"{enh}_saturation_mutagenesis_effects.tsv")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Saturation mutagenesis for enhancer sequences')

    parser.add_argument("--id", type=str, required=True, help="model ID")
    parser.add_argument("--enh", type=str, required=True, help="Enhancer name")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--images_save_dir", type=str, default="temp_figs/", help="Directory to save images")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for processing")
    parser.add_argument("--model_dir", type=str, default="regression_models/", help="Directory containing the model")

    args = parser.parse_args()

    enh = args.enh
    MODEL_DIR = args.model_dir
    device = args.device
    model_id = args.id


    if not torch.cuda.is_available() and device == "cuda":
        print("CUDA is not available, using CPU instead")
    device = torch.device(device if torch.cuda.is_available() else 'cpu')


    enhancer = load_wt(enh.strip())
    LABELS = preset_labels("mm_v_all")
    # enhancer = pad_samples([(enhancer, "MM")], LABELS)[0][0]
    enhancer = simple_pad(enhancer, 250, jitter=0)
    enhancer = one_hot_encode(enhancer).T  # Shape (4, sequence_length)
    enhancer = torch.tensor(enhancer, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)

    model_dir = os.path.join(MODEL_DIR, enh, model_id)
    model = SignalPredictor1D(1)

    # Load model
    model = load_trained_model(enh, model_dir, model, device=device)
    model.to(device)
    model.eval()

    # Load WT sequence
    WT_SEQ = load_wt(enh.strip())
    WT_SEQ = simple_pad(WT_SEQ, 250, jitter=0)

    # run saturation mutagenesis
    with torch.no_grad():
        sat_mut, deletion_effects, WT_value = saturation_mutagenesis(model, enh, device, BATCH_SIZE=args.batch_size, WT_SEQ=WT_SEQ)
    plot_saturation_mutagenesis_logo(sat_mut, deletion_effects, WT_value, enh, WT_SEQ, args.images_save_dir)
    
    # save as dataframe
    save_sat_mut(sat_mut, deletion_effects, WT_value, enh, WT_SEQ, args.images_save_dir)
