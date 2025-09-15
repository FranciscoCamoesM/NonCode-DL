import os
from data_preprocess import load_wt, pad_samples, preset_labels, get_dataloc, get_seq_label_pairs, get_coverage_seq_label_pairs, EnhancerDataset
from data_preprocess import load_trained_model


import matplotlib.pyplot as plt

from models import SignalPredictor1D
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from tangermeme.deep_lift_shap import deep_lift_shap
from tangermeme.plot import plot_logo, plot_heatmap

import argparse

parser = argparse.ArgumentParser(description='Interpolate DeepLIFT SHAP values for enhancer sequences')


parser.add_argument("--id", type=str, required=True, help="model ID")
parser.add_argument("--enh", type=str, required=True, help="Enhancer name")
parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
parser.add_argument("--images_save_dir", type=str, default="temp_figs/", help="Directory to save images")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size for processing")
parser.add_argument("--model_dir", type=str, default="regression_models/", help="Directory containing the model")
parser.add_argument('--n_refs', type=int, default=100, help='Number of references to use for DeepLift SHAP')

args = parser.parse_args()

SAVE_DIR = args.model_dir
N_REFS = args.n_refs
model_id = args.id
output_dir = args.images_save_dir
enh = args.enh

if model_id is None:
    model_id = input("Enter the model id: ")


LABELS = preset_labels("mm_v_all")
N_CLASSES = len(list(LABELS.values())[0])


# Load the model

if args.device == "cuda":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("CUDA is not available, using CPU instead")
elif args.device == "cpu":
    device = torch.device('cpu')


model_dir = os.path.join(SAVE_DIR, enh, model_id)
model = SignalPredictor1D(1)

# Load model
model = load_trained_model(enh, model_dir, model, device=device)
model.to(device)
model.eval()



# load original enhancer
enhancer = load_wt(enh.strip())

enhancer_len = len(enhancer)
pads = 250//2 - enhancer_len//2
pads = [pads, 250 - pads - enhancer_len]

enhancer = pad_samples([(enhancer, "MM")], LABELS)[0][0]
enhancer = torch.tensor(enhancer, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)


# load training data
dataloc = get_dataloc(enh)
X = []
y = []
for i in range(-5, 6):
    if enh != "E25E10R3":
        dataset = get_seq_label_pairs(enh_name = enh, local_path = dataloc, jiggle=i)
    else:
        dataset = get_coverage_seq_label_pairs(enh_name = enh, local_path = dataloc, jiggle=i, coverage=10)
    indexes = np.random.choice(len(dataset), N_REFS, replace=False)
    X.extend(np.array(list(dataset.keys()))[indexes])
    y.extend(np.array(list(dataset.values()))[indexes])
    print(f"Loaded {len(X)} samples from {enh} with jiggle {i}")
dataset = EnhancerDataset(X, y, LABELS)
print(f"Dataset_object created with {len(dataset)} samples")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=min(N_REFS, 256), shuffle=True, num_workers=4)

print(f"References loaded")

references = []
for x, y in dataloader:
    references.extend(x.float().detach().cpu().numpy().tolist())
    print(f"References shape: {len(references)}")
    if len(references) >= N_REFS:
        references = references[:N_REFS]
        references = torch.tensor(references, dtype=torch.float32)
        break


references = references.unsqueeze(0)
references = references.repeat(enhancer.shape[0], 1, 1, 1)



print(f"Input shape: {enhancer.shape}, Reference shape: {references.shape}")


# get the deep lift shap values
# from tangermeme.utils import characters
# for seq in references[0]:
#     print(characters(seq))
reference_shap_values = deep_lift_shap(model, enhancer, hypothetical=True, references=references, device=device)



fig, ax = plt.subplots(2, 1, figsize=(len(reference_shap_values[0][0])//15, 6))
plot_logo(reference_shap_values[0].cpu().detach().numpy(), ax=ax[0])
plot_logo(reference_shap_values[0].cpu().detach().numpy() * enhancer[0].cpu().detach().numpy(), ax=ax[1])
xticks = np.arange(0, reference_shap_values.shape[2], step=25)
ax[0].set_xticks(xticks, xticks - 25)
ax[1].set_xticks(xticks, xticks - 25)
plt.savefig(f"{output_dir}/{enh}_{model_id}_deeplift_shap.png")
plt.close()




fig, ax = plt.subplots(3, 1, figsize=(len(reference_shap_values[0][0])//15, 8), sharex=True)

# Compute the signal
wt_signal = reference_shap_values[0].cpu().detach().numpy() * enhancer[0].cpu().detach().numpy()
wt_signal = np.sum(wt_signal, axis=0)

# Plot the signal
# fill the padded area with black
ax[0].fill_between([0, pads[0]], min(wt_signal), max(wt_signal), color="black", alpha=0.15)
ax[0].fill_between([250 - pads[1], 250], min(wt_signal), max(wt_signal), color="black", alpha=0.15)


ax[0].plot([0, len(wt_signal)], [0, 0], color="r", alpha=0.4)
ax[0].plot(wt_signal)
ax[0].set_title("SHAP Values of WT enhancer")

# Plot the SHAP values
v = np.max(np.abs(reference_shap_values[0].cpu().detach().numpy()))
im = ax[1].imshow(reference_shap_values[0].cpu().detach().numpy(), cmap="coolwarm", aspect="auto", vmin=-v, vmax=v)
ax[1].set_yticks([0, 1, 2, 3], ["A", "C", "G", "T"])
ax[1].set_title("SHAP Values")


cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.6, pad=0.02)
cbar.set_label("SHAP Value Intensity")


att_energy = reference_shap_values[0].cpu().detach().numpy() ** 2
att_energy = np.sum(att_energy, axis=0)**0.5
ax[2].plot(att_energy)
ax[2].set_title("Euclidean norm of SHAP values")
ax[2].fill_between([0, pads[0]], min(att_energy), max(att_energy), color="black", alpha=0.15)
ax[2].fill_between([250 - pads[1], 250], min(att_energy), max(att_energy), color="black", alpha=0.15)
ax[2].set_xlabel("Position")


plt.savefig(f"{output_dir}/{enh}_{model_id}_deeplift_heatmap.png")
# plt.show()
plt.close()

print(f"Saved figures to {output_dir}/{enh}_{model_id}_deeplift_shap.png and {output_dir}/{enh}_{model_id}_deeplift_heatmap.png")

# save the shap values to a numpy file
np.save(f"{output_dir}/{enh}_{model_id}_deeplift_shap.npy", reference_shap_values[0].cpu().detach().numpy())
print(f"Saved SHAP values to {output_dir}/{enh}_{model_id}_deeplift_shap.npy")

print("All done!")