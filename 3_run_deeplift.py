from data_preprocess import *
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

parser = argparse.ArgumentParser(description='Train a model to predict enhancer activity')

parser.add_argument('--id', type=int, default=None, help='Batch size for training')

args = parser.parse_args()

SAVE_DIR = 'saved_models'
model_id = args.id

if model_id is None:
    model_id = input("Enter the model id: ")


## find model name with corresponding id
possible_models = []
for file in os.listdir(SAVE_DIR):
    if file.startswith(str(model_id)):
        if file.endswith(".pt"):
            possible_models.append(file)

if len(possible_models) == 0:
    print("No model found with that id")
    exit()
if len(possible_models) > 1:
    print("Multiple models found with that id")
    for i, model in enumerate(possible_models):
        print(f"{i}: {model}")

    model_id = input("Enter the model index: ")
    model_name = possible_models[int(model_id)]
else:
    model_name = possible_models[0]

model_name = model_name.split(".")[0]  # remove the .pt, to use the name for the other files

del possible_models     # clean up

print(f"Using model {model_name}")


## get parameters for the model
with open(f"{SAVE_DIR}/{model_name}_params.txt", "r") as f:
    params = f.readlines()

params = [p.strip() for p in params]
params = {p.split(": ")[0]: p.split(": ")[1] for p in params}

ENHANCER_NAME = params["Enhancer"]
LABEL_TITLE = params["Label"]  
BATCH_SIZE = int(params["Batch Size"])
LEARNING_RATE = float(params["Learning Rate"])
N_EPOCHS = int(params["Epochs"])
WEIGHT_DECAY = float(params["Weight Decay"])
# if "Momentum" in params:
#     MOM = float(params["Momentum"])
OPTIM = params["Optimizer"]


LABELS = preset_labels(LABEL_TITLE)
N_CLASSES = len(list(LABELS.values())[0])


# Load the model

model = SignalPredictor1D(num_classes=N_CLASSES)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(f"{SAVE_DIR}/{model_name}.pt"))
model.to(device)

model.eval()


class Wrapper(nn.Module):
    def __init__(self, model):
        super(Wrapper, self).__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)

        out = out[:,1] - out[:,0]
        return out.unsqueeze(1)


model = Wrapper(model)





# load original enhancer
with open(f"original_enhancers.txt", "r") as f:
    enhancer = f.readlines()
    # format is >EnhancerName\n
    #               sequence\n
    # so select the line after the one with the enhancer name
    enhancer = enhancer[enhancer.index(f">{ENHANCER_NAME}\n") + 1].strip()

enhancer_len = len(enhancer)
pads = 250//2 - enhancer_len//2
pads = [pads, 250 - pads - enhancer_len]

enhancer = pad_samples([(enhancer, "MM")], LABELS)[0][0]
enhancer = torch.tensor(enhancer, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)

# get the deep lift shap values
reference_shap_values = deep_lift_shap(model, enhancer, hypothetical=True)


fig, ax = plt.subplots(2, 1, figsize=(len(reference_shap_values[0][0])//15, 6))
plot_logo(reference_shap_values[0].cpu().detach().numpy(), ax=ax[0])
plot_logo(reference_shap_values[0].cpu().detach().numpy() * enhancer[0].cpu().detach().numpy(), ax=ax[1])
plt.savefig(f"explicability_figs/{ENHANCER_NAME}_{model_id}_shap.png")
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


plt.savefig(f"explicability_figs/{ENHANCER_NAME}_{model_id}_heatmap.png")
# plt.show()
plt.close()