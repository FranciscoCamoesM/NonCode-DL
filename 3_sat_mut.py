from data_preprocess import *
from data_preprocess import simple_pad, one_hot_encode
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
parser.add_argument('--w', type=int, default=3, help='Sie of the window to use for windowed mutagenesis. With 8 takes around 2 hours, with 7 around 30 minutes, and with 5 around 2 minutes.')
parser.add_argument('--n_top', type=int, default=250, help='Number of top and bottom kmers to save')
parser.add_argument('--max_n_delete', type=int, default=15, help='Maximum number of deletions to do')
parser.add_argument('--save_dir', type=str, default='saved_models', help='Directory where the model is saved')
parser.add_argument('--enh', type=str, default=None, help='Enhancer name to train on')
parser.add_argument('--label', type=str, default=None, help='Label title to train on')
parser.add_argument('--o', type=str, default='temp_figs', help='Directory where the images are saved')

args = parser.parse_args()

SAVE_DIR = args.save_dir
model_id = args.id
n_top = args.n_top
max_n_delete = args.max_n_delete + 1
images_save_dir = args.o

os.makedirs(images_save_dir, exist_ok=True)

if model_id is None:
    model_id = input("Enter the model id: ")


## find model name with corresponding id
possible_models = []
for file in os.listdir(SAVE_DIR):
    if file.startswith(str(model_id)):
        if file.endswith(".pt") or file.endswith(".pth"):
            possible_models.append(file)

if len(possible_models) < 15:
    print("Found the following models:")
    for i, model in enumerate(possible_models):
        print(f"{i}: {model}")
    print("-"*50, "\n")


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

model_name_ext = model_name.split(".")[1]
model_name = model_name.split(".")[0]  # remove the .pt, to use the name for the other files

del possible_models     # clean up

print(f"Using model {model_name}")


## get parameters for the model

if args.enh is None or args.label is None:
    if os.path.exists(f"{SAVE_DIR}/{model_name}_params.txt"):
        with open(f"{SAVE_DIR}/{model_name}_params.txt", "r") as f:
            params = f.readlines()

        params = [p.strip() for p in params]
        params = {p.split(": ")[0]: p.split(": ")[1] for p in params}

        ENHANCER_NAME = params["Enhancer"]
        LABEL_TITLE = params["Label"]  
        BATCH_SIZE = int(params["Batch Size"])

    else:
        print("No parameters file found for this model")
        params = {}
        ENHANCER_NAME = input("Enter the enhancer name: ").strip()
        LABEL_TITLE = input("Enter the label name: ").strip()
        BATCH_SIZE = int(input("Enter the batch size: ").strip())

else:
    ENHANCER_NAME = args.enh
    LABEL_TITLE = args.label
    BATCH_SIZE = 128


LABELS = preset_labels(LABEL_TITLE)
N_CLASSES = len(list(LABELS.values())[0])


# Load the model

model = SignalPredictor1D(num_classes=N_CLASSES)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(f"{SAVE_DIR}/{model_name}.{model_name_ext}", map_location=device))
model.to(device)

model.eval()


class Wrapper(nn.Module):
    def __init__(self, model):
        super(Wrapper, self).__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)

        # during training for mm_v_p, the label 1 was loss, so we need to take the negative of the output
        return -out


model = Wrapper(model)


# load original enhancer
with open(f"original_enhancers.txt", "r") as f:
    enhancers = f.readlines()
    # format is >EnhancerName\n
    #               sequence\n
    # so select the line after the one with the enhancer name
    WT_SEQ = enhancers[enhancers.index(f">{ENHANCER_NAME}\n") + 1].strip()


WT_SEQ = simple_pad(WT_SEQ, 250)
WT_SEQ = WT_SEQ.upper()

print(f"Using wild-type sequence: {WT_SEQ}.")

saturation_mut_data = [one_hot_encode(WT_SEQ).T]  # Start with the wild-type sequence
for i in range(len(WT_SEQ)):
  for nuc in ["A", "C", "G", "T"]:
    if WT_SEQ[i] != nuc:
        saturation_mut_data.append(one_hot_encode(WT_SEQ[:i] + nuc + WT_SEQ[i+1:]).T)
    else:
        saturation_mut_data.append(one_hot_encode("G" + WT_SEQ[:i] + WT_SEQ[i+1:]).T)

saturation_mut_data = np.array(saturation_mut_data)

data_loader = torch.utils.data.DataLoader(dataset=saturation_mut_data,
                                             batch_size=BATCH_SIZE,
                                             shuffle=False)

sat_mut = []
for images in data_loader:
  images = images.to(device).float()
  output = model(images)
  sat_mut.extend(output.cpu().detach().numpy())

sat_mut = np.array(sat_mut)

WT_value= sat_mut[0]
sat_mut = sat_mut[1:]  # Remove the wild-type sequence output

sat_mut = sat_mut.reshape(len(WT_SEQ), 4).T - WT_value

deletion_effects = sat_mut * one_hot_encode(WT_SEQ).T
sat_mut = sat_mut * (1 - one_hot_encode(WT_SEQ).T)  # Apply the deletion effects to the saturation mutation data


import tangermeme.plot as tp


fig, ax = plt.subplots(3, 1, figsize=(18, 15))
ax[0].set_xticks(np.arange(10)*25, np.arange(10)*25-25)
tp.plot_logo(sat_mut, ax[0])
ax[0].set_title(f"Saturation Mutation Logo for {ENHANCER_NAME} (Centered on WT)")
ax[1].plot(np.arange(6)*50, np.arange(6)*0, color='red', linestyle='--', linewidth=1)
ax[1].plot(np.sum(deletion_effects * one_hot_encode(WT_SEQ).T, axis=0), color='black', linewidth=2)
ax[1].set_title(f"Deletion Effects for {ENHANCER_NAME}")
ax[1].set_xlabel("Position")
ax[1].set_ylabel("Deletion Effect")
ax[1].grid(True)
ax[1].set_xticks(np.arange(10)*25, np.arange(10)*25-25)
ax[1].set_xlim(0, 250)
tp.plot_logo(deletion_effects, ax[2])
ax[2].set_title(f"Deletion Effects for {ENHANCER_NAME}")
ax[2].set_xlabel("Position")
ax[2].set_ylabel("Deletion Effect")
ax[2].grid(True)
ax[2].set_xticks(np.arange(10)*25, np.arange(10)*25-25)
plt.savefig(f"{images_save_dir}/{ENHANCER_NAME}_{model_id}_saturation_mutation_logo.png", dpi=150)
print(f"Saturation mutation logo saved as {images_save_dir}/{ENHANCER_NAME}_{model_id}_saturation_mutation_logo.png")