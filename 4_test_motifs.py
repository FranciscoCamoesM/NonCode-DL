import h5py
filename = "F:\OneDrive\Ambiente de Trabalho\i3s ml\NonCode\modisco\\114324_0307_E25E10_mm_v_all.modisco"
filename = "modisco\\204746_0310_E22P3B3_mm_v_all.modisco" 

with h5py.File(filename, "r") as f:
    # Print all root level object names (aka keys) 
    # these can be group or dataset names 
    print("Keys: %s" % f.keys())
    # get first object name/key; may or may NOT be a group
    a_group_key = list(f.keys())[0]

    # get the object type for a_group_key: usually group or dataset
    print(type(f[a_group_key])) 

    # If a_group_key is a group name, 
    # this gets the object names in the group and returns as a list
    data = list(f[a_group_key])

    # If a_group_key is a dataset name, 
    # this gets the dataset values and returns as a list
    data = list(f[a_group_key])
    # preferred methods to get dataset values:
    ds_obj = f[a_group_key]      # returns as a h5py dataset object
    ds_arr = f[a_group_key][()]




exit()

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


enhancer = pad_samples([(enhancer, "MM")], LABELS)[0][0]
enhancer = torch.tensor(enhancer, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)

# get the deep lift shap values
reference_shap_values = deep_lift_shap(model, enhancer, hypothetical=True)

plot_logo(reference_shap_values[0].cpu().detach().numpy())
plot_heatmap(reference_shap_values[0].cpu().detach().numpy(), cmap="coolwarm")

plot_logo(reference_shap_values[0].cpu().detach().numpy() * enhancer[0].cpu().detach().numpy())
plot_heatmap(reference_shap_values[0].cpu().detach().numpy() * enhancer[0].cpu().detach().numpy(), cmap="coolwarm")

