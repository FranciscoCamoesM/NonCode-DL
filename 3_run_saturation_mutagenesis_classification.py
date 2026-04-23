from data_preprocess import *
from data_preprocess import simple_pad, one_hot_encode
import matplotlib.pyplot as plt

from models import SignalPredictor1D
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from satmut_functions import saturation_mutagenesis, save_sat_mut, plot_saturation_mutagenesis_logo

from tangermeme.deep_lift_shap import deep_lift_shap
from tangermeme.plot import plot_logo

import argparse


# example way to run the script from command line:
# python 3_run_saturation_mutagenesis_classification.py --id 192161453 --enh E25E102 --label mm_v_all --o "temp_figs_XAI_class" --save_dir "../NonCode/saved_models_final/"

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=int, default=None, help='Batch size for training')
parser.add_argument('--save_dir', type=str, default='saved_models', help='Directory where the model is saved')
parser.add_argument('--enh', type=str, default=None, help='Enhancer name to train on')
parser.add_argument('--label', type=str, default=None, help='Label title to train on')
parser.add_argument('--o', type=str, default='temp_figs', help='Directory where the images are saved')

args = parser.parse_args()

SAVE_DIR = args.save_dir
model_id = args.id
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


# This is called a Wrapper. It's basically a new model that contains the original model, but can be used to modify the input or the output.
# In this case, it is used to take the negative of the output, because during training for loss-of-function classification models, the label 1 meant loss of
# function, meaning that higher output values actually correspond to lower function. By using this wrapper, we can swap the meaning of the output,
# making the interpretation of the XAI more intuitive (higher values correspond to higher function).
class Wrapper(nn.Module):
    def __init__(self, model):
        super(Wrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # runs the original model
        out = self.model(x)

        # returns the negative of the output
        return -out


model = Wrapper(model)


sat_mut, deletion_effects, WT_value = saturation_mutagenesis(model, ENHANCER_NAME, device, BATCH_SIZE=BATCH_SIZE)
plot_saturation_mutagenesis_logo(sat_mut, deletion_effects, WT_value, ENHANCER_NAME, WT_SEQ=None, images_save_dir=images_save_dir)


# save the saturation mutation data to a file
np.save(f"{images_save_dir}/{ENHANCER_NAME}_{model_id}_saturation_mutation.npy", sat_mut + deletion_effects)