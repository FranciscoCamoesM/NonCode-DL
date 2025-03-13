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


fig, ax = plt.subplots(2, 1, figsize=(len(reference_shap_values[0][0])//15, 6))
plot_logo(reference_shap_values[0].cpu().detach().numpy(), ax=ax[0])
plot_logo(reference_shap_values[0].cpu().detach().numpy() * enhancer[0].cpu().detach().numpy(), ax=ax[1])
plt.savefig(f"explicability_figs/{ENHANCER_NAME}_{model_id}_shap.png")

fig, ax = plt.subplots(2, 1, figsize=(len(reference_shap_values[0][0])//15, 6))
plot_heatmap(reference_shap_values[0].cpu().detach().numpy() * enhancer[0].cpu().detach().numpy(), cmap="coolwarm", ax=ax[0])
plot_heatmap(reference_shap_values[0].cpu().detach().numpy(), cmap="coolwarm", ax=ax[1])
plt.savefig(f"explicability_figs/{ENHANCER_NAME}_{model_id}_heatmap.png")


# do saturation mutagenesis on the WT enhancer

dataset = []
for i in range(enhancer.shape[2]):
    enhancer_mut = enhancer.clone()
    enhancer_mut[0, :, i] = 0
    for j in range(4):
        enhancer_mut[0, j, i] = 1
        dataset.append(enhancer_mut[0])

loader = DataLoader(dataset, batch_size=32, shuffle=False)

y = []
for X in loader:
    out = model(X.to(device)).cpu().detach().numpy()
    y.extend(list(out))

y = np.array(y)
# reorganize the output to be in the same format as the shap values
y_reshaped = []
for i in range(enhancer.shape[2]):
    y_reshaped.append(y[i*4:(i+1)*4])

y = np.array(y_reshaped)


plt.close()

fig, ax = plt.subplots(1, 1, figsize=(len(reference_shap_values[0][0])//15, 6))
plt.imshow(y[0].T, cmap="coolwarm", aspect="auto")
plt.colorbar()
plt.savefig(f"explicability_figs/{ENHANCER_NAME}_{model_id}_saturation.png")









exit()

# Load the data
print("Loading data")
dataset = get_seq_label_pairs(enh_name = ENHANCER_NAME)

X = list(dataset.keys())
y = list(dataset.values())

dataset = EnhancerDataset(X, y, LABELS)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Data loaded")


# implement deeplift

# get the first batch
X, y = next(iter(loader))
X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)[:20]
# y = torch.tensor(y, dtype=torch.float32).to(device)

print(X.shape, enhancer.shape)
# get the deep lift shap values
shap_values = deep_lift_shap(model, X, hypothetical=True)


for i, v in enumerate(shap_values):
    fig, ax = plt.subplots(2, 1, figsize=(len(v[0])//15, 6))
    plot_logo(v.cpu().detach().numpy(), ax=ax[0])
    plot_logo((v.cpu().detach().numpy()) * X[i].cpu().detach().numpy(), ax=ax[1])
    plt.show()
    # plot_heatmap(v.cpu().detach().numpy() - (reference_shap_values[0].cpu().detach().numpy()), cmap="coolwarm")
