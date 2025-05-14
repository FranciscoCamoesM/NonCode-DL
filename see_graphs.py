import matplotlib.pyplot as plt
import numpy as np
import os

import argparse


parser = argparse.ArgumentParser(description='Plot the distribution of mutations')
parser.add_argument('--id', type=int, default=0, help='ID of the enhancer')
parser.add_argument('--type', type=str, default='all', help='Type of the plot')


args = parser.parse_args()
ENHANCER_ID = args.id
PLOT_TYPE = args.type


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


if PLOT_TYPE == 'all':
    PLOT_TYPE = ['train_losses', 'val_losses']

elif PLOT_TYPE == 'train_losses':
    PLOT_TYPE = ['train_losses']
elif PLOT_TYPE == 'val_losses':
    PLOT_TYPE = ['val_losses']
elif PLOT_TYPE == 'cm':
    PLOT_TYPE = ['cm']


fig, axs = plt.subplots(len(PLOT_TYPE), 1, figsize=(15, 10))
fig.suptitle(f"{ENHANCER_NAME} - {LABEL_TITLE}", fontsize=20)

# get np data
location = f"{SAVE_DIR}/training_history/"

for i, plot_type in enumerate(PLOT_TYPE):
    data = np.load(f"./{location}/{model_name}_{plot_type}.npy", allow_pickle=True)
    axs[i].plot(data)
    axs[i].set_title("Train Losses")
    axs[i].set_xlabel("Epochs")
    axs[i].set_ylabel("Loss")

# save figure
plt.savefig(f"./temp_figs/{model_name}_{PLOT_TYPE}.png")

print(f"Saved figure to ./temp_figs/{model_name}_{PLOT_TYPE}.png")