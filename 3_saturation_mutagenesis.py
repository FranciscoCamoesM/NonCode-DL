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
parser.add_argument('--w', type=int, default=250, help='Window size surrounding the peak to consider for motif analysis.')
parser.add_argument('--n', type=int, default=100000, help='The maximum number of sequences per metacluster to consider for motif analysis.')
parser.add_argument('--z', type=int, default=20, help='The size of the seqlet cores, corresponding to `sliding_window_size`.')
parser.add_argument('--n_data', type=int, default=None, help='The number of sequences to use for motif analysis.')
args = parser.parse_args()

SAVE_DIR = 'saved_models'
model_id = args.id
MODISCO_W = args.w
MODISCO_N = args.n
MODISCO_Z = args.z
N_DATA = args.n_data


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





enhancer = load_wt(ENHANCER_NAME)


enhancer = pad_samples([(enhancer, "MM")], LABELS)[0][0]
enhancer = torch.tensor(enhancer, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)

# get the deep lift shap values
reference_shap_values = deep_lift_shap(model, enhancer, hypothetical=True)


# Load the data
print("Loading data")
dataset = get_seq_label_pairs(enh_name = ENHANCER_NAME)

X = list(dataset.keys())
y = list(dataset.values())

dataset = EnhancerDataset(X, y, LABELS)

if N_DATA == None:
    N_DATA = len(dataset)

loader = DataLoader(dataset, batch_size=N_DATA, shuffle=False)

print("Data loaded")


# implement deeplift

import time
# get the first batch
X, y = next(iter(loader))
# y = torch.tensor(y, dtype=torch.float32).to(device)
X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
data_np = X.cpu().detach().numpy()

#remove non-onehot encoded sequences
mask = np.sum(data_np, axis=(1, 2)) == len(data_np[0][0])
data_np = data_np[mask]
X = X[mask]

# get the deep lift shap values
start = time.time()
hyp_scores = deep_lift_shap(model, X, hypothetical=True)
print("Calculated deep lift shap values")
print("Time taken: ", time.time() - start)


import modiscolite

print("Saving data")
print(hyp_scores.shape, data_np.shape)
np.save("./modisco/temp_files/{}.hyp_scores.npy".format(model_name), hyp_scores)
np.save("./modisco/temp_files/{}.onehot.npy".format(model_name), data_np)

print("Running modisco")
# os.system('python "F:/Users/frank/anaconda3/envs/DL_torch/Scripts/modisco" motifs -s ./modisco/temp_files/{}.onehot.npy -a ./modisco/temp_files/{}.hyp_scores.npy -n 10000 -w 250 -o ./modisco/{}.modisco'.format(model_name, model_name, model_name))
# os.system('python "F:/Users/frank/anaconda3/envs/DL_torch/Scripts/modisco" report -i ./modisco/{}.modisco -o ./modisco/{}.report'.format(model_name, model_name))
os.system('modisco motifs -s ./modisco/temp_files/{}.onehot.npy -a ./modisco/temp_files/{}.hyp_scores.npy -n {} -w {} -z {} -o ./modisco/{}.modisco'.format(model_name, model_name, MODISCO_N, MODISCO_W, MODISCO_Z, model_name))
os.system('modisco report -i ./modisco/{}.modisco -o ./modisco/{}.report'.format(model_name, model_name))






print("Cleaning up")
# # for Linux
# os.system("rm ./modisco/temp_files/{}.hyp_scores.npy".format(model_name))
# os.system("rm ./modisco/temp_files/{}.onehot.npy".format(model_name))
# for Windows
os.system("del ./modisco/temp_files/{}.hyp_scores.npy".format(model_name))
os.system("del ./modisco/temp_files/{}.onehot.npy".format(model_name))


print("Done")