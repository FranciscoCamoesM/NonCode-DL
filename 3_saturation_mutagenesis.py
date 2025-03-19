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
parser.add_argument('--w', type=int, default=3, help='Sie of the window to use for windowed mutagenesis. With 8 takes around 2 hours, with 7 around 30 minutes, and with 5 around 2 minutes.')
parser.add_argument('--n_top', type=int, default=25, help='Number of top and bottom kmers to save')

args = parser.parse_args()

SAVE_DIR = 'saved_models'
model_id = args.id
n_top = args.n_top

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
# pads = 250//2 - enhancer_len//2
# pads = [pads, 250 - pads - enhancer_len]

enhancer = pad_samples([(enhancer, "MM")], LABELS)[0][0]
enhancer = torch.tensor(enhancer, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)



# do saturation mutagenesis on the WT enhancer

dataset = []
for i in range(enhancer.shape[2]):
    enhancer_mut = enhancer.clone()
    for j in range(4):
        enhancer_mut[0, :, i] = 0
        enhancer_mut[0, j, i] = 1
        dataset.append(enhancer_mut[0].clone())

loader = DataLoader(dataset, batch_size=32, shuffle=False)

y = []
for X in loader:
    out = model(X.to(device)).cpu().detach().numpy()
    y.extend(list(out))

y = np.array(y)

# reorganize the output to be in the same format as the shap values
y_reshaped = []
for i in range(enhancer.shape[2]):
    y_reshaped.append(y[i*4:(i+1)*4, 0])

y = np.array(y_reshaped)

fig, ax = plt.subplots(2, 1, figsize=(len(y)//15, 6), sharex=True)

ax[0].set_title("Saturation Mutagenesis")
# plot_logo(y.T, ax=ax[2])


# normalize by the WT output
wt_out = model(enhancer.to(device)).cpu().detach().numpy()
print(wt_out)
# y = y - wt_out


ax[0].imshow((y - wt_out).T, cmap="coolwarm", aspect="auto")
ax[0].set_yticks([0, 1, 2, 3], ["A", "C", "G", "T"])


plot_logo((y - wt_out).T, ax=ax[1])

# plot_logo((np.log(y / wt_out)).T, ax=ax[0])


plt.savefig(f"explicability_figs/{ENHANCER_NAME}_{model_id}_saturation.png")
plt.close()




window_size = args.w


def mutate_enhancer(enhancer, i=0, indices=None, dataset=None, window_size=3):
    if dataset is None:
        dataset = []
    if indices is None:
        indices = []
    
    # Base case: when all positions have been set
    if len(indices) == window_size:
        enhancer_mut = enhancer.clone()
        for pos, j in enumerate(indices):
            enhancer_mut[0, :, i + pos] = 0  # Reset column
            enhancer_mut[0, j, i + pos] = 1  # Set specific position
        dataset.append(enhancer_mut[0].clone())
        return
    
    # Recursive case: iterate over 4 possibilities for each position
    for j in range(4):
        mutate_enhancer(enhancer, i, indices + [j], dataset)
    
    return dataset

def gen_kmer(k=3):
    if k == 1:
        return [[i] for i in range(4)]
    else:
        kmers = []
        for i in range(4):
            for kmer in gen_kmer(k-1):
                kmers.append([i] + kmer)
        return kmers

def drag_kmers(enhancer, k=3):
    dataset = []

    possible_kmers = gen_kmer(k)

    for kmer in possible_kmers:
        for pos in range(enhancer.shape[2] - k + 1):
            enhancer_mut = enhancer.clone()
            for i, j in enumerate(kmer):
                enhancer_mut[0, :, pos + i] = 0
                enhancer_mut[0, j, pos + i] = 1
            dataset.append(enhancer_mut[0].clone())
    return dataset
    

possible_kmers = gen_kmer(window_size)

y = []
for kmer in tqdm(possible_kmers):
    batch = []
    for l in range(enhancer.shape[2] - window_size + 1):
        enhancer_mut = enhancer.clone()
        for pos, j in enumerate(kmer):
            enhancer_mut[0, :, l+pos] = 0  # Reset column
            enhancer_mut[0, j, l+pos] = 1  # Set specific position
        batch.append(enhancer_mut[0].clone())
    batch = torch.stack(batch)
    out = model(batch.to(device)).cpu().detach().numpy()
    y.append(list(out))

y_reshaped = np.array(y)[:,:,0].T
print(y_reshaped.shape)

v = np.max(np.abs(y_reshaped - wt_out))

plt.imshow(y_reshaped - wt_out, cmap="coolwarm", aspect="auto", interpolation="none", vmin=-v, vmax=v)
plt.xlabel("Kmer")
plt.ylabel("Position")
plt.xticks(np.arange(4**window_size, step=4**(window_size-1)))
plt.colorbar()
plt.savefig(f"explicability_figs/{ENHANCER_NAME}_{model_id}_{window_size}_kmers.png")
# plt.show()
plt.close()


# save the results to a file
np.save(f"explicability_figs/points/{ENHANCER_NAME}_{model_id}_{window_size}_kmers.npy", y_reshaped)

# sort by sum over L and plot ranked
sums = np.mean(y_reshaped, axis=0)
sorted_indices = np.argsort(sums)
print(f"we are sorting {len(sorted_indices)}")

# save to top and the bottom kmers
def id_to_kmer(id, window_size):
    nucs = ["A", "C", "G", "T"]
    return str.join("", [nucs[int(possible_kmers[id][i])] for i in range(window_size)])

with open(f"explicability_figs/{ENHANCER_NAME}_{model_id}_{window_size}_kmers_sorted.txt", "w") as f:
    f.write(f"Top {n_top} kmers\n")
    for i in sorted_indices[-n_top:][::-1]:
        f.write(f"{id_to_kmer(i, window_size)}\n")
    f.write("\n\n\n")
    f.write(f"Bottom {n_top} kmers\n")
    for i in sorted_indices[:n_top]:
        f.write(f"{id_to_kmer(i, window_size)}\n")


with open(f"explicability_figs/{ENHANCER_NAME}_{model_id}_{window_size}_kmers_values_sorted.txt", "w") as f:
    f.write(F"WT output: {wt_out[0][0]}\n\n")
    f.write(f"Top {n_top} kmers\n")
    for i in sorted_indices[-n_top:][::-1]:
        f.write(f"{id_to_kmer(i, window_size)}: {sums[i]}\n")
    f.write("\n\n\n")
    f.write(f"Bottom {n_top} kmers\n")
    for i in sorted_indices[:n_top]:
        f.write(f"{id_to_kmer(i, window_size)}: {sums[i]}\n")


sorted_sums = sums[sorted_indices]
sorted_y = y_reshaped[:, sorted_indices]

plt.plot(sorted_sums)
plt.xlabel("K-mer")
plt.ylabel("Sum of SHAP values")
# plt.show()
plt.close()

plt.imshow(sorted_y - wt_out, cmap="coolwarm", aspect="auto", interpolation="none", vmin=-v, vmax=v)
plt.xlabel("K-mer")
plt.ylabel("Position")
plt.xticks(np.arange(4**window_size, step=4**(window_size-1)))
plt.colorbar()
plt.savefig(f"explicability_figs/{ENHANCER_NAME}_{model_id}_{window_size}_kmers_sorted.png")
# plt.show()
plt.close()




## do deletions and insertions

# reload the unpadded enhancer
with open(f"original_enhancers.txt", "r") as f:
    enhancer = f.readlines()
    enhancer = enhancer[enhancer.index(f">{ENHANCER_NAME}\n") + 1].strip()



f2d = []

# deletions
for n_delete in range(0, 16):

    data = []
    for l in range((len(enhancer)) - n_delete + 1):
        enhancer_mut = enhancer[:l] + enhancer[l+n_delete:]

        # print(l)
        # print(enhancer)
        # print(enhancer_mut)
        data.append(enhancer_mut)


    data = pad_samples([(d, "MM") for d in data], LABELS)[0]
    data = np.array(data)
    data = torch.tensor(data, dtype=torch.float32).permute(0, 2, 1)


    # run the model
    out = model(data.to(device)).cpu().detach().numpy()


    # print(out.shape)
    
    out = out[:,0] - wt_out[0][0]

    new_line = np.zeros((202))
    new_line[:len(out)] = out
    f2d.append(new_line)



f2d = np.array(f2d)
v = np.max(np.abs(f2d))
plt.imshow(f2d, cmap="coolwarm", aspect="auto", interpolation="none", vmin=-v, vmax=v)
plt.colorbar()
plt.savefig(f"explicability_figs/{ENHANCER_NAME}_{model_id}_deletions.png")
# plt.show()
plt.close()

plt.plot(np.mean(f2d, axis=0))
plt.xlabel("Position")
plt.ylabel("Mean attribution")
plt.savefig(f"explicability_figs/{ENHANCER_NAME}_{model_id}_deletions_mean.png")
# plt.show()
plt.close()

print(f2d.shape)

mean_triangular = []
for n in range(f2d.shape[1]):
    coords = []
    for i in range(n + 1):
            for j in range(n - i, n + 1):

                coords.append((i, j))

    
    v = 0
    c = 0
    for i, j in coords:
        if i < f2d.shape[0] and j < f2d.shape[1]:
            v += f2d[i, j]
            c += 1

    mean_triangular.append(v/c)

plt.plot(mean_triangular)
plt.xlabel("Position")
plt.ylabel("Mean attribution")
plt.savefig(f"explicability_figs/{ENHANCER_NAME}_{model_id}_deletions_mean_triangular.png")
# plt.show()
plt.close()
    
