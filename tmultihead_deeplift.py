from data_preprocess import *
import matplotlib.pyplot as plt
import os

from models import SignalPredictor1D, SignalPredictor_Abstract
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import seaborn as sns


def gen_sampler_weights_multidataset(data, n_classes, n_datasets):
    weights = []
    if n_classes <= 2:
        class_count = []
        for i in range(n_datasets):
            class_count.append([0, 0])
        for _, label, dataset_id in data:
            class_count[dataset_id][int(label)] += 1

        for _, label, dataset_id in data:
            weights.append(1/class_count[dataset_id][int(label)])

    else:
        class_count = []
        for i in range(n_datasets):
            class_count.append([0, 0, 0, 0])
        for _, label, dataset_id in data:
            class_count[dataset_id][np.argmax(label)] += 1

        for _, label, dataset_id in data:
            weights.append(1/class_count[dataset_id][np.argmax(label)])

    return torch.DoubleTensor(weights)


### BEGGINIGN OF THE CODE ###

import argparse
parser = argparse.ArgumentParser(description='Train a neural network to predict enhancer activity')
parser.add_argument('--id', type=int, default=0, help='ID of the run')
parser.add_argument('--enh', type=str, default='Onze17', help='Enhancer name to train on')
parser.add_argument('--label', type=str, default='mm_v_all', help='Label title to train on')
parser.add_argument('--subset', type=float, default=1.0, help='Fraction of the data to use')
parser.add_argument('--bs', type=int, default=64, help='Batch size for training')
parser.add_argument('--lr', type=float, default=2.5e-6, help='Learning rate for training')
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train for')
parser.add_argument('--wd', type=float, default=9e-6, help='Weight decay for the optimizer')
parser.add_argument('--mom', type=float, default=None, help='Momentum for the optimizer')
parser.add_argument('--optim', type=str, default='adam', help='Optimizer to use')
parser.add_argument('--best_val', type=bool, default=True, help='Whether to save the best model based on validation accuracy')
parser.add_argument('--datafolder', type=str, default='Fasta_Pool3', help='Folder where the data is stored')
parser.add_argument('--optuna', type=bool, default=False, help='Whether to use optuna for hyperparameter optimization')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--pad', type=bool, default=True, help='Whether to pad the sequences with cassette or leave it with Ns')
args = parser.parse_args()

PADDING = args.pad
CHR_NAME = args.enh
LABEL_TITLE = args.label
SUBSET = args.subset
BATCH_SIZE = args.bs
LEARNING_RATE = args.lr
N_EPOCHS = args.epochs
WEIGHT_DECAY = args.wd
MOM = args.mom
OPTIM = args.optim
SUBSET = args.subset
BEST_VAL = args.best_val
DATA_FOLDER = args.datafolder
SEED = args.seed

SAVE_DIR = 'saved_models'


def count_multi(dl):
    memory_labels = []
    memory_classes = []
    for i, (data, target, dataset) in enumerate(dl):
        if N_CLASSES <= 2:
            targets = target.numpy()[:, 0] > 0.5
        else:
            np.argmax(target.numpy(), axis=1)
        
        dataset = dataset.numpy()

        memory_classes.extend(dataset)

        memory_labels.extend(targets)

        if i > 10:
            break

    #count
    from collections import Counter
    print(Counter(memory_labels))
    print(Counter(memory_classes))


if CHR_NAME == "ANK":
    DATASETS = [ "E24A3", "E22P1A3", "E25B2", "E25B3", "E25E10"] ## ANK
elif CHR_NAME == "Onze17":
    DATASETS = ["E18C2", "E22P3F2", "E24C3"]
elif CHR_NAME == "ALL":
    DATASETS = ["E25E10", "E24A3", "E22P1A3", "E25B2", "E25B3", "E25E102", "E22P1B925", "E22P1C1", "E23H5", "E22P3B3", "E22P1D10", "E22P3B3", "E22P1H9", "E22P2F4", "E22P3G3", "E22P3G3", "E18C2", "E22P3F2", "E24C3"]

DATASETS = list(set(DATASETS))


TEST_DATASETS = None
LABEL_TYPE = "mm_v_all"
encoding_dict = preset_labels(LABEL_TYPE)

print(encoding_dict)

class MultiDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.dataset_names = list(datasets.keys())
        
        self.seqs = []
        self.labels = []
        self.dataset_ids = []
        for i, dataset in enumerate(self.datasets.values()):
            self.seqs.extend(dataset.sequences)
            self.labels.extend(dataset.labels)
            self.dataset_ids.extend([i] * len(dataset.sequences))
        self.len = len(self.seqs)



    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        label = self.labels[idx]
        dataset = self.dataset_ids[idx]
        # Convert to tensors
        seq = torch.tensor(seq, dtype=torch.float32).permute(1, 0)
        label = torch.tensor(label, dtype=torch.float32)

        return seq, label, dataset

def build_dataset(dataset):
    datasets= {}
    for dataset in DATASETS:
        out = get_seq_label_pairs(enh_name=dataset, local_path = get_dataloc(dataset))
        seqs, labels = out.keys(), out.values()

        # Create a custom dataset class
        datasets[dataset] = EnhancerDataset(seqs, labels, encoding_dict = encoding_dict)

        print(f"Dataset {dataset} has {len(datasets[dataset])} samples.")

    # Create a custom dataset class
    multi_dataset = MultiDataset(datasets)
    del datasets
    return multi_dataset


multi_dataset = build_dataset(DATASETS)
if TEST_DATASETS is not None:
    test_datasets = build_dataset(TEST_DATASETS)
else:
    train_indices, test_indices = train_test_split(range(len(multi_dataset)), test_size=0.2, random_state=SEED)
    train_dataset = torch.utils.data.Subset(multi_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(multi_dataset, test_indices)
    multi_dataset = train_dataset
    print(f"Train dataset has {len(multi_dataset)} samples.")
    print(f"Test dataset has {len(test_dataset)} samples.")


train_indices, val_indices = train_test_split(range(len(multi_dataset)), test_size=0.2, random_state=SEED)
train_dataset = torch.utils.data.Subset(multi_dataset, train_indices)
val_dataset = torch.utils.data.Subset(multi_dataset, val_indices)
print(f"Train dataset has {len(train_dataset)} samples.")
print(f"Validation dataset has {len(val_dataset)} samples.")
N_CLASSES = 1

train_weights = gen_sampler_weights_multidataset(train_dataset, N_CLASSES, len(DATASETS))
val_weights = gen_sampler_weights_multidataset(val_dataset, N_CLASSES, len(DATASETS))
train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_dataset), replacement=True)
val_sampler = torch.utils.data.sampler.WeightedRandomSampler(val_weights, len(val_dataset), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


count_multi(train_loader)
count_multi(test_loader)






# find model with id
chosen_model = []
for filename in os.listdir(SAVE_DIR):
    if filename.startswith(f"{args.id}_") and filename.endswith(".pt"):
        print("Found model: ", filename)
        chosen_model.append(filename)
    
if len(chosen_model) == 0:
    raise ValueError(f"Model not found with id {args.id}")
if len(chosen_model) == 1:
    print("Using model: ", chosen_model[0])
    chosen_model = chosen_model[0]
else:
    print("Multiple models found with id: ", args.id)
    for i, model in enumerate(chosen_model):
        print(f"{i}: {model}")
    model_id = input("Enter the model index: ")
    chosen_model = chosen_model[int(model_id)]
    print("Using model: ", chosen_model)



# Load the model
model_dict = torch.load(f"{SAVE_DIR}/{chosen_model}")
model = SignalPredictor_Abstract(num_datasets = model_dict["deabstract.weight"].shape[0])
model.load_state_dict(model_dict)
model.abstract = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
if N_CLASSES <= 2:
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()
if OPTIM == 'adam':
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
elif OPTIM == 'sgd':
    optimizer = SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOM)
else:
    raise ValueError("Invalid optimizer")


if N_CLASSES <= 2:
    def count_corrects(output, target):
        preds = torch.round(torch.sigmoid(output))
        return torch.sum(preds == target)
else:
    def count_corrects(output, target):
        preds = torch.argmax(output, dim=1)
        return torch.sum(preds == torch.argmax(target, dim=1))



print("Running on: ", device)

# create a confusion matrix
model.eval()
all_preds = []
all_targets = []
all_dataset_ids = []
if N_CLASSES <= 2:
    for i, (data, target, dataset_ids) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        data = data.float()
        target = target.float()
        output = model(data, dataset_ids)
        preds = torch.round(torch.sigmoid(output))
        all_preds.extend(preds.cpu().detach().numpy())
        all_targets.extend(target.cpu().detach().numpy())
        all_dataset_ids.extend(dataset_ids)
else:
    for i, (data, target, dataset_ids) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        data = data.float()
        target = target.float()
        output = model(data, dataset_ids)
        preds = torch.argmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        all_preds.extend(preds.cpu().detach().numpy())
        all_targets.extend(target.cpu().detach().numpy())

def optimize_threshold(preds, labels):
    thresholds = np.arange(-10, 10, 0.1)
    best_threshold = 0.5
    best_f1 = 0.0
    best_signal = 1
    for threshold in thresholds:
        for signal in [1, -1]:
            preds_bin = (signal * preds >= threshold).astype(int)
            f1 = f1_score(labels, preds_bin)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_signal = signal
    return best_threshold, best_signal


import os
import datetime
now = datetime.datetime.now()


plt_title = 'Confusion Matrix of enhancer ' + str(CHR_NAME) + ' on ' + str(LABEL_TITLE)
if SUBSET < 1.0:
    plt_title += f" (Subset: {SUBSET*100}%)"


# do a CM for each dataset
for i in range(len(DATASETS)):
    interesting_preds = []	
    interesting_targets = []
    for j in range(len(all_dataset_ids)):
        if all_dataset_ids[j] == i:
            interesting_preds.append(all_preds[j])
            interesting_targets.append(all_targets[j])
    interesting_preds = np.array(interesting_preds)
    interesting_targets = np.array(interesting_targets)

    if len(interesting_targets) == 0:
        print("No samples in dataset", DATASETS[i])
        continue

    best_threshold, signal = optimize_threshold(interesting_preds, interesting_targets)
    # print("\n\nBest threshold for dataset", DATASETS[i], f": {len(interesting_targets)} samples")
    # print("Best threshold: ", best_threshold)
    # print("Signal: ", signal)
    interesting_preds = (interesting_preds * signal >= best_threshold).astype(int)
    if sum(interesting_preds) == 0 or sum(interesting_preds) == len(interesting_preds):
        print("All predictions are the same, skipping dataset", DATASETS[i])
        continue

    cm = confusion_matrix(interesting_targets, interesting_preds)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]         #normalize the confusion matrix
    # print(cm)

    print(f"{DATASETS[i]} , {cm[0][0]}, {cm[1][1]}")


model.abstract = True
model.eval()

from tangermeme.deep_lift_shap import deep_lift_shap
from tangermeme.utils import one_hot_encode
from tangermeme.plot import plot_logo

if CHR_NAME == "ANK":
    WT_seq = "ACCGCGGCTCCAGGGGCGAGAGGAGCCAGCCCTGTCTCTCTCTCCTGGCCCTGGGGGCGCCCGCCGCGAGGGCGGGGCCGGGATTGTGCTGATTCTGGCCTCCCTGGGCCGGAGGCTCTAGTGGAACTTAAGGCTCCTCCCTGATGGCACCGAGGCGAGGAACTGCCAGCTGTCTGTCTCCTTCCTGCCTTGACCCAGAGC"
elif CHR_NAME == "Onze17":
    WT_seq = "TAGTCTCTGTGATTCAGGTGGATCTGGCTCCCAAAGTGGCACGTGACCCAGGACAGGCCAATCCTCACATTCCACACCGCTGGCCACAGTGATTTGTTCAGGGGTGGGCACATGACCAAACTGGACCAGTGAGACTCAGCCCTGGGACTTTTGCTGAACTATGGGGAAAGAGGCACTTTTTTTCTGGCAGAGTTGTTGAGC"
elif CHR_NAME == "ALL":
    WT_seq = "AAAAGGCAGGATGCCTTCTAGTGCCGCCTTTGGCAGATGTGGACTGAGGAACTGTCTATTCCAGCACTGGGAGGGGCCGGCTGGTGGAGCTGTGGGTCCCTGAGGAAATGGCCAAGGGCCAGGGCAGTCCCTGGGCAGATGCTGCAGGCCTGGGGGCTCCAGAGCCCATCCGAGTGCCAGATGTCCAGCTGTAGCCTGGCC"


WT_seq = one_hot_encode(WT_seq).unsqueeze(0).float()

print(WT_seq.shape)

hyp_shap_values = deep_lift_shap(model, WT_seq, hypothetical=True, device=torch.device("cpu"), n_shuffles=100)
shap_values = hyp_shap_values[0].cpu().detach().numpy() * WT_seq.cpu().detach().numpy()

print(shap_values.shape)


plt.close("all")

fig, ax = plt.subplots(2, 1, figsize=(15, 10))
plot_logo(hyp_shap_values[0], ax=ax[0])
plot_logo(shap_values[0], ax=ax[1])
plt.title('SHAP values for sequence')
plt.savefig(f"temp_figs/multi_shap.png")
plt.show()