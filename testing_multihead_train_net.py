from data_preprocess import *
import matplotlib.pyplot as plt

from models import SignalPredictor1D, SignalPredictor_Abstract
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm


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
else:
    DATASETS = [CHR_NAME]
    print("#"*20)
    print("Warning: Only one dataset selected. This is not recommended for training a model.")
    print("#"*20, "\n")

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



# Train the model
model = SignalPredictor_Abstract(num_datasets = len(DATASETS))
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


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_model = None
    best_acc = 0.0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    with tqdm(total=num_epochs) as pbar:
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            running_corrects = 0
            running_total = 0
            for i, (data, target, dataset_ids) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                data = data.float()
                target = target.float()

                optimizer.zero_grad()
                output = model(data, dataset_ids)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                running_corrects += count_corrects(output, target)
                running_total += len(target)
            train_losses.append(train_loss / len(train_loader))
            train_accs.append(running_corrects.item() / running_total)

            model.eval()
            val_loss = 0.0
            running_corrects = 0
            running_total = 0
            for i, (data, target, dataset_ids) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                data = data.float()
                target = target.float()
                output = model(data, dataset_ids)
                loss = criterion(output, target)
                val_loss += loss.item()
                running_corrects += count_corrects(output, target)
                running_total += len(target)
            val_losses.append(val_loss / len(val_loader))
            val_accs.append(running_corrects.item() / running_total)
            if val_accs[-1] > best_acc:
                best_acc = val_accs[-1]
                best_model = model.state_dict()

            # print(f'Epoch {epoch}, Train Loss: {train_losses[-1]}, Test Loss: {val_losses[-1]}, Train Acc: {train_accs[-1]}, Test Acc: {val_accs[-1]} {"<-- new best!" if val_accs[-1] > best_acc else ""}')
            pbar.set_postfix({'train_acc': train_accs[-1], 'val_acc': val_accs[-1], 'train_loss': train_losses[-1], 'val_loss': val_losses[-1]})
            pbar.update(1)
            pbar.set_description(f'Epoch {epoch+1}/{num_epochs}')
            pbar.refresh()

    if BEST_VAL:
        model.load_state_dict(best_model)

    return train_losses, val_losses

train_losses, val_losses = train(model, train_loader, val_loader, criterion, optimizer, num_epochs=N_EPOCHS)

if not args.optuna:
    fig, ax = plt.subplots( 1, 2)
    ax[0].plot(train_losses, label='Train Loss')
    ax[1].plot(val_losses, label='Test Loss', c = 'orange')
    ax[0].set_xlabel('Epoch')
    ax[1].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[1].set_ylabel('Loss')
    ax[0].legend()
    ax[1].legend()

    plt.show()


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

from sklearn.metrics import confusion_matrix
import seaborn as sns

print(all_preds[:5])
print(all_targets[:5])

cm = confusion_matrix(all_targets, all_preds)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]         #normalize the confusion matrix

if not args.optuna:
    sns.heatmap(cm, annot=True)
    plt.set_cmap('gnuplot')
    plt.vmin = 0
    plt.vmax = 1
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix of enhancer ' + str(CHR_NAME) + ' on ' + str(LABEL_TITLE))
    plt.show()


# Save the model
import os
import datetime
now = datetime.datetime.now()

model_name = f"5{now.strftime('%H%M%S_%m%d')}_MULTILABEL_{CHR_NAME}_{LABEL_TITLE}"

if not args.optuna:
    # save model, confustion matrix, training history, and parameters all in different files
    torch.save(model.state_dict(), f"{SAVE_DIR}/{model_name}.pt")
    np.save(f"{SAVE_DIR}/training_history/{model_name}_cm.npy", cm)
    np.save(f"{SAVE_DIR}/training_history/{model_name}_train_losses.npy", train_losses)
    np.save(f"{SAVE_DIR}/training_history/{model_name}_val_losses.npy", val_losses)

plt_title = 'Confusion Matrix of enhancer ' + str(CHR_NAME) + ' on ' + str(LABEL_TITLE)
if SUBSET < 1.0:
    plt_title += f" (Subset: {SUBSET*100}%)"

plt.figure()
sns.heatmap(cm, annot=True)
plt.set_cmap('gnuplot')
plt.vmin = 0
plt.vmax = 1
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(plt_title)

if not args.optuna:
    plt.savefig(f"{SAVE_DIR}/plots/{model_name}_cm.png")

    with open(f"{SAVE_DIR}/{model_name}_params.txt", 'w') as f:
        f.write(f"Type: Multilable\n")
        f.write(f"Enhancer: {CHR_NAME}\n")
        f.write(f"Clones: {DATASETS}\n")
        f.write(f"Label: {LABEL_TITLE}\n")
        f.write(f"Padding: {PADDING}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Epochs: {N_EPOCHS}\n")
        f.write(f"Subset: {SUBSET} (number of training samples: {len(train_dataset)})\n")
        f.write(f"Weight Decay: {WEIGHT_DECAY}\n")
        if MOM is not None:
            f.write(f"Momentum: {MOM}\n")
        f.write(f"Optimizer: {OPTIM}\n")


print("Model saved as: ", model_name)
print("Model saved in: ", SAVE_DIR)

print(cm)
with open(f"{SAVE_DIR}/{model_name}_multi_cm.txt", 'w') as f:
    f.write(f"Confusion Matrix of enhancer {CHR_NAME} on {LABEL_TITLE}\n")
    f.write(f"Data from clones: {DATASETS}\n")
    f.write(f"Confusion Matrix:\n")
    f.write(f"{cm}\n\n")

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

        cm = confusion_matrix(interesting_targets, interesting_preds)
        print("Confusion Matrix for dataset", DATASETS[i], f": {len(interesting_targets)} samples")
        f.write(f"Confusion Matrix for dataset {DATASETS[i]}: {len(interesting_targets)} samples\n")

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]         #normalize the confusion matrix
        f.write(f"{cm}\n\n")
        print(cm)


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
    WT_seq = "TAGTCTCTGTGATTCAGGTGGATCTGGCTCCCAAAGTGGCACGTGACCCAGGACAGGCCAATCCTCACATTCCACACCGCTGGCCACAGTGATTTGTTCAGGGGTGGGCACATGACCAAACTGGACCAGTGAGACTCAGCCCTGGGACTTTTGCTGAACTATGGGGAAAGAGGCACTTTTTTTCTGGCAGAGTTGTTGAGC"


WT_seq = one_hot_encode(WT_seq).unsqueeze(0).float()

print(WT_seq.shape)

hyp_shap_values = deep_lift_shap(model, WT_seq, hypothetical=True, device=torch.device("cpu"))
shap_values = hyp_shap_values[0].cpu().detach().numpy() * WT_seq.cpu().detach().numpy()

print(shap_values.shape)


plt.close("all")

# plt.imshow(shap_values[0].cpu().detach().numpy(), cmap='hot', interpolation='none', aspect='auto')
# plt.colorbar()
# plt.title('SHAP values for sequence')
# plt.savefig(f"temp_figs/multi_shap.png")

fig, ax = plt.subplots(2, 1, figsize=(15, 10))
plot_logo(hyp_shap_values[0], ax=ax[0])
plot_logo(shap_values[0], ax=ax[1])
plt.title('SHAP values for sequence')
plt.savefig(f"temp_figs/multi_shap_{model_name}.png")
plt.show()


## compute confusion matrices for all datasets

# get shape of subplot
num_rows = int(np.ceil(len(DATASETS)**0.5))
num_cols = int(np.ceil(len(DATASETS) / num_rows))
fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 20))
axs = axs.flatten()

for i in range(len(DATASETS)):

    interesting_preds = []	
    interesting_targets = []
    for j in range(len(all_dataset_ids)):
        if all_dataset_ids[j] == i:
            interesting_preds.append(all_preds[j])
            interesting_targets.append(all_targets[j])
    interesting_preds = np.array(interesting_preds)
    interesting_targets = np.array(interesting_targets)

    cm = confusion_matrix(interesting_targets, interesting_preds)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]         #normalize the confusion matrix
    # sns.heatmap(cm, annot=True, ax=axs[i], cbar=False, cmap='gnuplot', vmin=0, vmax=1)
    axs[i].imshow(cm, cmap='gnuplot', vmin=0, vmax=1)
    # write the values in the confusion matrix
    for j in range(len(cm)):
        for k in range(len(cm)):
            axs[i].text(k, j, f"{cm[j][k]:.2f}", ha='center', va='center', color='black')
    axs[i].set_title(DATASETS[i])
    axs[i].set_xlabel('Predicted')
    axs[i].set_ylabel('True')
    axs[i].set_xticklabels([0, 1], rotation=0)
    axs[i].set_yticklabels([0, 1], rotation=0)  

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/plots/{model_name}_cm_datasets.png")
plt.show()