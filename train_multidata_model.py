import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from models import SignalPredictor_Abstract, SignalPredictor1D
from data_preprocess import *


DATASETS = ["E25E10", "E24A3", "E22P1A3", "E25B2", "E25B3"]

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
        seq = torch.tensor(seq, dtype=torch.float32)
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
    train_indices, test_indices = train_test_split(range(len(multi_dataset)), test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.Subset(multi_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(multi_dataset, test_indices)
    multi_dataset = train_dataset
    print(f"Train dataset has {len(multi_dataset)} samples.")
    print(f"Test dataset has {len(test_dataset)} samples.")


train_indices, val_indices = train_test_split(range(len(multi_dataset)), test_size=0.2, random_state=42)
train_dataset = torch.utils.data.Subset(multi_dataset, train_indices)
val_dataset = torch.utils.data.Subset(multi_dataset, val_indices)
print(f"Train dataset has {len(train_dataset)} samples.")
print(f"Validation dataset has {len(val_dataset)} samples.")



model = SignalPredictor_Abstract(num_datasets=len(DATASETS))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()
criterion = nn.BCEWithLogitsLoss()
LR = 2.5e-6
WD = 9e-6
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
batch_size = 64
num_epochs = 400


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


##### TESTING WITH TRADITIONAL DATASET #####
SUBSET = 1.0
ENHANCER_NAME = DATASETS[0]
DATA_FOLDER = get_dataloc(ENHANCER_NAME)
SEED = 42
PADDING = True
LABELS = preset_labels(LABEL_TYPE)

# Load the data
dataset = get_seq_label_pairs(enh_name = ENHANCER_NAME, local_path = DATA_FOLDER)

# Split the data
X = list(dataset.keys())
y = list(dataset.values())


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED)
del X_temp, y_temp


import random
random.seed(SEED)
# reduce the number of samples in the training set based on subset
if SUBSET < 1.0:
    sample_points = random.sample(range(len(X_train)), int(len(X_train) * SUBSET))
    X_train = [X_train[i] for i in sample_points]
    y_train = [y_train[i] for i in sample_points]


# Create a data loader
train_dataset = EnhancerDataset(X_train, y_train, LABELS, PADDING = PADDING)
val_dataset = EnhancerDataset(X_val, y_val, LABELS, PADDING = PADDING)
test_dataset = EnhancerDataset(X_test, y_test, LABELS, PADDING = PADDING)

###########









for epoch in range(num_epochs):
    running_loss = 0.0
    best_loss = 1e10
    best_model = None

    # Training
    model.train()
    for i, (inputs, labels, dataset_ids) in enumerate(train_loader):
        inputs = inputs.permute(0, 2, 1).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs, dataset_ids)
        # outputs = model(inputs)
        loss = criterion(outputs, labels)
        # for l, o in zip(labels, outputs):
        #     print(o, l)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Validation
    model.eval()
    running_loss = 0.0
    for i, (inputs, labels, dataset_ids) in enumerate(val_loader):
        inputs = inputs.permute(0, 2, 1).to(device)
        labels = labels.to(device)

        # outputs = model(inputs, dataset_ids)
        outputs = model(inputs)
        val_loss = criterion(outputs, labels)
        
        running_loss += val_loss.item()

    print(f"Validation Loss: {running_loss/len(val_loader):.4f}")

    if running_loss < best_loss:
        best_loss = running_loss
        best_model = model.state_dict()
        print(f"Best model saved at epoch {epoch+1} with loss {best_loss:.4f}")
        torch.save(best_model, "temp_best_model.pth")

# Load the best model
model.load_state_dict(torch.load("temp_best_model.pth", weights_only=True))



# plot data abstract
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

model.eval()

preds = []
labels = []
with torch.no_grad():
    for i, (inputs, label, dataset_ids) in enumerate(test_loader):
        inputs = inputs.permute(0, 2, 1).to(device)
        outputs = model(inputs, dataset_ids)
        preds.extend(outputs.cpu().detach().numpy())
        labels.extend(label.numpy())
preds = np.array(preds)
labels = np.array(labels)
preds = (preds > 0.5).astype(int)
cm = confusion_matrix(labels.flatten(), preds.flatten())
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# Plot confusion matrix 
plt.figure(figsize=(10, 5))
plt.imshow(cm, aspect='auto', cmap='gnuplot', interpolation='none', vmax=1, vmin=0)
for i in range(2):
    for j in range(2):
        plt.text(j, i, f"{cm[i, j]:.2f}", ha='center', va='center', color='black')
plt.colorbar()
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.legend()
plt.savefig("temp_figs/multidataset_outputs.png")
plt.show()


print("Confusion Matrix:")
print(cm)

# do a confusion matrix for each dataset
for i in range(len(DATASETS)):
    preds = []
    labels = []
    with torch.no_grad():
        for j, (inputs, label, dataset_ids) in enumerate(test_loader):
            inputs = inputs.permute(0, 2, 1).to(device)
            outputs = model(inputs, dataset_ids)
            preds.extend(outputs.cpu().detach().numpy())
            labels.extend(label.numpy())
    preds = np.array(preds)
    labels = np.array(labels)
    preds = (preds > 0.5).astype(int)
    cm = confusion_matrix(labels.flatten(), preds.flatten())
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # Plot confusion matrix 
    print("Confusion Matrix for dataset", DATASETS[i])
    print(cm)