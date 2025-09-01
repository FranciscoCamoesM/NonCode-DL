import os
from data_preprocess import *
import pandas as pd


filename = "LClibR2_merged.tsv"

dataloc = "MPRAs/MPRA_data/"

# load data with pandas
data = pd.read_csv(os.path.join(dataloc, filename), sep="\t")


print(data.head())


# Chekc if there are duplicate sequences
duplicate_sequences = data[data.duplicated(subset=["Sequence"], keep=False)]
if not duplicate_sequences.empty:
    print("Duplicate sequences found:")
    print(duplicate_sequences)
else:
    print("No duplicate sequences found.")



data["Seq_len"] = data["Sequence"].apply(lambda x: len(x))
data["GC_content"] = data["Sequence"].apply(lambda x: (x.count("G") + x.count("C")) / len(x) if len(x) > 0 else 0)
print("Data loaded successfully.")


data["Padded_Sequence"] = data["Sequence"].apply(lambda x: simple_pad(x, max_length=500, pre_and_post_padding = ["N"*250, "N"*250]))
data["Pad_seq_len"] = data["Padded_Sequence"].apply(lambda x: len(x))
print("Padded sequences created successfully.")


print(data.head())

print(data[["Seq_len", "Pad_seq_len"]].head())


# remove sequences with 0 gDNA_UMIs
print(f"Data before removing sequences with 0 gDNA_UMIs: {data.shape[0]}")
data = data[data["gDNA_UMIs"] > 0]
data = data[data["RNA/gDNA_ratio"] > 0]
print(f"Data after removing sequences with 0 gDNA_UMIs: {data.shape[0]}")

# aply log transformation to data["RNA/gDNA_ratio"]
data["RNA/gDNA_ratio"] = data["RNA/gDNA_ratio"].apply(lambda x: np.log(x) if x > 0 else 0)

print(data.head())



# plot ratio vs gc content
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(data["GC_content"], data["RNA/gDNA_ratio"], alpha=0.5)
plt.xlabel("GC Content")
plt.ylabel("Log Ratio of RNA to gDNA UMIs")
plt.title("GC Content vs Log Ratio of RNA to gDNA UMIs")
plt.grid()
plt.tight_layout()
plt.savefig("temp_figs/.mpra_gc_content_vs_ratio.png")
print("GC Content vs Log Ratio of RNA to gDNA UMIs plot saved as temp_figs/.mpra_gc_content_vs_ratio.png")











# dataset is a list of tuples (Padded_Sequence, gDNA_UMIs)
dataset = list(zip(data["Padded_Sequence"], data["RNA/gDNA_ratio"]))

dataset = [(one_hot_encode(seq), label) for seq, label in dataset]


# normalize the labels
labels = [label for _, label in dataset]
max_label = max(labels)
min_label = min(labels)
dataset = [(seq,2*((label - min_label) / (max_label - min_label)) - 1) for seq, label in dataset]



# plot the distribution of labels
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.hist(labels, bins=200, color='blue', edgecolor='black')
plt.xlabel("Ratio of RNA to gDNA UMIs")
plt.ylabel("Frequency") 
# plt.xscale('log')
plt.title("Distribution of RNA to gDNA UMIs Ratio")
plt.grid()
plt.tight_layout()
plt.savefig("temp_figs/.mpra_data_dist.png")


# Create the model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F



class SignalPredictor1D(nn.Module):
    # 5 convolutional layers, 3 fully connected layers
    def __init__(self, num_classes=1):
        super(SignalPredictor1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, 5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, 5, padding=2)
        self.conv4 = nn.Conv1d(256, 512, 5, padding=2)
        self.conv5 = nn.Conv1d(512, 1024, 5, padding=2)

        self.fc1 = nn.Linear(6144, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape from (batch_size, seq_length, num_channels) to (batch_size,
        # num_channels, seq_length)
        x = F.relu(self.conv1(x))
        # x = F.max_pool1d(x, kernel_size=3, stride=3)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, kernel_size=3, stride=3)
        x = F.relu(self.conv3(x))
        x = F.max_pool1d(x, kernel_size=3, stride=3)
        x = F.relu(self.conv4(x))
        x = F.max_pool1d(x, kernel_size=3, stride=3)
        x = F.relu(self.conv5(x))
        x = F.max_pool1d(x, kernel_size=3, stride=3)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class MPRADataSet(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        seq, label = self.dataset[idx]
        seq_tensor = torch.tensor(seq).float()
        return seq_tensor, label
    

model = SignalPredictor1D(num_classes=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00000001, weight_decay=0)

from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

train_dataset = MPRADataSet(train_data)
val_dataset = MPRADataSet(val_data)
test_dataset = MPRADataSet(test_data)

print(f"After splitting, train set size: {len(train_dataset)}, validation set size: {len(val_dataset)}, test set size: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


# Training loop
num_epochs = 2500
losses_history = []
from tqdm import tqdm
for epoch in tqdm(range(num_epochs)):
    model.train()
    running_loss = 0.0
    for seqs, labels in train_loader:
        seqs, labels = seqs.to(device), labels.to(device).float()
        
        optimizer.zero_grad()
        outputs = model(seqs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * seqs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for seqs, labels in val_loader:
            seqs, labels = seqs.to(device), labels.to(device).float()
            outputs = model(seqs)
            loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item() * seqs.size(0)

    val_loss /= len(val_loader.dataset)
    # print(f"Validation Loss: {val_loss:.4f}")

    losses_history.append((epoch_loss, val_loss))


print("Training complete.")

# plot the training and validation losses
import matplotlib.pyplot as plt
train_losses, val_losses = zip(*losses_history)
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='orange')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Losses")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("temp_figs/.mpra_training_validation_loss.png")
plt.show()


# plot and save it in temp_figs
import matplotlib.pyplot as plt
import numpy as np

test_results = []
model.eval()
with torch.no_grad():
    for seqs, labels in test_loader:
        seqs, labels = seqs.to(device), labels.to(device).float()
        outputs = model(seqs)
        test_results.extend(zip(outputs.cpu().numpy(), labels.cpu().numpy()))

test_outputs, test_labels = zip(*test_results)
test_outputs = np.array(test_outputs)
test_labels = np.array(test_labels)

print("Tested model on test set:", len(test_outputs), "samples.")

plt.figure(figsize=(10, 6))
plt.scatter(test_labels, test_outputs, alpha=0.5)
plt.plot(np.arange(0, int(max(test_labels.max(), test_outputs.max())) + 1), 
         np.arange(0, int(max(test_labels.max(), test_outputs.max())) + 1), 
         color='red', linestyle='--')
plt.xlabel("True Labels")
# plt.xscale('log')
# plt.yscale('log')
plt.ylabel("Predicted Outputs")
plt.title("Model Predictions vs True Labels")
plt.grid()
plt.tight_layout()
plt.savefig("temp_figs/.mpra_model_predictions_vs_true_labels.png")

print("Model predictions vs true labels plot saved as temp_figs/.mpra_model_predictions_vs_true_labels.png")