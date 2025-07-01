from data_preprocess import *
import argparse
import os

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--beta", type=float, default=0.18, help="Beta value for label interpolation")
parser.add_argument("--cov", type=int, default=18, help="Coverage of data to use")
parser.add_argument("--enh", type=str, default="E25E10", help="Enhancer name")
args = parser.parse_args()


AA = 1
BB = args.beta

lv =  {"MM": -AA, "NM": -BB, "NP":BB, "PP":AA}
enh_name =  args.enh

images_save_dir = f"images_interpolating_labels/{enh_name}/{args.cov}/{args.beta}"
if not os.path.exists(images_save_dir):
    os.makedirs(images_save_dir, exist_ok=True)
else:
    print(f"Already done")
    exit(0)

BATCH_SIZE = 256
JIGGLE = 1

data = get_interpolated_seq_label_pairs(coverage=args.cov, label_values=lv, enh_name=enh_name, local_path=get_dataloc(enh_name))

print("Number of sequences:", len(data))

values = list(data.values())
seqs = list(data.keys())

# # normalize values
# sorted_values = sorted(values)
# percentiles = int(len(sorted_values) * 0.35 / 100)
# v_mean = np.mean(sorted_values[percentiles:-percentiles])
# v_std = np.std(sorted_values[percentiles:-percentiles])
# values = [(v - v_mean) / v_std for v in values]
# del v_mean, v_std, sorted_values, percentiles


print(values[:10])  # Print first 10 values for verification

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.hist(values, bins=120, color='blue', alpha=0.7, edgecolor='black')
plt.title('Distribution of Labels')
plt.xlabel('Label Value')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.axvline(x=0, color='red', linestyle='--', label='Zero Line')
plt.legend()
plt.savefig(f"{images_save_dir}/{enh_name}_label_distribution.png", dpi=150)
print(f"Label distribution plot saved as {images_save_dir}/{enh_name}_label_distribution.png")

def get_gc_content(sequence):
    """Calculate the GC content of a DNA sequence."""
    gc_count = sequence.count('G') + sequence.count('C')
    return (gc_count / len(sequence)) * 100 if len(sequence) > 0 else -0


# plot gc content of the sequences vs the label values
gc_content = [get_gc_content(seq) for seq in seqs]
plt.figure(figsize=(10, 6))
plt.scatter(gc_content, values, alpha=0.5, color='green')
plt.title('GC Content vs Label Values')
plt.xlabel('GC Content (%)')
plt.ylabel('Label Value')
plt.grid()
plt.axhline(y=0, color='red', linestyle='--', label='Zero Line')
plt.legend()
plt.savefig(f"{images_save_dir}/{enh_name}_gc_vs_label_values.png", dpi=150)

def pad_with_jiggles(seqs, jiggle_lim):
    padded_seqs = []
    for j in range(-jiggle_lim, jiggle_lim + 1):
        padded_seqs.extend(simple_pad_batch(seqs, jiggle=j))
    return padded_seqs

def pad_dataset(d, jiggle_lim):
    seqs, values = list(zip(*d))
    padded_seqs = pad_with_jiggles(seqs, jiggle_lim)
    padded_values = values * (2 * jiggle_lim + 1)
    padded_seqs = [one_hot_encode(seq) for seq in padded_seqs]
    return list(zip(padded_seqs, padded_values))


# print(seqs[:10])  # Print first 10 sequences for verification
dataset = zip(seqs, values)


from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
class CustomDataset(Dataset):
    def __init__(self, data):
        self.seqs = np.array([seq for seq, _ in data])
        self.labels = np.array([label for _, label in data])

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        label = self.labels[idx]
        return seq.T, label
    

# Split the dataset into training and test sets
train_seqs, test_seqs = train_test_split(list(dataset), test_size=0.2, random_state=42)
train_seqs, val_seqs = train_test_split(train_seqs, test_size=0.2, random_state=42)
print(f"Training sequences: {len(train_seqs)}, Validation sequences: {len(val_seqs)}, Test sequences: {len(test_seqs)}")

# Create DataLoader for training, validation, and test sets
train_seqs = pad_dataset(train_seqs, JIGGLE)
train_dataset = CustomDataset(train_seqs)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
val_seqs = pad_dataset(val_seqs, JIGGLE)
val_dataset = CustomDataset(val_seqs)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,  num_workers=4)
test_seqs = pad_dataset(test_seqs, JIGGLE)
test_dataset = CustomDataset(test_seqs)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,  num_workers=4)

from models import SignalPredictor1D
import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ComplexModel(torch.nn.Module):
    def __init__(self, num_classes=1):
        super(ComplexModel, self).__init__()
        self.conv1 = nn.Conv1d(4, 128, kernel_size=7, padding=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(128, 128, kernel_size=7, padding=3)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(128, 128, kernel_size=7, padding=3)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv1d(128, 128, kernel_size=7, padding=3)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)

        self.conv5 = nn.Conv1d(128, 128*2, kernel_size=15, padding=0)
        self.relu5 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.conv6 = nn.Conv1d(128*2, 128*4, kernel_size=25, padding=0)
        self.relu6 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        self.pool3 = nn.MaxPool1d(2)

        # self.conv7 = nn.Conv1d(128 * 4, 128 * 8, kernel_size=5, padding=1)
        # self.bn7 = nn.BatchNorm1d(128 * 8)
        # self.relu7 = nn.ReLU()
        # self.conv8 = nn.Conv1d(128 * 8, 128 * 8, kernel_size=5, padding=1)
        # self.bn8 = nn.BatchNorm1d(128 * 8)
        # self.relu8 = nn.ReLU()
        # self.pool4 = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(128 * 24 * 2, 128*4)
        self.relufc1 = nn.ReLU()
        self.fc2 = nn.Linear(128*4, 128)
        self.relufc2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 1)

        # self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.dropout1(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.dropout2(x)
        x = self.pool3(x)

        # x = self.conv7(x)
        # x = self.bn7(x)
        # x = self.relu7(x)
        # x = self.conv8(x)
        # x = self.bn8(x)
        # x = self.relu8(x)
        # x = self.pool4(x)

        # print("Shape after conv layers:", x.shape)  # Debugging line

        x = x.view(-1, 128 * 24 * 2)
        x = self.fc1(x)
        x = self.relufc1(x)
        x = self.fc2(x)
        x = self.relufc2(x)
        x = self.fc3(x)

        return x


model = ComplexModel(num_classes=1)
# model = SignalPredictor1D(num_classes=1)
# # load pretrained model if available
# model_path = f"saved_models_try_2/E25E102/models/154171846_mm_v_all_C3_E25E102.pt"
# if os.path.exists(model_path):
#     print(f"Loading pretrained model from {model_path}")
#     model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
# else:
#     print(f"Pretrained model not found at {model_path}, starting from scratch.")
model.to(device)


# Train the model
import torch.optim as optim

num_epochs = 2000
lr = 1e-6  # Initial learning rate
# lr = 1e-5  # Initial learning rate
wd = 1 * 1e-6
optimizer = optim.Adam(model.parameters(), weight_decay=wd, lr=lr)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=25, verbose=True, min_lr=1e-8)
import torch.nn as nn
from tqdm import tqdm
criterion = nn.MSELoss()

class LabelFrequency:
    def __init__(self, labels):
        self.label_freq = {}
        # create the bins
        bins = np.linspace(min(labels), max(labels), num=100)
        hist, _ = np.histogram(labels, bins=bins)
        for i in range(len(bins) - 1):
            bin_label = (bins[i] + bins[i + 1]) / 2
            self.label_freq[bin_label] = hist[i]

        # invert the values to make it reverse proportional to the frequency
        self.keys = np.array(list(self.label_freq.keys()))
        self.label_freq = {i: 1 / (abs(v)**0.5 + 1e-6) for i, (k, v) in enumerate(self.label_freq.items())}  # Add small constant to avoid division by zero

    def __call__(self, labels):
        closest_bins = np.argmin(np.abs(self.keys[:, None] - labels.cpu().detach().numpy()[None, :]), axis=0).tolist()
        
        return torch.tensor([self.label_freq[closest_bin] for closest_bin in closest_bins], dtype=torch.float32, device=device)

label_freq = LabelFrequency(values)

def weighted_mse(y_pred, y_true):
    weights = label_freq(y_true)
    return torch.mean(weights * (y_pred - y_true)**2)

# criterion = weighted_mse  # Use the custom weighted MSE loss


epoch_loss = 0.0
val_epoch_loss = 0.0
best_val_loss = float('inf')
best_model_dict = None
patience = 100  # Early stopping patience

history = []

with tqdm(range(num_epochs), desc="Training Epochs") as pbar:
    for epoch in pbar:
        pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}, Best Val Loss: {best_val_loss:.4f}, Current LR: {optimizer.param_groups[0]['lr']:.3e}")
        pbar.set_postfix(loss=0.0)
        model.train()
        running_loss = 0.0
        for seqs, labels in train_loader:
            seqs = seqs.float().to(device)
            labels = torch.tensor(labels, dtype=torch.float32).to(device)

            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)  # Gradient clipping
            optimizer.step()

            running_loss += loss.item() * seqs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        with torch.no_grad():
            model.eval()
            val_running_loss = 0.0
            for seqs, labels in val_loader:
                seqs = seqs.float().to(device)
                labels = torch.tensor(labels, dtype=torch.float32).to(device)

                outputs = model(seqs)
                loss = criterion(outputs.squeeze(), labels)
                val_running_loss += loss.item() * seqs.size(0)


            val_epoch_loss = val_running_loss / len(val_loader.dataset)
        lr_scheduler.step(val_running_loss)
        

        history.append((epoch_loss, val_epoch_loss))

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_model_dict = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

#load the best model
if best_model_dict is not None:
    model.load_state_dict(best_model_dict)
    print("Loaded the best model with validation loss:", best_val_loss)

# Evaluate the model
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for seqs, labels in test_loader:
        seqs = seqs.float().to(device)
        labels = torch.tensor(labels, dtype=torch.float32).to(device)

        outputs = model(seqs)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(outputs.squeeze().cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
pierce_corr = np.corrcoef(y_true, y_pred)[0, 1]
print(f"Test MSE: {mse:.4f}, R^2: {r2:.4f}, Pearson Correlation: {pierce_corr:.4f}")
# Save the model

plt.figure(figsize=(10, 6))
plt.scatter(y_true, y_pred, alpha=0.5, color='blue')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
plt.title('True vs Predicted Labels')
plt.text(0.05, 1.01, f'Pearson Correlation: {pierce_corr:.4f}\nMSE: {mse:.4f}\nR^2: {r2:.4f}', transform=plt.gca().transAxes)
plt.xlabel('True Labels')
plt.ylabel('Predicted Labels')
plt.grid()
plt.savefig(f"{images_save_dir}/{enh_name}_true_vs_predicted_labels.png", dpi=150)
plt.show()
print(f"True vs Predicted labels plot saved as {images_save_dir}/{enh_name}_true_vs_predicted_labels.png")

plt.figure(figsize=(10, 6))
plt.plot([x[0] for x in history], label='Training Loss')
plt.plot([x[1] for x in history], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')  # Use logarithmic scale for better visibility
plt.legend()
plt.grid()
plt.savefig(f"{images_save_dir}/{enh_name}_tloss_over_epochs.png", dpi=150)
print(f"Training and validation loss plot saved as {images_save_dir}/{enh_name}_tloss_over_epochs.png")



WT_SEQ = "TTATGGATCCTCGGTTCACGCAATGACCGCGGCTCCAGGGGCGAGAGGAGCCAGCCCTGTCTCTCTCTCCTGGCCCTGGGGGCGCCCGCCGCGAGGGCGGGGCCGGGATTGTGCTGATTCTGGCCTCCCTGGGCCGGAGGCTCTAGTGGAACTTAAGGCTCCTCCCTGATGGCACCGAGGCGAGGAACTGCCAGCTGTCTGTCTCCTTCCTGCCTTGACCCAGAGCAGTTGATCCGGTCCTGGATCCATAA"

def one_hot_encode(seq):
    seq = seq.upper()
    encoding_dict = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
        "N": [0, 0, 0, 0]
    }

    encoding = []
    for i, char in enumerate(seq):
        encoding.append(encoding_dict[char])

    return np.array(encoding).T


saturation_mut_data = [one_hot_encode(WT_SEQ)]  # Start with the wild-type sequence
for i in range(len(WT_SEQ)):
  for nuc in ["A", "C", "G", "T"]:
    if WT_SEQ[i] != nuc:
        saturation_mut_data.append(one_hot_encode(WT_SEQ[:i] + nuc + WT_SEQ[i+1:]))
    else:
        saturation_mut_data.append(one_hot_encode("G" + WT_SEQ[:i] + WT_SEQ[i+1:]))

saturation_mut_data = np.array(saturation_mut_data)

data_loader = torch.utils.data.DataLoader(dataset=saturation_mut_data,
                                             batch_size=BATCH_SIZE,
                                             shuffle=False)

sat_mut = []
for images in data_loader:
  images = images.to(device).float()
  output = model(images)
  sat_mut.extend(output.cpu().detach().numpy())

sat_mut = np.array(sat_mut)

WT_value= sat_mut[0]
sat_mut = sat_mut[1:]  # Remove the wild-type sequence output

sat_mut = sat_mut.reshape(len(WT_SEQ), 4).T - WT_value

deletion_effects = sat_mut * one_hot_encode(WT_SEQ)
sat_mut = sat_mut * (1 - one_hot_encode(WT_SEQ))  # Apply the deletion effects to the saturation mutation data


import tangermeme.plot as tp


fig, ax = plt.subplots(3, 1, figsize=(18, 15))
ax[0].set_xticks(np.arange(10)*25, np.arange(10)*25-25)
tp.plot_logo(sat_mut, ax[0])
ax[0].set_title(f"Saturation Mutation Logo for {enh_name} (Centered on WT)")
ax[1].plot(np.arange(6)*50, np.arange(6)*0, color='red', linestyle='--', linewidth=1)
ax[1].plot(np.sum(deletion_effects * one_hot_encode(WT_SEQ), axis=0), color='black', linewidth=2)
ax[1].set_title(f"Deletion Effects for {enh_name}")
ax[1].set_xlabel("Position")
ax[1].set_ylabel("Deletion Effect")
ax[1].grid(True)
ax[1].set_xticks(np.arange(10)*25, np.arange(10)*25-25)
ax[1].set_xlim(0, 250)
tp.plot_logo(deletion_effects, ax[2])
ax[2].set_title(f"Deletion Effects for {enh_name}")
ax[2].set_xlabel("Position")
ax[2].set_ylabel("Deletion Effect")
ax[2].grid(True)
ax[2].set_xticks(np.arange(10)*25, np.arange(10)*25-25)
plt.savefig(f"{images_save_dir}/{enh_name}_saturation_mutation_logo.png", dpi=150)
print(f"Saturation mutation logo saved as {images_save_dir}/{enh_name}_saturation_mutation_logo.png")
