import os
import numpy as np
from data_preprocess import *
import matplotlib.pyplot as plt
from data_preprocess import import_indiv_enhancers, get_dataloc, simple_pad_batch
from data_preprocess import one_hot_encode

# um mau
# E22P1C1	2152	6691	4350	258
# um bom
# E22P1D10	867	26445	31755	6627
# o melhor
# E25E10 74226	107443	103524	7799


# enh_name = 'E22P1D10'
# BIN_CELLS = np.array([867, 26445, 31755, 6627])  # MM, NM, NP, PP
# enh_name = 'E22P1C1'
# BIN_CELLS = np.array([2152, 6691, 4350, 258])  # MM, NM, NP, PP
enh_name = 'E25E10'
BIN_CELLS = np.array([74226, 107443, 103524, 7799])  # MM, NM, NP, PP

ALFA = 1
BETA = 0.25
COVERAGE = 5
label_values = {'MM': -ALFA, 'NM': -BETA, 'NP': BETA, 'PP': ALFA}  # MM, NM, NP, PP
JIGGLE = 0
BATCH_SIZE = 256
num_epochs = 2000
lr = 1e-6  # Initial learning rate
wd = 0 * 1e-6

OLD_INTERPOLATION = False

images_save_dir = 'temp_figs'

interpolated_seqs = get_interpolated_seq_label_pairs(enh_name=enh_name, coverage=COVERAGE, label_values=label_values, local_path=get_dataloc(enh_name), BIN_CELLS=BIN_CELLS)
non_weighted_interpolated_seqs = get_interpolated_seq_label_pairs(coverage=COVERAGE, label_values=label_values, enh_name=enh_name, local_path=get_dataloc(enh_name))






plt.hist(list(interpolated_seqs.values()), bins=150, alpha=0.7, label='Weighted Interpolated Sequences')
plt.hist(list(non_weighted_interpolated_seqs.values()), bins=150, alpha=0.7, label='Non-Weighted Interpolated Sequences')
plt.xlabel('Interpolated Sequence Value')
plt.ylabel('Frequency')
plt.title(f'Histogram of Interpolated Sequences for {enh_name}')
plt.legend()
plt.savefig(f'temp_figs/.interpolated_histogram.png')
plt.show()

if OLD_INTERPOLATION:
    interpolated_seqs = non_weighted_interpolated_seqs



########### COPIED FROM interpolating_labels.py ###########

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
dataset = zip(list(interpolated_seqs.keys()), list(interpolated_seqs.values()))


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

values = list(interpolated_seqs.values())

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


from interpolate_sat_mut import saturation_mutagenesis

# Run saturation mutagenesis
saturation_mutagenesis(model, enh_name, device, images_save_dir, BATCH_SIZE=BATCH_SIZE)

from interpolate_deeplift import interpol_deep_lift
# Run DeepLIFT
interpol_deep_lift(model, enh_name, device, images_save_dir, BATCH_SIZE=BATCH_SIZE)