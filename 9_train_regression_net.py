import matplotlib.pyplot as plt
from data_preprocess import *
from sklearn.metrics import mean_squared_error, r2_score
import os
import argparse
from datetime import datetime
from models import SignalPredictor1D
import torch
import torch.nn as nn
import torch.optim as optim
from regression_training_funcs import CustomDataset, train, test
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader



# Define the values associated with each bin
AA = 1
BB = .45/.95
lv =  {"MM": -AA, "NM": -BB, "NP":BB, "PP":AA}


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--coverage', type=int, default=75, help='Coverage value')
parser.add_argument('--enh', type=str, default="E25E102", help='Enhancer name')
parser.add_argument('--wd', type=float, default=1e-6, help='Weight decay')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--clip_grad', type=bool, default=True, help='Whether to apply gradient clipping during training')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
parser.add_argument('--jitter', type=int, default=5, help='Jitter value')
parser.add_argument('--es_patience', type=int, default=200, help='Early stopping patience')

args = parser.parse_args()

COVERAGE = args.coverage
ENH = args.enh
BIN_CELLS = get_bin_cells(ENH)  # get the number of bin cells for the specified enhancer, which is needed for data interpolation
WD = args.wd
LR = args.lr
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
JITTER = args.jitter
ES_patience = args.es_patience
CLIP_GRAD = args.clip_grad

ENH_LIST = [[COVERAGE, ENH, BIN_CELLS]] # set dataset loading paramenters inside a list. To train a model on multiple datasets, simply add more lists with different parameters to this list, e.g. ENH_LIST = [[75, "E25E102", 10], [15, "E22P1D10", 10]]

# load data for all datasets specified in ENH_LIST
data = []
for enh_data in ENH_LIST:
    coverage, enh_name, bin_cells = enh_data
    # data.append(get_interpolated_seq_label_pairs(coverage=coverage, label_values=lv, enh_name=enh_name, local_path=get_dataloc(enh_name), BIN_CELLS=bin_cells))
    data.append(get_interpolated_seq_label_pairs(coverage=coverage, label_values=lv, enh_name=enh_name, local_path=os.path.join("../NonCode", get_dataloc(enh_name)), BIN_CELLS=bin_cells))

# check if there are enough sequences for training
print("Number of sequences:", sum(len(d) for d in data))
if sum(len(d) for d in data) < 1000:
    raise ValueError(f"Not enough sequences for training: {sum(len(d) for d in data)}. At least 1000 sequences are required.")

# assign a name for model based on datasets used
if len(ENH_LIST)== 1:
    enh_name = ENH_LIST[0][1]
else:
    enh_name = "mixed"

# save_dir = f"regression_models/{enh_name}/"
save_dir = os.path.join("regression_models", enh_name)

# current date
current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = os.path.join(save_dir, current_date)

seqs = []
values = []
for i, d in enumerate(data):
    seqs.extend(list(d.keys()))
    temp_v = list(d.values())
    # temp_v = normalize_values(temp_v)
    temp_v = list(zip(temp_v, [i] * len(temp_v)))
    values.extend(temp_v)


dataset = zip(seqs, values)    

# Split the dataset into training and test sets
# important to do before applying jitter, otherwise we will have data leakage
train_seqs, test_seqs = train_test_split(list(dataset), test_size=0.2, random_state=42)
train_seqs, val_seqs = train_test_split(train_seqs, test_size=0.2, random_state=42)
print(f"Training sequences: {len(train_seqs)}, Validation sequences: {len(val_seqs)}, Test sequences: {len(test_seqs)}")

# Create DataLoader for training, validation, and test sets
train_seqs = pad_dataset_jitter_range(train_seqs, JITTER)
train_dataset = CustomDataset(train_seqs)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_seqs = pad_dataset_jitter_range(val_seqs, JITTER)
val_dataset = CustomDataset(val_seqs)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_seqs = pad_dataset_jitter_range(test_seqs, JITTER)
test_dataset = CustomDataset(test_seqs)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SignalPredictor1D(num_classes=len(ENH_LIST))
model.to(device)


# Train the model
num_epochs = EPOCHS
lr = LR
wd = WD
optimizer = optim.Adam(model.parameters(), weight_decay=wd, lr=lr)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=15, verbose=True, min_lr=1e-8)
criterion = nn.MSELoss()

model, history, epochs_trained = train(model, optimizer, criterion, train_loader, val_loader, device, num_epochs=num_epochs, patience=ES_patience, clip_grad=CLIP_GRAD, N_LABELS=len(ENH_LIST), lr_scheduler=lr_scheduler)

# Evaluate the model
y_true, y_pred, y_color, mse, r2, pierson_corr = test(model, test_loader, device, N_LABELS=len(ENH_LIST))

# Save the model
os.makedirs(save_dir, exist_ok=True)
torch.save(model.state_dict(), f"{save_dir}/{enh_name}_trained_model.pt")
print(f"Model saved as {save_dir}/{enh_name}_trained_model.pt")

# save parameters in a csv file
with open(f"{save_dir}/{enh_name}_params.txt", "w") as f:
    f.write(f"Coverage: {COVERAGE}\n")
    f.write(f"Learning Rate: {lr}\n")
    f.write(f"Weight Decay: {wd}\n")
    f.write(f"Clip Gradients: {CLIP_GRAD}\n")
    f.write(f"Max Epochs: {num_epochs}\n")
    f.write(f"Early Stopping Patience: {ES_patience}\n")
    f.write(f"Epochs Trained: {epochs_trained}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Jitter: {JITTER}\n")
    f.write(f"ENH List: {ENH_LIST}\n")
    f.write(f"Pierson Correlation: {pierson_corr}\n")
    f.write(f"Mean Squared Error: {mse}\n")
    f.write(f"R^2 Score: {r2}\n")
print(f"Parameters saved as {save_dir}/{enh_name}_params.txt")

# Generate and save plots
# plot true vs predicted labels
plt.figure(figsize=(10, 6))
if len(ENH_LIST) == 1:
    plt.scatter(y_true, y_pred, alpha=0.5, s=50, edgecolors='k')
else:
    plt.scatter(y_true, y_pred, alpha=0.5, c=y_color, cmap='jet', s=50, edgecolors='k')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
plt.title('True vs Predicted Labels')
plt.text(0.05, 1.01, f'Pearson Correlation: {pierson_corr:.4f}\nMSE: {mse:.4f}\nR^2: {r2:.4f}', transform=plt.gca().transAxes)
plt.xlabel('True Labels')
plt.ylabel('Predicted Labels')
plt.grid()
plt.savefig(f"{save_dir}/{enh_name}_true_vs_predicted_labels.png", dpi=150)
plt.show()
print(f"True vs Predicted labels plot saved as {save_dir}/{enh_name}_true_vs_predicted_labels.png")

# plot training and validation loss over epochs
plt.figure(figsize=(10, 6))
plt.plot([x[0] for x in history], label='Training Loss')
plt.plot([x[1] for x in history], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')  # Use logarithmic scale for better visibility
plt.legend()
plt.grid()
plt.savefig(f"{save_dir}/{enh_name}_tloss_over_epochs.png", dpi=150)
print(f"Training and validation loss plot saved as {save_dir}/{enh_name}_tloss_over_epochs.png")