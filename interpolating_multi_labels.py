import matplotlib.pyplot as plt
from data_preprocess import *

AA = 1
BB = .475
COVERAGE = 5
lv =  {"MM": -AA, "NM": -BB, "NP":BB, "PP":AA}



ENH_LIST = [#[75, "E22P1A3", np.array([23731, 69026, 57444, 15199])],
                [75, "E25E102", np.array([74226,107443,103524,7799])],
                [75, "E25E10R3", np.array([68519,111643,60318,5687])],
                # [10, "E25B2", np.array([14866,63461,30286,3838])]
                ]

# ENH_LIST = [
#                 [7, "E18C2", np.array([3135, 4108, 3866, 501])],
#                 [7, "E22P3F2", np.array([1957, 8165, 7623, 689])],
#                 [10, "E24C3", np.array([11765, 31087, 21297, 2339])]
#                 ]


# ENH_LIST = [
#                 [25, "E22P1D10", np.array([8867, 26445, 31755, 6627])],
#                 # [5, "E22P3B3", np.array([8931, 9735, 7685, 635])],
#                 # [25, "E22P3B3R3", np.array([12001, 17423, 10126, 1234])]
#                 ]

ENH_LIST = [
                [25, "E22P1F10", np.array([33005, 50888, 40371, 4975])],
                ]




# ENH_LIST = [ 
#                 # [100, "E22P1A3", np.array([23731, 69026, 57444, 15199])],
#                 # # [10, "E24A3", np.array([1, 1, 1, 1])],
#                 # # [25, "E25E10", np.array([1, 1, 1, 1])],
#                 # [100, "E25E102", np.array([74226,107443,103524,7799])],
#                 # [100, "E25E10R3", np.array([68519,111643,60318,5687])],
#                 # [10, "E25B2", np.array([14866,63461,30286,3838])],
#                 # # [10, "E25B3", np.array([9327,39125,25464,2188])],
#                 # [1, "E18C2", np.array([3135, 4108, 3866, 501])],
#                 [50, "E22P3F2", np.array([1957, 8165, 7623, 689])],
#                 [100, "E24C3", np.array([11765, 31087, 21297, 2339])],
#                 [25, "E22P1D10", np.array([8867, 26445, 31755, 6627])],
#                 # [3, "E22P3B3", np.array([8931, 9735, 7685, 635])],
#                 [100, "E22P3B3R3", np.array([12001, 17423, 10126, 1234])],
#                 [25, "E22P1C1", np.array([2152, 6691, 4350, 258])],
#                 [50, "E22P3B4", np.array([5112, 24376, 19383, 2478])],
#                 [20, "E22P2F4", np.array([4919, 16352, 9135, 534])],
#                 [20, "E22P3G3", np.array([3644, 10704, 6332, 443])]]

ENH_LIST = [ENH_LIST[0]]


BATCH_SIZE = 256
JIGGLE = 3

data = []
for enh_data in ENH_LIST:
    coverage, enh_name, bin_cells = enh_data
    data.append(get_interpolated_seq_label_pairs(coverage=coverage, label_values=lv, enh_name=enh_name, local_path=get_dataloc(enh_name), BIN_CELLS=bin_cells))

if len(ENH_LIST)== 1:
    enh_name = ENH_LIST[0][1]
else:
    enh_name = "mixed"

save_dir = f"regression_models/{enh_name}/"
os.makedirs(save_dir, exist_ok=True)


print("Number of sequences:", sum(len(d) for d in data))

# # normalize values
# def normalize_values(values, percentiles=35):
#     sorted_values = sorted(values)
#     percentiles = int(len(sorted_values) * percentiles / 100)
#     v_mean = np.mean(sorted_values[percentiles:-percentiles])
#     v_std = np.std(sorted_values[percentiles:-percentiles])
#     values = [(v - v_mean) / v_std for v in values]

#     return np.array(values).tolist()


seqs = []
values = []
for i, d in enumerate(data):
    seqs.extend(list(d.keys()))
    temp_v = list(d.values())
    # temp_v = normalize_values(temp_v)
    temp_v = list(zip(temp_v, [i] * len(temp_v)))
    values.extend(temp_v)



def get_gc_content(sequence):
    """Calculate the GC content of a DNA sequence."""
    gc_count = sequence.count('G') + sequence.count('C')
    return (gc_count / len(sequence)) * 100 if len(sequence) > 0 else -0


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
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, label = self.data[idx]
        return seq.T, label
    

# Split the dataset into training and test sets
train_seqs, test_seqs = train_test_split(list(dataset), test_size=0.2, random_state=42)
train_seqs, val_seqs = train_test_split(train_seqs, test_size=0.2, random_state=42)
print(f"Training sequences: {len(train_seqs)}, Validation sequences: {len(val_seqs)}, Test sequences: {len(test_seqs)}")

# Create DataLoader for training, validation, and test sets
train_seqs = pad_dataset(train_seqs, JIGGLE)
train_dataset = CustomDataset(train_seqs)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_seqs = pad_dataset(val_seqs, JIGGLE)
val_dataset = CustomDataset(val_seqs)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_seqs = pad_dataset(test_seqs, JIGGLE)
test_dataset = CustomDataset(test_seqs)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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
        self.fc3 = nn.Linear(128, num_classes)

        # self.normalization = nn.Linear(1, num_classes)

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

        # print(f"Shape after conv layers: {x.shape}")

        x = x.view(-1, 128 * 24 * 2)
        x = self.fc1(x)
        x = self.relufc1(x)
        x = self.fc2(x)
        x = self.relufc2(x)
        x = self.fc3(x)
        # x = self.normalization(x)

        return x
    
    def abstract(self, x):
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

        # print(f"Shape after conv layers: {x.shape}")

        x = x.view(-1, 128 * 24 * 2)
        x = self.fc1(x)
        x = self.relufc1(x)
        x = self.fc2(x)
        x = self.relufc2(x)
        x = self.fc3(x)
        x = torch.mean(x, dim=1, keepdim=True)  # Average over the batch dimension

        return x


# model = ComplexModel(num_classes=len(ENH_LIST))
model = SignalPredictor1D(num_classes=len(ENH_LIST))
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

num_epochs = 5000
lr = 1e-6  # Initial learning rate
# lr = 1e-5  # Initial learning rate
wd = 0 * 1e-6
optimizer = optim.Adam(model.parameters(), weight_decay=wd, lr=lr)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=15, verbose=True, min_lr=1e-8)
import torch.nn as nn
from tqdm import tqdm
criterion = nn.MSELoss()

def train(model, optimizer, criterion, train_loader, val_loader, num_epochs=5000, patience=100):
    epoch_loss = 0.0
    val_epoch_loss = 0.0
    best_val_loss = float('inf')
    best_model_dict = None

    history = []

    with tqdm(range(num_epochs), desc="Training Epochs") as pbar:
        for epoch in pbar:
            pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}, Best Val Loss: {best_val_loss:.4f}, Current LR: {optimizer.param_groups[0]['lr']:.3e}")
            pbar.set_postfix(loss=0.0)
            model.train()
            running_loss = 0.0
            for seqs, y in train_loader:
                seqs = seqs.float().to(device)
                labels, classes = y
                labels = torch.tensor(labels, dtype=torch.float32).to(device)

                optimizer.zero_grad()

                class_mask = np.zeros((len(labels), len(ENH_LIST)))
                for i, c in enumerate(classes):
                    class_mask[i,c] = 1
                
                class_mask = torch.tensor(class_mask, dtype=torch.float32).to(device)

                outputs = model(seqs) 
                
                labels = labels.unsqueeze(dim=1) * class_mask + (1 - class_mask) * outputs.detach()

                # print(list(zip(outputs.detach().cpu().numpy().tolist(), labels.cpu().detach().numpy().tolist()))[:5])
                
                loss = criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)  # Gradient clipping
                optimizer.step()

                running_loss += loss.item() * seqs.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)

            with torch.no_grad():
                model.eval()
                val_running_loss = 0.0
                for seqs, y in val_loader:
                    seqs = seqs.float().to(device)
                    labels, classes = y
                    labels = torch.tensor(labels, dtype=torch.float32).to(device)

                    class_mask = np.zeros((len(labels), len(ENH_LIST)))
                    for i, c in enumerate(classes):
                        class_mask[i,c] = 1

                    class_mask = torch.tensor(class_mask, dtype=torch.float32).to(device)

                    outputs = model(seqs) * class_mask
                    outputs = outputs.sum(dim=1)  # Sum over the classes

                    loss = criterion(outputs, labels)
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

    return model, history

model, history = train(model, optimizer, criterion, train_loader, val_loader, num_epochs=num_epochs)

# Evaluate the model
def test(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    y_color = []
    with torch.no_grad():
        for seqs, y in test_loader:
            seqs = seqs.float().to(device)
            labels, classes = y
            labels = torch.tensor(labels, dtype=torch.float32).to(device)

            outputs = model.abstract(seqs)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.squeeze().cpu().numpy())
            y_color.extend(classes.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_color = np.array(y_color)

    print(f"Shapes: y_true: {y_true.shape}, y_pred: {y_pred.shape}, y_color: {y_color.shape}")


    from sklearn.metrics import mean_squared_error, r2_score
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pierce_corr = np.corrcoef(y_true, y_pred)[0, 1]
    print(f"Test MSE: {mse:.4f}, R^2: {r2:.4f}, Pearson Correlation: {pierce_corr:.4f}")

    return y_true, y_pred, y_color, mse, r2, pierce_corr

# Save the model

y_true, y_pred, y_color, mse, r2, pierce_corr = test(model, test_loader)

plt.figure(figsize=(10, 6))
plt.scatter(y_true, y_pred, alpha=0.5, c=y_color, cmap='jet', edgecolors='k', s=50)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
plt.title('True vs Predicted Labels')
plt.text(0.05, 1.01, f'Pearson Correlation: {pierce_corr:.4f}\nMSE: {mse:.4f}\nR^2: {r2:.4f}', transform=plt.gca().transAxes)
plt.xlabel('True Labels')
plt.ylabel('Predicted Labels')
plt.grid()
plt.savefig(f"{save_dir}/{enh_name}_true_vs_predicted_labels.png", dpi=150)
plt.show()
print(f"True vs Predicted labels plot saved as {save_dir}/{enh_name}_true_vs_predicted_labels.png")

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
for seqs in data_loader:
  seqs = seqs.to(device).float()
  output = model.abstract(seqs)
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
plt.savefig(f"{save_dir}/{enh_name}_saturation_mutation_logo.png", dpi=150)
print(f"Saturation mutation logo saved as {save_dir}/{enh_name}_saturation_mutation_logo.png")


exit()

#now change it to a different architecture
class NewComplexModel(torch.nn.Module):
    def __init__(self, old_model, num_classes=1):
        # change only the last layer to a linear layer
        super(NewComplexModel, self).__init__()
        self.conv1 = old_model.conv1
        self.relu1 = old_model.relu1
        self.conv2 = old_model.conv2
        self.relu2 = old_model.relu2
        self.pool1 = old_model.pool1
        self.conv3 = old_model.conv3
        self.relu3 = old_model.relu3
        self.conv4 = old_model.conv4
        self.relu4 = old_model.relu4
        self.pool2 = old_model.pool2

        self.conv5 = old_model.conv5
        self.relu5 = old_model.relu5
        self.dropout1 = old_model.dropout1
        self.conv6 = old_model.conv6
        self.relu6 = old_model.relu6
        self.dropout2 = old_model.dropout2
        self.pool3 = old_model.pool3

        self.fc1 = old_model.fc1
        self.relufc1 = old_model.relufc1
        self.fc2 = old_model.fc2
        self.relufc2 = old_model.relufc2
        self.abstract_fc = nn.Linear(128, 1)  # Change the last layer
        self.fc3 = nn.Linear(1, num_classes)

    def abstract(self, x):
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

        x = x.view(-1, 128 * 24 * 2)
        x = self.fc1(x)
        x = self.relufc1(x)
        x = self.fc2(x)
        x = self.relufc2(x)

        return self.abstract_fc(x)
    
    def forward(self, x):
        x = self.abstract(x)
        return self.fc3(x)
    

# Create the new model with the same weights as the old model
new_model = NewComplexModel(num_classes=len(ENH_LIST), old_model=model)
new_model.to(device)

# freeze the weights of the new model
for param in new_model.parameters():
    param.requires_grad = False

# Set the last layer to be trainable
for param in new_model.fc3.parameters():
    param.requires_grad = True
# Train the new model
optimizer = optim.Adam(new_model.fc3.parameters(), weight_decay=wd, lr=1e-4)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=25, verbose=True, min_lr=1e-8)  

new_model, history = train(new_model, optimizer, criterion, train_loader, val_loader, num_epochs=num_epochs)

print("Training complete. Evaluating the new model...")
# Evaluate the new model
new_model.eval()

y_true, y_pred, y_color, mse, r2, pierce_corr = test(new_model, test_loader)

plt.figure(figsize=(10, 6))
plt.scatter(y_true, y_pred, alpha=0.5, c=y_color, cmap='jet', edgecolors='k', s=50)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
plt.title('True vs Predicted Labels (New Model)')
plt.text(0.05, 1.01, f'Pearson Correlation: {pierce_corr:.4f}\nMSE: {mse:.4f}\nR^2: {r2:.4f}', transform=plt.gca().transAxes)
plt.xlabel('True Labels')
plt.ylabel('Predicted Labels')
plt.grid()
plt.savefig(f"{save_dir}/{enh_name}_true_vs_predicted_labels_new_model.png", dpi=150)
plt.show()
print(f"True vs Predicted labels plot for new model saved as {save_dir}/{enh_name}_true_vs_predicted_labels_new_model.png")
