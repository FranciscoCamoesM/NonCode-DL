from data_preprocess import *
import matplotlib.pyplot as plt

from models import SignalPredictor1D
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm


def gen_sampler_weights(data):
    class_count = [0, 0, 0, 0]
    for _, label in data:
        class_count[np.argmax(label)] += 1

    weights = []
    for _, label in data:
        weights.append(1/class_count[np.argmax(label)])

    return torch.DoubleTensor(weights)


### BEGGINIGN OF THE CODE ###

import argparse
parser = argparse.ArgumentParser(description='Train a neural network to predict enhancer activity')
parser.add_argument('--enh', type=str, default='E25E10', help='Enhancer name to train on')
parser.add_argument('--label', type=str, default='mm_v_all', help='Label title to train on')
parser.add_argument('--subset', type=float, default=1.0, help='Fraction of the data to use')
parser.add_argument('--bs', type=int, default=64, help='Batch size for training')
parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate for training')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train for')
parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay for the optimizer')
parser.add_argument('--mom', type=float, default=None, help='Momentum for the optimizer')
parser.add_argument('--optim', type=str, default='adam', help='Optimizer to use')
parser.add_argument('--best_val', type=bool, default=True, help='Whether to save the best model based on validation accuracy')
parser.add_argument('--datafolder', type=str, default='Fasta_Pool2', help='Folder where the data is stored')
args = parser.parse_args()

ENHANCER_NAME = args.enh
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

SAVE_DIR = 'saved_models'



LABELS = preset_labels(LABEL_TITLE)
N_CLASSES = len(list(LABELS.values())[0])


# Load the data
dataset = get_seq_label_pairs(enh_name = ENHANCER_NAME, local_path = DATA_FOLDER)

# Split the data
X = list(dataset.keys())
y = list(dataset.values())


if SUBSET < 1.0:
    X = X[:int(len(X) * SUBSET)]
    y = y[:int(len(y) * SUBSET)]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
del X_temp, y_temp


import random
random.seed(4120)
# reduce the number of samples in the training set based on subset
if SUBSET < 1.0:
    sample_points = random.sample(range(len(X_train)), int(len(X_train) * SUBSET))
    X_train = [X_train[i] for i in sample_points]
    y_train = [y_train[i] for i in sample_points]


# Create a data loader
train_dataset = EnhancerDataset(X_train, y_train, LABELS)
val_dataset = EnhancerDataset(X_val, y_val, LABELS)
test_dataset = EnhancerDataset(X_test, y_test, LABELS)

train_weights = gen_sampler_weights(train_dataset)
val_weights = gen_sampler_weights(val_dataset)
train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_dataset), replacement=True)
val_sampler = torch.utils.data.sampler.WeightedRandomSampler(val_weights, len(val_dataset), replacement=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


print("Train size: ", len(train_dataset))

# test the data loader
for i, (data, target) in enumerate(train_loader):
    print(data.shape)
    break

def count(dl):
    memory = []
    for i, (data, target) in enumerate(dl):
        targets = np.argmax(target.numpy(), axis=1)
        memory.extend(targets)

        if i > 10:
            break

    #count
    from collections import Counter
    print(Counter(memory))

count(train_loader)
count(test_loader)




# Train the model
model = SignalPredictor1D(num_classes=N_CLASSES)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.CrossEntropyLoss()
if OPTIM == 'adam':
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
elif OPTIM == 'sgd':
    optimizer = SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOM)
else:
    raise ValueError("Invalid optimizer")


print("Running on: ", device)


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_model = None
    best_acc = 0.0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0.0
        running_corrects = 0
        running_total = 0
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = data.float().permute(0, 2, 1)
            target = target.float()

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            running_corrects += torch.sum(torch.argmax(output, dim=1) == torch.argmax(target, dim=1))
            running_total += len(target)
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(running_corrects / running_total)

        model.eval()
        val_loss = 0.0
        running_corrects = 0
        running_total = 0
        for i, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            data = data.float().permute(0, 2, 1)
            target = target.float()
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            running_corrects += torch.sum(torch.argmax(output, dim=1) == torch.argmax(target, dim=1))
            running_total += len(target)
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(running_corrects / running_total)
        if val_accs[-1] > best_acc:
            best_acc = val_accs[-1]
            best_model = model.state_dict()

        

        print(f'Epoch {epoch}, Train Loss: {train_losses[-1]}, Test Loss: {val_losses[-1]}, Train Acc: {train_accs[-1]}, Test Acc: {val_accs[-1]} {"<-- new best!" if val_accs[-1] > best_acc else ""}')

    if BEST_VAL:
        model.load_state_dict(best_model)

    return train_losses, val_losses

train_losses, val_losses = train(model, train_loader, val_loader, criterion, optimizer, num_epochs=N_EPOCHS)

# fig, ax = plt.subplots( 1, 2)
# ax[0].plot(train_losses, label='Train Loss')
# ax[1].plot(val_losses, label='Test Loss', c = 'orange')
# ax[0].set_xlabel('Epoch')
# ax[1].set_xlabel('Epoch')
# ax[0].set_ylabel('Loss')
# ax[1].set_ylabel('Loss')
# ax[0].legend()
# ax[1].legend()

# plt.show()


# create a confusion matrix
model.eval()
all_preds = []
all_targets = []
for i, (data, target) in enumerate(test_loader):
    data, target = data.to(device), target.to(device)
    data = data.float().permute(0, 2, 1)
    target = target.float()
    output = model(data)
    preds = torch.argmax(output, dim=1)
    target = torch.argmax(target, dim=1)
    all_preds.extend(preds.cpu().numpy())
    all_targets.extend(target.cpu().numpy())

from sklearn.metrics import confusion_matrix
import seaborn as sns

print(all_preds[:5])
print(all_targets[:5])

cm = confusion_matrix(all_targets, all_preds)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]         #normalize the confusion matrix
# sns.heatmap(cm, annot=True)
# plt.set_cmap('gnuplot')
# plt.vmin = 0
# plt.vmax = 1
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix of enhancer ' + str(ENHANCER_NAME) + ' on ' + str(LABEL_TITLE))
# plt.show()


# Save the model
import os
import datetime
now = datetime.datetime.now()

model_name = f"{now.strftime('%H%M%S_%m%d')}_{ENHANCER_NAME}_{LABEL_TITLE}"

# save model, confustion matrix, training history, and parameters all in different files
torch.save(model.state_dict(), f"{SAVE_DIR}/{model_name}.pt")
np.save(f"{SAVE_DIR}/training_history/{model_name}_cm.npy", cm)
np.save(f"{SAVE_DIR}/training_history/{model_name}_train_losses.npy", train_losses)
np.save(f"{SAVE_DIR}/training_history/{model_name}_val_losses.npy", val_losses)

plt_title = 'Confusion Matrix of enhancer ' + str(ENHANCER_NAME) + ' on ' + str(LABEL_TITLE)
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
plt.savefig(f"{SAVE_DIR}/plots/{model_name}_cm.png")

with open(f"{SAVE_DIR}/{model_name}_params.txt", 'w') as f:
    f.write(f"Enhancer: {ENHANCER_NAME}\n")
    f.write(f"Label: {LABEL_TITLE}\n")
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



