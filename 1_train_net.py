from data_preprocess import *
from training_funcs import train, gen_sampler_weights
import matplotlib.pyplot as plt

from models import SignalPredictor1D
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm



### BEGGINIGN OF THE CODE ###

import argparse
parser = argparse.ArgumentParser(description='Train a neural network to predict enhancer activity')
parser.add_argument('--enh', type=str, default='E25E10', help='Enhancer name to train on')
parser.add_argument('--label', type=str, default='mm_v_all', help='Label title to train on')
parser.add_argument('--subset', type=float, default=1.0, help='Fraction of the data to use')
parser.add_argument('--bs', type=int, default=256, help='Batch size for training')
parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate for training')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train for')
parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay for the optimizer')
parser.add_argument('--mom', type=float, default=None, help='Momentum for the optimizer')
parser.add_argument('--optim', type=str, default='adam', help='Optimizer to use')
parser.add_argument('--best_val', type=bool, default=True, help='Whether to save the best model based on validation accuracy')
parser.add_argument('--optuna', type=bool, default=False, help='Whether to use optuna for hyperparameter optimization')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--pad', type=bool, default=True, help='Whether to pad the sequences with cassette or leave it with Ns')
parser.add_argument('--coverage', type=int, default=1, help='Coverage for the enhancer')
parser.add_argument('--jiggle', type=int, default=None, help='Jiggle the sequences by this many bases to both sides')
args = parser.parse_args()

PADDING = args.pad
ENHANCER_NAME = args.enh
LABEL_TITLE = args.label
SUBSET = args.subset
BATCH_SIZE = args.bs
LEARNING_RATE = args.lr
N_EPOCHS = args.epochs
WEIGHT_DECAY = args.wd
MOM = args.mom
OPTIM = args.optim
BEST_VAL = args.best_val
DATA_FOLDER = get_dataloc(ENHANCER_NAME)
COVERAGE = args.coverage
JIGGLE = args.jiggle
SEED = args.seed

SAVE_DIR = 'saved_models_final'



LABELS = preset_labels(LABEL_TITLE)
N_CLASSES = len(list(LABELS.values())[0])


# Load the data
if COVERAGE < 2:
    dataset = get_seq_label_pairs(enh_name = ENHANCER_NAME, local_path = DATA_FOLDER)
else:
    dataset = get_coverage_seq_label_pairs(enh_name = ENHANCER_NAME, local_path = DATA_FOLDER, coverage = COVERAGE)

# Split the data
X = list(dataset.keys())
y = list(dataset.values())


if SUBSET < 1.0:
    X = X[:int(len(X) * SUBSET)]
    y = y[:int(len(y) * SUBSET)]

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
X_train, X_val, y_train, y_val,  = train_test_split(X_temp, y_temp, test_size=0.2, random_state=SEED)
del X_temp, y_temp


import random
random.seed(SEED)
# reduce the number of samples in the training set based on subset
if SUBSET < 1.0:
    sample_points = random.sample(range(len(X_train)), int(len(X_train) * SUBSET))
    X_train = [X_train[i] for i in sample_points]
    y_train = [y_train[i] for i in sample_points]


# Create a data loader
if JIGGLE is None:
    train_dataset = EnhancerDataset(X_train, y_train, LABELS, PADDING = PADDING)
    val_dataset = EnhancerDataset(X_val, y_val, LABELS, PADDING = PADDING)
    test_dataset = EnhancerDataset(X_test, y_test, LABELS, PADDING = PADDING)
else:
    train_dataset = EnhancerDataset(X_train, y_train, LABELS, PADDING = PADDING, jiggle_range=JIGGLE)
    val_dataset = EnhancerDataset(X_val, y_val, LABELS, PADDING = PADDING, jiggle_range=JIGGLE)
    test_dataset = EnhancerDataset(X_test, y_test, LABELS, PADDING = PADDING, jiggle_range=JIGGLE)

train_weights = gen_sampler_weights(train_dataset, N_CLASSES)
val_weights = gen_sampler_weights(val_dataset, N_CLASSES)
train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_dataset), replacement=True)
val_sampler = torch.utils.data.sampler.WeightedRandomSampler(val_weights, len(val_dataset), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


print("Train size: ", len(train_dataset))

# test the data loader
for i, (data, target) in enumerate(train_loader):
    # print(data.shape)
    print(target.shape)
    break





# Train the model
model = SignalPredictor1D(num_classes=N_CLASSES)
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


print("Running on: ", device)



train_losses, val_losses, train_accs, val_accs = train(model, train_loader, val_loader, criterion, optimizer, num_epochs=N_EPOCHS, device=device, BEST_VAL=BEST_VAL, binary=(N_CLASSES <= 2), early_stopping_patience=100)

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
if N_CLASSES <= 2:
    for i, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        data = data.float()
        target = target.float()
        output = model(data)
        preds = torch.round(torch.sigmoid(output))
        all_preds.extend(preds.cpu().detach().numpy())
        all_targets.extend(target.cpu().detach().numpy())
else:
    for i, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        data = data.float()
        target = target.float()
        output = model(data)
        preds = torch.argmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        all_preds.extend(preds.cpu().detach().numpy())
        all_targets.extend(target.cpu().detach().numpy())

from sklearn.metrics import confusion_matrix
import seaborn as sns

print(all_preds[:5])
print(all_targets[:5])

cm = confusion_matrix(all_targets, all_preds)
un_normalized_cm = cm.copy()  # Keep the unnormalized confusion matrix for saving
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]         #normalize the confusion matrix

if not args.optuna:
    sns.heatmap(cm, annot=True)
    plt.set_cmap('gnuplot')
    plt.vmin = 0
    plt.vmax = 1
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix of enhancer ' + str(ENHANCER_NAME) + ' on ' + str(LABEL_TITLE))
    plt.show()
    

# Save the model
import os
import datetime
now = datetime.datetime.now()


model_id = now.strftime('%j%H%M%S_%m%d')

model_name = f"{model_id}_{ENHANCER_NAME}_{LABEL_TITLE}"


if not args.optuna:
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

if not args.optuna:
    plt.savefig(f"{SAVE_DIR}/plots/{model_name}_cm.png")
    plt.close()


    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss', c = 'orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss of enhancer ' + str(ENHANCER_NAME) + ' on ' + str(LABEL_TITLE))
    if SUBSET < 1.0:
        plt.title(f"Loss of enhancer {ENHANCER_NAME} on {LABEL_TITLE} (Subset: {SUBSET*100}%)")
    plt.savefig(f"{SAVE_DIR}/plots/{model_name}_loss.png")
    plt.close()



    with open(f"{SAVE_DIR}/{model_name}_params.txt", 'w') as f:
        f.write(f"Enhancer: {ENHANCER_NAME}\n")
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
        f.write(f"Coverage: {COVERAGE}\n")
        if JIGGLE is not None:
            f.write(f"Jiggle: {JIGGLE}\n")

        # record the metrics
        f.write(f"Confusion Matrix: {cm.tolist()}\n")
        f.write(f"Unnormalized Confusion Matrix: {un_normalized_cm.tolist()}\n")
        f.write(f"Number of Epochs: {len(train_losses)}\n")
        f.write(f"Train Losses: {train_losses}\n")
        f.write(f"Validation Losses: {val_losses}\n")



print("Model saved as: ", model_name)
print("Model saved in: ", SAVE_DIR)


# save F1 score for optuna
if args.optuna:
    from sklearn.metrics import f1_score
    # f1 = f1_score(all_targets, all_preds, average='weighted')
    f1= min(cm[0][0], cm[1][1])
    with open(f"optuna_log/{ENHANCER_NAME}_f1.txt", 'w') as f:
        f.write(str(f1))
    print("F1 score: ", f1)

