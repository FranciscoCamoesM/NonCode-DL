import pickle
from data_preprocess import *
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from training_funcs import gen_sampler_weights, train, test


import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import argparse

def pad_wrapper(dataset, encoding_dict, jiggle=0):
    # Pad the sequences and labels
    padded_sequences, padded_labels = pad_samples(dataset, encoding_dict=encoding_dict, jiggle=jiggle)
    
    # return list of tuples
    return list(zip(padded_sequences, padded_labels))


parser = argparse.ArgumentParser(description="Generate processed dataset for enhancer classification.")
parser.add_argument("--enh", type=str, default="E25E102", help="Enhancer name")
parser.add_argument("--coverage", type=int, default=5, help="Coverage")
parser.add_argument("--label", type=str, default="mm_v_all", help="Label type")
parser.add_argument("--lr", type=float, default=0.000005, help="Learning rate")
parser.add_argument("--wd", type=float, default=0.001, help="Weight decay")
parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument("--save_dir", type=str, default="saved_models_try_2", help="Directory to save the model")
parser.add_argument("--sample_type", type=str, default="upsample", help="Type of sampling to use: upsample or downsample")
parser.add_argument("--jiggle", type=int, default=10, help="Jiggle range for padding")

args = parser.parse_args()

def gc_content(seq):
    seq = seq.upper()
    g = seq.count("G")
    c = seq.count("C")
    return (g + c) / len(seq)


ENH = args.enh
data_path = get_dataloc(ENH)
COVERAGE = args.coverage
LABEL = args.label
SAVE_DIR = args.save_dir

BATCH_SIZE = args.batch_size
LR = args.lr
WD = args.wd
N_EPOCHS = args.epochs
JIGGLE_RANGE = args.jiggle
sample_type = args.sample_type # "upsample" or "downsample"

#save sequences in processed_data/
data_dir = os.path.join("processed_data", ENH)
data_dir = os.path.join(data_dir, str(COVERAGE))
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    os.makedirs(os.path.join(data_dir, "train"))
    os.makedirs(os.path.join(data_dir, "val"))
    os.makedirs(os.path.join(data_dir, "test"))

# check if the dataset already exists
if sample_type == "upsample":
    extension = ""
elif sample_type == "downsample":
    extension = "_downsampled"

savefile = os.path.join(data_dir, "train", f"train_dataset_{LABEL}_{COVERAGE}{extension}.pkl")

if os.path.exists(savefile):
    print("Dataset already exists. Loading from disk.")
    with open(os.path.join(data_dir, "train", f"train_dataset_{LABEL}_{COVERAGE}{extension}.pkl"), "rb") as f:
        train_dataset_padded = pickle.load(f)
    with open(os.path.join(data_dir, "val", f"val_dataset_{LABEL}_{COVERAGE}{extension}.pkl"), "rb") as f:
        val_dataset_padded = pickle.load(f)
    with open(os.path.join(data_dir, "test", f"test_dataset_{LABEL}_{COVERAGE}{extension}.pkl"), "rb") as f:
        test_dataset_padded = pickle.load(f)
    print("Dataset loaded from disk.")

else:
    data = get_coverage_seq_label_pairs(enh_name=ENH, local_path=data_path, coverage=COVERAGE)
    x, y = data.keys(), data.values()
    x = list(x)
    y = list(y)

    if LABEL == "mm_v_all":
        data = {x[i]: int(y[i] != "MM") for i in range(len(x))}
    else:
        raise ValueError("Invalid label type. Please use 'mm_v_all' or 'other_label'.")


    content = {}
    for label in set(data.values()):
        content[label] = []

    print("Number of files: ", len(data))
    for x, y in data.items():
        # print("Processing: ", y)
        content[y].append(gc_content(x))

    # Plotting the GC content
    fig, ax = plt.subplots(3, 1, figsize=(15, 22))
    for label, gc_values in content.items():
        ax[0].hist(gc_values, bins=200, alpha=0.5, label=label, density=True)
        print(f"Label: {label}, GC Content: {np.mean(gc_values):.2f} ± {np.std(gc_values):.2f}; Count: {len(gc_values)}")
    ax[0].set_title("GC Content Distribution before sampling")
    ax[0].set_xlabel("GC Content")
    ax[0].set_ylabel("Frequency")
    ax[0].legend()
    # plt.savefig("temp_figs//gc_content_distribution.png")

    # print("figure saved as temp_figs/gc_content_distribution.png")

    data = get_coverage_seq_label_pairs(enh_name=ENH, local_path=data_path, coverage=COVERAGE)
    x, y = data.keys(), data.values()
    x = list(x)
    y = list(y)


    # sample mm to have a distribution similar to the other labels
    if sample_type == "upsample":
        mm_sequences = []
        other_sequences = []
        sampled_seqs = []
        for x, y in data.items():
            if y == "MM" or y == 0:
                mm_sequences.append([x, gc_content(x), 0])
            else:
                other_sequences.append([x, gc_content(x)])
        # each entry will have the sequence, the gc content, and the numbe rof times it is being upsampled

        for seq in mm_sequences:
            sampled_seqs.append(seq)
            seq.append(seq[2] + 1)
            del seq[2]


        with tqdm(total=len(other_sequences), desc="Sampling sequences", unit="sequences") as pbar:
            while len(sampled_seqs) < len(other_sequences):
                # sample a random sequence from the other sequences
                seq = random.choice(other_sequences)
                # sample a random sequence from the mm sequences with the same gc content
                tol = 0
                options = []
                while len(options) == 0:
                    tol += 0.001
                    options = [s for s in mm_sequences if abs(s[1] - seq[1]) <= tol]
                    if tol > 0.15:
                        print("No options found for this sequence")
                        break
                if len(options) == 0:
                    print("No options found for this sequence")
                    continue

                # if some have been more sampled than others, sample from the less sampled
                min_sample = min([s[2] for s in options])
                options = [s for s in options if s[2] == min_sample]

                # sample a random sequence from the options
                seq2 = random.choice(options)
                # upsample the sequence
                sampled_seqs.append(seq2)

                # increment the number of times it has been sampled
                seq2.append(seq2[2] + 1)
                del seq2[2]

                pbar.update(1)

    elif sample_type == "downsample":
        mm_sequences = []
        other_sequences = []
        sampled_seqs = []
        for x, y in data.items():
            if y == "MM" or y == 0:
                mm_sequences.append([x, gc_content(x)])
            else:
                other_sequences.append([x, gc_content(x), y])

        # for each element in mm_sequences, sample the closest element in other_sequences
        with tqdm(total=len(mm_sequences), desc="Sampling sequences", unit="sequences") as pbar:
            while len(sampled_seqs) < len(mm_sequences):
                # sample a random sequence from the mm sequences
                seq = random.choice(mm_sequences)
                # sample a random sequence from the other sequences with the same gc content
                tol = 0
                options = []
                while len(options) == 0:
                    tol += 0.001
                    options = [s for s in other_sequences if abs(s[1] - seq[1]) <= tol]
                    if tol > 0.15:
                        print("No options found for this sequence")
                        break
                if len(options) == 0:
                    print("No options found for this sequence")
                    continue

                # sample a random sequence from the options
                seq2 = random.choice(options)
                # downsample the sequence
                sampled_seqs.append(seq2)

                # remove the sequence from the other sequences
                other_sequences.remove(seq2)

                pbar.update(1)
    else:
        raise ValueError("Invalid sample type. Please use 'upsample' or 'downsample'.")

    # plot the distribution of the sampled sequences
    if sample_type == "upsample":
        ax[1].hist([s[1] for s in sampled_seqs], bins=200, alpha=0.5, label="Sampled MM", density=True)
        ax[1].hist([s[1] for s in other_sequences], bins=200, alpha=0.5, label="Sampled Other", density=True)
    elif sample_type == "downsample":
        ax[1].hist([s[1] for s in mm_sequences], bins=200, alpha=0.5, label="MM", density=True)
        ax[1].hist([s[1] for s in sampled_seqs], bins=200, alpha=0.5, label="Sampled Other", density=True)

    ax[1].set_title("GC Content Distribution after sampling")
    ax[1].set_xlabel("GC Content")
    ax[1].set_ylabel("Frequency")
    ax[1].legend()
    # plt.savefig("temp_figs//gc_content_distribution_sampled.png")
    # print("figure saved as temp_figs/gc_content_distribution_sampled.png")


    # plot number of times each sequence has been sampled
    if sample_type == "upsample":
        ax[2].hist([s[2] for s in mm_sequences], bins=200, alpha=0.5, label="Sampled MM", density=True)
        ax[2].set_title("Number of times each sequence has been sampled")
        ax[2].set_xlabel("Number of times sampled")
        ax[2].set_ylabel("Frequency")
        ax[2].legend()
    plt.savefig(f"{data_dir}/gc_content_distribution.png")
    plt.close()
    print("figure saved as gc_content_distribution.png in ", data_dir)


    if sample_type == "upsample":
        dataset = [[s[0], "MM"] for s in sampled_seqs]
        dataset += [[x, y] for x, y in data.items() if y != "MM"]
    elif sample_type == "downsample":
        dataset = [[s[0], "MM"] for s in mm_sequences]
        dataset += [[s[0], s[2]] for s in sampled_seqs]


    print(len(dataset), "sequences in the dataset")
    print(len(list(set([d[0] for d in dataset]))), "unique sequences in the dataset")

    unique_data = list(set([d[0] for d in dataset]))

    unique_train_data, unique_val_data = train_test_split(unique_data, test_size=0.2, random_state=42)
    unique_train_data, unique_test_data = train_test_split(unique_train_data, test_size=0.2, random_state=42)

    train_data = []
    val_data = []
    test_data = []
    for d in dataset:
        if d[0] in unique_train_data:
            train_data.append(d)
        elif d[0] in unique_val_data:
            val_data.append(d)
        elif d[0] in unique_test_data:
            test_data.append(d)
        else:
            raise ValueError("Data not in train or val set")
    print("Number of training sequences: ", len(train_data))
    print("Number of val sequences: ", len(val_data))
    print("Number of test sequences: ", len(test_data))
    print("Total number of sequences: ", len(val_data) + len(train_data) + len(test_data)), " = ", len(dataset)

    del dataset
    del unique_data
    del unique_train_data
    del unique_val_data
    del unique_test_data
    del data
    del sampled_seqs
    del mm_sequences
    del other_sequences


    train_dataset_padded = []
    val_dataset_padded = []
    test_dataset_padded = []

    for j in tqdm(range(-JIGGLE_RANGE, JIGGLE_RANGE + 1)):
        train_dataset_padded += pad_wrapper(train_data, encoding_dict=preset_labels(LABEL), jiggle=j)
        val_dataset_padded += pad_wrapper(val_data, encoding_dict=preset_labels(LABEL), jiggle=j)
        test_dataset_padded += pad_wrapper(test_data, encoding_dict=preset_labels(LABEL), jiggle=j)

    # # subsample the val set to increase speed
    # val_dataset_padded = random.sample(val_dataset_padded, int(len(val_dataset_padded) * 0.2))

    print("Number of training sequences after padding: ", len(train_dataset_padded))
    print("Number of val sequences after padding: ", len(val_dataset_padded))
    print("Number of test sequences after padding: ", len(test_dataset_padded))
    print("Ratio of val to total sequences: ", len(val_dataset_padded) / (len(val_dataset_padded) + len(train_dataset_padded)))


    if sample_type == "upsample":
        train_dataset_padded_loc = os.path.join(data_dir, "train", f"train_dataset_{LABEL}_{COVERAGE}.pkl")
        val_dataset_padded_loc = os.path.join(data_dir, "val", f"val_dataset_{LABEL}_{COVERAGE}.pkl")
        test_dataset_padded_loc = os.path.join(data_dir, "test", f"test_dataset_{LABEL}_{COVERAGE}.pkl")
    elif sample_type == "downsample":
        train_dataset_padded_loc = os.path.join(data_dir, "train", f"train_dataset_{LABEL}_{COVERAGE}_downsampled.pkl")
        val_dataset_padded_loc = os.path.join(data_dir, "val", f"val_dataset_{LABEL}_{COVERAGE}_downsampled.pkl")
        test_dataset_padded_loc = os.path.join(data_dir, "test", f"test_dataset_{LABEL}_{COVERAGE}_downsampled.pkl")
    with open(train_dataset_padded_loc, "wb") as f:
        pickle.dump(train_dataset_padded, f)
    with open(val_dataset_padded_loc, "wb") as f:
        pickle.dump(val_dataset_padded, f)
    with open(test_dataset_padded_loc, "wb") as f:
        pickle.dump(test_dataset_padded, f)






from models import SignalPredictor1D

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class EnhancerDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.sequences, self.labels = zip(*data)

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        seq = self.sequences[idx].T
        label = self.labels[idx]

        seq = torch.tensor(seq, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return seq, label
    

def calc_accuracy(y_true, y_pred, running_corrects, running_total):
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > 0.5).float()
    correct = (y_pred == y_true).float().sum()

    return correct.item() + running_corrects, y_true.size(0) + running_total

def update_confusion_matrix(y_true, y_pred, TP, TN, FP, FN):
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > 0.5).float()

    TP += ((y_true == 1) & (y_pred == 1)).sum().item()
    TN += ((y_true == 0) & (y_pred == 0)).sum().item()
    FP += ((y_true == 0) & (y_pred == 1)).sum().item()
    FN += ((y_true == 1) & (y_pred == 0)).sum().item()

    return TP, TN, FP, FN


train_dataset = EnhancerDataset(train_dataset_padded)
val_dataset = EnhancerDataset(val_dataset_padded)
test_dataset = EnhancerDataset(test_dataset_padded)

train_weights = gen_sampler_weights(train_dataset, 2)
val_weights = gen_sampler_weights(val_dataset, 2)
train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_dataset), replacement=True)
val_sampler = torch.utils.data.sampler.WeightedRandomSampler(val_weights, len(val_dataset), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Save the dataset

# example usage
model = SignalPredictor1D(num_classes=1)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
criterion = torch.nn.BCEWithLogitsLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model.to(device)
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

train_losses, val_losses, train_accs, val_accs = train(model, train_loader, val_loader, criterion, optimizer, device=device, num_epochs=N_EPOCHS, BEST_VAL=True)
test_loss, test_acc, confusion_matrix = test(model, test_loader, criterion, device=device, return_matrix=True)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Confusion Matrix: {confusion_matrix}")

history["train_loss"] = train_losses
history["val_loss"] = val_losses
history["train_acc"] = train_accs
history["val_acc"] = val_accs


print("Training complete.")

# Calc Confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np




# Save the model
SAVE_DIR = os.path.join(SAVE_DIR, ENH)
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    os.makedirs(os.path.join(SAVE_DIR, "cm"))
    os.makedirs(os.path.join(SAVE_DIR, "history"))
    os.makedirs(os.path.join(SAVE_DIR, "models"))

now = datetime.now()
day = now.strftime("%j")
hour = now.hour
minute = now.minute
second = now.second
second = minute*60+second
second = str(second).zfill(4)
savetime = f"{day}{hour}{second}"

model_name = f"{savetime}_{LABEL}_C{COVERAGE}_{ENH}"
torch.save(model.state_dict(), os.path.join(SAVE_DIR, "models", model_name + ".pt"))
print(f"Model saved as {model_name} in {SAVE_DIR}")


fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(history["train_loss"], label="Train Loss")
ax[0].plot(history["val_loss"], label="Val Loss")
ax[0].set_title("Loss")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].set_yscale("log")
ax[0].legend()
ax[1].plot(history["train_acc"], label="Train Accuracy")
ax[1].plot(history["val_acc"], label="Val Accuracy")
ax[1].set_title("Accuracy")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Accuracy")
ax[1].legend()
if sample_type == "upsample":
    plt.savefig(os.path.join(SAVE_DIR, "history", f"history_{model_name}.png"))
elif sample_type == "downsample":
    plt.savefig(os.path.join(SAVE_DIR, "history", f"history_{model_name}_downsampled.png"))
plt.close()
print("History saved as history.png in ", SAVE_DIR)


TP = confusion_matrix[1][1]
FP = confusion_matrix[0][1]
FN = confusion_matrix[1][0]
TN = confusion_matrix[0][0]


acc = (TP + TN) / (TP + TN + FP + FN)
print(f"Test Accuracy: {acc:.4f}")
print(f"TP: {TP}, FP: {FP}\nFN: {FN}, TN: {TN}")


cm = np.array([[TN, FP], [FN, TP]])
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=False,
            xticklabels=["Predicted Negative", "Predicted Positive"],
            yticklabels=["True Negative", "True Positive"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(SAVE_DIR, "cm", f"confusion_matrix_{model_name}.png"))
plt.close()
print("Confusion matrix saved as confusion_matrix.png in ", SAVE_DIR)
# plt.show()


params = {
    "Enhancer": ENH,
    "Label": LABEL,
    "Batch Size": BATCH_SIZE,
    "Learning Rate": LR,
    "Weight Decay": WD,
    "Epochs": N_EPOCHS,
    "Coverage": COVERAGE,
    "Jiggle": JIGGLE_RANGE,
}

with open(os.path.join(SAVE_DIR, f"{model_name}_params.txt"), "w") as f:
    for key, value in params.items():
        f.write(f"{key}: {value}\n")
print("Parameters saved as params.txt in ", SAVE_DIR)