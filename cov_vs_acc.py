import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

from models import SignalPredictor1D
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data_preprocess import one_hot_encode, one_hot_decode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load test data with highest coverage
loc = "processed_data/E25E102/100/test/test_dataset_mm_v_all_100.pkl"





class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, label = self.data[idx]
        return seq.T, label
    
def f1_score(TP, TN, FP, FN):
    if TP + FP == 0:
        return 0
    if TP + FN == 0:
        return 0
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2/(1/precision + 1/recall)
    return f1

def f1_sym(TP, TN, FP, FN):
    return f1_score(TP, TN, FP, FN)/2 + f1_score(TN, TP, FN, FP)/2



with open(loc, "rb") as f:
    test_dataset = pickle.load(f)


print("test_dataset", len(test_dataset))

test_seqs = []
test_labels = []
for seq, label in test_dataset:
    test_seqs.append(one_hot_decode(seq))
    test_labels.append(label)


print("Len with duplicates:", len(test_seqs))
# test_seqs = list(set(test_seqs))
print("Len with duplicates:", len(test_seqs))
print("test_seqs[0]", test_seqs[0])


models_to_test = [
    ["saved_models_try_2/E25E102/models/155160463_mm_v_all_C1_E25E102.pt", 1],
    ["saved_models_try_2/E25E102/models/154190360_mm_v_all_C2_E25E102.pt", 2],
    ["saved_models_try_2/E25E102/models/154171846_mm_v_all_C3_E25E102.pt", 3],
    ["saved_models_try_2/E25E102/models/154153348_mm_v_all_C5_E25E102.pt", 5],
    ["saved_models_try_2/E25E102/models/154162385_mm_v_all_C10_E25E102.pt", 10],
    ["saved_models_try_2/E25E102/models/154170127_mm_v_all_C25_E25E102.pt", 25],
    ["saved_models_try_2/E25E102/models/154170637_mm_v_all_C100_E25E102.pt", 100],
    

]


trianing_data_temp = "processed_data/E25E102/{}/train/train_dataset_mm_v_all_{}_downsampled.pkl"

graph_points = []

for model_path, coverage in models_to_test:
    print("model_path", model_path)
    print("coverage", coverage)
    
    # load model
    model = SignalPredictor1D(num_classes=1)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()
    
    # load training data
    training_data_loc = trianing_data_temp.format(coverage, coverage)
    with open(training_data_loc, "rb") as f:
        training_dataset = pickle.load(f)
    training_seqs = []
    training_labels = []
    for seq, label in training_dataset:
        training_seqs.append(one_hot_decode(seq))
        training_labels.append(label)

    # print("Len with duplicates:", len(training_seqs))
    training_seqs = list(set(training_seqs))
    # print("Len with duplicates:", len(training_seqs))


    # test set:
    model_test_set = []
    for i in range(len(test_seqs)):
        seq = test_seqs[i]
        if seq in training_seqs:
            continue

        seq = one_hot_encode(seq)

        model_test_set.append([seq, test_labels[i]])

    print(f"Started with {len(test_seqs)} test sequences and ended with {len(model_test_set)} test sequences")

    # model_test_set = CustomDataset(training_dataset)
    model_test_set = CustomDataset(model_test_set)
    model_test_loader = DataLoader(model_test_set, batch_size=64, shuffle=False)


    # test model
    y_true = []
    y_pred = []
    with torch.no_grad():
        for seq, label in tqdm(model_test_loader):
            seq = seq.float().to(device)
            label = label.to(device)

            output = model(seq)
            output = torch.sigmoid(output).cpu().numpy()
            y_pred.extend(output)
            y_true.extend(label.cpu().numpy())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # build confusion matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred.round())
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("*"*50)
    print("cm", cm)
    print(np.mean(y_pred), np.mean(y_true))
    print(f"F1 score: {f1_sym(cm[0][0], cm[1][1], cm[0][1], cm[1][0])}")
    print("*"*50)

    graph_points.append([coverage, f1_sym(cm[0][0], cm[1][1], cm[0][1], cm[1][0]), len(list(set(training_seqs)))])


graph_points = sorted(graph_points, key=lambda x: x[0])



# save vlaues to a text file
with open("temp_figs/.coverage_vs_f1_score.txt", "w") as f:
    # write header
    f.write("Coverage\tF1 Score\tTraining Set Size\n")
    for coverage, f1, train_size in graph_points:
        f.write(f"{coverage}\t{f1}\t{train_size}\n")


import cov_vs_acc_plot_only