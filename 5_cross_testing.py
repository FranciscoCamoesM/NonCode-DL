import os
import torch
import torch.nn as nn
from models import *
from data_preprocess import get_seq_label_pairs, EnhancerDataset, preset_labels, get_dataloc
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np

model_list = [135230, 170349, 105959, 152018, 135916, 164632, 121329, 112258, 170543, 164155, 145614, 160800,161039,180122,175800,104335, 104814, 105933, 131953, 162929]


# model_list = [121329, 112258, 170543, 164155]
LABELS = preset_labels("mm_v_all")
print(LABELS)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModelInfo():
    # init model id, and fetch model info about enh, dataloc, etc.
    def __init__(self, model_id):
        self.model_id = model_id
        self.model_info = self.get_model_info(model_id)
        self.enh = self.model_info['Enhancer']
        self.dataloc = get_dataloc(self.enh)

        for f in os.listdir('saved_models'):
            if f.startswith(str(model_id)):
                if f.endswith('.pt'):
                    self.model_path = 'saved_models/' + f
                    break
        
        if self.model_path is None:
            raise ValueError(f"Model path not found for model id: {model_id}")

    def get_model_info(self, model_id):
        for f in os.listdir('saved_models'):
            # find "saved_models/103432_0327_E22P3G3_mm_v_all_params.txt"
            if f.startswith(str(model_id)):
                if f.endswith('params.txt'):
                    with open('saved_models/' + f, 'r') as file:
                        lines = file.readlines()
                        model_info = {}
                        for line in lines:
                            s = line.strip().split(': ')
                            key, value = s[0], s[1]
                            model_info[key] = value
                        return model_info
        return None
    
    def load_model(self):
        model = SignalPredictor1D(1)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        return model


model_info_list = [ModelInfo(model_id) for model_id in model_list]

different_clones = list(set([model_info.enh for model_info in model_info_list]))
print(different_clones)

f1_results = -1* np.ones((len(different_clones), len(model_info_list)))
min_acc_results = -1* np.ones((len(different_clones), len(model_info_list)))

for clone in different_clones:
    print(clone)

    #load data of clone
    dataset = get_seq_label_pairs(enh_name = clone, local_path = get_dataloc(clone))
    # Split the data
    X = list(dataset.keys())
    y = list(dataset.values())

    dataset = EnhancerDataset(X, y, LABELS)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    #load model
    for model_info in model_info_list:
        print(f"Running model {model_info.model_id} on {clone}")
        model = model_info.load_model()
        model.eval()
        model.to(device)

        cm = [[0, 0], [0, 0]]

        # get the output of the model
        for batch in dataloader:
            X, y = batch
            X = X.to(device).float()
            y = y.to(device).long()

            output = model(X)
            output = (output > 0.5).long().squeeze(1).cpu().numpy()

            for i in range(len(output)):
                # print(f"y: {y[i].item()}, output: {output[i]}")
                cm[y[i].item()][output[i]] += 1
                # print(f"cm: {cm[y[i].item()][output[i]]}")

        print(cm)
        # calculate f1 score
        mock_y_true = []
        mock_y_pred = []
        for i in range(2):
            for j in range(2):
                for _ in range(cm[i][j]):
                    mock_y_true.append(i)
                    mock_y_pred.append(j)
        F1 = f1_score(mock_y_true, mock_y_pred, average='weighted')
        print(f"F1 score: {F1}")
        f1_results[different_clones.index(clone)][model_info_list.index(model_info)] = F1

        L0_acc = cm[0][0] / (cm[0][0] + cm[0][1] + 1e-10)
        L1_acc = cm[1][1] / (cm[1][0] + cm[1][1] + 1e-10)
        min_acc = min(L0_acc, L1_acc)

        min_acc_results[different_clones.index(clone)][model_info_list.index(model_info)] = min_acc


# calculate homology score
homology_matrix = np.zeros((len(different_clones), len(different_clones)))
WT_seqs = {}
for i in range(len(different_clones)):
    # load sequence from original_enhancers.txt
    next_line = False
    with open('original_enhancers.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if different_clones[i] in line:
                next_line = True
                continue
            if next_line:
                seq = line.strip()
                break

    WT_seqs[different_clones[i]] = seq

for i in range(len(different_clones)):
    for j in range(len(different_clones)):
        if i == j:
            homology_matrix[i][j] = 1
        else:
            # calculate homology score
            homology_score = 0
            for k in range(len(WT_seqs[different_clones[i]])):
                if WT_seqs[different_clones[i]][k] == WT_seqs[different_clones[j]][k]:
                    homology_score += 1
            homology_matrix[i][j] = homology_score / len(WT_seqs[different_clones[i]])

# calculate kmer composition for each WT
def generate_kmers(k, prefix=""):
    bases = ["A", "C", "G", "T"]
    
    if k == 0:
        return [prefix]
    
    kmers = []
    for base in bases:
        kmers.extend(generate_kmers(k - 1, prefix + base))
    
    return kmers

k = 3
compositions = {}
kmer_dict = {kmer: 0 for kmer in generate_kmers(k)}
for i in range(len(different_clones)):
    seq_kmer_dict = kmer_dict.copy()
    seq = WT_seqs[different_clones[i]]
    for j in range(len(seq) - k + 1):
        kmer = seq[j:j + k]
        seq_kmer_dict[kmer] += 1
    # normalize
    total = sum(seq_kmer_dict.values())
    for kmer in seq_kmer_dict:
        seq_kmer_dict[kmer] /= total
    compositions[different_clones[i]] = list(seq_kmer_dict.values())


# # order the different clones based on homology
# homology_order = []
# for i in range(len(different_clones)):
#     for j in range(len(different_clones)):
#         if homology_matrix[i][j] == 1:
#             homology_order.append(i)
#             break
# different_clones = [different_clones[i] for i in homology_order]
# homology_matrix = homology_matrix[homology_order, :][:, homology_order]

#using kmer composition to order the different clones
coordinates = np.array(list(compositions.values()))

# start with the point furthest from the origin, and then find the point closest to the last point that is not already in the list
# by doing this, we can find the order of the points that are closest to each other
distances = np.linalg.norm(coordinates, axis=1)
start_index = np.argmax(distances)
homology_order = [start_index]
while len(homology_order) < len(coordinates):
    last_index = homology_order[-1]
    last_point = np.array(list(compositions.values())[last_index])
    distances = np.linalg.norm(coordinates - last_point, axis=1)
    distances[homology_order] = np.inf
    next_index = np.argmin(distances)
    homology_order.append(next_index)

print(different_clones)
different_clones = [different_clones[i] for i in homology_order]
homology_matrix = homology_matrix[homology_order, :][:, homology_order]
f1_results = f1_results[homology_order, :]
min_acc_results = min_acc_results[homology_order, :]
print(different_clones)


# plot homology matrix
plt.imshow(homology_matrix, cmap='gnuplot', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(different_clones)), different_clones, rotation=90)
plt.yticks(range(len(different_clones)), different_clones)
plt.xlabel('Clone')
plt.ylabel('Clone')
plt.title('Homology Score')
plt.tight_layout()
plt.savefig('cross_testing_figs/homology_matrix.png')
plt.close()


# order models based on the order of the clones
model_order = []
for clone in different_clones:
    for model_info in model_info_list:
        if model_info.enh == clone:
            model_order.append(model_info_list.index(model_info))

f1_results = f1_results[:, model_order]
min_acc_results = min_acc_results[:, model_order]
model_info_list = [model_info_list[i] for i in model_order]



plt.imshow(f1_results, cmap='gnuplot', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(model_info_list)), [str(model_info.model_id) for model_info in model_info_list], rotation=90)
plt.yticks(range(len(different_clones)), different_clones)
plt.xlabel('Model ID')
plt.ylabel('Clone')
plt.title('F1 Score')
plt.tight_layout()
plt.savefig('cross_testing_figs/cross_testing_f1.png')
plt.close()

plt.imshow(min_acc_results, cmap='gnuplot', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(model_info_list)), [str(model_info.model_id) for model_info in model_info_list], rotation=90)
plt.yticks(range(len(different_clones)), different_clones)
plt.xlabel('Model ID')
plt.ylabel('Clone')
plt.title('Min Accuracy')
plt.tight_layout()
plt.savefig('cross_testing_figs/cross_testing_min_acc.png')
plt.close()