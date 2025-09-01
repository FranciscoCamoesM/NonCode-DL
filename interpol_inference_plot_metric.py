import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from data_preprocess import *
from models import SignalPredictor1D
from data_preprocess import get_interpolated_seq_label_pairs, get_dataloc, load_trained_model, one_hot_encode, simple_pad

folder = "reg_model_inference/chr8:41511631-41511831"
enhancer_number = 5
folder = "reg_model_inference/chr20:43021876-43022076"
enhancer_number = 12

# try to load metrics .txt
frame = pd.read_csv(os.path.join(folder, 'metrics.txt'), sep='\t')

print("Metrics:")
print(frame)

n_cols = len(frame["Training Enh"].unique())

pearson_results = np.zeros((n_cols, n_cols))
spearman_results = np.zeros((n_cols, n_cols))

enh_ids = list(frame["Training Enh"].unique())

for i, enh1 in enumerate(enh_ids):
    for j, enh2 in enumerate(enh_ids):
        pearson_results[i, j] = frame[(frame["Training Enh"] == enh2) & (frame["Inference Enh"] == enh1)]["Pearson Corr"]
        spearman_results[i, j] = frame[(frame["Training Enh"] == enh2) & (frame["Inference Enh"] == enh1)]["Spearman Corr"]



print("Pearson Correlation Results:")
print(pearson_results)

print("Spearman Correlation Results:")
print(spearman_results)


# save plots
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(pearson_results, annot=True, fmt=".2f", cmap="Oranges", xticklabels=enh_ids, yticklabels=enh_ids, ax=ax)
plt.title("Pearson Correlation Results", fontsize=16)
plt.xlabel("Training Dataset", fontsize=14)
plt.ylabel("Test Dataset", fontsize=14)
plt.savefig(os.path.join(folder, 'results_matrix_pearson.png'), dpi=150)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(spearman_results, annot=True, fmt=".2f", cmap="Greens", xticklabels=enh_ids, yticklabels=enh_ids, ax=ax)
plt.title(f"Generalization between models of Enhancer {enhancer_number}", fontsize=16)
plt.xlabel("Training Dataset", fontsize=14)
plt.ylabel("Test Dataset", fontsize=14)
plt.savefig(os.path.join(folder, 'results_matrix_spearman.png'), dpi=150)