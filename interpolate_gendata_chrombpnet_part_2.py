import os
import matplotlib.pyplot as plt
from data_preprocess import center_and_cossim

import argparse
parser = argparse.ArgumentParser(description="Plot predictions from ChromBPNet.")
parser.add_argument("--enh", type=str, default="E25E10", help="Name of the dataset.")
parser.add_argument("--cov", type=int, default=100, help="Coverage of the dataset.")
args = parser.parse_args()

dataset_name = args.enh
dataset_coverage = args.cov

folder_to_read = "chrombpnet_experiments/predictions"
file_to_read = f"{dataset_name}_{dataset_coverage}_preds.txt"
save_location = "chrombpnet_experiments/plots"

# Read the predictions from the file
data = []
file_to_read = os.path.join(folder_to_read, file_to_read)
with open(file_to_read, "r") as f:
    for line in f:
        elements = line.strip().split("\t")
        if len(elements) != 3:
            Warning(f"Line '{line.strip()}' does not have 3 elements, skipping.")
            continue
        seq_id, label, pred = elements
        data.append((float(label), float(pred)))

plot_name = f"{dataset_name}_{dataset_coverage}_predictions.png"
save_path = os.path.join(save_location, plot_name)
os.makedirs(save_location, exist_ok=True)

# calculate metrics
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

true_labels = [label for label, _ in data]
predicted_values = [pred for _, pred in data]
mse = mean_squared_error(true_labels, predicted_values)
r2 = r2_score(true_labels, predicted_values)
corr = pearsonr(true_labels, predicted_values)[0]
spearman_corr = spearmanr(true_labels, predicted_values)[0]
cossim = center_and_cossim(true_labels, predicted_values)

print(f"Pearson Correlation Coefficient: {corr:.4f}")
print(f"Spearman Correlation Coefficient: {spearman_corr:.4f}")
# print(f"Cosine Similarity: {cossim:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(*zip(*data), alpha=0.5)
plt.title(f"ChromBPNet Predictions for {dataset_name} with Coverage {dataset_coverage}")
plt.xlabel("True Labels")
plt.ylabel("Predicted Values")
plt.text(0, 1, f"Pear. Corr: {corr:.4f}\nSpear. Corr: {spearman_corr:.4f}"
         , transform=plt.gca().transAxes, fontsize=9, verticalalignment='bottom', horizontalalignment='right')
plt.grid(True)
plt.savefig(save_path, dpi=150)
plt.close()

