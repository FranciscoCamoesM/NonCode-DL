import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


savedir = "model_comparison/testeeee_"
savedir = "model_comparison/chr8:41511631-41511831"

pearson_corr = {}
spearman_corr = {}
auc_score = {}

for file in os.listdir(savedir):
    if file.endswith("_metrics.txt"):
        with open(os.path.join(savedir, file), 'r') as f:
            lines = f.readlines()
            pearson_corr_elem = float(lines[0].split(': ')[1].strip())
            spearman_corr_elem = float(lines[1].split(': ')[1].strip())
            auc_score_elem = float(lines[2].split(': ')[1].strip())
            ref_dataset = lines[3].split(': ')[1].strip().split('(')[1][:-1].strip()
            test_dataset = lines[4].split(': ')[1].strip().split('(')[1][:-1].strip()

        combination = f"{ref_dataset} vs {test_dataset}"

        pearson_corr[combination] = pearson_corr_elem
        spearman_corr[combination] = spearman_corr_elem
        auc_score[combination] = auc_score_elem

        # print(f"Processed {combination}:")
        # print(f"  Pearson Correlation: {pearson_corr_elem:.2f}")
        # print(f"  Spearman Correlation: {spearman_corr_elem:.2f}")
        # print(f"  AUC Score: {auc_score_elem:.2f}")


# display them in a grid

rows = []
cols = []
for combination in pearson_corr.keys():
    row, col = combination.split(' vs ')
    rows.append(row)
    cols.append(col)

rows = sorted(set(rows))
cols = sorted(set(cols))

pearson_grid = np.zeros((len(rows), len(cols)))
spearman_grid = np.zeros((len(rows), len(cols)))
auc_grid = np.zeros((len(rows), len(cols)))

for i, row in enumerate(rows):
    for j, col in enumerate(cols):
        combination = f"{row} vs {col}"
        if combination in pearson_corr and row != col:
            pearson_grid[i, j] = pearson_corr[combination]
            spearman_grid[i, j] = spearman_corr[combination]
            auc_grid[i, j] = auc_score[combination]
        else:
            # If the combination is not found, fill with a placeholder value
            pearson_grid[i, j] = -10
            spearman_grid[i, j] = -10
            if combination in auc_score:
                auc_grid[i, j] = auc_score[combination]
            else:
                auc_grid[i, j] = -10




def plot_heatmap(data, title, xlabel, ylabel, **kwargs):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=True, fmt=".2f", cbar_kws={'label': 'Score'}, square=True, linewidths=0.5, **kwargs)
    # write nan values as 'N/A' across the heatmap
    for text in plt.gca().texts:
        # print(text.get_text())
        if text.get_text() == '-10.00':
            text.set_text('N/A')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(ticks=np.arange(len(cols)) + 0.5, labels=cols, rotation=45)
    plt.yticks(ticks=np.arange(len(rows)) + 0.5, labels=rows, rotation=0)
    plt.tight_layout()
    plt.savefig(f"temp_figs/.comapre_{title.replace(' ', '_').lower()}.png")
    plt.show()

    print(f"Heatmap for {title} saved as temp_figs/.{title.replace(' ', '_').lower()}.png")

plot_heatmap(pearson_grid, 'Pearson Correlation Coefficient', "Test Model's Dataset", "Reference Model's Dataset", cmap='Blues', vmin=-0.3, vmax=1)
plot_heatmap(spearman_grid, 'Spearman Correlation Coefficient', "Test Model's Dataset", "Reference Model's Dataset", cmap='Greens', vmin=-0.25, vmax=1)
plot_heatmap(auc_grid, 'ROC AUC Score of Tested Model', "Test Model's Dataset", "Reference Model's Dataset", cmap='Oranges', vmin=0.4, vmax=1)

# replace -10 with NaN for better readability in the heatmap
pearson_grid[pearson_grid == -10] = 0
spearman_grid[spearman_grid == -10] = 0
auc_grid[auc_grid == -10] = 0

# print means of each test model's dataset
print("Means of each test model's dataset:")
for col in cols:
    pearson_mean = np.mean(pearson_grid[:, cols.index(col)])
    spearman_mean = np.mean(spearman_grid[:, cols.index(col)])
    auc_mean = np.mean(auc_grid[:, cols.index(col)])
    print(f"{col}: Pearson: {pearson_mean:.2f}, Spearman: {spearman_mean:.2f}, AUC: {auc_mean:.2f}, total: {(pearson_mean**2 + spearman_mean**2 + auc_mean**2):.2f}")

# print means of each reference model's dataset
print("\nMeans of each reference model's dataset:")
for row in rows:
    pearson_mean = np.mean(pearson_grid[rows.index(row), :])
    spearman_mean = np.mean(spearman_grid[rows.index(row), :])
    auc_mean = np.mean(auc_grid[rows.index(row), :])
    print(f"{row}: Pearson: {pearson_mean:.2f}, Spearman: {spearman_mean:.2f}, AUC: {auc_mean:.2f}, total: {(pearson_mean**2 + spearman_mean**2 + auc_mean**2):.2f}")