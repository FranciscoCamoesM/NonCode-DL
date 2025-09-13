import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

folder = "regression_models/"

df = []

for enh_folder in tqdm(os.listdir(folder)):
    for model in os.listdir(os.path.join(folder, enh_folder)):
        if not os.path.isdir(os.path.join(folder, enh_folder, model)):
            continue
        for file in tqdm(os.listdir(os.path.join(folder, enh_folder, model))):
            if file.endswith(".txt"):
                print(f"Processing file: {file}")
                ID = model
                ENH = enh_folder

                file_path = os.path.join(folder, enh_folder, model, file)
                coverage = -1
                jitter = -1
                for line in open(file_path):
                    if "ENH List" in line:
                        coverage = int(line.split(":")[1].split(",")[0].split("[")[2].strip())
                        # print(f"Coverage: {coverage}")

                    if "Jitter" in line:
                        jitter = int(line.split(":")[1].strip())
                        # print(f"Jitter: {jitter}")

                    if line.startswith("Pierson Correlation"):
                        Pearson_corr = float(line.split(":")[1].strip())
                        # print(f"Pearson Correlation: {Pearson_corr}")
                    
                    if line.startswith("Mean Squared Error"):
                        mse = float(line.split(":")[1].strip())
                        
                    if line.startswith("R^2 Score"):
                        r2_score = float(line.split(":")[1].strip())

                    if line.startswith("Weight Decay"):
                        weight_decay = float(line.split(":")[1].strip())
                        # print(f"Weight Decay: {weight_decay}")

                    if line.startswith("Learning Rate"):
                        learning_rate = float(line.split(":")[1].strip())
                        # print(f"Learning Rate: {learning_rate}")

                    if line.startswith("Epochs"):
                        epochs = int(line.split(":")[1].strip())
                        # print(f"Epochs: {epochs}")

                    if line.startswith("Validation Pierson Correlation"):
                        val_Pearson_corr = float(line.split(":")[1].strip())
                        # print(f"Validation Pearson Correlation: {val_Pearson_corr}")

                    if line.startswith("Validation Mean Squared Error"):
                        val_mse = float(line.split(":")[1].strip())
                        # print(f"Validation Mean Squared Error: {val_mse}")

                    if line.startswith("Validation R^2 Score"):
                        val_r2_score = float(line.split(":")[1].strip())
                        # print(f"Validation R^2 Score: {val_r2_score}")


                df.append({
                    "ID": ID,
                    "ENH": ENH,
                    "Coverage": coverage,
                    "Jitter": jitter,
                    "Weight Decay": weight_decay,
                    "Learning Rate": learning_rate,
                    "Epochs": epochs,
                    "Test Pearson Correlation": Pearson_corr,
                    "Test Mean Squared Error": mse,
                    "Test R^2 Score": r2_score,
                    "Validation Pearson Correlation": val_Pearson_corr,
                    "Validation Mean Squared Error": val_mse,
                    "Validation R^2 Score": val_r2_score,
                })


df = pd.DataFrame(df)

# remove the entries for ENH  E22P3B3R3 with coverage of 25
df = df[~((df["ENH"] == "E22P3B3R3") & (df["Coverage"] == 25))]

# print(len(df), "files processed")
# # print(df["Coverage"].unique())
# # count coverages
# coverage_counts = df["Coverage"].value_counts()
# print(coverage_counts)

# # nem unique ENH for each coverage
# for coverage in df["Coverage"].unique():
#     subset = df[df["Coverage"] == coverage]
#     print(f"Coverage {coverage} has {len(subset['ENH'].unique())} unique ENH: {subset['ENH'].unique()}")

# # Coverage 15 has 3 unique ENH: ['E22P1D10' 'E22P3B4' 'E24C3']
# # Coverage 5 has 2 unique ENH: ['E23H5' 'E22P1C1']
# # Coverage 10 has 3 unique ENH: ['E22P2F4' 'E22P3B3' 'E22P1B925']
# # Coverage 100 has 3 unique ENH: ['E25E102' 'E25E10R3' 'E22P1F10']
# # Coverage 25 has 3 unique ENH: ['E25E10' 'E22P1A3' 'E22P3B3R3']
# # Coverage 50 has 1 unique ENH: ['E22P3B3R3']

# coverage_counts = df["ENH"].value_counts()
# print(coverage_counts)

# exit()


# group by ENH, for each ENH chose row with best Validation Pearson Corr
# best_models = df.loc[df.groupby("ENH")["Validation Pearson Correlation"].idxmax()]
# best_models = df.loc[df.groupby("ENH")["Validation Mean Squared Error"].idxmax()]
best_models = df.copy()
best_models = best_models.drop(columns=["Jitter", "Epochs", "Weight Decay", "Learning Rate"])

# set negatives to 0 in Pearson and r^2
# best_models["Validation Pearson Correlation"] = best_models["Validation Pearson Correlation"].clip(lower=0)
# best_models["Validation R^2 Score"] = best_models["Validation R^2 Score"].clip(lower=0)

# best_models["Mixed Metric"] = best_models["Validation Pearson Correlation"] * best_models["Validation R^2 Score"] / best_models["Validation Mean Squared Error"]
# best_models = best_models.loc[best_models.groupby("ENH")["Mixed Metric"].idxmax()]

## for each ENH, normalize each of the three validation metrics, by subtracting the lowest and diving my the highest, and adding it to a col called Normalized (metric_name)
# initialize cols with nans
best_models["Normalized Pearson Correlation"] = float("nan")
best_models["Normalized Inverted Mean Squared Error"] = float("nan")
best_models["Normalized R^2 Score"] = float("nan")

best_models["Inverted Mean Squared Error"] = 1/(1+best_models["Validation Mean Squared Error"])

for enh in best_models["ENH"].unique():
    subset = best_models[best_models["ENH"] == enh]
    min_val = subset[["Validation Pearson Correlation", "Inverted Mean Squared Error", "Validation R^2 Score"]].min()
    max_val = subset[["Validation Pearson Correlation", "Inverted Mean Squared Error", "Validation R^2 Score"]].max()
    best_models.loc[best_models["ENH"] == enh, "Normalized Pearson Correlation"] = (subset["Validation Pearson Correlation"] - min_val["Validation Pearson Correlation"]) / (max_val["Validation Pearson Correlation"] - min_val["Validation Pearson Correlation"])
    best_models.loc[best_models["ENH"] == enh, "Normalized Inverted Mean Squared Error"] = (subset["Inverted Mean Squared Error"] - min_val["Inverted Mean Squared Error"]) / (max_val["Inverted Mean Squared Error"] - min_val["Inverted Mean Squared Error"])
    best_models.loc[best_models["ENH"] == enh, "Normalized R^2 Score"] = (subset["Validation R^2 Score"] - min_val["Validation R^2 Score"]) / (max_val["Validation R^2 Score"] - min_val["Validation R^2 Score"])

print(best_models.head(25))


# plot all values for all enhancers
n_rows = np.ceil(np.sqrt(len(best_models["ENH"].unique())))
n_cols = np.ceil(len(best_models["ENH"].unique()) / n_rows)

print(n_rows, n_cols, len(best_models["ENH"].unique()))


fig, axs = plt.subplots(int(n_rows), int(n_cols), figsize=(24, 20), dpi=100)
axs = axs.flatten()
for i, ENH in enumerate(best_models["ENH"].unique()):
    subset = best_models[best_models["ENH"] == ENH]
    metric1 = subset["Normalized Pearson Correlation"]
    metric2 = subset["Normalized Inverted Mean Squared Error"]
    metric3 = subset["Normalized R^2 Score"]
    axs[i].scatter(metric1, metric2, s=10*(1+metric3)**2, c=metric3, label="Normalized R^2 Score", cmap='viridis', alpha=0.7, edgecolors='k')
    axs[i].set_title(f"Hyperparameter Tuning Results for {ENH}", fontsize=16)
    axs[i].legend(fontsize=12)
    axs[i].set_xlabel("Normalized Pearson Correlation", fontsize=14)
    axs[i].set_ylabel("Normalized Inverted MSE", fontsize=14)
    fig.colorbar(axs[i].collections[0], ax=axs[i], orientation='vertical', pad=0.01)
plt.tight_layout()
plt.savefig("temp_figs/.best_models_normalized_metrics.png")
plt.show()


best_models["Evaluation Metric"] = best_models["Normalized Pearson Correlation"]**2 + best_models["Normalized Inverted Mean Squared Error"]**2 + best_models["Normalized R^2 Score"]**2

best_models = best_models.loc[best_models.groupby("ENH")["Evaluation Metric"].idxmax()]

print(best_models[["ENH", "Normalized Pearson Correlation", "Normalized Inverted Mean Squared Error", "Normalized R^2 Score", "Evaluation Metric"]])
print(best_models[["ID", "ENH", "Test Pearson Correlation", "Test Mean Squared Error", "Test R^2 Score"]])



# ### plot the metrics

# fig, axs = plt.subplots(1, 3, figsize=(18, 8), dpi=100)
# metrics = ["Test Pearson Correlation", "Test Mean Squared Error", "Test R^2 Score"]

# for i, metric in enumerate(metrics):
#     if metric in best_models.columns:
#         sns.boxplot(data=best_models, y=metric, ax=axs[i], color="#A5D9ED")
#         sns.stripplot(data=best_models, y=metric, ax=axs[i], color='black', alpha=0.8)
#         axs[i].set_yticklabels(np.round(axs[i].get_yticks(), 2), fontsize=16)
#         axs[i].set_ylabel("")
#         axs[i].spines['right'].set_visible(False)
#         axs[i].spines['top'].set_visible(False)
#         if i == 0:
#             axs[i].set_ylabel("Score", fontsize=18)

#         metric = metric.replace("Test ", "")

#         if metric == "R^2 Score":
#             metric = "R-squared Score"
#         axs[i].set_xlabel(metric, fontsize=18)
#     else:
#         print(f"Column name '{metric}' not found in DataFrame.")

# plt.tight_layout()
# plt.savefig("temp_figs/best_models_test_features_violin_plots.png")
# plt.show()



print(best_models.columns)


# Set up a matplotlib figure with three subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))  # Larger width to accommodate three plots

# 1. Scatter plot for Test Pearson Correlation vs Coverage
sns.scatterplot(data=best_models, x="Coverage", y="Test Pearson Correlation", ax=axes[0])
axes[0].set_xlabel("Coverage", fontsize=14)
axes[0].set_ylabel("Pearson Correlation", fontsize=14)
axes[0].set_xscale("log")
axes[0].set_xticks(best_models["Coverage"].unique())
axes[0].set_xticklabels(best_models["Coverage"].unique(), fontsize=12, rotation=45)
axes[0].text(0.05, 0.95, f"Spearman Correlation: {best_models['Test Pearson Correlation'].corr(best_models['Coverage'], method='spearman'):.2f}", transform=axes[0].transAxes, fontsize=12, verticalalignment='top')

# 2. Scatter plot for R-squared vs Coverage
sns.scatterplot(data=best_models, x="Coverage", y="Test R^2 Score", ax=axes[1])
axes[1].set_xlabel("Coverage", fontsize=14)
axes[1].set_ylabel("R-squared Score", fontsize=14)
axes[1].set_xscale("log")
axes[1].set_xticks(best_models["Coverage"].unique())
axes[1].set_xticklabels(best_models["Coverage"].unique(), fontsize=12, rotation=45)
axes[1].text(0.05, 0.95, f"Spearman Correlation: {best_models['Test R^2 Score'].corr(best_models['Coverage'], method='spearman'):.2f}", transform=axes[1].transAxes, fontsize=12, verticalalignment='top')

# 3. Scatter plot for MSE vs Coverage
sns.scatterplot(data=best_models, x="Coverage", y="Test Mean Squared Error", ax=axes[2])
axes[2].set_xlabel("Coverage", fontsize=14)
axes[2].set_ylabel("Mean Squared Error", fontsize=14)
axes[2].set_xscale("log")
axes[2].set_xticks(best_models["Coverage"].unique())
axes[2].set_xticklabels(best_models["Coverage"].unique(), fontsize=12, rotation=45)
axes[2].text(0.05, 0.95, f"Spearman Correlation: {best_models['Test Mean Squared Error'].corr(best_models['Coverage'], method='spearman'):.2f}", transform=axes[2].transAxes, fontsize=12, verticalalignment='top')

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("temp_figs/best_models_metrics_vs_coverage.png")
plt.show()




sizes = [
["E22P1H9",	"chr17:17720324-17720524",	35,	9],
["E22P1G5",	"chr4:6280601-6280801",	106,	79],
["E23G1",	"chr15:40635821-40636021",	219,	139],
["E22P3G3",	"chr11:2858382-2858582",	1473,	875],
["E25B3",	"chr8:41511631-41511831",	4315,	2082],
["E22P3B3R2",	"chr20:43021876-43022076",	4656,	2420],
["E25B2",	"chr8:41511631-41511831",	9916,	4173],
["E22P1E6",	"chr9:4126263-4126463",	181748,	1793],
["E24A3",	"chr8:41511631-41511831",	36329,	11768],
["E22P3G3",	"chr11:2858382-2858582",	54754,	9801],
["E18C2",	"chr11:17429230-17429430",	118118,	14592],
["E22P3F2",	"chr11:17429230-17429430",	90280,	18559],
["E23H5",	"chr4:6280601-6280801",	126500,	18578],
["E22P1C1",	"chr7:127258124-127258324",	141719,	26254],
["E22P1B925",	"chr7:44197990-44198190",	207756,	22172],
["E22P2F4",	"chr11:2858382-2858582",	193969,	48584],
["E22P3B3",	"chr20:43021876-43022076",	175464,	46996],
["E22P1D10",	"chr20:43021876-43022076",	221292,	52802],
["E24C3",	"chr11:17429230-17429430",	323086,	45938],
["E22P3B4",	"chr7:127258124-127258324",	245137,	103270],
["E25E10",	"chr8:41511631-41511831",	352183,	76020],
["E22P1A3",	"chr8:41511631-41511831",	607008,	121619],
["E22P3B3R3",	"chr20:43021876-43022076",	1155818,	547671],
["E22P1F10",	"chr11:45921374-45921574",	1987076,	876407],
["E25E102",	"chr8:41511631-41511831",	2408560,	400742],
["E25E10R3",	"chr8:41511631-41511831",	4846834,	1600258],
]

# add sizes to the DataFrame
sizes_df = pd.DataFrame(sizes, columns=["ENH", "Location", "Size_Reads", "N_seqs"])
best_models = best_models.merge(sizes_df, on="ENH", how="left")

# sort by Test R^2
best_models = best_models.sort_values(by="Test Pearson Correlation", ascending=False)

print(best_models)

best_models["coverage_ratio"] = best_models["Size_Reads"] / best_models["N_seqs"]

# plot metrics agaisnt Seize_Reads and N_seqs in two subplots
fig, axs = plt.subplots(3, 3, figsize=(16, 12), dpi=100)
for i, metric in enumerate(["Test Pearson Correlation", "Test Mean Squared Error", "Test R^2 Score"]):
    sns.scatterplot(data=best_models, x="Size_Reads", y=metric, ax=axs[i, 0])
    axs[i, 0].text(0.05, 0.95, f"Spearman Correlation: {best_models[metric].corr(best_models['Size_Reads'], method='spearman'):.2f}", transform=axs[i, 0].transAxes, fontsize=12, verticalalignment='top')
    sns.scatterplot(data=best_models, x="N_seqs", y=metric, ax=axs[i, 1])
    axs[i, 1].text(0.05, 0.95, f"Spearman Correlation: {best_models[metric].corr(best_models['N_seqs'], method='spearman'):.2f}", transform=axs[i, 1].transAxes, fontsize=12, verticalalignment='top')
    sns.scatterplot(data=best_models, x="coverage_ratio", y=metric, ax=axs[i, 2])
    axs[i, 2].text(0.05, 0.95, f"Spearman Correlation: {best_models[metric].corr(best_models['coverage_ratio'], method='spearman'):.2f}", transform=axs[i, 2].transAxes, fontsize=12, verticalalignment='top')

    metric = metric.replace("Test ", "")
    if metric == "R^2 Score":
        metric = "R-squared Score"

    axs[i, 0].set_ylabel(metric, fontsize=14)
    axs[i, 0].set_xscale("log")
    
    axs[i, 1].set_ylabel(metric, fontsize=14)
    axs[i, 1].set_xscale("log")

    axs[i, 2].set_ylabel(metric, fontsize=14)

    if i == 2:
        axs[i, 0].set_xlabel("Number of Total Reads", fontsize=16)
        axs[i, 1].set_xlabel("Number of Unique Sequences", fontsize=16)
        axs[i, 2].set_xlabel("Average Coverage", fontsize=16)
    else:
        axs[i, 0].set_xlabel("", fontsize=14)
        axs[i, 1].set_xlabel("", fontsize=14)
        axs[i, 2].set_xlabel("", fontsize=14)



plt.tight_layout()
plt.savefig("temp_figs/best_models_metrics_vs_sizes.png")
plt.show()

# plot only spearma corr
fig, axs = plt.subplots(1, 3, figsize=(16, 4), dpi=100)

sns.scatterplot(data=best_models, x="Size_Reads", y="Test Pearson Correlation", ax=axs[0])
axs[0].text(0.05, 0.95, f"Spearman Correlation: {best_models["Test Pearson Correlation"].corr(best_models['Size_Reads'], method='spearman'):.2f}", transform=axs[0].transAxes, fontsize=12, verticalalignment='top')
sns.scatterplot(data=best_models, x="N_seqs", y="Test Pearson Correlation", ax=axs[1])
axs[1].text(0.05, 0.95, f"Spearman Correlation: {best_models["Test Pearson Correlation"].corr(best_models['N_seqs'], method='spearman'):.2f}", transform=axs[1].transAxes, fontsize=12, verticalalignment='top')
sns.scatterplot(data=best_models, x="coverage_ratio", y="Test Pearson Correlation", ax=axs[2])
axs[2].text(0.05, 0.95, f"Spearman Correlation: {best_models["Test Pearson Correlation"].corr(best_models['coverage_ratio'], method='spearman'):.2f}", transform=axs[2].transAxes, fontsize=12, verticalalignment='top')

axs[0].set_ylabel("Pearson Correlation", fontsize=14)
axs[0].set_xscale("log")

axs[1].set_ylabel("Pearson Correlation", fontsize=14)
axs[1].set_xscale("log")

axs[2].set_ylabel("Pearson Correlation", fontsize=14)

axs[0].set_xlabel("Number of Total Reads", fontsize=16)
axs[1].set_xlabel("Number of Unique Sequences", fontsize=16)
axs[2].set_xlabel("Average Coverage", fontsize=16)

plt.tight_layout()
plt.savefig("temp_figs/best_models_pearson_vs_sizes.png")
plt.show()



id_of_interes=[
    "2025-07-19_05-26-53",
    "2025-07-19_22-21-41",
    "2025-07-20_11-22-03",
    "2025-07-19_15-27-12",
    "2025-07-18_12-55-41",
    "2025-07-19_23-40-35",
    "2025-07-20_01-20-46",
    "2025-07-19_07-29-06",
    "2025-07-19_18-16-26",
    "2025-07-20_09-56-28",
    "2025-07-19_16-27-35",
    "2025-07-19_03-10-32",
    "2025-07-18_15-17-32",
    "2025-07-18_17-22-54"
]
    



# # divide them by enhancer
# enhncaers = df["ENH"].unique()
# print("Enhancers:", enhncaers)

# for enh in enhncaers:
#     print(f"There are {len(df[df['ENH'] == enh])} models for enhancer {enh}")

#     # print 3 best according to each of the 3 metrics
#     print("Best models by Pearson Correlation:")
#     print(df[df["ENH"] == enh].sort_values(by="Pearson Correlation", ascending=False).head(5)[["ID", "Pearson Correlation"]])
#     print("Best models by Mean Squared Error:")
#     print(df[df["ENH"] == enh].sort_values(by="Mean Squared Error", ascending=True).head(5)[["ID", "Mean Squared Error"]])
#     print("Best models by R^2 Score:")
#     print(df[df["ENH"] == enh].sort_values(by="R^2 Score", ascending=False).head(5)[["ID", "R^2 Score"]])
#     print("\n\n")

#     input("Press Enter to continue...")




location_mapping = {
"E18C2":	"chr11:17429230-17429430",
"E22P3F2":	"chr11:17429230-17429430",
"E24C3":	"chr11:17429230-17429430",
"E22P2F4":	"chr11:2858382-2858582",
"E22P3G3":	"chr11:2858382-2858582",
"E22P1F10":	"chr11:45921374-45921574",
"E23G1":	"chr15:40635821-40636021",
"E22P1H9":	"chr17:17720324-17720524",
"E22P1D10":	"chr20:43021876-43022076",
"E22P3B3":	"chr20:43021876-43022076",
"E22P3B3R2":	"chr20:43021876-43022076",
"E22P3B3R3":	"chr20:43021876-43022076",
"E22P1G5":	"chr4:6280601-6280801",
"E23H5":	"chr4:6280601-6280801",
"E22P1C1":	"chr7:127258124-127258324",
"E22P3B4":	"chr7:127258124-127258324",
"E22P1B925":	"chr7:44197990-44198190",
"E22P1A3":	"chr8:41511631-41511831",
"E24A3":	"chr8:41511631-41511831",
"E25B2":	"chr8:41511631-41511831",
"E25B3":	"chr8:41511631-41511831",
"E25E10":	"chr8:41511631-41511831",
"E25E102":	"chr8:41511631-41511831",
"E25E10R3":	"chr8:41511631-41511831",
"E22P1E6":	"chr9:4126263-4126463",
}


location_colors = {
    "chr11:17429230-17429430": "e69138",
    "chr11:2858382-2858582": "76a5af",
    "chr11:45921374-45921574": "e06666",
    "chr15:40635821-40636021": "ffff00",
    "chr17:17720324-17720524": "cc4125",
    "chr20:43021876-43022076": "6d9eeb",
    "chr4:6280601-6280801": "93c47d",
    "chr7:127258124-127258324": "8e7cc3",
    "chr7:44197990-44198190": "ff00ff",
    "chr8:41511631-41511831": "ffe599",
    "chr9:4126263-4126463": "e6b8af",
}






