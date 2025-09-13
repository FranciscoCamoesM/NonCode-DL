import pandas as pd
import os
from tqdm import tqdm

folder = "saved_models_final/"

df = []

for file in tqdm(os.listdir(folder)):
    if file.endswith(".txt"):
        ID = file.split("_")[0]
        ENH = file.split("_")[2]

        file_path = os.path.join(folder, file)
        # print(f"ID: {ID}, ENH: {ENH}")
        for line in open(file_path):
            if "Coverage" in line:
                coverage = int(line.split(":")[1].strip())
                # print(f"Coverage: {coverage}")

            if "Jitter" in line:
                jitter = int(line.split(":")[1].strip())
                # print(f"Jitter: {jitter}")

            if "Validation Confusion Matrix" in line:
                matrix = line.split(":")[1].strip()
                matrix = matrix.replace(" ", "").replace("[", "").replace("]", "")
                matrix = [int(x) for x in matrix.split(",")]
                if len(matrix) != 4:
                    print(f"Unexpected matrix length in file {file}: {matrix}")
                    continue
                TN = matrix[0]
                FP = matrix[1]
                FN = matrix[2]
                TP = matrix[3]
                # print(f"TN: {TN}, FP: {FP}, FN: {FN}, TP: {TP}")

            if "Weight Decay" in line:
                weight_decay = float(line.split(":")[1].strip())
                # print(f"Weight Decay: {weight_decay}")

            if "Learning Rate" in line:
                learning_rate = float(line.split(":")[1].strip())
                # print(f"Learning Rate: {learning_rate}")

            if "Epochs" in line:
                epochs = int(line.split(":")[1].strip())
                # print(f"Epochs: {epochs}")

            if "Subset" in line:
                n_samples = int(line.split(":")[2].strip()[:-1])
                # print(f"Number of training samples: {n_samples}")

            if "Validation Precision" in line:
                validation_precision = float(line.split(":")[1].strip())
                # print(f"Validation Precision: {validation_precision}")
            if "Validation Recall" in line:
                validation_recall = float(line.split(":")[1].strip())
                # print(f"Validation Recall: {validation_recall}")
            if "Validation Accuracy" in line:
                validation_accuracy = float(line.split(":")[1].strip())
                # print(f"Validation Accuracy: {validation_accuracy}")
            if "Validation F1 Score" in line:
                validation_f1_score = float(line.split(":")[1].strip())
                # print(f"Validation F1 Score: {validation_f1_score}")
            if "Validation ROC AUC" in line:
                validation_roc_auc = float(line.split(":")[1].strip())
                # print(f"Validation ROC AUC: {validation_roc_auc}")

            if "Test Confusion Matrix" in line:
                test_matrix = line.split(":")[1].strip()
                test_matrix = test_matrix.replace(" ", "").replace("[", "").replace("]", "")
                test_matrix = [int(x) for x in test_matrix.split(",")]
                if len(test_matrix) != 4:
                    print(f"Unexpected test matrix length in file {file}: {test_matrix}")
                    continue
                TN_test = test_matrix[0]
                FP_test = test_matrix[1]
                FN_test = test_matrix[2]
                TP_test = test_matrix[3]
                # print(f"Test TN: {TN_test}, FP: {FP_test}, FN: {FN_test}, TP: {TP_test}")

            if "Test Precision" in line:
                test_precision = float(line.split(":")[1].strip())
                # print(f"Test Precision: {test_precision}")
            if "Test Recall" in line:
                test_recall = float(line.split(":")[1].strip())
                # print(f"Test Recall: {test_recall}")
            if "Test Accuracy" in line:
                test_accuracy = float(line.split(":")[1].strip())
                # print(f"Test Accuracy: {test_accuracy}")
            if "Test F1 Score" in line:
                test_f1_score = float(line.split(":")[1].strip())
                # print(f"Test F1 Score: {test_f1_score}")
            if "Test ROC AUC" in line:
                test_roc_auc = float(line.split(":")[1].strip())
                # print(f"Test ROC AUC: {test_roc_auc}")


        df.append({
            "ID": ID,
            "ENH": ENH,
            "Coverage": coverage,
            "Jitter": jitter,
            "Weight Decay": weight_decay,
            "Learning Rate": learning_rate,
            "Epochs": epochs,
            "TN": TN,
            "FP": FP,
            "FN": FN,
            "TP": TP,
            "Test TN": TN_test,
            "Test FP": FP_test,
            "Test FN": FN_test,
            "Test TP": TP_test,
            "Validation Precision": validation_precision,
            "Validation Recall": validation_recall,
            "Validation Accuracy": validation_accuracy,
            "Validation F1 Score": validation_f1_score,
            "Validation ROC AUC": validation_roc_auc,
            "Test Precision": test_precision,
            "Test Recall": test_recall,
            "Test Accuracy": test_accuracy,
            "Test F1 Score": test_f1_score,
            "Test ROC AUC": test_roc_auc,
            "Number of Training Samples": n_samples
        })

        
        ID = None
        ENH = None
        coverage = None
        jitter = None
        weight_decay = None
        learning_rate = None
        epochs = None
        TN = None
        FP = None
        FN = None
        TP = None
        TN_test = None
        FP_test = None
        FN_test = None
        TP_test = None
        validation_precision = None
        validation_recall = None
        validation_accuracy = None
        validation_f1_score = None
        validation_roc_auc = None
        test_precision = None
        test_recall = None
        test_accuracy = None
        test_f1_score = None
        test_roc_auc = None
        n_samples = None


df = pd.DataFrame(df)


# remove rows with NaN in F1 Score
df = df.dropna(subset=["Validation F1 Score", "Test F1 Score"])

# # remove enhancer E23G7
# df = df[df["ENH"] != "E23G7"]

# sort by ROC AUC and then by F1 Score
df = df.sort_values(by=["Validation ROC AUC", "Validation F1 Score"], ascending=[False, False], ignore_index=True)

# sort by F1 Score and then by ROC AUC
# df = df.sort_values(by=["Validation F1 Score", "Validation ROC AUC"], ascending=[False, False], ignore_index=True)


print(df.head(25))

# params df
params_df = df[["ID", "ENH", "Coverage", "Jitter", "Weight Decay", "Learning Rate", "Epochs"]].copy()
params_df["lr_wd"] = params_df["Learning Rate"].astype(str) + "_" + params_df["Weight Decay"].astype(str)
# count how many unique lr_wd combinations there are
unique_lr_wd = params_df["lr_wd"].nunique()
print(f"Number of unique lr_wd combinations: {unique_lr_wd}")
# print count of each lr_wd combination
print(params_df["lr_wd"].value_counts())


final_table = []
# for each combination of enhancer and coverage, print the best model
for enh in df["ENH"].unique():
    for coverage in df["Coverage"].unique():
        subset = df[(df["ENH"] == enh) & (df["Coverage"] == coverage)]
        if not subset.empty:
            best_model = subset.iloc[0]
            # print(f"Best model for ENH: {enh}, Coverage: {coverage}")
            # print(best_model[["ID", "F1 Score", "Weight Decay", "Learning Rate", "Epochs"]])
            # print("-" * 50, "\n\n")
            final_table.append({
                "ENH": enh,
                "Coverage": coverage,
                "ID": best_model["ID"],
                "Precision": best_model["Test Precision"],
                "Recall": best_model["Test Recall"],
                "Accuracy": best_model["Test Accuracy"],
                "F1 Score": best_model["Test F1 Score"],
                "ROC AUC": best_model["Test ROC AUC"],
                # "Weight Decay": best_model["Weight Decay"],
                # "Learning Rate": best_model["Learning Rate"],
                # "Epochs": best_model["Epochs"],
            })



    # input()

final_table = pd.DataFrame(final_table)
print(final_table.head(25))



# reorgazine final_table to have only one row per ENH, # with columns for each coverage
final_table = final_table.pivot(index="ENH", columns="Coverage", values=["ID", "Precision", "Recall", "Accuracy", "F1 Score", "ROC AUC"]).reset_index()
# flatten the multiindex columns
final_table.columns = ['_'.join(map(str, col)).strip() for col in final_table.columns.values]
# rename the columns
final_table.columns = [col.replace("ENH_", "ENH") for col in final_table.columns]

print(final_table.columns)

# # reorganize the columns
new_order = ['ENH', 'ID_1', 'Precision_1', 'Recall_1', 'Accuracy_1', 'F1 Score_1', 'ROC AUC_1',
             'ID_3', 'Precision_3', 'Recall_3', 'Accuracy_3', 'F1 Score_3', 'ROC AUC_3',
             'ID_10', 'Precision_10', 'Recall_10', 'Accuracy_10', 'F1 Score_10', 'ROC AUC_10']

final_table = final_table[new_order]

# sort by ENH
final_table = final_table.sort_values(by="ENH", ascending=True, ignore_index=True)

# print the final table
print(final_table)



exit()











# sort by f1 score
final_table = final_table.sort_values(by="F1 Score", ascending=False, ignore_index=True)
# # print only E25E102
# if "E25E102" in final_table["ENH"].values:
#     final_table = final_table[final_table["ENH"] == "E25E102"]
#     print(final_table.head(25))
# else:
#     print("E25E102 not found in final table.")
#     print(final_table.head(25))
#     print("Exiting...")

# see which models have None in F1 Score
rows = []
for index, row in final_table.iterrows():
    if row["F1 Score"] is None:
        rows.append(index)

nna_df = final_table.loc[rows]


# count how many times each ENH appears in nna_df
nna_counts = nna_df["ENH"].value_counts()
print("ENH counts in nna_df with None F1 Score:")
print(nna_counts)

# count how many times each coverage appears in nna_df
nna_coverage_counts = nna_df["Coverage"].value_counts()
print("Coverage counts in nna_df with None F1 Score:")
print(nna_coverage_counts)

# count how many times each lr_wd appears in nna_df
nna_df["lr_wd"] = nna_df["Learning Rate"].astype(str) + "_" + nna_df["Weight Decay"].astype(str)
nna_lr_wd_counts = nna_df["lr_wd"].value_counts()
print("lr_wd counts in nna_df with None F1 Score:")
print(nna_lr_wd_counts)

print(nna_df)
exit()

# sort by coverage, then sort by enhancer
final_table = final_table.sort_values(by=["ENH", "Coverage"], ascending=[True, True], ignore_index=True)

#chang e lr and wd to scientific notation
final_table["Learning Rate"] = final_table["Learning Rate"].apply(lambda x: f"{x:.2e}")
final_table["Weight Decay"] = final_table["Weight Decay"].apply(lambda x: f"{x:.2e}")
# save the final table to a csv file
print(final_table.head(25))
# exit()


interesting_enhancers = [
    "E24A3",
    "E25B2",
    "E25B3",
    "E25E10",
    "E25E102",
    "E25E10R3"
]

interesting_enhancers = None


# for each ENH, print metrics for each coverage
for enh in final_table["ENH"].unique():
    if interesting_enhancers and enh not in interesting_enhancers:
        continue
    print(f"Metrics for ENH: {enh}")
    ids = []
    subset = final_table[final_table["ENH"] == enh]
    for coverage in subset["Coverage"].unique():
        coverage_subset = subset[subset["Coverage"] == coverage]
        print(f"Coverage: {coverage}")
        print(coverage_subset[["ID", "Precision", "Recall", "F1 Score", "Accuracy", "Weight Decay", "Learning Rate", "Epochs"]])
        # print("ID:", coverage_subset["ID"].values[0], coverage_subset["Precision"].values[0], coverage_subset["Recall"].values[0], coverage_subset["F1 Score"].values[0], coverage_subset["Accuracy"].values[0])
        print("-" * 50)
        ids.append(coverage_subset["ID"].values[0])
    print("\n\n")

    response = input("Save these cms? (y/n): ")
    if response.lower() == 'y':
        # get cms at "saved_models_final/plots/" that start with the ids and copy them to "confusion_matrices_temp/"
        for id in ids:
            for file in os.listdir("saved_models_final/plots/"):
                if file.startswith(id) and file.endswith(".png"):
                    src = os.path.join("saved_models_final/plots/", file)
                    dst = os.path.join("confusion_matrices_temp/", file)
                    print(f"Copying {src} to {dst}")
                    os.system(f"cp {src} {dst}")

            # find params and move them as well
            for file in os.listdir("saved_models_final/"):
                if file.startswith(id) and file.endswith("params.txt"):
                    src = os.path.join("saved_models_final/", file)
                    dst = os.path.join("confusion_matrices_temp/", file)
                    print(f"Copying {src} to {dst}")
                    os.system(f"cp {src} {dst}")

        
        
        
        
        print("Confusion matrices saved.")



    input("Press Enter to continue to the next enhancer...")


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


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

points = []
for index, row in final_table.iterrows():
    location = location_mapping.get(row["ENH"], "Unknown")
    color = location_colors.get(location, "k")
    points.append((row["Accuracy"], row["F1 Score"], color))

plt.figure(figsize=(12, 8))
for accuracy, f1_score, color in points:
    plt.scatter(f1_score, accuracy, color=f"#{color}", alpha=0.7, s=100)
plt.title("F1 Score vs Accuracy for Different Enhancers")
plt.xlabel("F1 Score")
plt.ylabel("Accuracy")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("temp_figs/.f1_score_vs_coverage.png")
plt.show()


points = []
for index, row in final_table.iterrows():
    location = location_mapping.get(row["ENH"], "Unknown")
    color = location_colors.get(location, "k")
    points.append((row["Precision"], row["Recall"], color))

plt.figure(figsize=(12, 8))
for precision, recall, color in points:
    plt.scatter(recall, precision, color=f"#{color}", alpha=0.7, s=100)
plt.title("Precision vs Recall for Different Enhancers")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("temp_figs/.precision_vs_recall.png")
plt.show()