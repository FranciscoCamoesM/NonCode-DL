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

            if "Jiggle" in line:
                jiggle = int(line.split(":")[1].strip())
                # print(f"Jiggle: {jiggle}")

            if "Unnormalized Confusion Matrix" in line:
                # print("Unnormalized Confusion Matrix:")
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

            elif "Confusion Matrix" in line:
                print("Confusion Matrix:")
                matrix = line.split(":")[1].strip()
                matrix = matrix.replace(" ", "").replace("[", "").replace("]", "")
                matrix = [float(x) for x in matrix.split(",")]
                if len(matrix) != 4:
                    print(f"Unexpected matrix length in file {file}: {matrix}")
                    continue
                Norm_TN = matrix[0]
                Norm_FP = matrix[1]
                Norm_FN = matrix[2]
                Norm_TP = matrix[3]
                print(f"TN: {Norm_TN}, FP: {Norm_FP}, FN: {Norm_FN}, TP: {Norm_TP}")

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


        df.append({
            "ID": ID,
            "ENH": ENH,
            "Coverage": coverage,
            "Jiggle": jiggle,
            "Weight Decay": weight_decay,
            "Learning Rate": learning_rate,
            "Epochs": epochs,
            "TN": TN,
            "FP": FP,
            "FN": FN,
            "TP": TP,
            "Norm_TN": Norm_TN,
            "Norm_FP": Norm_FP,
            "Norm_FN": Norm_FN,
            "Norm_TP": Norm_TP,
            "N_training_samples": n_samples
        })


df = pd.DataFrame(df)

# calculate f1 score
df["Precision"] = df["TP"] / (df["TP"] + df["FP"])
df["Recall"] = df["TP"] / (df["TP"] + df["FN"])
df["F1 Score"] = 2 * (df["Precision"] * df["Recall"]) / (df["Precision"] + df["Recall"])

# remove rows with NaN in F1 Score
df = df.dropna(subset=["F1 Score"])

# sort by F1 Score
df = df.sort_values(by="F1 Score", ascending=False, ignore_index=True)

print(df.head(25))
# print(df.tail(15))

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
                "F1 Score": best_model["F1 Score"],
                "Weight Decay": best_model["Weight Decay"],
                "Learning Rate": best_model["Learning Rate"],
                "Epochs": best_model["Epochs"]
            })

    # input()

final_table = pd.DataFrame(final_table)

# save the final table to a csv file
print(final_table.head(25))