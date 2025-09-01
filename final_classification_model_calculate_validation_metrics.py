import pandas as pd
import os
from tqdm import tqdm


# this script has already been used. no need to run it again. however, it would not hurt because  it checks if the validation set size is already calculated and skips those files.

folder = "saved_models_final/"

df = []

for file in tqdm(os.listdir(folder)):
    if file.endswith(".txt"):
        ID = file.split("_")[0]
        ENH = file.split("_")[2]

        file_path = os.path.join(folder, file)
        # print(f"ID: {ID}, ENH: {ENH}")
        validation_set_size = None
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

            if "Validation Set Size" in line:
                validation_set_size = int(line.split(":")[1].strip())
                # print(f"Validation Set Size: {validation_set_size}")                


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
            "N_training_samples": n_samples,
            "filename": file,
            "Validation_Set_Size": validation_set_size

        })


df = pd.DataFrame(df)

# calculate f1 score
df["Precision"] = df["TP"] / (df["TP"] + df["FP"])
df["Recall"] = df["TP"] / (df["TP"] + df["FN"])
df["F1 Score"] = 2 * (df["Precision"] * df["Recall"]) / (df["Precision"] + df["Recall"])
df["Accuracy"] = (df["TP"] + df["TN"]) / (df["TP"] + df["TN"] + df["FP"] + df["FN"])

# remove rows with NaN in F1 Score
df = df.dropna(subset=["F1 Score"])

# # remove enhancer E23G7
# df = df[df["ENH"] != "E23G7"]

# sort by F1 Score
df = df.sort_values(by="F1 Score", ascending=False, ignore_index=True)

# sort by enhancer
df = df.sort_values(by=["ENH", "Coverage"], ascending=[True, False], ignore_index=True)

print(df.head(25))
print(df.tail(15))


import time
from data_preprocess import get_dataloc, EnhancerDataset, get_coverage_seq_label_pairs, preset_labels, load_trained_model
from sklearn.model_selection import train_test_split
import torch
from models import SignalPredictor1D
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

previous_enh = None
previous_coverage = None


with tqdm(total=len(df)) as pbar:
    for entry in df.itertuples():
        if entry.Validation_Set_Size is not None and entry.Validation_Set_Size > 0:
            print(f"File {entry.filename} has already been processed. Validation Set Size: {entry.Validation_Set_Size}")
            pbar.update(1)
            continue

        if previous_enh != entry.ENH or previous_coverage != entry.Coverage:
            # if previous_enh != entry.ENH:
            #     print(f"Processing new enhancer: {entry.ENH} instead of {previous_enh}")
            # if previous_coverage != entry.Coverage:
            #     print(f"Processing new coverage level: {entry.Coverage} instead of {previous_coverage}")
            dataset = get_coverage_seq_label_pairs(coverage = entry.Coverage, enh_name = entry.ENH, local_path = get_dataloc(entry.ENH))
            print(f"Dataset size: {len(dataset)}")
            previous_enh = entry.ENH
            previous_coverage = entry.Coverage

            X = list(dataset.keys())
            y = list(dataset.values())

            X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   # separate training and validation set from the test set
            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)  # further split training set into training and validation sets
            

            del X_train, y_train, X_temp, y_temp
            del dataset

            LABELS = preset_labels('mm_v_all')

            validation_set = EnhancerDataset(X_val, y_val, LABELS, PADDING = True)
            validation_dataloader = torch.utils.data.DataLoader(validation_set, batch_size=256, shuffle=False, num_workers=4)

            test_set = EnhancerDataset(X_test, y_test, LABELS, PADDING = True)
            test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4)

            # print(f"Validation set size: {len(validation_set)}")

            print(f"Test set size: {len(test_set)}")
            print(f"Validation set size: {len(validation_set)}")


        # load model with correct ID
        # print(f"Loading model from file: {entry.filename}")
        model_loc = entry.filename.replace("_params.txt", ".pt")
        model_loc = os.path.join(folder, model_loc)


        model = SignalPredictor1D(1)
        model.load_state_dict(torch.load(model_loc, weights_only=True))
        model.to(device)
        model.eval()

        # run on a test batch from validation set
        labels = []
        predicitons = []

        with torch.no_grad():
            for batch in validation_dataloader:
                X_batch, y_batch = batch
                X_batch = X_batch.float()
                y_batch = y_batch.float()

                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                # run the model
                outputs = model(X_batch)

                if y_batch.shape[0] > 1:
                    labels.extend(y_batch.squeeze().detach().cpu().numpy().tolist())
                    predicitons.extend(outputs.squeeze().detach().cpu().numpy().tolist())
                else:
                    labels.append(float(y_batch.squeeze().detach().cpu().numpy()))
                    predicitons.append(float(outputs.squeeze().detach().cpu().numpy()))
            

        labels = [int(x) for x in labels]
        # check if there are both 0 and 1 in labels
        if len(set(labels)) < 2:
            print(f"Skipping file {entry.filename} because it has only one class in the validation set.")
            pbar.update(1)
            continue

        roc_auc = roc_auc_score(labels, predicitons)

        # calculate confusion matrix
        predicitons = [1 if x > 0 else 0 for x in predicitons]
        cm = confusion_matrix(labels, predicitons)

        # calculate F1 score
        precision = precision_score(labels, predicitons)
        recall = recall_score(labels, predicitons)
        f1 = 2 / (1 / precision + 1 / recall)
        acc = accuracy_score(labels, predicitons)



        # calculate predictions on the test set
        labels_test = []
        predictions_test = []
        with torch.no_grad():
            for batch in test_dataloader:
                X_batch, y_batch = batch
                X_batch = X_batch.float()
                y_batch = y_batch.float()

                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                # run the model
                outputs = model(X_batch)

                if len(outputs.shape) > 1:
                    outputs = outputs.squeeze()
                if len(y_batch.shape) > 1:
                    y_batch = y_batch.squeeze()

                labels_test.extend(y_batch.detach().cpu().numpy().tolist())
                predictions_test.extend(outputs.detach().cpu().numpy().tolist())

        # calculate auroc on the test set
        labels_test = [int(x) for x in labels_test]
        roc_auc_test = roc_auc_score(labels_test, predictions_test)

        # calculate confusion matrix on the test set
        predictions_test = [1 if x > 0 else 0 for x in predictions_test]
        cm_test = confusion_matrix(labels_test, predictions_test)
        # calculate F1 score on the test set
        precision_test = precision_score(labels_test, predictions_test)
        recall_test = recall_score(labels_test, predictions_test)
        f1_test = 2 / (1 / precision_test + 1 / recall_test)
        acc_test = accuracy_score(labels_test, predictions_test)


        # add each to the params file
        with open(os.path.join(folder, entry.filename), "a") as f:
            f.write(f"Validation ROC AUC: {roc_auc}\n")
            f.write(f"Validation Confusion Matrix: {cm.tolist()}\n")
            f.write(f"Validation Precision: {precision}\n")
            f.write(f"Validation Recall: {recall}\n")
            f.write(f"Validation F1 Score: {f1}\n")
            f.write(f"Validation Accuracy: {acc}\n")
            f.write(f"Test ROC AUC: {roc_auc_test}\n")
            f.write(f"Test Confusion Matrix: {cm_test.tolist()}\n")
            f.write(f"Test Precision: {precision_test}\n")
            f.write(f"Test Recall: {recall_test}\n")
            f.write(f"Test F1 Score: {f1_test}\n")
            f.write(f"Test Accuracy: {acc_test}\n")
            f.write(f"Validation Set Size: {len(validation_set)}\n")
            f.write(f"Test Set Size: {len(test_set)}\n")



        pbar.update(1)


exit()
