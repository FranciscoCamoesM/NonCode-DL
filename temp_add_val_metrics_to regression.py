import pandas as pd
import os
from tqdm import tqdm
from data_preprocess import get_bin_cells, get_interpolated_seq_label_pairs, get_dataloc, one_hot_encode, simple_pad_batch
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from models import SignalPredictor1D
from sklearn.model_selection import train_test_split

folder = "regression_models/"
exit()
df = []
AA = 1
BB = .45/.95
lv =  {"MM": -AA, "NM": -BB, "NP":BB, "PP":AA}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pad_with_jiggles(seqs, jiggle_lim):
    padded_seqs = []
    for j in range(-jiggle_lim, jiggle_lim + 1):
        padded_seqs.extend(simple_pad_batch(seqs, jiggle=j))
    return padded_seqs
def pad_dataset(d, jiggle_lim):
    seqs, values = list(zip(*d))
    padded_seqs = pad_with_jiggles(seqs, jiggle_lim)
    padded_values = values * (2 * jiggle_lim + 1)
    padded_seqs = [one_hot_encode(seq) for seq in padded_seqs]
    return list(zip(padded_seqs, padded_values))

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, label = self.data[idx]
        return seq.T, label


for enh_folder in tqdm(os.listdir(folder)):
    dataset = None
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
                jiggle = -1
                for line in open(file_path):
                    if "ENH List" in line:
                        coverage = int(line.split(":")[1].split(",")[0].split("[")[2].strip())
                        # print(f"Coverage: {coverage}")

                    if "Jiggle" in line:
                        jiggle = int(line.split(":")[1].strip())
                        # print(f"Jiggle: {jiggle}")

                    if "Pierson Correlation" in line:
                        pierson_corr = float(line.split(":")[1].strip())
                        # print(f"Pierson Correlation: {pierson_corr}")
                    
                    if "Mean Squared Error" in line:
                        mse = float(line.split(":")[1].strip())
                        
                    if "R^2 Score" in line:
                        r2_score = float(line.split(":")[1].strip())

                    if "Weight Decay" in line:
                        weight_decay = float(line.split(":")[1].strip())
                        # print(f"Weight Decay: {weight_decay}")

                    if "Learning Rate" in line:
                        learning_rate = float(line.split(":")[1].strip())
                        # print(f"Learning Rate: {learning_rate}")

                    if "Epochs" in line:
                        epochs = int(line.split(":")[1].strip())
                        # print(f"Epochs: {epochs}")


                df.append({
                    "ID": ID,
                    "ENH": ENH,
                    "Coverage": coverage,
                    "Jiggle": jiggle,
                    "Weight Decay": weight_decay,
                    "Learning Rate": learning_rate,
                    "Epochs": epochs,
                    "Pierson Correlation": pierson_corr,
                    "Mean Squared Error": mse,
                    "R^2 Score": r2_score,
                })


        if dataset is None:
            print(f"Loading dataset for {ENH} with coverage {coverage} and jiggle {jiggle}")
            BIN_CELLS = get_bin_cells(ENH)
            dataset = get_interpolated_seq_label_pairs(coverage=coverage, label_values=lv, enh_name=ENH, local_path=get_dataloc(ENH), BIN_CELLS=BIN_CELLS)
            x = dataset.keys()
            y = dataset.values()
            dataset = list(zip(x, y))
            dataset, test_seqs = train_test_split(dataset, test_size=0.2, random_state=42)
            train_seqs, dataset = train_test_split(dataset, test_size=0.2, random_state=42)
            
            dataset = pad_dataset(dataset, jiggle_lim=jiggle)

            dataset = CustomDataset(dataset)
            # del x, y, train_seqs, test_seqs
            dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

        # load model
        model_name = f"{ENH}_trained_model.pt"
        model_path = os.path.join(folder, enh_folder, model, model_name)
        print(f"Loading model from: {model_path}")
        # Load the model
        model = SignalPredictor1D(1)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        model.to(device)

        labels_log = []
        preds = []
        # run inference
        with torch.no_grad():
            for batch in tqdm(dataloader):
                inputs, labels = batch
                inputs = inputs.float().to(device)
                outputs = model(inputs)

                preds.extend(outputs.cpu().numpy())
                labels_log.extend(labels.cpu().numpy())

        if preds and labels_log:
            from sklearn.metrics import mean_squared_error, r2_score
            preds = np.array(preds).flatten()
            labels_log = np.array(labels_log).flatten()
            # Compute metrics here
            mse = mean_squared_error(labels_log, preds)
            r2 = r2_score(labels_log, preds)
            pierson_corr = np.corrcoef(labels_log, preds)[0, 1]
            
            # add to file
            with open(file_path, "a") as f:
                f.write(f"Coverage: {coverage}\n")
                f.write(f"Validation Pierson Correlation: {pierson_corr}\n")
                f.write(f"Validation Mean Squared Error: {mse}\n")
                f.write(f"Validation R^2 Score: {r2}\n")

            print(f"Metrics added to {file_path}")
