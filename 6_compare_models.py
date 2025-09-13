import os
import torch
import torch.nn as nn
from models import *
from data_preprocess import get_coverage_seq_label_pairs, EnhancerDataset, preset_labels, get_dataloc
from data_preprocess import get_extreme_seq_label_pairs
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import train_test_split
import numpy as np

class ModelInfo():
    # init model id, and fetch model info about enh, dataloc, etc.
    def __init__(self, model_id, saved_models_dir = "saved_models"):
        self.model_id = model_id
        self.saved_models_dir = saved_models_dir
        self.model_info = self.get_model_info(model_id)
        self.enh = self.model_info['Enhancer']
        self.dataloc = get_dataloc(self.enh)

        for f in os.listdir(self.saved_models_dir):
            if f.startswith(str(model_id)):
                if f.endswith('.pt'):
                    self.model_path = os.path.join(self.saved_models_dir, f)
                    break
        
        if self.model_path is None:
            raise ValueError(f"Model path not found for model id: {model_id}")

    def get_model_info(self, model_id):
        for f in os.listdir(self.saved_models_dir):
            if f.startswith(str(model_id)):
                if f.endswith('params.txt'):
                    with open(os.path.join(self.saved_models_dir, f), 'r') as file:
                        lines = file.readlines()
                        model_info = {}
                        for line in lines:
                            s = line.strip().split(': ')
                            key, value = s[0], s[1]
                            model_info[key] = value
                        return model_info
        raise ValueError(f"Model info not found for model id: {model_id}")
    
    def load_model(self):
        model = SignalPredictor1D(1)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        return model





PADDING = True



models = [205132638, 194102032] # chr7:127258124-127258324
savedir = "model_comparison/chr7:127258124-127258324"

models = [192153143, 188213332] #chr11:2858382-2858582
savedir = "model_comparison/chr11:2858382-2858582"

models = [181225653, 193164951, 182183523] # chr11:17429230-17429430
savedir = "model_comparison/chr11:17429230-17429430"

models = [205105342, 191161116, 191212610, 195115650] # chr20:43021876-43022076
savedir = "model_comparison/chr20:43021876-43022076"

models = [191162606, 191195004, 191163547, 182012424, 187134752, 191164107, 191165846] # ank
savedir = "model_comparison/chr8:41511631-41511831"

models = [205105342, 191212610]
savedir = "model_comparison/temp"

if not os.path.exists(savedir):
    os.makedirs(savedir)

          
SAVED_MODELS_DIR = "saved_models_final"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for model_ref_n in models:
    model_ref_info = ModelInfo(model_ref_n, saved_models_dir=SAVED_MODELS_DIR)
    model_ref = model_ref_info.load_model().to(device)
    ref_coverage = int(model_ref_info.model_info.get('Coverage', 2))
    print(f"Model Ref: {model_ref_info.enh}, Coverage: {ref_coverage}")


    ref_dataset = get_coverage_seq_label_pairs(coverage = ref_coverage, enh_name = model_ref_info.enh, local_path = model_ref_info.dataloc)
    X_ref = list(ref_dataset.keys())
    y_ref = list(ref_dataset.values())
    X_biased_ref, X_test_ref, y_biased_ref, y_test_ref = train_test_split(X_ref, y_ref, test_size=0.2, random_state=42)   # separate training and validation set from the test set

    # free up memory
    del ref_dataset, X_ref, y_ref, X_biased_ref, y_biased_ref


    for model_test_n in models:
        NAME = f"{model_ref_n}_{model_test_n}"


        if model_ref_n == model_test_n:
            with open(f'{savedir}/{NAME}_metrics.txt', 'w') as f:
                f.write(f"Pearson Correlation: NaN\n")
                f.write(f"Spearman Correlation: NaN\n")
                f.write(f"AUC Score: {float(model_ref_info.model_info["Test ROC AUC"]):.2f}\n")
                f.write(f"Model Ref: {model_ref_info.model_id} ({model_ref_info.enh})\n")
                f.write(f"Model Test: {model_ref_info.model_id} ({model_ref_info.enh})\n")
            continue
            

        model_test_info = ModelInfo(model_test_n, saved_models_dir=SAVED_MODELS_DIR)
        model_test = model_test_info.load_model().to(device)
        print(f"Model Ref: {model_ref_info.enh}, Model Test: {model_test_info.enh}")

        LABELS = preset_labels(model_ref_info.model_info['Label'])

        test_coverage = int(model_test_info.model_info.get('Coverage', -1))
        
        if test_coverage == -1:
            print(f"Test Coverage not found for model {model_test_info.model_id}, exiting...")
            exit()

        test_dataset = get_coverage_seq_label_pairs(coverage = test_coverage, enh_name = model_test_info.enh, local_path = model_test_info.dataloc)
        X_test = list(test_dataset.keys())
        y_test = list(test_dataset.values())
        X_biased_test, X_test_test, y_biased_test, y_test_test = train_test_split(X_test, y_test, test_size=0.2, random_state=42)   # separate training and validation set from the test set


        # remove biased sequences from the test set
        dataset = list(zip(X_test_ref, y_test_ref))
        # dataset = [(x, y) for x, y in dataset if x not in X_biased_test]

        if len(dataset) < 2:
            print(f"No sequences left after removing biased sequences for model {model_ref_info.model_id} vs {model_test_info.model_id}, skipping...")
            continue

        X, y = zip(*dataset)
        X = list(X)
        y = list(y)


        from collections import Counter
        print("Label Distribution:", Counter(y))
        dataset = EnhancerDataset(X, y, LABELS, PADDING = PADDING)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)

        # gewt output for each model
        model_ref_output = []
        model_test_output = []
        ref_labels = []
        for i, data in enumerate(dataloader):
            seqs, labels = data
            seqs = torch.tensor(seqs, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.float32)
            seqs = seqs.to(device)
            model_ref_output.append(model_ref(seqs).cpu().detach().numpy())
            model_test_output.append(model_test(seqs).cpu().detach().numpy())
            ref_labels.append(labels.cpu().detach().numpy())
        model_ref_output = np.concatenate(model_ref_output)
        model_test_output = np.concatenate(model_test_output)
        ref_labels = np.concatenate(ref_labels)
        print("Label Distribution:", Counter([str(i) for i in ref_labels]))

        # plot output

        fig = plt.figure(figsize=(8, 8))
        gs = GridSpec(4, 4)

        ax_main = fig.add_subplot(gs[1:4, 0:3])
        if len(model_ref_output) > 1000:
            ax_main.scatter(model_ref_output, model_test_output, c=ref_labels, cmap='bwr', alpha=0.1)
        else:
            ax_main.scatter(model_ref_output, model_test_output, c=ref_labels, cmap='bwr', alpha=0.5)
        ax_main.set_xlabel('Reference Model Output')
        ax_main.set_ylabel('Tested Model Output')
        if model_ref_info.enh == model_test_info.enh:
            ax_main.set_title(f'Reference Model {model_ref_info.model_id} ({model_ref_info.model_info["Label"]}) vs Tested Model {model_test_info.model_id} ({model_test_info.model_info["Label"]})')
            
        else:
            ax_main.set_title(f'Reference Model {model_ref_info.model_id} ({model_ref_info.enh}) vs Tested Model {model_test_info.model_id} ({model_test_info.enh})')

        positive_data_ref = model_ref_output[ref_labels == 1]
        positive_data_test = model_test_output[ref_labels == 1]
        negative_data_ref = model_ref_output[ref_labels == 0]
        negative_data_test = model_test_output[ref_labels == 0]

        ax_x_hist = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
        ax_x_hist.hist(positive_data_ref, bins=50, alpha=0.5, color= 'red', density=True)
        ax_x_hist.hist(negative_data_ref, bins=50, alpha=0.5, color= 'blue', density=True)
        ax_x_hist.set_ylabel('Frequency')
        if model_ref_info.enh == model_test_info.enh:
            ax_x_hist.set_title(f'Reference Model {model_ref_info.model_id} on {model_ref_info.model_info["Label"]}')
        else:
            ax_x_hist.set_title(f'Reference Model trained on {model_ref_info.enh}')

        ax_y_hist = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
        ax_y_hist.hist(positive_data_test, bins=50, alpha=0.5, orientation='horizontal', color= 'red', density=True)
        ax_y_hist.hist(negative_data_test, bins=50, alpha=0.5, orientation='horizontal', color= 'blue', density=True)
        ax_y_hist.set_xlabel('Frequency')
        if model_ref_info.enh == model_test_info.enh:
            ax_y_hist.set_title(f'Tested Model {model_test_info.model_id} on {model_ref_info.model_info["Label"]}')
        else:
            ax_y_hist.set_title(f'Tested Model trained on {model_test_info.enh}')

        # calculate correlation metrics
        from scipy.stats import pearsonr, spearmanr
        from sklearn.metrics import roc_auc_score

        pearson_corr = pearsonr(model_ref_output, model_test_output)[0][0]
        spearman_corr = spearmanr(model_ref_output, model_test_output)[0]
        if len(set([str(x) for x in ref_labels])) < 2:
            auc_score = 0
        else:
            auc_score = roc_auc_score(ref_labels, model_test_output)
        
        print(f"Pearson Correlation: {pearson_corr}, Spearman Correlation: {spearman_corr}, AUC Score: {auc_score}", "\n\n", "-"*50)

        ax_main.text(0.05, 0.95, f'Pearson Correlation: {pearson_corr:.2f}\nSpearman Correlation: {spearman_corr:.2f}\nAUC Score of Tested Model: {auc_score:.2f}',
                     transform=ax_main.transAxes, fontsize=14,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        # explain the colors in the scatter plot with a legend
        import matplotlib.patches as mpatches
        legend_elements = [
            mpatches.Patch(color='red', label='Positive Labels'),
            mpatches.Patch(color='blue', label='Negative Labels')
        ]
        ax_main.legend(handles=legend_elements, loc='lower right')


        plt.setp(ax_x_hist.get_xticklabels(), visible=False)
        plt.setp(ax_y_hist.get_yticklabels(), visible=False)

        plt.tight_layout()
        plt.savefig(f'{savedir}/{NAME}_model_ref_vs_model_test.png')
        plt.show()
        plt.close()

        # save metrics to a text file
        with open(f'{savedir}/{NAME}_metrics.txt', 'w') as f:
            f.write(f"Pearson Correlation: {pearson_corr:.2f}\n")
            f.write(f"Spearman Correlation: {spearman_corr:.2f}\n")
            f.write(f"AUC Score: {auc_score:.2f}\n")
            f.write(f"Model Ref: {model_ref_info.model_id} ({model_ref_info.enh})\n")
            f.write(f"Model Test: {model_test_info.model_id} ({model_test_info.enh})\n")

        

print("-"*50,"\n\n")
print(f"All models saved in {savedir}")