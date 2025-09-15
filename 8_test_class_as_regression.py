import os
import torch
import torch.nn as nn
from models import *
from data_preprocess import get_seq_label_pairs, EnhancerDataset, preset_labels, get_dataloc
from data_preprocess import get_extreme_seq_label_pairs
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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


savedir = "model_comaprison/final_model_test"

if not os.path.exists(savedir):
    os.makedirs(savedir)


PADDING = True

# if False:
#     temp = model_ref
#     model_ref = model_test
#     model_test = temp

# models = [162929, 161039, 171106] #20
# models = [135230, 145614, 162929, 131953, 105959, 105933, 104814] #ANK
# models = [104335, 152018, 175800]
# models = [112258, 121329, 170543]
SAVED_MODELS_DIR = "saved_models_final"
models = [180180434, 180202549, 180183942, 181002631, 180195946]
COVARAGE = 10
AA = 1
BB = 0.475
label_values = {"MM": -AA, "NM": -BB, "NP": BB, "PP": AA}

bin_cells = {"E22P1A3": np.array([23731, 69026, 57444, 15199]),
                "E24A3": np.array([1, 1, 1, 1]),
                "E25E10": np.array([1, 1, 1, 1]),
                "E25E102": np.array([74226,107443,103524,7799]),
                "E25E10R3": np.array([68519,111643,60318,5687]),
                "E25B2": np.array([14866,63461,30286,3838]),
                "E25B3": np.array([9327,39125,25464,2188])}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for model_ref_n in models:
    model_ref_info = ModelInfo(model_ref_n, saved_models_dir=SAVED_MODELS_DIR)
    model_ref = model_ref_info.load_model().to(device)

    interpolated_seqs_1 = get_interpolated_seq_label_pairs(enh_name=model_ref_info.enh, coverage=COVARAGE, local_path=get_dataloc(model_ref_info.enh), BIN_CELLS=bin_cells[model_ref_info.enh], label_values=label_values)
    

    LABELS = preset_labels(model_ref_info.model_info['Label'])


    dataset = get_seq_label_pairs(coverage = 2, enh_name = model_ref_info.enh, local_path = model_ref_info.dataloc)
    print("\nDataset Size:", len(dataset))
    # Split the data
    X = list(dataset.keys())
    y = list(dataset.values())
    from collections import Counter
    print("Label Distribution:", Counter(y))
    dataset = EnhancerDataset(X, y, LABELS, PADDING = PADDING)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

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

    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(4, 4)

    ax_main = fig.add_subplot(gs[1:4, 0:3])
    ax_main.scatter(model_ref_output, model_test_output, c=ref_labels, cmap='viridis', alpha=0.1)
    ax_main.set_xlabel('Model Ref Output')
    ax_main.set_ylabel('Model Test Output')
    if model_ref_info.enh == model_test_info.enh:
        ax_main.set_title(f'Model Ref {model_ref_info.model_id} ({model_ref_info.model_info["Label"]}) vs Model Test {model_test_info.model_id} ({model_test_info.model_info["Label"]})')
        
    else:
        ax_main.set_title(f'Model Ref {model_ref_info.model_id} ({model_ref_info.enh}) vs Model Test {model_test_info.model_id} ({model_test_info.enh})')

    positive_data_ref = model_ref_output[ref_labels == 1]
    positive_data_test = model_test_output[ref_labels == 1]
    negative_data_ref = model_ref_output[ref_labels == 0]
    negative_data_test = model_test_output[ref_labels == 0]

    ax_x_hist = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    ax_x_hist.hist(positive_data_ref, bins=50, alpha=0.5, color= 'yellow', density=True)
    ax_x_hist.hist(negative_data_ref, bins=50, alpha=0.5, color= 'purple', density=True)
    ax_x_hist.set_ylabel('Frequency')
    if model_ref_info.enh == model_test_info.enh:
        ax_x_hist.set_title(f'Model Ref {model_ref_info.model_id} on {model_ref_info.model_info["Label"]}')
    else:
        ax_x_hist.set_title(f'Model Ref trained on {model_ref_info.enh}')

    ax_y_hist = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
    ax_y_hist.hist(positive_data_test, bins=50, alpha=0.5, orientation='horizontal', color= 'yellow', density=True)
    ax_y_hist.hist(negative_data_test, bins=50, alpha=0.5, orientation='horizontal', color= 'purple', density=True)
    ax_y_hist.set_xlabel('Frequency')
    if model_ref_info.enh == model_test_info.enh:
        ax_y_hist.set_title(f'Model Test {model_test_info.model_id} on {model_ref_info.model_info["Label"]}')
    else:
        ax_y_hist.set_title(f'Model Test trained on {model_test_info.enh}')

    # calculate r_score
    from scipy.stats import pearsonr
    r_score = pearsonr(model_ref_output, model_test_output)[0][0]
    ax_main.text(0.05, 0.95, f'R: {r_score:.2f}', transform=ax_main.transAxes, fontsize=14,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.setp(ax_x_hist.get_xticklabels(), visible=False)
    plt.setp(ax_y_hist.get_yticklabels(), visible=False)

    plt.tight_layout()
    plt.savefig(f'{savedir}/{NAME}_model_ref_vs_model_test.png')
    plt.show()
    plt.close()

    continue