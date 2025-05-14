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
    def __init__(self, model_id):
        self.model_id = model_id
        self.model_info = self.get_model_info(model_id)
        self.enh = self.model_info['Enhancer']
        self.dataloc = get_dataloc(self.enh)

        for f in os.listdir('saved_models'):
            if f.startswith(str(model_id)):
                if f.endswith('.pt'):
                    self.model_path = 'saved_models/' + f
                    break
        
        if self.model_path is None:
            raise ValueError(f"Model path not found for model id: {model_id}")

    def get_model_info(self, model_id):
        for f in os.listdir('saved_models'):
            # find "saved_models/103432_0327_E22P3G3_mm_v_all_params.txt"
            if f.startswith(str(model_id)):
                if f.endswith('params.txt'):
                    with open('saved_models/' + f, 'r') as file:
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


savedir = "model_comaprison/E25E10_different_labels"

if not os.path.exists(savedir):
    os.makedirs(savedir)


model_ref = 175800
model_test = 170543

PADDING = True

# if False:
#     temp = model_ref
#     model_ref = model_test
#     model_test = temp

# models = [162929, 161039, 171106] #20
# models = [135230, 145614, 162929, 131953, 105959, 105933, 104814] #ANK
# models = [104335, 152018, 175800]
# models = [112258, 121329, 170543]
models = [162929, 152732]

for model_ref_n in models:
    for model_test_n in models:
        if model_ref_n == model_test_n:
            continue
            
        if PADDING:
            NAME = f"pad_{model_ref_n}_{model_test_n}"
        else:
            NAME = f"nopad_{model_ref_n}_{model_test_n}"


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            
        model_ref_info = ModelInfo(model_ref_n)
        model_test_info = ModelInfo(model_test_n)
        print(f"Model Ref: {model_ref_info.enh}, Model Test: {model_test_info.enh}")
        model_ref = model_ref_info.load_model().to(device)
        model_test = model_test_info.load_model().to(device)

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

        fig = plt.figure(figsize=(10, 10))
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

        plt.setp(ax_x_hist.get_xticklabels(), visible=False)
        plt.setp(ax_y_hist.get_yticklabels(), visible=False)

        plt.tight_layout()
        plt.savefig(f'{savedir}/{NAME}_model_ref_vs_model_test.png')
        plt.show()
        plt.close()


        # fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        # ax[0].scatter(model_ref_output, ref_labels, c=ref_labels, cmap='viridis', alpha=0.1)
        # ax[0].set_xlabel('Model Ref Output')
        # ax[0].set_ylabel('Labels')
        # ax[0].set_title('Model Ref Output vs Labels')
        # ax[1].scatter(model_test_output, ref_labels, c=ref_labels, cmap='viridis', alpha=0.1)
        # ax[1].set_xlabel('Model Test Output')
        # ax[1].set_ylabel('Labels')
        # ax[1].set_title('Model Test Output vs Labels')
        # plt.savefig(f'{savedir}/{NAME}_model_test_vs_labels.png')
        # plt.show()


        best_f1 = 0
        best_threshold = 0
        best_sign = 0
        h = []
        for threshold in np.arange(-25, 15, 0.1):
            f1 = f1_score(ref_labels, model_test_output > threshold)
            h.append((threshold, f1))
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_sign = 1

        for threshold in np.arange(-15, 25, 0.1):
            f1 = f1_score(ref_labels, -model_test_output > threshold)
            h.append((threshold, -f1))
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_sign = -1

        print(f"Model Test F1 Score: {best_f1}, threshold: {best_threshold}, sign: {best_sign}")

        plt.figure(figsize=(10, 5))
        plt.plot([x[0] for x in h if x[1] > 0], [x[1] for x in h if x[1] > 0], label='F1 Score')
        plt.plot([x[0] for x in h if x[1] < 0], [x[1] for x in h if x[1] < 0], label='Negative F1 Score')
        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.title('F1 Score vs Threshold')
        plt.axhline(y=best_f1, color='r', linestyle='--')
        plt.axhline(y=-best_f1, color='r', linestyle='--')
        plt.axvline(x=best_threshold, color='g', linestyle='--')
        plt.savefig(f'{savedir}/{NAME}_f1_score_vs_threshold.png')
        plt.show()


        # compute confusion matrix
        from sklearn.metrics import confusion_matrix

        confusion_ref = confusion_matrix(ref_labels, model_ref_output > 0.5)
        confusion_test = confusion_matrix(ref_labels, model_test_output > 0.5)
        confusion_test_best = confusion_matrix(ref_labels, best_sign*model_test_output > best_threshold)

        confusion_ref = confusion_ref.astype('float') / confusion_ref.sum(axis=1)[:, np.newaxis]
        confusion_test = confusion_test.astype('float') / confusion_test.sum(axis=1)[:, np.newaxis]
        confusion_test_best = confusion_test_best.astype('float') / confusion_test_best.sum(axis=1)[:, np.newaxis]

        # plot confusion matrix
        fig, ax = plt.subplots(1, 3, figsize=(20, 10))
        ax[0].imshow(confusion_ref, cmap='Blues', vmin=0, vmax=1)
        ax[0].set_xlabel('Predicted')
        ax[0].set_ylabel('True')
        ax[0].set_xticks([0, 1])
        ax[0].set_xticklabels(['Negative', 'Positive'])
        ax[0].set_yticks([0, 1])
        ax[0].set_yticklabels(['Negative', 'Positive'])
        ax[1].imshow(confusion_test, cmap='Blues', vmin=0, vmax=1)
        ax[1].set_xlabel('Predicted')
        ax[1].set_ylabel('True')
        ax[1].set_xticks([0, 1])
        ax[1].set_xticklabels(['Negative', 'Positive'])
        ax[1].set_yticks([0, 1])
        ax[1].set_yticklabels(['Negative', 'Positive'])
        ax[2].imshow(confusion_test_best, cmap='Blues', vmin=0, vmax=1)
        ax[2].set_xlabel('Predicted')
        ax[2].set_ylabel('True')
        ax[2].set_xticks([0, 1])
        ax[2].set_xticklabels(['Negative', 'Positive'])
        ax[2].set_yticks([0, 1])
        ax[2].set_yticklabels(['Negative', 'Positive'])
        if model_ref_info.enh == model_test_info.enh:
            ax[0].set_title(f'Confusion Matrix of Model {model_ref_info.model_id} on task {model_ref_info.model_info["Label"]}')
            ax[1].set_title(f'Confusion Matrix of Model {model_test_info.model_id} on task {model_ref_info.model_info["Label"]}')
            ax[2].set_title(f'Confusion Matrix of Model {model_test_info.model_id} on task {model_ref_info.model_info["Label"]} (Best Threshold)')
        else:
            ax[0].set_title(f'Confusion Matrix of Model Trained on {model_ref_info.enh} ({model_ref_info.enh})')
            ax[1].set_title(f'Confusion Matrix on dataset {model_ref_info.enh} of Model Trained on {model_test_info.enh}')
            ax[2].set_title(f'Confusion Matrix on dataset {model_ref_info.enh} of Model Trained on {model_test_info.enh} (Best Threshold)')


        # write the numbers on the confusion matrix
        for i in range(2):
            for j in range(2):
                ax[0].text(j, i, f'{confusion_ref[i, j]:.2f}', ha='center', va='center', color='black')
                ax[1].text(j, i, f'{confusion_test[i, j]:.2f}', ha='center', va='center', color='black')
                ax[2].text(j, i, f'{confusion_test_best[i, j]:.2f}', ha='center', va='center', color='black')
        plt.tight_layout()
        plt.savefig(f'{savedir}/{NAME}_confusion_matrix.png')
        plt.show()