from data_preprocess import *
import matplotlib.pyplot as plt

from models import SignalPredictor1D
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm


def gen_sampler_weights(data):
    class_count = [0, 0, 0, 0]
    for _, label in data:
        class_count[np.argmax(label)] += 1

    weights = []
    for _, label in data:
        weights.append(1/class_count[np.argmax(label)])

    return torch.DoubleTensor(weights)


### BEGGINIGN OF THE CODE ###

import argparse
parser = argparse.ArgumentParser(description='Train a neural network to predict enhancer activity')
parser.add_argument('--enh', type=str, default='E25E10', help='Enhancer name to train on')
parser.add_argument('--label', type=str, default='mm_v_all', help='Label title to train on')
parser.add_argument('--subset', type=float, default=1.0, help='Fraction of the data to use')
parser.add_argument('--bs', type=int, default=64, help='Batch size for training')
parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate for training')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train for')
parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay for the optimizer')
parser.add_argument('--mom', type=float, default=None, help='Momentum for the optimizer')
parser.add_argument('--optim', type=str, default='adam', help='Optimizer to use')
parser.add_argument('--best_val', type=bool, default=True, help='Whether to save the best model based on validation accuracy')
args = parser.parse_args()

ENHANCER_NAME = args.enh
LABEL_TITLE = args.label
SUBSET = args.subset
BATCH_SIZE = args.bs
LEARNING_RATE = args.lr
N_EPOCHS = args.epochs
WEIGHT_DECAY = args.wd
MOM = args.mom
OPTIM = args.optim
SUBSET = args.subset
BEST_VAL = args.best_val

SAVE_DIR = 'saved_models'



LABELS = preset_labels(LABEL_TITLE)
N_CLASSES = len(list(LABELS.values())[0])


# Load the data
dataset = get_seq_label_pairs(enh_name = ENHANCER_NAME)

# Split the data
X = list(dataset.keys())
y = list(dataset.values())

print(y[:25], list(set(y)))
X_minus = []
X_plus = []

for i in tqdm(range(len(X))):
    if y[i] == 'MM':
        X_minus.append(X[i])
    else:
        X_plus.append(X[i])

print(len(X_minus), len(X_plus))

def measure_nuc_freq(X):
    nucs = 'ACGT'
    freqs = []
    for x in X:
        freq = {nuc: 0 for nuc in nucs}
        for nuc in x:
            if nuc != 'N':
                freq[nuc] += 1
        freqs.append(freq)
    
    freqs = {nuc: [f[nuc] for f in freqs] for nuc in nucs}
    return freqs

freqs_minus = measure_nuc_freq(X_minus)
freqs_plus = measure_nuc_freq(X_plus)

def plot_nuc_freq(freqs_minus, freqs_plus):
    fig_axis = plt.subplots(2, 2, figsize=(10, 10))
    fig, ax = fig_axis
    nucs = 'ACGT'
    for i, nuc in enumerate(nucs):
        ax[i//2][i%2].hist(freqs_minus[nuc], bins=20, alpha=0.5, label='Minus', color='blue', density=False)
        ax[i//2][i%2].hist(freqs_plus[nuc], bins=20, alpha=0.5, label='Plus', color='red', density=False)
        ax[i//2][i%2].legend()
        ax[i//2][i%2].set_xlabel(f'Number of {nuc} nucleotides')
        ax[i//2][i%2].set_ylabel('Frequency')
        ax[i//2][i%2].set_title(f'Nucleotide {nuc}')
    plt.tight_layout()
    plt.show()


    fig_axis = plt.subplots(2, 2, figsize=(10, 10))
    fig, ax = fig_axis
    nucs = 'ACGT'
    for i, nuc in enumerate(nucs):
        ax[i//2][i%2].hist(freqs_minus[nuc], bins=20, alpha=0.5, label='Minus', color='blue', density=True)
        ax[i//2][i%2].hist(freqs_plus[nuc], bins=20, alpha=0.5, label='Plus', color='red', density=True)
        ax[i//2][i%2].legend()
        ax[i//2][i%2].set_xlabel(f'Number of {nuc} nucleotides')
        ax[i//2][i%2].set_ylabel('Frequency')
        ax[i//2][i%2].set_title(f'Nucleotide {nuc}')
    plt.tight_layout()

    plt.show()

import numpy as np

# describe probability function of minus for each nucleotide

def describe_prob(freqs):
    nucs = 'ACGT'
    probs = {nuc: (np.mean(freqs[nuc]), np.std(freqs[nuc])) for nuc in nucs}
    return probs

probs_minus = describe_prob(freqs_minus)

seq_probs = []
for seq in X_plus:
    dists = [seq.count(nuc) for nuc in 'ACGT']
    avg_prob = 0
    # print(seq, dists)
    for i in range(4):
        # calculate the probability of the nucleotide based on the distribution of the minus class
        # print(probs_minus['ACGT'[i]], dists[i])

        # gaussian approximation
        prob = (dists[i] - probs_minus['ACGT'[i]][0]) / probs_minus['ACGT'[i]][1]
        prob = 1/np.sqrt(2*np.pi) * np.exp(-0.5*prob**2)
        avg_prob += prob**2
        if i==1:
            avg_prob += 2 * prob**2

    # print(avg_prob)
    seq_probs.append(avg_prob)

# close all figures
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
# ax.hist(seq_probs, bins=20, alpha=0.5, label='Plus', color='red', density=True)
# plt.show()

# plot_nuc_freq(freqs_minus, freqs_plus)

# sort X_plus by the probability of being a minus class
X_plus = [x for _, x in sorted(zip(seq_probs, X_plus), reverse=True)]
seq_probs = sorted(seq_probs, reverse=True)

PERCENT = 0.5

# split the data
X_plus = X_plus[:2*len(X_minus)]
X_plus = X_plus[:int(PERCENT*len(X_plus))]


# plot the distribution of both classes again
freqs_plus = measure_nuc_freq(X_plus)
freqs_minus = measure_nuc_freq(X_minus)

# plot_nuc_freq(freqs_minus, freqs_plus)


#SAVE DATA IN A FILE
filename_minus = 'bias_corrected_files\\ultraplex_demux_Unknown_CB761-002F0002_1.fq.gz_{}{}{}.fa'.format(ENHANCER_NAME, int(100*PERCENT), 'MM')
filename_plus = 'bias_corrected_files\\ultraplex_demux_Unknown_CB761-002F0002_1.fq.gz_{}{}{}.fa'.format(ENHANCER_NAME, int(100*PERCENT), 'PP')

with open(filename_minus, 'w') as f:
    for i, x in enumerate(X_minus):
        f.write(f'>seq_{i}\n{x}\n')

with open(filename_plus, 'w') as f:
    for i, x in enumerate(X_plus):
        f.write(f'>seq_{i}\n{x}\n')