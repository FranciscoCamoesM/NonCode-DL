import argparse
from Bio import pairwise2
import os


parser = argparse.ArgumentParser(description='allign motif with respective WT sequence')
parser.add_argument('--enh', type=int, default=None, help='Batch size for training')
parser.add_argument('--motif', type=str, default=None, help='Motif sequence')

args = parser.parse_args()

enh = args.enh
motif = args.motif

if enh is None:
    raise ValueError('Please provide the enhancer number')

if motif is None:
    raise ValueError('Please provide the motif sequence')

# Load the WT sequence
name = None
all_enhancers = []
with open(f'./original_enhancers.txt', 'r') as f:
    if name is None:
        name = f.readline().strip()[1:]
    else:
        enhancer = f.readline().strip()
        all_enhancers[name] = enhancer
        name = None


enhancer = all_enhancers[enh]

# Load the motif sequence
motif = motif.upper()


