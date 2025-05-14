#!/bin/bash

echo "Running training scripts for E23H3 with different hyperparameters..."

# for W in 0 1e-12 1e-5; do
#     for L in 17e-6 10e-6 5e-6; do
#         python 1_train_net.py --datafolder Fasta_Pool4 --enh E24C3 --lr $L --wd $W --epochs 50
#     done
# done

# for W in 0 1e-12 1e-5; do
#     for L in 2.5e-5 2e-5 18e-6; do
#         python 1_train_net.py --datafolder Fasta_Pool4 --enh E25B2 --lr $L --wd $W --epochs 50
#     done
# done

for W in 0; do
    for L in 10e-6; do
        python 1_train_net.py --datafolder Fasta_Pool4 --enh E25E102 --lr $L --wd $W --epochs 50
    done
done

echo "Training complete!"
