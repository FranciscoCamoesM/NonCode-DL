import os
from data_preprocess import load_wt, pad_samples, get_dataloc, get_seq_label_pairs, get_coverage_seq_label_pairs, EnhancerDataset
from data_preprocess import load_trained_model, one_hot_encode, simple_pad


import matplotlib.pyplot as plt

from models import SignalPredictor1D
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd

from tangermeme.deep_lift_shap import deep_lift_shap
from satmut_functions import plot_deeplift

import argparse

# example way to run the script from command line:
# python 11_run_deeplift_regression.py --id 2025-07-18_15-17-32 --enh E25E102 --device cuda --images_save_dir deeplift_logos/ --batch_size 256 --model_dir ../NonCode/regression_models/ --n_refs 100


def run_deeplift_regression(model_id, enh, device, n_refs, model_save_dir, output_dir, DPI=150):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # if device is a string, convert
    if isinstance(device, str):
        # Set device
        if device == "cuda":
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if not torch.cuda.is_available():
                print("CUDA is not available, using CPU instead")
        elif device == "cpu":
            device = torch.device('cpu')
        else:
            raise ValueError(f"Invalid device. Please choose 'cuda' or 'cpu'. Got {device} instead.")


    # Load model
    model = SignalPredictor1D(1)
    model_dir = os.path.join(model_save_dir, enh, model_id)
    model = load_trained_model(enh, model_dir, model, device=device)
    model.eval()

    # Load WT enhancer
    WT_seq = load_wt(enh.strip())
    WT_seq = simple_pad(WT_seq)
    WT_enhancer_OH = one_hot_encode(WT_seq).T  # Shape (4, sequence_length)
    WT_enhancer_tensor = torch.tensor(WT_enhancer_OH, dtype=torch.float32).unsqueeze(0) # we add a dimenstion with size 1 using unsqueeze to represent the number of samples, which in our case is only 1: the WT enhancer
    device_name = device.type if isinstance(device, torch.device) else str(device)
    attr_shap_values = deep_lift_shap(model, WT_enhancer_tensor, hypothetical=True, device=device_name, n_shuffles=n_refs)

    attr_shap_values = attr_shap_values.cpu().detach().numpy()[0]  # Get only the first sample (WT enhancer) and convert to numpy array
    fig_save_loc = os.path.join(output_dir, f"{enh}_{model_id}_deeplift_shap.png")

    plot_deeplift(attr_shap_values, fig_save_loc, WT_enhancer=WT_enhancer_OH, enh_name=enh, DPI=DPI)

    # save the shap values to a csv file
    values_out_file = os.path.join(output_dir, f"{enh}_{model_id}_deeplift_shap.tsv")
    save_df = pd.DataFrame(attr_shap_values.T, columns=["A", "C", "G", "T"])
    save_df["WT_Nucleotide"] = [nuc for nuc in WT_seq]
    save_df["Position"] = range(-24, len(WT_seq)-24)
    # reorder columns to have Position and WT_Nucleotide first
    save_df = save_df[["Position", "WT_Nucleotide", "A", "C", "G", "T"]]
    save_df.to_csv(values_out_file, index=False, sep="\t")

    print(f"Saved SHAP values to {values_out_file}")
    print(f"Saved DeepLIFT SHAP plot to {fig_save_loc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate DeepLIFT SHAP values for enhancer sequences')

    parser.add_argument("--id", type=str, required=True, help="model ID")
    parser.add_argument("--enh", type=str, required=True, help="Enhancer name")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--images_save_dir", type=str, default="temp_figs/", help="Directory to save images")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for processing")
    parser.add_argument("--model_dir", type=str, default="regression_models/", help="Directory containing the model")
    parser.add_argument('--n_refs', type=int, default=100, help='Number of references to use for DeepLift SHAP')
    parser.add_argument('--DPI', type=int, default=150, help='DPI for saved images')

    args = parser.parse_args()

    SAVE_DIR = args.model_dir
    N_REFS = args.n_refs
    model_id = args.id
    output_dir = args.images_save_dir
    enh = args.enh
    DPI = args.DPI

    if model_id is None:
        model_id = input("Enter the model id: ")

        
    # Set device
    if args.device == "cuda":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            print("CUDA is not available, using CPU instead")
    elif args.device == "cpu":
        device = torch.device('cpu')

    run_deeplift_regression(model_id=model_id, enh=enh, device=device, n_refs=N_REFS, model_save_dir=SAVE_DIR, output_dir=output_dir, DPI=DPI)
