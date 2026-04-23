import os
import torch
import numpy as np
import pandas as pd
import tangermeme.plot as tp
import matplotlib.pyplot as plt
from data_preprocess import one_hot_encode, load_wt, simple_pad, load_trained_model, pad_samples
from models import SignalPredictor1D
import argparse
from satmut_functions import saturation_mutagenesis, save_sat_mut, plot_saturation_mutagenesis_logo


# example way to run the script from command line:
# python 13_run_saturation_mutagenesis_regression.py --id 2025-07-18_15-17-32 --enh E25E102 --device cuda --o saturation_mutation_logos/ --batch_size 256 --model_dir ../NonCode/regression_models/



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Saturation mutagenesis for enhancer sequences')

    parser.add_argument("--id", type=str, required=True, help="model ID")
    parser.add_argument("--enh", type=str, required=True, help="Enhancer name")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--o", type=str, default="temp_figs/", help="Directory to save images")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for processing")
    parser.add_argument("--model_dir", type=str, default="regression_models/", help="Directory containing the model")
    parser.add_argument("--DPI", type=int, default=150, help="DPI for saving images")

    args = parser.parse_args()

    enh = args.enh
    MODEL_DIR = args.model_dir
    device = args.device
    model_id = args.id

    run_saturation_mutagenesis(model_id=model_id, enh=enh, device=device, output_dir=args.o, batch_size=args.batch_size, model_save_dir=MODEL_DIR, DPI=args.DPI)


def run_saturation_mutagenesis(model_id, enh, device, output_dir, batch_size, model_save_dir, DPI=150):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not torch.cuda.is_available() and device == "cuda":
        print("CUDA is not available, using CPU instead")
    device = torch.device(device if torch.cuda.is_available() else 'cpu')


    enhancer = load_wt(enh.strip())
    enhancer = simple_pad(enhancer, 250, jitter=0)
    enhancer = one_hot_encode(enhancer).T  # Shape (4, sequence_length)
    enhancer = torch.tensor(enhancer, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)

    model_dir = os.path.join(model_save_dir, enh, model_id)
    model = SignalPredictor1D(1)

    # Load model
    model = load_trained_model(enh, model_dir, model, device=device)
    model.to(device)
    model.eval()

    # Load WT sequence
    WT_SEQ = load_wt(enh.strip())
    WT_SEQ = simple_pad(WT_SEQ, 250, jitter=0)

    # run saturation mutagenesis
    with torch.no_grad():
        sat_mut, deletion_effects, WT_value = saturation_mutagenesis(model, enh, device, BATCH_SIZE=batch_size, WT_SEQ=WT_SEQ)
    plot_saturation_mutagenesis_logo(sat_mut, deletion_effects, WT_value, enh, output_dir, model_id, WT_SEQ=WT_SEQ, DPI=DPI)

    # save as dataframe
    save_sat_mut(sat_mut, deletion_effects, WT_value, enh, WT_SEQ, output_dir) 
