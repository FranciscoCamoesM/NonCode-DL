## This script is used to load a trained model and perform inference on new data. It can take as input both a fasta file or a simple csv or txt file with the sequences to be predicted.


import argparse
import torch
import os
from data_preprocess import simple_pad_batch, one_hot_encode, load_fasta, load_seqs_csv, load_trained_model
from models import SignalPredictor1D
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser(description="Inference script for trained models.")
parser.add_argument("--id", type=int, required=True, help="ID of the training run to analyze.")
parser.add_argument("--seqs", type=str, required=True, help="Path to the input file containing sequences to predict. Can be a fasta file or a simple csv or txt file with one sequence per line.")
parser.add_argument("--model_dir", type=str, default="trained_models", help="Directory where trained models are saved.")
parser.add_argument("--input_format", type=str, default=None, help="Format of the input file. Can be 'fasta', 'fa', 'csv', or 'txt'. If not provided, will be inferred from the file extension.")
parser.add_argument("--o", type=str, default="inference_results", help="Directory to save inference results.")
parser.add_argument("--plot", action="store_true", help="Whether to generate a histogram plot of the predicted values.")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size for inference.")
parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on. Can be 'cpu' or 'cuda'.")
args = parser.parse_args()



input_file = args.seqs
model_saved_dir = os.path.join(args.model_dir, f"run_{args.id}")
output_dir = args.o
input_format = args.input_format
BATCH_SIZE = args.batch_size
device = args.device
PLOT_HISTOGRAM = args.plot
NUM_CLASSES = 1  # regression task



os.makedirs(output_dir, exist_ok=True)

## Assign the input type for proper sequence parsing.
equivalence_dict = {
    "fasta": "fasta",
    "fa": "fasta",
    "csv": "csv",
    "txt": "csv",
    "tsv": "csv"
}

if input_format is None:
    ext = os.path.splitext(input_file)[1][1:].lower()  # get extension, remove dot and convert to lower case
    input_format = equivalence_dict.get(ext, None)
    if input_format is None:
        raise ValueError(f"Could not infer input format from file extension {ext}. Please provide the input format using the --input_format argument.")
else:
    input_format = equivalence_dict.get(input_format.lower(), None)
    if input_format is None:
        raise ValueError(f"Invalid input format {input_format}. Supported formats are: {', '.join(equivalence_dict.keys())}.")


if input_format == "fasta":
    file_info = load_fasta(input_file)
    sequences, names = zip(*file_info)  # unzip the list of tuples into two lists: sequences and names
    del file_info  # free memory
elif input_format == "csv":
    sequences = load_seqs_csv(input_file)
    names = None



model = SignalPredictor1D(num_classes=NUM_CLASSES)
load_trained_model(training_dataset[1], os.path.join(MODEL_DIR, str(model_id)), model, device='cpu')

# turn sequences into one hot encoding and pad them to 250 bp
sequences_padded = simple_pad_batch(sequences, max_length=250)
sequences_oh = [one_hot_encode(seq) for seq in sequences_padded]
sequences_oh = torch.tensor(sequences_oh, dtype=torch.float32).permute(0, 2, 1)  # (B, C, L)

model.to(device)
model.eval()
results = []
with torch.no_grad():
    for i in tqdm(range(0, len(sequences_oh), BATCH_SIZE)):
        batch_seqs = sequences_oh[i:i+BATCH_SIZE].to(device)
        batch_preds = model(batch_seqs).cpu().numpy()
        results.extend(batch_preds)

# save results in a csv file
results_df = pd.DataFrame(results)
if names is not None:
    results_df["Name"] = names
results_df["Sequence"] = sequences

file_basename = os.path.basename(input_file)
file_basename = os.path.splitext(file_basename)[0]  # remove extension
out_file = os.path.join(output_dir, f"{file_basename}_inference_{args.id}.csv")
results_df.to_csv(out_file, index=False)
print(f"Inference completed. Results saved to {out_file}.")

if PLOT_HISTOGRAM:
    print("Generating histogram plot of predicted values...")
    results = np.array(results).flatten()  # flatten to eliminate extra dimensions 
    plt.hist(results, bins=50)
    plt.xlabel("Predicted Values")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Predicted Values by Model from Run {args.id}")
    plt.savefig(os.path.join(output_dir, f"{file_basename}_inference_{args.id}_histogram.png"))
    plt.close()
    print("Histogram plot saved.")

print("Inference script finished.")