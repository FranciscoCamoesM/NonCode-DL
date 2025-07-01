import torch
import pickle
import numpy as np
import pandas as pd
from models import SignalPredictor1D
from data_preprocess import one_hot_encode, one_hot_decode, simple_pad_batch


model_loc = "saved_models_try_2/E25E102/models/154162385_mm_v_all_C10_E25E102.pt"
variants_loc = "variants_data/diagram_consortium/SNPs/variants_DIAGRAM.txt"

ENH = "E25E102"
zero_position = 41511631    # for ANK1, this is the zero position in the genome
chrom = 8
LEN = 201


# # Load the model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = SignalPredictor1D(num_classes=1)
# model.load_state_dict(torch.load(model_loc, map_location=device, weights_only=True))
# model.to(device)
# model.eval()


# Load the variants data
# variants_df = pd.read_csv(variants_loc)
lines = []
with open(variants_loc, "r") as f:
    for line in f:
        lines.append(line.strip().split("\t"))

# create dataframe with header based on the first line
header = lines[0]
# Remove any leading BOM character and strip whitespace
header = [col.strip().replace('\ufeff', '') for col in header]
print("Header columns:", header)
variants_df = pd.DataFrame(lines[1:], columns=header)  

# filter dataframe for variants on chromosome 8
variants_df = variants_df[variants_df["CHROMOSOME"] == str(chrom)]
print("Filtered Variants DataFrame shape:", variants_df.shape)
variants_df = variants_df[variants_df["POSITION"].astype(int) >= zero_position]
print("Filtered Variants DataFrame shape:", variants_df.shape)
variants_df = variants_df[variants_df["POSITION"].astype(int) < zero_position + LEN]
print("Filtered Variants DataFrame shape:", variants_df.shape)

if variants_df.empty:
    raise ValueError(f"No variants found in chromosome {chrom} within the specified range.")

print("Variants DataFrame shape:", variants_df.shape)
print(variants_df.head())
print("Variants DataFrame columns:", variants_df.columns)
print(variants_df.columns.tolist())


# Now processing the 'varId' column should not throw an error
# variants_df["normalized_position"] = variants_df["position"].astype(int) - zero_position
# print(variants_df.head())



exit()

# get WT sequence
wt_seq = None
with open(f"original_enhancers.txt", "r") as f:
    for line in f:
        if not(line.startswith('>')):
            continue
        line = line[1:].strip()  # Remove the leading '>' character
        if line == ENH:
            wt_seq = next(f).strip()

if wt_seq is None:
    raise ValueError(f"Could not find sequence for enhancer {ENH} in original_enhancers.txt")

print("WT sequence length:", len(wt_seq))



variant_seqs_str = [wt_seq]  # Start with the wild-type sequence
for variant in variants_df.itertuples():
    # print(variant)
    pos = variant.normalized_position
    ref = variant.reference
    alt = variant.alt

    # remove quotes from ref and alt if they exist
    ref = ref.strip('"').strip("'")
    alt = alt.strip('"').strip("'")

    if pos < 0 or pos >= len(wt_seq):
        print(f"Skipping variant at position {pos} (out of bounds)")
        continue

    if ref != wt_seq[pos]:
        print(f"\nSkipping variant at position {pos} (reference mismatch: {ref} != {wt_seq[pos]})")
        
        print(f"Landscape:\n{wt_seq[pos-2:pos+3]}")  # Print a small landscape around the variant
        continue

    # Create the variant sequence
    variant_seq = wt_seq[:pos] + alt + wt_seq[pos+1:]

    variant_seqs_str.append(variant_seq)


# one-hot encode the sequences
variant_seqs_str = simple_pad_batch(variant_seqs_str)
variant_seqs_oh = [one_hot_encode(seq).T for seq in variant_seqs_str]


from torch.utils.data import DataLoader, Dataset
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq, 0  # Returning 0 as a placeholder label
    
# Create DataLoader for the variants
variant_dataset = CustomDataset(variant_seqs_oh)
variant_loader = DataLoader(variant_dataset, batch_size=64, shuffle=False)

print(f"Dataset size: {variant_dataset.__len__()} sequences")


# Run the model on the variants
predictions = []
with torch.no_grad():
    for batch in variant_loader:
        seqs, _ = batch  # Ignore labels
        seqs = seqs.to(device)
        preds = model(seqs.float())
        preds = preds.cpu().numpy()
        predictions.extend(preds.flatten().tolist())

wt_preds = predictions[0]  # The first prediction corresponds to the wild-type sequence
predictions = predictions[1:]  # Exclude the wild-type prediction from the results

print("Predictions for variants:")
# for seq, pred in zip(variant_seqs_str, predictions):
#     print(f"Sequence: {seq}, Prediction: {pred:.4f}")

# Add predictions to the DataFrame
variants_df["prediction"] = predictions
variants_df["relative_prediction"] = variants_df["prediction"] - wt_preds

print(variants_df[["normalized_position", "reference", "alt", "pValue", "prediction", "relative_prediction"]])