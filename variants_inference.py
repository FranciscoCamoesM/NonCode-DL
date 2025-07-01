import torch
import pickle
import numpy as np
import pandas as pd
from models import SignalPredictor1D
from data_preprocess import one_hot_encode, one_hot_decode, simple_pad_batch


model_loc = "saved_models_try_2/E25E102/models/154171846_mm_v_all_C3_E25E102.pt"

variants_loc = "variants_data/t2d_hugeamp/SNPs/ANK1_SNPs_T2D.csv"
# variants_loc = "variants_data/t2d_hugeamp/SNPs/ANK1_SNPs_Triglycerides.csv"
# variants_loc = "variants_data/t2d_hugeamp/SNPs/ANK1_SNPs_HbA1c.csv"

ENH = "E25E102"
zero_position = 41511631    # for ANK1, this is the zero position in the genome

pvalue_threshold = 0.1  # p-value threshold for filtering variants


# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SignalPredictor1D(num_classes=1)
model.load_state_dict(torch.load(model_loc, map_location=device, weights_only=True))
model.to(device)
model.eval()


# Load the variants data
lines = []
with open(variants_loc, "r") as f:
    for line in f:
        lines.append(line.strip().split(",")[:15])

# create dataframe with header based on the first line
header = lines[0]
# Remove any leading BOM character and strip whitespace
header = [col.strip().replace('\ufeff', '') for col in header]
# print("Header columns:", header)
variants_df = pd.DataFrame(lines[1:], columns=header)  

# Convert position to integer and center it on the zero position
variants_df["normalized_position"] = variants_df["position"].astype(int) - zero_position



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

# print("WT sequence length:", len(wt_seq))

# filter variants based on pvalue threshold
variants_df["pValue"] = variants_df["pValue"].astype(float)
variants_df = variants_df[variants_df["pValue"] < pvalue_threshold]

# Create variant sequences
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

print(f"Dataset size: {variant_dataset.__len__() - 1} variants")


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

# print("Predictions for variants:")
# for seq, pred in zip(variant_seqs_str, predictions):
#     print(f"Sequence: {seq}, Prediction: {pred:.4f}")

# Add predictions to the DataFrame
variants_df["prediction"] = predictions
variants_df["relative_prediction"] = variants_df["prediction"] - wt_preds
variants_df["-Log10(pValue)"] = -np.log10(variants_df["pValue"])

print(variants_df[["normalized_position", "reference", "alt", "pValue", "prediction", "relative_prediction", "-Log10(pValue)"]])




# exit()

# create a set of 50 random variants
n_variants = 50

import random
random.seed(42)
random_variants = []
for _ in range(n_variants):
    position = random.randint(0, len(wt_seq) - 1)
    ref = wt_seq[position]
    alt = random.choice([base for base in "ACGT" if base != ref])  # Choose a different base
    random_variants.append({
        "normalized_position": position,
        "reference": ref,
        "alt": alt,
    })

# Convert to DataFrame
random_variants_df = pd.DataFrame(random_variants)  

# Add predictions for random variants
random_variant_seqs_str = [wt_seq]  # Start with the wild-type sequence
for variant in random_variants_df.itertuples():
    pos = variant.normalized_position
    ref = variant.reference
    alt = variant.alt

    if pos < 0 or pos >= len(wt_seq):
        print(f"Skipping random variant at position {pos} (out of bounds)")
        continue

    if ref != wt_seq[pos]:
        print(f"\nSkipping random variant at position {pos} (reference mismatch: {ref} != {wt_seq[pos]})")
        continue

    # Create the random variant sequence
    variant_seq = wt_seq[:pos] + alt + wt_seq[pos+1:]
    random_variant_seqs_str.append(variant_seq)
# one-hot encode the random sequences
random_variant_seqs_str = simple_pad_batch(random_variant_seqs_str)
random_variant_seqs_oh = [one_hot_encode(seq).T for seq in random_variant_seqs_str]
# Create DataLoader for the random variants
random_variant_dataset = CustomDataset(random_variant_seqs_oh)
random_variant_loader = DataLoader(random_variant_dataset, batch_size=64, shuffle=False)
# Run the model on the random variants
random_predictions = []
with torch.no_grad():
    for batch in random_variant_loader:
        seqs, _ = batch  # Ignore labels
        seqs = seqs.to(device)
        preds = model(seqs.float())
        preds = preds.cpu().numpy()
        random_predictions.extend(preds.flatten().tolist())

# Add predictions to the random variants DataFrame
random_wt_preds = random_predictions[0]  # The first prediction corresponds to the wild-type sequence
random_predictions = random_predictions[1:]  # Exclude the wild-type prediction from the results
random_variants_df["prediction"] = random_predictions
random_variants_df["relative_prediction"] = random_variants_df["prediction"] - random_wt_preds  
# Print the random variants DataFrame
print("\nRandom Variants DataFrame:")
print(random_variants_df[["normalized_position", "reference", "alt", "prediction", "relative_prediction"]])


random_values = random_variants_df["relative_prediction"].values
values = variants_df["relative_prediction"].values

#mka ethem abs
values = np.abs(values)
random_values = np.abs(random_values)

print("\nRelative predictions for variants:")
print("Variants:", values)
print("Random Variants:", random_values)




# values = [ -1.255948  ,  -0.160011  ,  -0.322829  ,  2.059054  ,  0.524357  ,  0.044208  ,  -0.04687  ,  0.513456  ,  -1.69289  ,  0.002122  ,  -0.231908  ,  0.524028  ,  -1.392017  ,  -0.279357  ,  2.111962  ,  0.136065  ,  -0.570127  ,  -1.99799  ,  0.002122  ,  2.059054  ,]


# plot each distribution in violin plot
import matplotlib.pyplot as plt
import seaborn as sns   
plt.figure(figsize=(10, 6))
sns.violinplot(data=[values, random_values], inner="quartile", palette=["blue", "orange"])
# show points on top of the violin plot
sns.stripplot(data=[values, random_values], color='black', alpha=0.5, size=3, jitter=True)
plt.xticks([0, 1], ["Variants", "Random Variants"])
plt.title("Distribution of Relative Predictions")
plt.ylabel("Relative Prediction")
plt.grid()
plt.savefig(f"temp_figs/{ENH}_relative_predictions_violin_plot.png", dpi=300)

# measure if they are significantly different using t-test
from scipy import stats
t_stat, p_value = stats.ttest_ind(values, random_values)
print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}") 