import numpy as np
from sklearn.model_selection import train_test_split
from data_preprocess import *

model_id = 151121
training_dataset = [100, "E22P1A3", np.array([23731, 69026, 57444, 15199])]
val_dataset = [100, "E25E102", np.array([68519, 111643, 60318, 5687])]
AA = 1
BB = 0.475
lv = {"MM": -AA, "NM":-BB, "NP":BB, "PP":AA}
MODEL_DIR = 'saved_models'
NUM_CLASSES = 1


training_coverage = training_dataset[0]
training_enh = training_dataset[1]
training_bin_cells = training_dataset[2]

training_enh_data_dict = get_interpolated_seq_label_pairs(coverage=training_coverage, label_values=lv, enh_name=training_enh, local_path=get_dataloc(training_enh), BIN_CELLS=training_bin_cells)
training_enh_values = training_enh_data_dict.values()
training_enh_keys = training_enh_data_dict.keys()
training_enh_data = zip(training_enh_keys, training_enh_values)
training_enh_data = list(training_enh_data)

train_seqs, test_seqs = train_test_split(training_enh_data, test_size=0.2, random_state=42)
train_seqs, val_seqs = train_test_split(train_seqs, test_size=0.2, random_state=42)

inference_coverage = val_dataset[0]
inference_enh = val_dataset[1]
inference_bin_cells = val_dataset[2]

inference_data = get_interpolated_seq_label_pairs(coverage=inference_coverage, label_values=lv, enh_name=inference_enh, local_path=get_dataloc(inference_enh), BIN_CELLS=inference_bin_cells)
inference_values = inference_data.values()
inference_keys = inference_data.keys()
inference_data = zip(inference_keys, inference_values)
inference_data = list(inference_data)


# remove sequences present in training data from inference data
inference_data_filtered = []
common_seqs = []
for seq in inference_data:
    if seq[0] not in training_enh_keys:
        inference_data_filtered.append(seq)
    else:
        # print(seq[0], "lll", seq[1], training_enh_data_dict[seq[0]])
        common_seqs.append((seq[0], seq[1], training_enh_data_dict[seq[0]]))


# calc r score for common sequences
r_score = np.corrcoef([s[1] for s in common_seqs], [s[2] for s in common_seqs])[0, 1]
print(f"R score for common sequences: {r_score:.4f}")


model = SignalPredictor1D(num_classes=NUM_CLASSES)
load_trained_model(model_id, MODEL_DIR, model, device='cpu')

