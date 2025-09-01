import numpy as np
from sklearn.model_selection import train_test_split
from data_preprocess import *
from models import SignalPredictor1D
from data_preprocess import get_interpolated_seq_label_pairs, get_dataloc, load_trained_model, one_hot_encode, simple_pad

AA = 1
BB = 0.45/.95
lv = {"MM": -AA, "NM":-BB, "NP":BB, "PP":AA}
NUM_CLASSES = 1

enhancer_list = [[25, "E22P1A3"], [100, "E25E102"], [100, "E25E10R3"], [25, "E25E10"]]
model_ids = {"E22P1A3": "2025-07-19_05-26-53",
             "E25E102": "2025-07-18_15-17-32",
             "E25E10": "2025-07-19_03-10-32",
             "E25E10R3": "2025-07-18_17-22-54"
             }
save_dir = "reg_model_inference/chr8:41511631-41511831"



# enhancer_list = [[15, "E22P1D10"], [10, "E22P3B3"], [50, "E22P3B3R3"]]
# model_ids = {"E22P1D10": "2025-07-19_15-27-12",
#              "E22P3B3": "2025-07-20_01-20-46",
#              "E22P3B3R3": "2025-07-19_07-29-06"
#              }
# save_dir = "reg_model_inference/chr20:43021876-43022076"









os.makedirs(save_dir, exist_ok=True)

for training_dataset in enhancer_list:
    model_id = model_ids[training_dataset[1]]
    MODEL_DIR = 'regression_models/'+ training_dataset[1] + '/'
    for inference_dataset in enhancer_list:


        training_coverage = training_dataset[0]
        training_enh = training_dataset[1]
        training_bin_cells = get_bin_cells(training_enh)

        training_enh_data_dict = get_interpolated_seq_label_pairs(coverage=training_coverage, label_values=lv, enh_name=training_enh, local_path=get_dataloc(training_enh), BIN_CELLS=training_bin_cells)
        training_enh_values = training_enh_data_dict.values()
        training_enh_keys = training_enh_data_dict.keys()
        training_enh_data = zip(training_enh_keys, training_enh_values)
        training_enh_data = list(training_enh_data)

        train_seqs, test_seqs = train_test_split(training_enh_data, test_size=0.2, random_state=42)
        train_seqs, val_seqs = train_test_split(train_seqs, test_size=0.2, random_state=42)
        print(f"Training data size: {len(train_seqs)}, Validation data size: {len(val_seqs)}, Test data size: {len(test_seqs)}")
        train_seqs= [seq for seq, _ in train_seqs]

        inference_coverage = inference_dataset[0]
        inference_enh = inference_dataset[1]
        inference_bin_cells = get_bin_cells(inference_enh)

        inference_data = get_interpolated_seq_label_pairs(coverage=inference_coverage, label_values=lv, enh_name=inference_enh, local_path=get_dataloc(inference_enh), BIN_CELLS=inference_bin_cells)
        inference_values = inference_data.values()
        inference_keys = inference_data.keys()
        inference_data = zip(inference_keys, inference_values)
        inference_data = list(inference_data)

        if len(inference_data) == 0:
            raise ValueError(f"No data found for inference with coverage {inference_coverage} and enhancer {inference_enh}. Please check the data availability.")
        print(f"Inference data size: {len(inference_data)}")

        # remove sequences present in training data from inference data
        inference_data_filtered = []
        common_seqs = []
        for seq in inference_data:
            if seq[0] not in train_seqs:
                inference_data_filtered.append(seq)
            else:
                # print(seq[0], "lll", seq[1], training_enh_data_dict[seq[0]])
                common_seqs.append((seq[0], seq[1], training_enh_data_dict[seq[0]]))

        if len(inference_data_filtered) == 0:
            raise ValueError(f"All sequences in inference data are present in training data. Please check the data availability or adjust the inference dataset.")



        # one hot encode the sequences
        for i in range(len(inference_data_filtered)):
            seq, label = inference_data_filtered[i]
            seq = simple_pad(seq, 250)
            one_hot_seq = one_hot_encode(seq).T
            inference_data_filtered[i] = (one_hot_seq, label)

        if len(common_seqs) > 0:
            # calc r score for common sequences
            r_score = np.corrcoef([s[1] for s in common_seqs], [s[2] for s in common_seqs])[0, 1]
            print(f"R score for common sequences: {r_score:.4f}")


        model = SignalPredictor1D(num_classes=NUM_CLASSES)
        load_trained_model(training_dataset[1], os.path.join(MODEL_DIR, str(model_id)), model, device='cpu')

        from torch.utils.data import DataLoader

        class InterpolDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                seq, label = self.data[idx]
                seq_tensor = torch.tensor(seq, dtype=torch.float32)
                label_tensor = torch.tensor(label, dtype=torch.float32)
                return seq_tensor, label_tensor
            
        # Create DataLoader for inference data
        inference_dataset = InterpolDataset(inference_data_filtered)
        inference_loader = DataLoader(inference_dataset, batch_size=256, shuffle=False)


        # Run inference
        model.eval()
        predictions = []
        with torch.no_grad():
            for seqs, labels in inference_loader:
                seqs = seqs.to('cpu')  # Move to CPU if necessary
                outputs = model(seqs)
                predictions.extend(outputs.cpu().numpy().flatten().tolist())

        true_values = [label for _, label in inference_data_filtered]


        from scipy.stats import pearsonr, spearmanr
        pearson_corr = pearsonr(true_values, predictions)[0]
        spearman_corr = spearmanr(true_values, predictions)[0]
        print(f"Pierceon correlation coefficient: {pearson_corr:.4f}")
        print(f"Spearman correlation coefficient: {spearman_corr:.4f}")

        # Plot predicted vs true values
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5), dpi=150)
        plt.scatter(true_values, predictions, alpha=0.5)
        plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], color='red', linestyle='--')
        plt.title(f"Predicted vs True Values for model {model_id} (trained on {training_enh} - cov {training_coverage})\nInference on {inference_enh} - cov {inference_coverage}")
        plt.xlabel("True Values", fontsize=12)
        plt.ylabel("Predicted Values", fontsize=12)
        plt.grid()
        plt.text(0.05, 0.95, f"Pier. Corr: {pearson_corr:.4f}\nSpearman Corr: {spearman_corr:.4f}",
                transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        plt.savefig(os.path.join(save_dir, f"predicted_vs_true_{model_id}_({training_enh}_to_{inference_enh}).png"), dpi=150)
        plt.show()
        print(f"Predicted vs True Values plot saved as {os.path.join(save_dir, f'predicted_vs_true_{model_id}_({training_enh}_to_{inference_enh}).png')}")



        # ###############
        # print("#"*50)
        # print("WARNING\t"* 10)
        # print("   Keep in mind that by excluding thre training seqs, we are enriching the dataset with sequences that differ from the trianing data")
        # print("   Say the inference dataset has 3000 seqs. 2000 real sequences, 1000 biased sequences. Same thing for the training dataset.")
        # print("   Say from the 2000 in real of each, 1000 are the same. So we have 1000 common sequences.")
        # print("   When we remove 80% of the common sequences, we are left with 2200 sequences, 1000 of which are biased, so almost 50% of the dataset is biased instead of 33%")
        # print("WARNING\t"* 10)
        # print("#"*50)

        # save metrics in metrics file

        if not(os.path.exists(os.path.join(save_dir, 'metrics.txt'))):
            with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
                f.write(f"Model ID\tTraining Enh\tTraining Coverage\tInference Enh\tInference Coverage\tPearson Corr\tSpearman Corr\n")
                f.write(f"{model_id}\t{training_enh}\t{training_coverage}\t{inference_enh}\t{inference_coverage}\t{pearson_corr:.4f}\t{spearman_corr:.4f}\n")
        else:
            with open(os.path.join(save_dir, 'metrics.txt'), 'a') as f:
                f.write(f"{model_id}\t{training_enh}\t{training_coverage}\t{inference_enh}\t{inference_coverage}\t{pearson_corr:.4f}\t{spearman_corr:.4f}\n")