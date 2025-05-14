from data_preprocess import import_files, get_seq_label_pairs



import os


datafolder = "Fasta_Pool2"
interest = "E25E10"    

x = get_seq_label_pairs(local_path=datafolder, enh_name=interest)

dataset = {}
unique_lables = list(set(x.values()))
for l in unique_lables:
    d = [seq for seq in x if x[seq] == l]
    print(l+":",len(d),"\n")
    dataset[l] = d

# save each label to a separate file
for l in dataset:
    with open(f"meme_seqs/{interest}_{l}.txt", "w") as f:
        for i, seq in enumerate(dataset[l]):
            f.write(f">{i}\n{seq}\n")
    print(f"Saved {len(dataset[l])} sequences for label {l} to file meme_seqs/{interest}_{l}.txt")



