import os
from data_preprocess import *


locs = ["Fasta_Pool2", "Fasta_Pool3", "Fasta_Pool4", "Fasta_Pool5"]

all_files = []
for loc in locs:
    all_files_loc = import_files(loc)

    for file in all_files_loc:
        all_files.append([file.file_enh, len(file), loc])

print(f"Number of enhancers: {len(np.unique(all_files))}")
print(f"Number of files: {len(all_files)}")
print(all_files)

# pint sequences per enhancer
enhancer_counts = {}
for file in all_files:
    if file[0] not in enhancer_counts:
        enhancer_counts[file[0]] = 0
    enhancer_counts[file[0]] += file[1]

df_data = []
sen_enhancers = set()
for enh in all_files:
    if enh[0] in sen_enhancers:
        continue


    indiv = get_seq_label_pairs(enh_name = enh[0], local_path=enh[2])
    seq_count_1 = len(indiv)
    indiv = get_coverage_seq_label_pairs(enh_name = enh[0], local_path=enh[2], coverage=3)
    seq_count_3 = len(indiv)
    indiv = get_coverage_seq_label_pairs(enh_name = enh[0], local_path=enh[2], coverage=10)
    seq_count_10 = len(indiv)
    indiv = get_coverage_seq_label_pairs(enh_name = enh[0], local_path=enh[2], coverage=100)
    seq_count_100 = len(indiv)
    # print(f"Enhancer {enh[0]} has {sum(seq_count)} unique sequences")
    df_data.append({
        "Enhancer": enh[0],
        "Number of Unique Sequences": seq_count_1,
        "Number of Unique Sequences (Coverage 3)": seq_count_3,
        "Number of Unique Sequences (Coverage 10)": seq_count_10,
        "Number of Unique Sequences (Coverage 100)": seq_count_100,
        "Number of Reads": enhancer_counts[enh[0]],
    })
    sen_enhancers.add(enh[0])


import pandas as pd
df = pd.DataFrame(df_data)
#sort by enhancer name
df = df.sort_values(by="Enhancer", ascending=True, ignore_index=True)

print(df)

print(df.loc[df["Enhancer"] == "E22P3G3"])

# # plot unique sequences per enhancer vs total sequences
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 6))
# plt.scatter(df["Number of unique sequences"], df["Number of sequences"])
# plt.title("Unique Sequences vs Total Sequences per Enhancer")
# plt.xlabel("Number of Unique Sequences")
# plt.ylabel("Total Number of Sequences")
# plt.xscale("log")
# plt.yscale("log")
# plt.grid(True)
# plt.show()