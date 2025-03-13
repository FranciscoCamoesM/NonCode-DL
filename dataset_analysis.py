import numpy as np
import os
import matplotlib.pyplot as plt
from data_preprocess import import_files, load_wt
from tqdm import tqdm


SAVE_DIR = 'dataset_analysis'

from Bio import pairwise2
from Bio.pairwise2 import format_alignment

def count_mutations(wt_seq, mut_seq):
    # Align sequences
    alignments = pairwise2.align.globalxx(wt_seq, mut_seq)

    if not alignments:
        return None, None, None

    # Assume the best alignment is the first one
    aligned_wt, aligned_mut, score, start, end = alignments[0]

    # # eliminate regions with
    # # A-CG            ACG
    # # AT-G  to become ATG

    # for i in range(len(aligned_wt)-1):
    #     if aligned_wt[i] == "-" and aligned_mut[i + 1] == "-":
    #         aligned_wt = aligned_wt[:i] + aligned_wt[i+1:]
    #         aligned_mut = aligned_mut[:i+1] + aligned_mut[i+2:]

    #     elif aligned_wt[i + 1] == "-" and aligned_mut[i] == "-":
    #         aligned_wt = aligned_wt[:i+1] + aligned_wt[i+2:]
    #         aligned_mut = aligned_mut[:i] + aligned_mut[i+1:]


    return aligned_wt, aligned_mut


from collections import defaultdict
def mutation_distribution(wt_seq, mut_seqs, enh):
    mut_counts = {}
    distribution = defaultdict(int)
    for mut_seq in tqdm(mut_seqs):
        aligned_wt, aligned_mut = count_mutations(wt_seq, mut_seq)
        if aligned_mut is None:
            continue

        al_idx = 0
        wt_idx = 0
        while True:
            if aligned_wt[al_idx] != aligned_mut[al_idx]:
                distribution[wt_idx] += 10
    
            if aligned_wt[al_idx] != "-":
                wt_idx += 1


            al_idx += 1


            if al_idx >= len(aligned_wt):
                break
            if wt_idx >= len(wt_seq):
                wt_idx -= 1

    plt.figure(figsize=(15, 10))
    plt.bar(distribution.keys(), distribution.values())
    plt.yscale("log")
    # plt.show()
    plt.savefig(os.path.join(SAVE_DIR, f"{enh}_mutation_distribution.png"))

    return mut_counts






# Load the data
local_path = "Fasta_Pool2"
all_files = import_files(local_path)


enh_list = []
for file in all_files:
    enh_list.append(file.file_enh)
    

enh_list = np.unique(enh_list)

print(f"Number of enhancers: {len(enh_list)}")

for enh in enh_list:
    mutation_entries = {}
    for file in all_files:
        if file.file_enh != enh:
            continue

        file.load()
        for entry in file.contents:
            if entry[1] not in mutation_entries:
                mutation_entries[entry[1]] = []
            mutation_entries[entry[1]].append(file.file_class)

    print(f"Enhancer {enh} has {len(mutation_entries)} mutations")

    mut_counts = {}

    for mut in mutation_entries:
        mut_counts[mut] = len(mutation_entries[mut])


    bar_graph = {"1": 0, "2 or 3": 0, "4 to 10": 0, "10 to 100": 0, "100 to 1000": 0, "1000+": 0}
    for mut in mut_counts:
        if mut_counts[mut] == 1:
            bar_graph["1"] += 1
        elif mut_counts[mut] == 2 or mut_counts[mut] == 3:
            bar_graph["2 or 3"] += 1
        elif mut_counts[mut] >= 4 and mut_counts[mut] <= 10:
            bar_graph["4 to 10"] += 1
        elif mut_counts[mut] >= 10 and mut_counts[mut] <= 100:
            bar_graph["10 to 100"] += 1
        elif mut_counts[mut] >= 100 and mut_counts[mut] <= 1000:
            bar_graph["100 to 1000"] += 1
        else:
            bar_graph["1000+"] += 1

    print(bar_graph)

    # plot in %
    bar_graph_perc = {}
    total = sum(bar_graph.values())
    for key in bar_graph:
        bar_graph_perc[key] = bar_graph[key] / total * 100



    # do both plots in the same figure
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    ax[0].bar(bar_graph.keys(), bar_graph.values())
    ax[0].set_xlabel("Number of mutations")
    ax[0].set_ylabel("Number of enhancers (log scale)")
    ax[0].set_title(f'Number of class entries per mutation in enhancer {enh}')
    # ax[0].set_yscale("log")

    ax[1].bar(bar_graph_perc.keys(), bar_graph_perc.values())
    ax[1].set_xlabel("Number of mutations")
    ax[1].set_ylabel("Percentage of enhancers")
    ax[1].set_title(f"Percentage of class entries per mutation")

    fig.suptitle(f"Enhancer {enh} mutation distribution")

    # plt.show()
    plt.close(fig)





    #analise mutation distribution
    
    # load wt
    wt_enh = load_wt(enh)


    count_mutations(wt_enh, list(mutation_entries.keys())[0])

    mutation_distribution(wt_enh, list(mutation_entries.keys()), enh)




        
        