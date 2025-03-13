import os
import numpy as np




def load_fasta(file):
    enh = []
    title = ""
    with open(file) as f:
        for line in f:
            if line[0] == ">":
                title = line.strip()
            else:
                enh.append([title, line.strip()])
    return enh


class File:
    def __init__(self, file, local_path):
        
        self.file = file
        self.local_path = local_path

        filename = file.split(".")[-2].split("_")[-1]
        self.filename = filename
        
        if filename[-2] == "P":
            self.file_class = "PP"
            self.file_enh = filename[:-2]
        elif filename[-2] == "M":
            self.file_class = "MM"
            self.file_enh = filename[:-2]
        else:
            self.file_class = filename[-1]
            self.file_class = "N" + self.file_class
        
            self.file_enh = filename[:-1]

        self.loaded = False
        self.contents = None

    def __str__(self):
        return f"{self.file_enh}, {self.file_class}"
    
    def load(self):
        self.contents = load_fasta(f"{self.local_path}/{self.file}")
        self.loaded = True

    def sequences(self):
        if not self.loaded:
            self.load()
        return [entry[1] for entry in self.contents]
    
    def __len__(self):
        if not self.loaded:
            self.load()
        return len(self.contents) 
    


save_dir = "Fasta_Pool2"
all_files = [File(file, save_dir) for file in os.listdir(save_dir) if file.endswith(".fa")]


# load wt sequences
wt = {
"E22P3B3":"AATTCTCTATCCACATAGTGTCCCTGTTGTCACAGATGACTCTCCACCTTTAGCATGGTATCCCTGTTCCAGCCAGGACAGAAGAGCAGGGAAAAGGACAAAAGAGGTATGAGCTACTGAGTTGGCTTCCCTTTTAAGGATCTTTCTTGGAAGCCCCACCCACAGCTTTCTCTTCCACCTCACTGGCCAGAACTCCATCGT",
"E24A3": "CATCAAGGAACTGAGGGCTCAGATCTTCGCAAATACTGTGGACAATGTCCACATCATTCTGCAGATCGACAATGCCCGTCTTGCTGCTGATGACTTCAGAGTCAAGTATGAGACAAGAGCTGGCCATGCGCCAGTCTGTGGAGAGCAACATCCATGGGCTCTGCAAGGTCATTGATGACACCAATGTCACTCTGCTGCAGC",
"E25E10": "CATCAAGGAACTGAGGGCTCAGATCTTCGCAAATACTGTGGACAATGTCCACATCATTCTGCAGATCGACAATGCCCGTCTTGCTGCTGATGACTTCAGAGTCAAGTATGAGACAAGAGCTGGCCATGCGCCAGTCTGTGGAGAGCAACATCCATGGGCTCTGCAAGGTCATTGATGACACCAATGTCACTCTGCTGCAGC",
}

from Bio import pairwise2
from tqdm import tqdm

def unique(list1):
    # intilize a null list
    unique_list = []
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


for file in all_files:
    wt_seq = wt[file.file_enh]
    wt_seq = wt_seq[10:-10]
    # print(wt_seq)


    seqs = file.sequences()

    print(len(unique(seqs)), len(seqs))
    
    alligned_seqs = []
    for seq in tqdm(seqs):
        # alignments = pairwise2.align.globalxx(wt_seq, seq)
        wt_align, seq_align, score, start, end = pairwise2.align.localms(wt_seq, seq, 1, -1, -1, -1)[0]

        new_wt = ""
        new_seq = ""
        for i in range(len(wt_align)):
            if wt_align[i] == "-":
                continue
            new_wt += wt_align[i]
            new_seq += seq_align[i]

        alligned_seqs.append(new_seq)

    # print(alligned_seqs)
    print(print(len(unique(alligned_seqs)), len(alligned_seqs)))


    #save as file in save_files
    with open(f"temp_files/alligned_{file.file_enh}_{file.file_class}", "w") as f:
        for i, seq in enumerate(alligned_seqs):
            f.write(f"{seq}\n")