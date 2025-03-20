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
    
    def __len__(self):
        if not self.loaded:
            self.load()
        return len(self.contents)    



def import_files(local_path = "Fasta_Pool2", **kwargs):
    all_files = []
    for file in os.listdir(local_path):
        if file.endswith(".fa"):
            
            # files are NAMEP, NAMEPP, NAMEMM, and NAMEM
            all_files.append(File(file, local_path))

    return all_files

def import_indiv_enhancers(enh_id = 0, enh_name = None, **kwargs):
    all_files = import_files(**kwargs)
    enhancer_names = [f.file_enh for f in all_files]
    enhancer_names = list(set(enhancer_names))

    if enh_name is None:
        print("Using enhancer ", enhancer_names[enh_id])
        enh_name = enhancer_names[enh_id]

    loaded_enhancers = []
    for file in all_files:
        if file.file_enh == enh_name:
            loaded_enhancers.append(file)

    return loaded_enhancers


def get_seq_label_pairs(**kwargs):
    all_files = import_indiv_enhancers(**kwargs)

    all_enhancer_labels = {}
    for file in all_files:
        file.load()
        for entry in file.contents:
            if entry[1] in all_enhancer_labels:
                all_enhancer_labels[entry[1]].append(file.file_class)
            else:
                all_enhancer_labels[entry[1]] = [file.file_class]


    all_enhancers_single_label = {}
    for enhancer in all_enhancer_labels:
        if len(set(all_enhancer_labels[enhancer])) == 1:
            all_enhancers_single_label[enhancer] = all_enhancer_labels[enhancer][0]

        else:
            # pick median
            labels = all_enhancer_labels[enhancer]
            # sort labels
            labels = sorted(labels)
            
            all_enhancers_single_label[enhancer] = labels[len(labels)//2]

    return all_enhancers_single_label



def one_hot_encode(seq):
    seq = seq.upper()
    encoding_dict = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
        "N": [0, 0, 0, 0]
    }

    encoding = []
    for i, char in enumerate(seq):
        encoding.append(encoding_dict[char])

    return np.array(encoding)

def one_hot_encode_label(label, encoding_dict):

    return np.array(encoding_dict[label])

def pad_samples(batch, encoding_dict):
    with open("padding_seq.txt", "r") as f:
        pre_line = f.readline()
        skip_line = f.readline()
        post_line = f.readline()

    pre_line = pre_line.strip()
    post_line = post_line.strip()

    sequences = [pre_line + item[0] + post_line for item in batch]
    labels = [item[1] for item in batch]

    max_length = 250
    padded_sequences = []
    # center sequences
    for seq in sequences:
        cutting_points = [len(seq)//2 - max_length//2, len(seq)//2 + max_length//2]
        padded_seq = seq[cutting_points[0]:cutting_points[1]]
        padded_sequences.append(padded_seq)

    # one hot encode
    one_hot_sequences = [one_hot_encode(seq) for seq in padded_sequences]
    one_hot_labels = [one_hot_encode_label(label, encoding_dict) for label in labels]

    return one_hot_sequences, one_hot_labels


from torch.utils.data import Dataset
import torch

class EnhancerDataset(Dataset):
    def __init__(self, seqs, labels, encoding_dict):
        self.sequences = seqs
        self.labels = labels

        self.sequences, self.labels = pad_samples(list(zip(self.sequences, self.labels)), encoding_dict)

        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        return seq, label
    

def preset_labels(name = "id"):
    if name == "id":
        return {
            "MM": [1, 0, 0, 0],
            "NM": [0, 1, 0, 0],
            "NP": [0, 0, 1, 0],
            "PP": [0, 0, 0, 1]
        }
    elif name == "mm_v_all":
        return {
            "MM": [1, 0],
            "NM": [0, 1],
            "NP": [0, 1],
            "PP": [0, 1]
        }
    elif name == "pp_v_all":
        return {
            "MM": [1, 1],
            "NM": [1, 1],
            "NP": [1, 1],
            "PP": [0, 1]
        }
    elif name == "m_v_p":
        return {
            "MM": [1, 0],
            "NM": [1, 0],
            "NP": [0, 1],
            "PP": [0, 1]
        }
    elif name == "3_class" or "3":
        return {
            "MM": [1, 0, 0],
            "NM": [0, 1, 0],
            "NP": [0, 0, 1],
            "PP": [0, 0, 1]
        }
    else:
        print("#"*20,"Invalid label preset", "#"*20)
        return None
    

def load_wt(ENHANCER_NAME):
    # load original enhancer
    with open(f"original_enhancers.txt", "r") as f:
        enhancer = f.readlines()
        # format is >EnhancerName\n
        #               sequence\n
        # so select the line after the one with the enhancer name
        enhancer = enhancer[enhancer.index(f">{ENHANCER_NAME}\n") + 1].strip()

    return enhancer