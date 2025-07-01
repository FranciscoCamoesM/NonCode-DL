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
    local_path = "../NonCode/" + local_path
    all_files = []
    for file in os.listdir(local_path):
        if file.endswith(".fa"):
            
            # files are NAMEP, NAMEPP, NAMEMM, and NAMEM
            if file.endswith("P.fa") or file.endswith("M.fa"):
                all_files.append(File(file, local_path))

    return all_files

def import_indiv_enhancers(enh_name = None, enh_id = None, **kwargs):
    all_files = import_files(**kwargs)
    enhancer_names = [f.file_enh for f in all_files]
    enhancer_names = list(set(enhancer_names))

    if enh_name is None:
        if enh_id is None:
            raise ValueError("No enhancer name or ID provided. Please provide either 'enh_name' or 'enh_id'.")
        print("**** No enhancer name provided. Using enhancer ID ", enh_id)
        print("Using enhancer ", enhancer_names[enh_id])
        enh_name = enhancer_names[enh_id]

    loaded_enhancers = []
    for file in all_files:
        if file.file_enh == enh_name:
            loaded_enhancers.append(file)
    
    if len(loaded_enhancers) == 0:
        raise ValueError(f"No Enhancer {enh_name} found in target directory. Check if variable 'local_path' is correct. (Currently: {kwargs.get('local_path', None)})")

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

def one_hot_decode(seq):
    decoding_dict = {
        0: "A",
        1: "C",
        2: "G",
        3: "T"
    }

    decoded_seq = ""
    for i in range(len(seq)):
        if np.sum(seq[i]) == 0:
            decoded_seq += "N"
        else:
            decoded_seq += decoding_dict[np.argmax(seq[i])]

    return decoded_seq


def one_hot_encode_label(label, encoding_dict):

    return np.array(encoding_dict[label])

def simple_pad_batch(batch, PADDING = True, jiggle = 0):
    if os.path.exists("padding_seq.txt") and PADDING:
        with open("padding_seq.txt", "r") as f:
            pre_line = f.readline()
            skip_line = f.readline()
            post_line = f.readline()
    else:
        print("Padding sequence file not found. Using padding with N's.")
        pre_line = "N" * 125
        skip_line = "N" * 0
        post_line = "N" * 125

    pre_line = pre_line.strip()
    post_line = post_line.strip()

    sequences = [simple_pad(seq, PADDING=PADDING, jiggle=jiggle, pre_and_post_padding=[pre_line, post_line]) for seq in batch]

    return sequences

def simple_pad(seq, PADDING = True, jiggle = 0, pre_and_post_padding = None):
    if pre_and_post_padding is None:
        if os.path.exists("padding_seq.txt") and PADDING:
            with open("padding_seq.txt", "r") as f:
                pre_line = f.readline()
                skip_line = f.readline()
                post_line = f.readline()
        else:
            print("Padding sequence file not found. Using padding with N's.")
            pre_line = "N" * 125
            skip_line = "N" * 0
            post_line = "N" * 125

    else:
        pre_line = pre_and_post_padding[0]
        post_line = pre_and_post_padding[1]

    
    pre_line = pre_line.strip()
    post_line = post_line.strip()

    seq = pre_line + seq + post_line
    max_length = 250
    cutting_points = [len(seq)//2 - max_length//2, len(seq)//2 + max_length//2]
    cutting_points[0] += jiggle
    cutting_points[1] += jiggle
    cutting_points[0] = max(0, cutting_points[0])
    cutting_points[1] = min(len(seq), cutting_points[1])
    padded_seq = seq[cutting_points[0]:cutting_points[1]]
    
    return padded_seq



def pad_samples(batch, encoding_dict, PADDING = True, jiggle = 0):
    if os.path.exists("padding_seq.txt") and PADDING:
        with open("padding_seq.txt", "r") as f:
            pre_line = f.readline()
            skip_line = f.readline()
            post_line = f.readline()
    else:
        print("Padding sequence file not found. Using padding with N's.")
        pre_line = "N" * 125
        skip_line = "N" * 0
        post_line = "N" * 125

    pre_line = pre_line.strip()
    post_line = post_line.strip()

    sequences = [pre_line + item[0] + post_line for item in batch]
    labels = [item[1] for item in batch]

    max_length = 250
    padded_sequences = []
    # center sequences
    for seq in sequences:
        cutting_points = [len(seq)//2 - max_length//2, len(seq)//2 + max_length//2]
        
        cutting_points[0] += jiggle
        cutting_points[1] += jiggle
        cutting_points[0] = max(0, cutting_points[0])
        cutting_points[1] = min(len(seq), cutting_points[1])
        padded_seq = seq[cutting_points[0]:cutting_points[1]]
        padded_sequences.append(padded_seq)

    # one hot encode
    one_hot_sequences = [one_hot_encode(seq) for seq in padded_sequences]
    one_hot_labels = [one_hot_encode_label(label, encoding_dict) for label in labels]

    return one_hot_sequences, one_hot_labels


from torch.utils.data import Dataset
import torch

class EnhancerDataset(Dataset):
    def __init__(self, seqs, labels, encoding_dict, jiggle_range = None, **kwargs):

        if jiggle_range is None:
            self.sequences, self.labels = pad_samples(list(zip(seqs, labels)), encoding_dict, **kwargs)
        else:
            self.sequences = []
            self.labels = []

            for j in range(-jiggle_range, jiggle_range + 1):
                temp_sequences, temp_labels = pad_samples(list(zip(seqs, labels)), encoding_dict, jiggle=j, **kwargs)
                self.sequences.extend(temp_sequences)
                self.labels.extend(temp_labels)
                

        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx].T
        label = self.labels[idx]
        
        return seq, label
    

def preset_labels(name = "id"):
    if name == "id" or name == "all_v_all":
        return {
            "MM": [1, 0, 0, 0],
            "NM": [0, 1, 0, 0],
            "NP": [0, 0, 1, 0],
            "PP": [0, 0, 0, 1]
        }
    elif name == "mm_v_all":
        return {
            "MM": [1],
            "NM": [0],
            "NP": [0],
            "PP": [0]
        }
    elif name == "pp_v_all":
        return {
            "MM": [1],
            "NM": [1],
            "NP": [1],
            "PP": [0]
        }
    elif name == "m_v_p":
        return {
            "MM": [1],
            "NM": [1],
            "NP": [0],
            "PP": [0]
        }
    elif name == "3_class" or name == "3":
        return {
            "MM": [1, 0, 0],
            "NM": [0, 1, 0],
            "NP": [0, 0, 1],
            "PP": [0, 0, 1]
        }
    else:
        raise ValueError(f"Invalid label preset: {name}. Available presets are: id, all_v_all, mm_v_all, pp_v_all, m_v_p, 3_class, 3")
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



def get_extreme_seq_label_pairs(coverage = 3, **kwargs):
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
        if len(set(all_enhancer_labels[enhancer])) <= coverage:
            continue

        labels = all_enhancer_labels[enhancer]
        
        labels = sorted(labels)
        
        if labels[-int(len(labels)/3)] == "MM" or labels[0] != "MM":
            all_enhancers_single_label[enhancer] = labels[0]


    return all_enhancers_single_label



def get_coverage_seq_label_pairs(coverage = 3, **kwargs):
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
        if len(all_enhancer_labels[enhancer]) < coverage:
            continue

        labels = all_enhancer_labels[enhancer]
        
        labels = sorted(labels)
        
        all_enhancers_single_label[enhancer] = labels[len(labels)//2]

    return all_enhancers_single_label



def get_interpolated_seq_label_pairs_old(enh_name, coverage = 3, label_values = {"MM": -1, "NM": -0.15, "NP":0.15, "PP":1}, **kwargs):
    all_files = import_indiv_enhancers(enh_name, local_path=get_dataloc(enh_name))

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
        if len(all_enhancer_labels[enhancer]) < coverage:
            continue

        labels = all_enhancer_labels[enhancer]
        
        labels = [label_values[label] for label in labels]

        all_enhancers_single_label[enhancer] = np.mean(labels)

    return all_enhancers_single_label


def get_interpolated_seq_label_pairs(enh_name, coverage=5, label_values = {"MM": -1, "NM": -0.15, "NP":0.15, "PP":1}, BIN_CELLS=None, **kwargs):
    if BIN_CELLS is None:
        return get_interpolated_seq_label_pairs_old(coverage=coverage, label_values=label_values, enh_name=enh_name, **kwargs)

    all_files = import_indiv_enhancers(enh_name=enh_name, local_path=get_dataloc(enh_name))

    all_enhancer_labels = {}
    for file in all_files:
        file.load()
        for entry in file.contents:
            if entry[1] in all_enhancer_labels:
                all_enhancer_labels[entry[1]].append(file.file_class)
            else:
                all_enhancer_labels[entry[1]] = [file.file_class]

    counted_enhancers = {}
    for enhancer in all_enhancer_labels:
        labels = all_enhancer_labels[enhancer]
        counted_enhancers[enhancer] = [labels.count('MM'), labels.count('NM'), labels.count('NP'), labels.count('PP')]


    # sums
    BIN_SEQS = np.array([0, 0, 0, 0])
    for enhancer in counted_enhancers:
        BIN_SEQS += np.array(counted_enhancers[enhancer])
    ratio = BIN_CELLS / BIN_SEQS
    # print(f'Total counts: {BIN_SEQS}, BIN_CELLS: {BIN_CELLS}')
    # print(f'Ratio: {1 / ratio} seq per cell')

    # filter out enhancers with less than COVERAGE
    counted_enhancers = {enhancer: counted_enhancers[enhancer] for enhancer in counted_enhancers if np.sum(counted_enhancers[enhancer]) >= coverage}

    weighted_seqs = {}
    for enhancer in counted_enhancers:
        weighted_seqs[enhancer] = np.array(counted_enhancers[enhancer]) * ratio

    interpolated_seqs = {}
    for enhancer in weighted_seqs:
        values = weighted_seqs[enhancer]
        interpolated_seqs[enhancer] = np.sum(values * np.array([label_values['MM'], label_values['NM'], label_values['NP'], label_values['PP']])) / np.sum(values)

    return interpolated_seqs




def get_dataloc(enh):
    dict = {"Fasta_Pool2":["E25E10", "E22P3B3", "E24A3"], 
            "Fasta_Pool3":["E18C2", "E22P1B925", "E22P1E6", "E22P2F4", "E22P3G3", "E23G7", "E23H5"], 
            "Fasta_Pool4":["E22P1A3","E22P1C1", "E22P1D10", "E22P1H9", "E22P3B3R2", "E22P3F2", "E24C3", "E25B2", "E25B3", "E25E102"], 
            "Fasta_InSilico": ["FAKE1", "FAKE2", "FAKE3", "FAKE4", "FAKE5", "FAKE6", "FAKE7", "FAKE8", "FAKE9", "FAKE10"],
            "Fasta_Pool5": ["E22P1E6", "E22P1F10", "E22P1G5", "E22P3B3R3", "E22P3B4", "E23D6", "E23G1", "E25E10R3"]
            }
    


    for key in dict:
        if enh in dict[key]:
            return key
    # if enh not in dict.values():
    raise ValueError(f"Enhancer {enh} not found in any data location.")



def load_trained_model(model_id, folder, model, device="cpu", **kwargs):
    possible_models = []
    for file in os.listdir(folder):
        if file.startswith(f"{model_id}_") and file.endswith(".pt"):
            possible_models.append(file)
        
    if len(possible_models) == 0:
        raise ValueError(f"No model found with ID {model_id} in folder {folder}.")
    if len(possible_models) > 1:
        print(f"Multiple models found with ID {model_id}. Please select one:")
        for i, model in enumerate(possible_models):
            print(f"{i}: {model}")
        model_id = int(input("Enter the model index: "))
        model_name = possible_models[model_id]
    else:
        model_name = possible_models[0]

    model_name = model_name.split(".")[0]  # remove the .pt, to use the name for the other files
    print(f"Using model {model_name}")
    
    model.load_state_dict(torch.load(f"{folder}/{model_name}.pt", map_location=device))
    return model

    
