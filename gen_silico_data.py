import os
import random
import numpy as np

WT_SEQ = "GTTACTGGCCATTCTTCTGGGCCTCAGAGCCAGGAGGGGCGGCTTTGTGGGGAAGCCCTTTCTGGCACCATTGTTTACTCCTGGGGTTGCTATGGTGATGCTGCTTGAGTGGTGATGCAGAGAAGAGGCCCTCACCTGTCCCTGGAGCCTCTTGTGACCCTTGGGCCCCTGGCTACTTTTCAGGTTAGACAGGCAGGGGAT"


TF_LIST = {"TGATGCTGC":3.5, "TGATGGAG":-1, "GTGA":1.5, "AAATTGTG":-3}
BB = 0.4
N_SEQS = 100000

random.seed(42)

def my_gaussian(min, max):
    mean = (min + max) / 2
    stddev = (max - min) / 6
    return np.random.normal(mean, stddev)


class SG:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.efficiency = my_gaussian(0.01, 0.2)
        print(f"Efficiency: {self.efficiency}")

    def cut(self, seq):
        if random.random() > self.efficiency:
            return seq
        
        cut_start = self.find_cut(seq, self.start)
        cut_end = self.find_cut(seq, self.end)

        if cut_start == -1 or cut_end == -1:
            return seq
        
        cut_start += min(int(my_gaussian(0,len(seq)/3)), int(my_gaussian(0,len(seq)/3)))
        cut_end += -min(int(my_gaussian(0,len(seq)/3)), int(my_gaussian(0,len(seq)/3)))

        if cut_start > cut_end:
            cut_start, cut_end = cut_end, cut_start

        if cut_start < 0:
            cut_start = 0
        if cut_end > len(seq):
            cut_end = len(seq)
        
        insertions = []
        while random.random() < 0.01:
            insertions.append(seq[random.randint(cut_start, cut_end)])
        insertions = ''.join(insertions)

        # print(f"Cutting {seq} at {cut_start} and {cut_end} with insertions {insertions}")

        new_seq = seq[:cut_start] + insertions + seq[cut_end:]

        return new_seq
    
    def find_cut(self, seq, motif):
        return seq.find(motif)

        
class SG2:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def cut(self, seq):
        if random.random() > 0.3:
            return seq
        start = self.start + int(my_gaussian(3, 12))
        end = self.end - int(my_gaussian(3, 12))

        start = max(0, min(start, len(seq)))
        end = max(0, min(end, len(seq)))
        if start > end:
            start, end = end, start
        if start == end:
            return seq

        return seq[:start] + seq[end:]


SG_LIST = [
    SG("CACCATTGTTTACTCC", "TCCCTGGAGCCTCTTGTG"),
    SG("TGCTGCTTGAGTGGTGATGCAGAG", "TAGACAGGC"),
    SG("CCATTCTTCT", "ACTTTTCAGGTTAGACAGG"),
    SG("TGGTGATGCTGCTTGAGTGG", "TCCCTGGAGCCTCTTGTG"),
    SG("CCATTCTTCT", "GCTATGGTGATG"),
    SG("TTACTCCTG", "CTTGTGACCCTTGGGCCC"),
    SG("CACCATTGTTTACTCC", "GCTTGAGTGGTGATGC"),
    # SG2(150, 175),
    # SG2(125, 161),
    # SG2(90, 130),
    # SG2(50, 80),
    # SG2(20, 60),
    ] 


def get_signal(seq, tfs):
    signal = 0
    for tf, value in tfs.items():
        if tf in seq:
            signal += value
        elif tf[:-1] in seq:
            signal += value*0.7
        elif tf[1:] in seq:
            signal += value*0.7
        elif tf[:-2] in seq or tf[2:] in seq or tf[1:-1] in seq:
            signal += value*0.3


    noise = my_gaussian(-2, 7)
    signal += noise
    return signal


def get_singal_batch(seqs, tfs):
    signals = []
    for seq in seqs:
        signal = get_signal(seq, tfs)
        signals.append(signal)
    return np.array(signals)


def generate_sequences(num_seqs, SG_LIST):
    sequences = []
    while len(sequences) < num_seqs:
        random.shuffle(SG_LIST)
        seq = WT_SEQ
        for sg in SG_LIST:
            seq = sg.cut(seq)

            if len(seq) < 125:
                break
        sequences.append(seq)
    return sequences

def generate_labels(sequences, tfs):
    labels = []
    for seq in sequences:
        label = get_signal(seq, tfs)
        labels.append(label)
    return np.array(labels)

def generate_data(num_seqs, tfs, SG_LIST):
    sequences = generate_sequences(num_seqs, SG_LIST)
    labels = generate_labels(sequences, tfs)
    return sequences, labels

dataset = generate_data(N_SEQS, TF_LIST, SG_LIST)

# for x, y in zip(*dataset):
#     print(len(x))
#     print(y)
#     print()
        


import matplotlib.pyplot as plt

#plot histogram of labels
plt.hist(dataset[1], bins=50)
plt.xlabel('Labels')
plt.ylabel('Frequency')
plt.title('Histogram of Labels')
plt.savefig('temp_figs/histogram_labels.png')
plt.show()
#plot histogram of sequences

mean_signal = -1
NN = 5000
for _ in range(NN):
    mean_signal += get_signal(WT_SEQ, TF_LIST)/NN

def facs(cells, thresh):
    negative = [i for i, cell in enumerate(cells) if cell <= thresh]
    positive = [i for i, cell in enumerate(cells) if cell > thresh]
    return positive, negative

def percentual_facs(seqs, labels, threshes = [BB, 1, 2 - BB]):
    classes = []
    for thresh in threshes:
        positive, negative = facs(labels, thresh * mean_signal)
        negative_cells = [seqs[i] for i in negative]
        negative_labels = [labels[i] for i in negative]
        classes.append((negative_cells, negative_labels))
        seqs = [seqs[i] for i in positive]
        labels = [labels[i] for i in positive]

    classes.append((seqs, labels))

    return classes

# test
seqs, labels = dataset
classes = percentual_facs(seqs, labels)
for i, (seqs, labels) in enumerate(classes):
    plt.hist(labels, bins=50)
    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Labels - Class {i}')
    plt.savefig(f'temp_figs/histogram_labels_class_{i}.png')
    plt.show()

print("Classes: ", len(classes))
print("mean signal: ", mean_signal)
for class_num, (seqs, labels) in enumerate(classes):
    print(f"Class {class_num}:")
    print(f"Number of sequences: {len(seqs)}")
    print(f"Mean label: {np.mean(labels)}")
    print()
    

# SAVE AS enhancer FAKE1.fasta

dataloc = "Fasta_InSilico"
filename_template = "Insilico_FAKE5{}.fa"

labels = ["MM", "M", "P", "PP"]

if not os.path.exists(dataloc):
    os.makedirs(dataloc)

for i, label in enumerate(labels):
    filename = os.path.join(dataloc, filename_template.format(label))
    with open(filename, "w") as f:
        seq, label = classes[i]
        for j, (seq, label) in enumerate(zip(seq, label)):
            f.write(f">{j}-{label}\n")
            f.write(f"{seq}\n")
print("Files saved in ", dataloc)

print(len(dataset[0]))
print(len(list(set(dataset[0]))))