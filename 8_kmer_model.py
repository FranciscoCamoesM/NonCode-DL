import os
import numpy
from tqdm import tqdm

enh = "E25E10"

def generate_kmer(k, prefix=""):
    bases = ["A", "C", "G", "T"]
    
    if k == 0:
        return [prefix]
    
    kmers = []
    for base in bases:
        kmers.extend(generate_kmer(k - 1, prefix + base))
    
    return kmers


print("Generating kmers")
ks = [1,3]
kmers = []
for k in ks:
    kmers.extend(generate_kmer(k))
print("Kmers generated")


datasets = {}
for file in os.listdir("meme_seqs"):
    if file.endswith(".txt"):
        if file.split("_")[0] != enh:
            continue
        with open("meme_seqs/" + file, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith(">"):
                    continue
                seq = line.strip()
                if datasets.get(file) is None:
                    datasets[file] = []
                datasets[file].append(seq)
print("Datasets loaded")
print(datasets.keys())

kmer_dict = {kmer: 0 for kmer in kmers}

kmer_datasets = {}
for dataset in datasets:
    kmer_datasets[dataset] = []
    for seq in tqdm(datasets[dataset]):
        seq_kmer_dict = kmer_dict.copy()
        for k in ks:
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i + k]
                if "N" in kmer:
                    continue
                seq_kmer_dict[kmer] += 1
        # normalize
        # total = sum(seq_kmer_dict.values())
        # for kmer in seq_kmer_dict:
        #     seq_kmer_dict[kmer] /= total
        kmer_datasets[dataset].append(list(seq_kmer_dict.values()))

print("Kmer datasets generated")


# build classifier with perceptron
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
import numpy as np

X = []
y = []
for d, dataset in enumerate(kmer_datasets):
    print("Dataset: ", d, dataset)
    for i in range(len(kmer_datasets[dataset])):
        X.append(kmer_datasets[dataset][i])
        y.append(dataset.split("_")[1].split(".")[0] == "MM")
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train test split done")
print(X_train.shape)
print(X_test.shape)

#balance the dataset
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_train, y_train = ros.fit_resample(X_train, y_train)
print("Dataset balanced")

model = Perceptron()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Model trained")

# print confusion matrix
from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(y_test, y_pred)
print("Accuracy: ", (confusion[1][1] / (confusion[1][1] + confusion[1][0]) + confusion[0][0] / (confusion[0][0] + confusion[0][1]))/2)
print(confusion)


# show feature importance
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
importances = model.coef_[0]
indices = np.argsort(importances)[::-1]
feature_names = []
for k in kmers:
    feature_names.append(k)
feature_names = np.array(feature_names)
plt.figure(figsize=(25, 10))
plt.title("Feature importances")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
plt.xlim([-1, len(importances)])
plt.tight_layout()
plt.savefig("temp_figs/feature_importances.png")
plt.show()