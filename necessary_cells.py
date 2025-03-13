from data_preprocess import import_files
import numpy as np
import os


# Load the data
local_path = "Fasta_Pool2"
all_files = import_files(local_path)


enh_list = []
for file in all_files:
    enh_list.append(file.file_enh)

    

enh_list = np.unique(enh_list)

print(f"Number of enhancers: {len(enh_list)}")



muts_to_uniq = {}
enh_per_class = {}


for enh in enh_list:
    mutation_entries = {}
    all_mutations = []
    for file in all_files:
        if file.file_enh != enh:
            continue

        file.load()
        sequences = [entry[1] for entry in file.contents]
        enh_per_class[str(file)] = len(sequences)
        all_mutations.extend(sequences)

    unique_all_mutations = np.unique(all_mutations)

    # print(len(unique_all_mutations), len(all_mutations))
    print(f"Enhancer {enh} has {len(unique_all_mutations)} mutations, {len(all_mutations)} total entries")
    muts_to_uniq[enh] = (len(unique_all_mutations), len(all_mutations))

print(enh_per_class)

import  matplotlib.pyplot as plt

# get linear regression
x = [entry[0] for entry in muts_to_uniq.values()]
y = [np.log(entry[1]) for entry in muts_to_uniq.values()]
x, y = zip(*sorted(zip(x, y)))

m, b = np.polyfit(x, y, 1)
print(f"polyfit: {m}x + {b}")
plt.plot(x, m*np.array(x) + b, 'k--', alpha = 0.5, label = f"y = e ^ ({m}x + {b:.2f})")
print(x,  m*np.array(x) + b)
plt.scatter(x,y)
plt.title("Number of unique mutations vs total number of entries")
plt.xlabel("Number of unique mutations")
plt.ylabel("Total number of entries")
plt.legend()
plt.tight_layout()
plt.show()

cells_used = {
    'E22P3B3, MM':	6828,
    'E22P3B3, NM':	17882,
    'E22P3B3, NP':	23407,
    'E22P3B3, PP':	1787,
    'E24A3, MM': 419,
    'E24A3, NM': 2970,
    'E24A3, NP': 1618,
    'E24A3, PP': 121,
    'E25E10, MM': 	6231,
    'E25E10, NM':	30599,
    'E25E10, NP':	31650,
    'E25E10, PP':	285,
}


# cells used vs total number of entries
cells_used_enh = {}

for enh in enh_list:
    total = 0
    for name, count in cells_used.items():
        if name.startswith(enh):
            total += count
    cells_used_enh[enh] = int(total)


print(cells_used_enh)
print(muts_to_uniq)
x = [entry for entry in cells_used_enh.values()]
y =  [entry[1] for entry in muts_to_uniq.values()]
print(x)
print(y)
plt.scatter(x,y)
plt.title("Number of cells used vs total number of entries")
plt.xlabel("Number of cells used")
plt.ylabel("Total number of entries")
plt.show()


# per class 
enh_to_color = {
    'E22P3B3': 'red',
    'E24A3': 'blue',
    'E25E10': 'green',
}

class_to_color = {
    'MM': 'red',
    'NM': 'blue',
    'NP': 'green',
    'PP': 'purple',
}



#linear regression
x = [enh_per_class[key] for key in enh_per_class.keys()]
y = [cells_used[key] for key in enh_per_class.keys()]


ids_to_remove = [9, 11, 3]
x = [x[i] for i in range(len(x)) if i not in ids_to_remove]
y = [y[i] for i in range(len(y)) if i not in ids_to_remove]


x = np.log(x)
x, y = zip(*sorted(zip(x, y)))
m, b = np.polyfit(x, y, 1)
print(f"polyfit: {m}x + {b}")
x = np.exp(x)
# plt.plot(np.exp(x), m*np.array(x) + b, 'k--', alpha = 0.5, label = f"y = {m:.2f}x + {b:.2f}")
plt.plot(np.linspace(min(x), max(x), 10), m*np.array(np.log(np.linspace(min(x), max(x), 10))) + b, 'k--', alpha = 0.5, label = f"y = {m:.2f}log(x) + {b:.2f}")


keys = list(enh_per_class.keys())
keys = [key for i, key in enumerate(enh_per_class.keys()) if i not in ids_to_remove]
# plt.scatter(np.log(x), np.log(y), c = [enh_to_color[key.split(", ")[0]] for key in keys])
plt.scatter(x, y, c = [enh_to_color[key.split(", ")[0]] for key in keys])
# for i, key in enumerate(keys):
#     plt.annotate(key, (x[i], y[i]))
# plt.scatter(x, y, c = [class_to_color[key.split(", ")[1]] for key in keys])
plt.title("Number of cells used vs total number of entries")
plt.xlabel("Number of entries")
plt.ylabel("Number of cells used")
plt.legend()
plt.show()