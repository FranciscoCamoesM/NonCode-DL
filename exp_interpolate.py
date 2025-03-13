from data_preprocess import *
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.show()



# Objective: See if most enhancers have more than one entry

all_files = import_files()
enhancer_names = [f.file_enh for f in all_files]
enhancer_names = list(set(enhancer_names))

loaded_enhancers = []

for enhancer in enhancer_names:
    # plt.bar([str(f) for f in all_files if f.file_enh == enhancer], [len(f) for f in all_files if f.file_enh == enhancer])
    # plt.xticks(rotation=90)
    # plt.xlabel("Number of enhancers")
    # plt.ylabel("File")
    # plt.title("Number of enhancers in each file for enhancer " + enhancer)
    # plt.show()

    enhancer_list = {}
    for file in all_files:
        file.load()
        if file.file_enh != enhancer:
            continue

        for entry in file.contents:
            if entry[1] in enhancer_list:
                enhancer_list[entry[1]].append(file.file_class)
            else:
                enhancer_list[entry[1]] = [file.file_class]

    
    loaded_enhancers.append(enhancer_list)




# # check how many of the same enhancer across different files
# for enhancer in loaded_enhancers:
#     plt.hist([len(enhancer[entry]) for entry in enhancer])
#     plt.xlabel("Number of files")
#     plt.ylabel("Number of enhancers")
#     plt.title("Number of enhancers in each file for enhancer ")
#     plt.show()

# # coclusion: most enhancers have only one entry. therefore we cant use https://www.biorxiv.org/content/10.1101/2023.12.20.572268v1
