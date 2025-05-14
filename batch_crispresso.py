import os
import argparse
import matplotlib.pyplot as plt
import math


argparser = argparse.ArgumentParser()
argparser.add_argument("--local_path", default="../NonCode/Fasta_Pool3")

local_path = argparser.parse_args().local_path

all_files = []
for file in os.listdir(local_path):
    if file.endswith(".fa"):
        all_files.append(file)



enh_names = []

for file in all_files:
    e = file.split(".")[-2].split("_")[-1]

    if e[-1] == "P" or e[-1] == "M":
        e = e[:-1]
        if e[-1] == "P" or e[-1] == "M":
            e = e[:-1]


    enh_names.append(e)

enh_names = list(set(enh_names))

print(enh_names)

# load original enhancers

file = "original_enhancers.txt"
wt_enhancers = {}
with open(file, "r") as f:
    for line in f:
        if line.startswith(">"):
            enhancer = line[1:].strip()
        else:
            seq = line.strip()
            wt_enhancers[enhancer] = seq



for enh in enh_names:

    interesting_files = [f for f in all_files if enh in f]

    # create unique enhancer files without duplicates
    all_enhs = {}
    for f in interesting_files:
        file_class = f.split(".")[-2].split(enh)[-1]
        with open(f"{local_path}/{f}", "r") as file:
            for line in file:
                if not(line.startswith(">")):
                    if line.strip() not in all_enhs:
                        all_enhs[line.strip()] = [file_class]
                    else:
                        all_enhs[line.strip()].append(file_class)

    # print unique enhancer files
    
    # class_counts = []
    # for enh_mut in all_enhs:
    #     class_counts.append(math.log(len(all_enhs[enh_mut]))/math.log(2))

    # plt.hist(class_counts, bins = 20, density=True)
    # plt.show()

    # chose the most common class
    enh_single_class = {}
    for enh_mut in all_enhs:
        enh_single_class[enh_mut] = max(set(all_enhs[enh_mut]), key = all_enhs[enh_mut].count)


    class_es_dist = {}
    for enh_mut in enh_single_class:
        if enh_single_class[enh_mut] not in class_es_dist:
            class_es_dist[enh_single_class[enh_mut]] = 1
        else:
            class_es_dist[enh_single_class[enh_mut]] += 1

    # plt.bar(class_es_dist.keys(), class_es_dist.values())
    # plt.show()



    save_loc = f"../unique_enhancers/"
    # check if save_loc exists
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    with open(f"{save_loc}unique_{enh}M.fa", "w") as f:
        for enh_mut in enh_single_class:
            if enh_single_class[enh_mut] == "M":
                f.write(f">{enh_mut}\n{enh_mut}\n")

    with open(f"{save_loc}unique_{enh}P.fa", "w") as f:
        for enh_mut in enh_single_class:
            if enh_single_class[enh_mut] == "P":
                f.write(f">{enh_mut}\n{enh_mut}\n")

    with open(f"{save_loc}unique_{enh}PP.fa", "w") as f:
        for enh_mut in enh_single_class:
            if enh_single_class[enh_mut] == "PP":
                f.write(f">{enh_mut}\n{enh_mut}\n")

    with open(f"{save_loc}unique_{enh}MM.fa", "w") as f:
        for enh_mut in enh_single_class:
            if enh_single_class[enh_mut] == "MM":
                f.write(f">{enh_mut}\n{enh_mut}\n")

    filename_template = "unique_{0}{1}.fa"
    print("python unique_fastq.py {} {} {} {}".format(*[save_loc + filename_template.format(enh, class_name) for class_name in ["M", "P", "PP", "MM"]]))

    # print("\n\n")
    # print("-"*20)
    # print(f"Enhancer {enh}")
    # print(wt_enhancers[enh])
    # print("")
    # batch file
    batch_name = "batch_" + enh + ".batch"
    # print(f"Batch file: {batch_name}")
    # print("n\tr1")
    for f in interesting_files:
        # print(f.split(".")[-2].split(enh)[-1] + "\t" + "./" + local_path + "/" + f)
        print(f.split(".")[-2].split(enh)[-1] + "\t" + save_loc + f"unique_{enh}{f.split('.')[-2].split(enh)[-1]}.fa")

    with open(batch_name, "w") as f:
        for file in interesting_files:
            f.write(f.split(".")[-2].split(enh)[-1] + "\t" + save_loc + f"unique_{enh}{f.split('.')[-2].split(enh)[-1]}.fastq\n")

    
    # print(f"CRISPRessoBatch --batch_settings {batch_name} --amplicon_seq {wt_enhancers[enh]} -p 4 --base_editor_output -w 20")
    os.system(f"CRISPRessoBatch --batch_settings {batch_name} --amplicon_seq {wt_enhancers[enh]} -p 4 --base_editor_output -w 20")

    print("-"*20)