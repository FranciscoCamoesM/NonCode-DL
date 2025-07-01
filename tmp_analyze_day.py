import os
import pandas as pd
import numpy as np



filenames = []
DAY = "0519"
important_params = ["Model_id", "Enhancer", "Label", "Coverage", "Jiggle", "TP", "TN", "Mean_Acc"]
df = pd.DataFrame(columns=important_params)

# for key in best_casedir.keys():
#     # get the value
#     value = best_casedir[key]
#     df = df._append({"Enhancer": key, "Label": "mm_v_all", "Coverage": 1, "Jiggle": 0, "TP": 0, "TN": 0, "Mean_Acc": value}, ignore_index=True)


for file in os.listdir("saved_models"):
    if file.endswith(".txt"):
        if file.split("_")[1] != DAY:
            continue

        # open the file
        params = {}

        with open(os.path.join("saved_models", file), "r") as f:
            lines = f.readlines()
            for line in lines:
                splits = line.split(":")
                if len(splits) != 2:
                    continue
                header, value = splits
                header = header.strip()
                value = value.strip()
                params[header] = value
        # print the parameters
        # print(params)

        smaller_dict = {}
        for key in important_params:
            if key in params:
                smaller_dict[key] = params[key]

        cm_file = file.split("_params.txt")[0] + "_cm.npy"

        # load the confusion matrix
        cm = np.load(os.path.join("saved_models", "training_history", cm_file))
        if cm.shape != (2, 2):
            print(f"Skipping {file} because confusion matrix is not 2x2. Shape is {cm.shape}")
            continue

        # print(cm[0,0], cm[1,1])

        smaller_dict["TP"] = cm[0, 0]
        smaller_dict["TN"] = cm[1, 1]
        smaller_dict["Mean_Acc"] = (cm[0, 0] + cm[1, 1]) / 2
        smaller_dict["Model_id"] = int(file.split("_")[0])

        # add the smaller dictionary to the dataframe
        df = df._append(smaller_dict, ignore_index=True)


# filter the dataframe for rows where the "Label" column is "mm_v_all"
print(f"Len of df immediately after appending: {len(df)}")

df = df[df["Label"] == "mm_v_all"]
# remove the "Label" column
df = df.drop(columns=["Label"])

print(f"Len of df after filtering Labels: {len(df)}")

#replace Nans in Jiggle with 0
df["Jiggle"] = df["Jiggle"].replace(pd.NA, 0)
# convert the "Jiggle" column to numeric
df["Jiggle"] = pd.to_numeric(df["Jiggle"], errors="coerce")
# convert the "Coverage" column to numeric
df["Coverage"] = pd.to_numeric(df["Coverage"], errors="coerce")


print(df.head())

# count how many times each value appears in the "Jiggle" column
jiggle_counts = df["Jiggle"].value_counts()
print(jiggle_counts)

# count how many times each value appears in the "Coverage" column 
coverage_counts = df["Coverage"].value_counts()
print(coverage_counts)

# count how many times each value appears in the "Enhancer" column
enhancer_counts = df["Enhancer"].value_counts()
print(enhancer_counts)

for Enhancer in df["Enhancer"].unique():
    # filter the dataframe for rows where the "Enhancer" column is equal to the current value
    filtered_df = df[df["Enhancer"] == Enhancer]
    # sort by acc
    filtered_df = filtered_df.sort_values(by=["Mean_Acc"], ascending=False)
    # sort by jiggle
    filtered_df = filtered_df.sort_values(by=["Jiggle"], ascending=False)
    # sort by coverage
    filtered_df = filtered_df.sort_values(by=["Coverage"], ascending=False)

    # print the filtered dataframe
    print(f"Filtered dataframe for {Enhancer}:")
    print(filtered_df)


exit()

# -------------------------------------------------------


# Grades
grades = [
  {"Effect of Adding Jiggle": "No effect", "Effect of Increasing Coverage": "Mixed",},
  {"Effect of Adding Jiggle": "No effect", "Effect of Increasing Coverage": "Needs Finetunign",},
  {"Effect of Adding Jiggle": "No effect", "Effect of Increasing Coverage": "Needs Finetunign", },
  {"Effect of Adding Jiggle": "Slight Loss (Probably No Effect)", "Effect of Increasing Coverage": "Detrimental (Needs Finetuning)", },
  {"Effect of Adding Jiggle": "No effect", "Effect of Increasing Coverage": "Detrimental",},
  {"Effect of Adding Jiggle": "Slight Loss", "Effect of Increasing Coverage": "Seems Detrimental",},
  {"Effect of Adding Jiggle": "No effect", "Effect of Increasing Coverage": "Seems Detrimental",},
  {"Effect of Adding Jiggle": "Slight Loss (Probably No Effect)", "Effect of Increasing Coverage": "Detrimental (Needs Finetuning)",},
  {"Effect of Adding Jiggle": "No effect", "Effect of Increasing Coverage": "Needs Finetunign",},
  {"Effect of Adding Jiggle": "No effect", "Effect of Increasing Coverage": "Slight Loss (Probably No Effect)",},
  {"Effect of Adding Jiggle": "No effect", "Effect of Increasing Coverage": "Needs Finetunign",},
  {"Effect of Adding Jiggle": "Improoved", "Effect of Increasing Coverage": "Needs Finetunign",},
  {"Effect of Adding Jiggle": "Needs Finetunign", "Effect of Increasing Coverage": "Terrible (Lost Data)",},
  {"Effect of Adding Jiggle": "Slight Loss (Probably No Effect)", "Effect of Increasing Coverage": "Needs Finetunign",},
  {"Effect of Adding Jiggle": "Slight Improvement", "Effect of Increasing Coverage": "Slight Loss (Probably No Effect)",},
  {"Effect of Adding Jiggle": "No effect", "Effect of Increasing Coverage": "Terrible (Lost Data)",},
  {"Effect of Adding Jiggle": "Slight Improvement", "Effect of Increasing Coverage": "Terrible (Lost Data)",},
  {"Effect of Adding Jiggle": "No effect", "Effect of Increasing Coverage": "Improoved", },
]

for i, enhancer in enumerate(best_casedir.keys()):
    grades[i]["Enhancer"] = enhancer


print("Grades:")
for i, grade in enumerate(grades):
    print(f"Enhancer {grade["Enhancer"]}: {grade["Effect of Adding Jiggle"]}, {grade["Effect of Increasing Coverage"]}")


unique_jiggle_grades = set([grade["Effect of Adding Jiggle"] for grade in grades])
print(f"\n\nUnique Jiggle Grades:")
for grade in unique_jiggle_grades:
    print(grade, sum([1 for g in grades if g["Effect of Adding Jiggle"] == grade]))
unique_coverage_grades = set([grade["Effect of Increasing Coverage"] for grade in grades])
print(f"\n\nUnique Coverage Grades:")
for grade in unique_coverage_grades:
    print(grade, sum([1 for g in grades if g["Effect of Increasing Coverage"] == grade]))