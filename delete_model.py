import argparse

parser = argparse.ArgumentParser(description='Train a model to predict enhancer activity')

parser.add_argument('--id', type=int, default=None, help='Batch size for training')

args = parser.parse_args()

SAVE_DIR = 'saved_models'
model_id = args.id

if model_id is None:
    model_id = input("Enter the model id: ")


## find model name with corresponding id
possible_models = []
for file in os.listdir(SAVE_DIR):
    if file.startswith(str(model_id)):
        if file.endswith(".pt"):
            possible_models.append(file)

if len(possible_models) == 0:
    print("No model found with that id")
    exit()
if len(possible_models) > 1:
    print("Multiple models found with that id")
    for i, model in enumerate(possible_models):
        print(f"{i}: {model}")

    model_id = input("Enter the model index: ")
    model_name = possible_models[int(model_id)]
else:
    model_name = possible_models[0]

model_name = model_name.split(".")[0]  # remove the .pt, to use the name for the other files

del possible_models     # clean up

print(f"Using model {model_name}")


print(model_name)

# delete files in the directory
import os
os.remove(f"{SAVE_DIR}/{model_name}.pt")
os.remove(f"{SAVE_DIR}/{model_name}_params.txt")
os.remove(f"{SAVE_DIR}/plots/{model_name}_cm.png")
os.remove(f"{SAVE_DIR}/training_history/{model_name}_cm.npy")
os.remove(f"{SAVE_DIR}/training_history/{model_name}_all_train_losses.npy")
os.remove(f"{SAVE_DIR}/training_history/{model_name}_all_val_losses.npy")