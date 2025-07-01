import argparse
import os

parser = argparse.ArgumentParser(description="Wrapper for saturation script")
parser.add_argument("--id", type=str, required=True, help="ID for the saturation run")
parser.add_argument("--w", type=str, default=None)

def main():
    args = parser.parse_args()
    training_id = args.id

    interest_models = []
    for enh in os.listdir("saved_models_try_2"):
        for model in os.listdir(os.path.join("saved_models_try_2", enh, "models")):
            if model.endswith(".pt") or model.endswith(".pth"):
                if training_id in model:
                    interest_models.append(os.path.join("saved_models_try_2", enh, "models", model))
    
    
    if not interest_models:
        print(f"No models found for training ID: {training_id}")
        return

    if len(interest_models) > 1:
        print(f"Multiple models found for training ID: {training_id}. Using the first one.")
        for i, model in enumerate(interest_models):
            print(f"{i}: {model}")

        selected_model = int(input("\nSelect the model to use (0 for first): "))
        if selected_model < 0 or selected_model >= len(interest_models):
            print("Invalid selection. Exiting.")
            return
        model_path = interest_models[selected_model]
    else:
        model_path = interest_models[0]

    print(f"Using model: {model_path}")

    enhancer_name = model_path.split(os.sep)[1].strip()
    print(f"Enhancer name: {enhancer_name}")

    path_to_model = os.path.join(*model_path.split(os.sep)[:-1])
    print(f"Path to model: {path_to_model}")

    label = "mm_v_all"

    if args.w is None:
        os.system(f"python 3_saturation_mutagenesis.py --id {training_id} --enh {enhancer_name} --save_dir {path_to_model} --label {label}")
    else:
        os.system(f"python 3_saturation_mutagenesis.py --id {training_id} --enh {enhancer_name} --save_dir {path_to_model} --label {label} --w {args.w}")
    




if __name__ == "__main__":
    main()