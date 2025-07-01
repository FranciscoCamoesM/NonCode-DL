# run optuna python 1_train_net.py --enh {ENHANCER_NAME} --lr {lr} --wd {wd} --epcohs 600 --dataloc {dataloc}

ENHANCER_NAME = "E22P3F2"
ENHANCER_NAMEZZZZZZ = ["E22P1E6", "E22P3B4", "E22P1F10", "E25E10R3", "E22P3B3R3"]
dataloc = "Fasta_Pool5"
label = 'mm_v_all'

import optuna
import os
import random

def objective(trial):
    lr = trial.suggest_float("lr", 1e-8, 1e-4, log=True)
    wd = 0 #trial.suggest_float("wd", 1e-12, 2e-3, log=True)
    cov = 3
    
    random_seed = random.randint(0, 100000)
    os.system(f"python 1_train_net.py --label {label} --enh {ENHANCER_NAME} --lr {lr} --wd {wd} --epochs 200 --optuna True --seed {random_seed} --best_val True --coverage {cov} --jiggle 5")

    # load f1 score from file

    with open(f"optuna_log/{ENHANCER_NAME}_f1.txt", "r") as f:  
        f1 = float(f.read())
    # delete f1 file
    os.remove(f"optuna_log/{ENHANCER_NAME}_f1.txt")

    return f1

storage = optuna.storages.RDBStorage("sqlite:///db.sqlite3", skip_table_creation=False)
for ENHANCER_NAME in ENHANCER_NAMEZZZZZZ:
    study = optuna.create_study(direction="maximize", storage=storage, study_name=ENHANCER_NAME+' . '+label, load_if_exists=True)
    study.optimize(objective, n_trials=500)

    # record best parameters
    best_params = study.best_params
    with open(f"optuna_log/{ENHANCER_NAME}_best_params.txt", "w") as f:
        f.write(f"Best parameters: {best_params}")
