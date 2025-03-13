@echo off
REM Activate the DL_torch conda environment
CALL conda activate DL_torch

REM Loop from 0.1 to 1.0 in increments of 0.1
FOR /L %%G IN (1,1,9) DO (
    REM Calculate subset by dividing the loop variable by 10
    SET /A "subset=%%G"
    SET /A "div=%%G * 10 / 10"
    SET /A "mod=%%G %% 10"
    REM Run the python script with the current subset value
    python train_net.py --epochs 50 --wd 1e-3 --subset 0.%%G
)