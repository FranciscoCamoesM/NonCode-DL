import os
import matplotlib.pyplot as plt
from tangermeme.plot import plot_logo
import numpy as np
from data_preprocess import load_wt, simple_pad, one_hot_encode

from p11_run_deeplift_regression import run_deeplift_regression
from p12_run_satmut_regression import run_saturation_mutagenesis

output_dir = "explicability_final_reg/deeplift"

enh_to_plot = []

enh_to_plot.append(["E25E10", "2025-07-19_03-10-32"])
enh_to_plot.append(["E22P1A3", "2025-07-19_05-26-53"])
enh_to_plot.append(["E25E102", "2025-07-18_15-17-32"])
enh_to_plot.append(["E25E10R3", "2025-07-18_17-22-54"])

enh_to_plot.append(["E22P1D10", "2025-07-19_15-27-12"])
enh_to_plot.append(["E22P1F10", "2025-07-18_12-55-41"])
enh_to_plot.append(["E23H5", "2025-07-20_09-56-28"])



values = []
for enh, model_id in enh_to_plot:
    run_deeplift_regression(model_id=model_id, enh=enh, device="cuda", output_dir="explicability_final_reg/deeplift", n_refs=100, model_save_dir="../NonCode/regression_models/", DPI=150)
    run_saturation_mutagenesis(model_id=model_id, enh=enh, device="cuda", output_dir="explicability_final_reg/satmut_logos/", batch_size=256, model_save_dir="../NonCode/regression_models/", DPI=150)

print("All done!")