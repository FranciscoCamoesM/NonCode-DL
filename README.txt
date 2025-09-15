# Deep Learning Models to Predict the Functional Impact of Genomic Variants in Cis-Regulatory Sequences

This repository contains the code used for training and evaluating the models developed for the project "Deep Learning Models to Predict the Functional Impact of Genomic Variants in Cis-Regulatory Sequences". The models are designed to analyze genomic sequences and predict the effects of variants on regulatory functions.

---

## Repository Structure

### Data Files
- `binned_cell_data.tsv` – Data with the number of cells in each FACS bin for each dataset (used for interpolation).
- `original_enhancers.txt` – WT enhancer sequences of each dataset.
- `padding_seq.txt` – Padding sequence (from the viral vector). 
- **Disclaimer** - The dataset files are not included in the repository due to size constraints. Please contact the authors to obtain the datasets.

### Classification Model Pipeline
- `1_train_classification_net.py` – Train a classification network.
- `2_run_deeplift_classification.py` – Run **DeepLIFT** interpretation for classification.
- `3_run_saturation_mutagenesis_classification.py` – Run saturation mutagenesis for classification.
- `4_classification_attributions_average.py` – Average saturation mutagenesis or **DeepLIFT** results across models.
- `6_compare_models.py` – Compare trained models on different datasets.
- `7_comparison_grids.py` – Generate grids for visual model comparisons. 
- `8_test_class_as_regression.py` – Evaluate classification models on regression tasks.

### Regression Model Pipeline
- `9_train_regression_net.py` – Train a regression network.
- `10_regression_model_inference.py` – Inference pipeline for regression models.
- `10_1_regression_inference_plots.py` – Generate plots for regression inferences.
- `11_run_deeplift_regression.py` – Run **DeepLIFT** interpretation for regression.
- `12_deeplift_regression_plots.py` – Visualization of regression DeepLIFT results.
- `12_reg_model_comparisons.py` – Compare regression models.
- `13_run_saturation_mutagenesis_regression.py` – Regression with saturation mutagenesis.

### Inference on ChromBPNet
- `chrombpnet_inference_part_1.py` – ChromBPNet inference (part 1).
- `chrombpnet_inference_part_2.py` – ChromBPNet inference (part 2).

### Utilities
- `0_dataset_analysis.py` – Exploratory analysis of dataset.
- `data_preprocess.py` – Helper functions for data preprocessing.
- `models.py` – Model architecture definitions.
- `training_funcs.py` – Helper functions for training.
- `DL_torch_env.yml` – Conda environment file with dependencies.

### Documentation
- `README.txt` – Additional notes.
- `.gitignore` – Git ignore configuration.

---

## ⚙️ Installation

Create the environment using the provided Conda YAML file:

```bash
conda env create -f DL_torch_env.yml
conda activate dl_torch
