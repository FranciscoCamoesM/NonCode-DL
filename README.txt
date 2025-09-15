# Deep Learning Models to Predict the Functional Impact of Genomic Variants in Cis-Regulatory Sequences

This repository contains the code used for training and evaluating the models developed for the project "Deep Learning Models to Predict the Functional Impact of Genomic Variants in Cis-Regulatory Sequences". The models are designed to analyze genomic sequences and predict the effects of variants on regulatory functions.

---

## Repository Structure

### Data Files
- `binned_cell_data.tsv` - Data with the number of cells in each FACS bin for each dataset (used for interpolation).
- `original_enhancers.txt` - WT enhancer sequences of each dataset.
- `padding_seq.txt` - Padding sequence (from the viral vector). 
- **Disclaimer** - The dataset files are not included in the repository due to size constraints. Please contact the authors to obtain the datasets.

### Classification Model Pipeline
- `1_train_classification_net.py` - Train a classification network.
- `2_run_deeplift_classification.py` - Run **DeepLIFT** interpretation for classification models.
- `3_run_saturation_mutagenesis_classification.py` - Run saturation mutagenesis for classification models.
- `4_classification_attributions_average.py` - Average saturation mutagenesis or **DeepLIFT** results across models.
- `6_compare_models.py` - Compare trained models on different datasets.
- `7_comparison_grids.py` - Generate grids for visual model comparisons. 
- `8_test_class_as_regression.py` - Evaluate classification models on regression tasks.

### Regression Model Pipeline
- `9_train_regression_net.py` - Train a regression network.
- `10_regression_model_inference.py` - Perform inference on any dataset using the trained regression models.
- `10_1_regression_inference_plots.py` - Generate plots for regression inferences.
- `11_run_deeplift_regression.py` - Run **DeepLIFT** interpretation for regression models.
- `12_deeplift_regression_plots.py` - Visualization of regression DeepLIFT results.
- `13_run_saturation_mutagenesis_regression.py` - Run saturation mutagenesis for regression models.
- `14_reg_model_comparisons.py` - Compare regression models.

### Inference on ChromBPNet
- `chrombpnet_inference_part_1.py` - Preprocess data for inference on ChromBPNet.
- `chrombpnet_inference_part_2.py` - Plot results and calculate metrics for ChromBPNet inference.
- `ChromBPNet/` - Directory containing the scripts to be run using ChromBPNet environment.
- `ChromBPNet/chrombpnet_env.yml` - Conda environment file for ChromBPNet.
- `ChromBPNet/training_protocol.sh` - Training protocol used to train ChromBPNet models.
- `ChromBPNet/inference_custom_seqs.sh` - Shell script to run inference on custom sequences using trained ChromBPNet models.
- `ChromBPNet/generate_padded_genome.py` - Generate mock padded genome to perform inference using ChromBPNet.
- `ChromBPNet/display_inference.py` - Displays inference results from ChromBPNet for a given regression dataset.

### Utilities
- `0_dataset_analysis.py` - Exploratory analysis of dataset.
- `data_preprocess.py` - Helper functions for data preprocessing.
- `models.py` - Model architecture definitions.
- `training_funcs.py` - Helper functions for training.
- `DL_torch_env.yml` - Conda environment file with dependencies.

### Documentation
- `README.txt` - Additional notes.
- `.gitignore` - Git ignore configuration.

---

## Usage Instructions

### Installation of the environment for Classification and Regression Models

Create the environment using the provided Conda YAML file:

```bash
conda env create -f DL_torch_env.yml
conda activate dl_torch
```

### ChromBPNet

ChromBPNet requires a separate environment. Create it using the provided YAML file:
```bash
conda env create -f ChromBPNet/chrombpnet_env.yml
conda activate chrombpnet
```

Using the DL_torch_env environment, run chrombpnet_inference_part_1.py on the desired dataset to preprocess its sequences. Then, switch to the chrombpnet environment to run inference_custom_seqs.sh. Finally, switch back to the DL_torch_env environment to run chrombpnet_inference_part_2.py for plotting and metrics.


