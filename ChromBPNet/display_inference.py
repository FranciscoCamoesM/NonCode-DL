import tables
import matplotlib.pyplot as plt
from plot import plot_logo
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Plot SHAP values")
parser.add_argument("--bed", type=str, default="seqs_predict_sorted.bed", help="Path to the sorted bed file")
parser.add_argument("--o", type=str, default="temp_fig.png", help="Output directory")
parser.add_argument("--i", type=str, default="numerical_inference_results_custom", help="Input directory")
parser.add_argument("--window", type=int, default=250, help="Window size")
parser.add_argument("--shift", type=int, default=125, help="Shift")
parser.add_argument("--tick_spacing", type=int, default=25, help="Tick spacing")
parser.add_argument("--save_output", action="store_true", help="Save the output to a file")
parser.add_argument("--outfile", type=str, default="custom_seqs_preds.txt", help="Output file for BigWig data")

args = parser.parse_args()


WINDOW_SIZE = args.window
SHIFT = args.shift
TICK_SPACING = args.tick_spacing
source = args.bed
destination = args.o
input_dir = args.i


if ".png" in destination:
    destination = destination.split(".png")[0]


import pyBigWig
import pandas as pd

output_prefix = f"{input_dir}/results"

def print_bed_file(filepath):
    """ Reads and prints the first few lines of a BED file """
    try:
        data = pd.read_csv(filepath, sep='\t', header=None)
        print("Contents of", filepath)
        print(data.head())  # Print first few lines of the file

        # check if files come from classification data
        if "-" in data[0][0]:
            return True
        else:
            return False
    except Exception as e:
        print(f"Failed to read {filepath}: {str(e)}")

def print_bigwig_file(filepath):
    """ Reads and prints some statistics from a BigWig file """
    try:
        bw = pyBigWig.open(filepath)
        if bw is None:
            print(f"Failed to open {filepath}.")
            return

        print("Statistics of", filepath)
        for chrom in bw.chroms():
            print(f"Stats for {chrom}: {bw.stats(chrom)}")
        bw.close()
    except Exception as e:
        print(f"Failed to read {filepath}: {str(e)}")

def save_bigwig_file(filepath, out_filepath = "custom_seqs_preds.txt"):
    """ Reads and prints some statistics from a BigWig file """
    try:
        bw = pyBigWig.open(filepath)
        if bw is None:
            print(f"Failed to open {filepath}.")
            return

        with open(out_filepath, 'w') as f:
            for chrom in bw.chroms():
                chrom_name = str(chrom).replace("_", "\t").replace("m", "-").replace("p", ".")
                if "last" in chrom_name:
                    continue  # Skip chromosomes that are not relevant
                f.write(f"{chrom_name}\t{bw.stats(chrom)[0]}\n")
        bw.close()
        print(f"BigWig data saved to {out_filepath}")

        
    except Exception as e:
        print(f"Failed to read {filepath}: {str(e)}")

def print_bigwig_classification(filepath):
    try:
        bw = pyBigWig.open(filepath)
        if bw is None:
            print(f"Failed to open {filepath}.")
            return

        print("Statistics of", filepath)
        dataset = []
        for chrom in bw.chroms():
            class_ = int(chrom.split("-")[1])
            # print(f"Stats for {class_}: {bw.stats(chrom)}")
            dataset.append([class_, bw.stats(chrom)[0]])
        bw.close()

        correlation = np.corrcoef([x[0] for x in dataset], [x[1] for x in dataset])[0, 1]
        # plt.scatter([x[0] for x in dataset], [x[1] for x in dataset], label=f"Pierson correlation: {correlation:.2f}")
        boxplot_data = []
        for i in range(max([x[0] for x in dataset]) + 1):
            boxplot_data.append([x[1] for x in dataset if x[0] == i])
        plt.boxplot(boxplot_data, positions=range(len(boxplot_data)), widths=0.6)
        plt.title(f"Pierson correlation: {correlation:.2f}")
        plt.xticks(range(len(boxplot_data)), [str(i) for i in range(len(boxplot_data))])
        plt.xlabel("Class")
        plt.ylabel("Predicted Accessibility")
        plt.grid()
        plt.savefig(f"{destination}_box.png")
        plt.close()
        
        # save the data to a CSV file
        df = pd.DataFrame(dataset, columns=["Class", "Value"])
        df.to_csv(f"{destination}_classification_data.csv", index=False)

        plt.violinplot(boxplot_data, positions=range(len(boxplot_data)), widths=0.6)
        plt.title(f"Pierson correlation: {correlation:.2f}")
        plt.xticks(range(len(boxplot_data)), [str(i) for i in range(len(boxplot_data))])
        plt.xlabel("Class")
        plt.ylabel("Predicted Accessibility")
        plt.grid()
        plt.savefig(f"{destination}_violin.png")
        plt.close()

        print(f"Scatter plot saved as {destination}_violin.png")



    except Exception as e:
        print(f"Failed to read {filepath}: {str(e)}")


def main(SAVE_OUTPUT = False):
    # Define file names
    files = {
        'chrombpnet_nobias_bw': f"{output_prefix}_chrombpnet_nobias.bw",
        'chrombpnet_nobias_bed': f"{output_prefix}_chrombpnet_nobias_preds.bed"
    }

    # Print contents of BED files
    for key, filepath in files.items():
        if filepath.endswith('.bed'):
            classification = print_bed_file(filepath)

    if classification:
        for key, filepath in files.items():
            if filepath.endswith('.bw'):
                print_bigwig_classification(filepath)
    else:
        # Print statistics of BigWig files
        for key, filepath in files.items():
            if filepath.endswith('.bw'):
                print_bigwig_file(filepath)
                if SAVE_OUTPUT:
                    save_bigwig_file(filepath, args.outfile)

if __name__ == '__main__':
    SAVE_OUTPUT = args.save_output
    main(SAVE_OUTPUT)
