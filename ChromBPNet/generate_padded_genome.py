import os

import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Generate a padded genome from a FASTA file.")
parser.add_argument("--input_fa", type=str, default="custom_seqs.fa", help="Input FASTA file.")
parser.add_argument("--out_fa", type=str, default=None, help="Output padded FASTA file.")
parser.add_argument("--out_bed", type=str, default=None, help="Output BED file.")
parser.add_argument("--generate_bed", action="store_false", help="Generate a BED file. Default is True when left unspecified.")
parser.add_argument("--total_length", type=int, default=2114, help="Total length of the padded genome.")
args = parser.parse_args()


# Assign arguments to variables
INPUT_FA = args.input_fa

file_title = os.path.splitext(os.path.basename(INPUT_FA))[0]
if args.out_fa is not None:
    OUT_FA = args.out_fa
else:
    OUT_FA = f"{file_title}_padded.fa"

if args.out_bed is not None:
    OUT_BED = args.out_bed
else:
    OUT_BED = f"{file_title}.bed"

generate_bed = args.generate_bed
TOTAL_LENGTH = args.total_length


# check if the input file exists
if not os.path.isfile(INPUT_FA):
    raise FileNotFoundError(f"Input file {INPUT_FA} does not exist. Please check the file path.")

# read the input FASTA file
sequences = []
with open(INPUT_FA, "r") as f:
    name = None
    for line in f:
        if line.startswith(">"):
            name = line.strip()[1:]
            if name == "":
                name = None
            continue
        else:
            sequences.append([name, line.strip()])
            name = None

if len(sequences) == 0:
    raise ValueError("No sequences found in the file. Please check if the FASTA file is properly formatted with '>' and sequence lines.")


# generate a padded genome
data_lines = []
c = 0
for i, (name, seq) in enumerate(sequences):
    print(f"Processing sequence {i+1}/{len(sequences)}: {name}")
    if name is None:
        name = f"seq_{c}"
        c += 1
    sequences[i][0] = name

for name, seq in sequences:    
    data_lines.append(f">{name}")


    padded_base = TOTAL_LENGTH - len(seq)
    if padded_base < 0:
        raise ValueError(f"Sequence {name} is longer than the total length of {TOTAL_LENGTH}. Please check the sequence length.")
    
    right_padding = padded_base // 2
    left_padding = padded_base - right_padding
    padded_seq = "N" * left_padding + seq + "N" * right_padding
    data_lines.append(padded_seq)

# write the padded genome to a new file
output_file = os.path.join(os.path.dirname(INPUT_FA), OUT_FA)
with open(output_file, "w") as f:
    for line in data_lines:
        f.write(line + "\n")
print(f"Padded genome written to {output_file}")

if generate_bed:
    # create a bed file
    output_file = os.path.join(os.path.dirname(INPUT_FA), OUT_BED)
    with open(output_file, "w") as f:
        for name, seq in sequences:
            # get the chromosome name and start and end positions
            chrom = name.split(":")[0]
            f.write(f"{chrom}\t1056\t1255\t{chrom}\t1000\t+\t1056\t1255\t0\t1\n")
    print(f"BED file written to {output_file}")

print("Done!")