# prefetch SRR7585374 - data from https://pmc.ncbi.nlm.nih.gov/articles/PMC6389269/
fastq-dump --split-files --gzip SRR7585374.sra



# download genome
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz -O data/genomes/hg38.fa.gz
wget https://www.encodeproject.org/files/GRCh38_EBV.chrom.sizes/@@download/GRCh38_EBV.chrom.sizes.tsv -O data/genomes/hg38.chrom.sizes
wget https://www.encodeproject.org/files/ENCFF356LFX/@@download/ENCFF356LFX.bed.gz -O data/genomes/blacklist.bed.gz

gunzip data/genomes/hg38.fa.gz
gunzip data/genomes/blacklist.bed.gz


bowtie2-build data/genomes/hg38.fa.gz data/genomes/hg38

# all in allignement.sh
   # bowtie2 -x hg38   -1 data/downloads/SRR7585374/SRR7585374_1.fastq.gz   -2 data/downloads/SRR7585374/SRR7585374_2.fastq.gz   --very-sensitive -X 2000 -p 8 | samtools view -bS - > data/alignments/SRR7585374.bam
   # samtools sort -@8 -o data/alignments/SRR7585374.sorted.bam data/alignments/SRR7585374.bam
   # samtools index data/alignments/SRR7585374.sorted.bam


# Relaxed peak calling (p-value 0.01)
macs2 callpeak  -t data/alignments/SRR7585374.sorted.bam  -f BAMPE  -g hs  -n SRR7585374_relaxed  --outdir data/peaks/  --pvalue 0.01


# Ensure that the peak regions do not intersect with the blacklist regions
bedtools slop -i data/genomes/blacklist.bed -g data/genomes/hg38.chrom.sizes -b 1057 > data/downloads/temp.bed
bedtools intersect -v -a data/peaks/SRR7585374_relaxed_peaks.narrowPeak -b data/downloads/temp.bed > data/peaks/SRR7585374_filtered_peaks.bed
rm data/downloads/temp.bed


# Create validation and test splits
head -n 24 data/genomes/hg38.chrom.sizes > data/genomes/hg38.chrom.subset.sizes
mkdir -p data/splits
chrombpnet prep splits -c data/genomes/hg38.chrom.subset.sizes -tcr chr1 chr3 chr6 -vcr chr8 chr20 -op data/splits/fold_0
chrombpnet prep splits -c data/genomes/hg38.chrom.subset.sizes -tcr chr2 chr8 chr9 chr16 -vcr chr12 chr17 -op data/splits/fold_1
chrombpnet prep splits -c data/genomes/hg38.chrom.subset.sizes -tcr chr4 chr11 chr12 chr15 chrY -vcr chr22 chr7 -op data/splits/fold_2
chrombpnet prep splits -c data/genomes/hg38.chrom.subset.sizes -tcr chr5 chr10 chr14 chr18 chr20 chr22 -vcr chr6 chr21 -op data/splits/fold_3
chrombpnet prep splits -c data/genomes/hg38.chrom.subset.sizes -tcr chr7 chr13 chr17 chr19 chr21 chrX -vcr chr10 chr18 -op data/splits/fold_4


# generate non-peak regions
chrombpnet prep nonpeaks -g data/genomes/hg38.fa -p data/peaks/SRR7585374_filtered_peaks.bed -c  data/genomes/hg38.chrom.sizes -fl data/splits/fold_0.json -br data/genomes/blacklist.bed -o data/output


### TRAINING ###

# Download the pre-trained bias model
mkdir bias_model
wget https://storage.googleapis.com/chrombpnet_data/input_files/bias_models/ATAC/ENCSR868FGK_bias_fold_0.h5 -O bias_model/ENCSR868FGK_bias_fold_0.h5


# Train a bias-factorized ChromBPNet model
mkdir chrombpnet_model
chrombpnet pipeline \
        -ibam data/alignments/SRR7585374.sorted.bam \
        -d "ATAC" \
        -g data/genomes/hg38.fa \
        -c data/genomes/hg38.chrom.sizes \
        -p data/peaks/SRR7585374_filtered_peaks.bed \
        -n data/output_negatives.bed \
        -fl data/splits/fold_0.json \
        -b bias_model/ENCSR868FGK_bias_fold_0.h5 \
        -es 5 \
        -o chrombpnet_model/


# # inference!
# create a fasta file with the sequences you want to predict

# then run the following commands
mkdir -p inference_results


## IMPORTANT: ENSURE BED FILE IS SORTED AND HAS 10 COLUMNS

# # you can do that manually and then run the following command:
# chrombpnet contribs_bw \
#   -m chrombpnet_model/models/chrombpnet_nobias.h5 \
#   -r seqs_predict.bed \
#   -g data/genomes/hg38.fa \
#   -c data/genomes/hg38.chrom.sizes \
#   -op inference_results/ \
#   -pc counts

# # and to view the results, run:
# python clear_plots.py
# python temp.py

# # or you can just run the following command:
# bash shap_pipeline.sh
# # wich will sort the bed file, run the chrombpnet contribs_bw command and then run the python scripts to view the results


# training was done using the hg38 genome, but the model can be used to predict on other genomes as well.
# The sequences I want to analyze were extracted from the hg19 genome, so I need to liftOver them to hg38.



