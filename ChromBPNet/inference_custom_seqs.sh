export LD_LIBRARY_PATH=/home/franciscom/miniconda3/envs/chrombpnet/lib:$LD_LIBRARY_PATH


python generate_padded_genome.py --input_fa custom_seqs.fa

# generate chrom sizes
samtools faidx custom_seqs_padded.fa
cut -f1,2 custom_seqs_padded.fa.fai > custom_seqs_padded.chrom.sizes

chrombpnet pred_bw \
  -cmb chrombpnet_model/models/chrombpnet_nobias.h5 \
  -r custom_seqs.bed \
  -g custom_seqs_padded.fa \
  -c custom_seqs_padded.chrom.sizes \
  -op numerical_inference_results_custom/results \
  -bs 256

python display_inference.py --save_output --outfile ../chrombpnet_experiments/predictions/E22P1A3_25_preds.txt 