library(BSgenome.Drerio.UCSC.danRer10)

library(rtracklayer)
library(Biostrings)

# Assuming the path to the BED file and it is correctly formatted as BED
bed_path <- "./original_enhancers.bed"


# Import the BED file
enhancers <- import.bed(bed_path)

# Using BSgenome to fetch sequences corresponding to enhancer regions
# Note: BSgenome.Drerio.UCSC.danRer10 should be loaded as it contains the genome information
fasta_sequences <- getSeq(BSgenome.Drerio.UCSC.danRer10, 
                          names=as.character(enhancers$name),
                          start=start(enhancers), 
                          end=end(enhancers), 
                          strand=strand(enhancers))

# Convert the DNAStringSet object to a FASTA format
writeXStringSet(fasta_sequences, file="enhancers.fasta")

# Check the output FASTA file
cat(readLines("enhancers.fasta"), sep="\n")