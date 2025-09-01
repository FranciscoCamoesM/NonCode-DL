import numpy as np
import matplotlib.pyplot as plt
from tangermeme.plot import plot_logo


seq1 = "ACGGT"
seq2 = "ATGGT"

# one-hot encoding for the sequences
def one_hot_encode(seq):
    encoding = np.zeros((len(seq), 4), dtype=int)
    for i, base in enumerate(seq):
        if base == 'A':
            encoding[i, 0] = 1
        elif base == 'C':
            encoding[i, 1] = 1
        elif base == 'G':
            encoding[i, 2] = 1
        elif base == 'T':
            encoding[i, 3] = 1
    return encoding

seq1_encoded = one_hot_encode(seq1)
seq2_encoded = one_hot_encode(seq2)



# Calculate the difference between the two sequences
consensus = seq1_encoded + seq2_encoded
consensus = consensus.T**2 / 2

print(consensus)


# Plot the consensus sequence
fig, ax = plt.subplots(figsize=(10, 4))
plot_logo(consensus, ax=ax)
# disable vertical ticks
ax.set_yticks([])
plt.savefig("consensus_sequence.png")
plt.close()