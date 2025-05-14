import numpy as np

from models import SignalPredictor1D
import torch


model = SignalPredictor1D(num_classes=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(f"saved_models/162929_0403_E25E102_mm_v_all.pt", map_location=device))
# model.load_state_dict(torch.load(f"saved_models/131953_0331_E25E102_mm_v_all.pt", map_location=device))

model.to(device)
model.eval()




sequences_unpadded = [  "ACCGCGGCTCCAGGGGCGAGAGGAGCCAGCCCTGTCTCTCTCTCCTGGCCCTGGGGGCGCCCGCCGCGAGGGCGGGGCCGGGATTGTGCTGATTCTGGCCTCCCTGGGCCGGAGGCTCTAGTGGAACTTAAGGCTCCTCCCTGATGGCACCGAGGCGAGGAACTGCCAGCTGTCTGTCTCCTTCCTGCCTTGACCCAGAGC",
                        "ACCGCGGCTCCAGGGGCGAGAGGATCCAGCCCTGTCTCTCTCTCCTGGCCCTGGGGGCGCCCGCCGCGAGGGCGGGGCCGGGATTGTGCTGATTCTGGCCTCCCTGGGCCGGAGGCTCTAGTGGAACTTAAGGCTCCTCCCTGATGGCACCGAGGCGAGGAACTGCCAGCTGTCTGTCTCCTTCCTGCCTTGACCCAGAGC",
                        "ACCGCGGCTCCAGGGGCGAGAGGAGCCAGCCCTGTCTCTCTCTCCTGGCCCTGGGGGCGCCCGCCGCGAGGGCGGTGCCGGGATTGTGCTGATTCTGGCCTCCCTGGGCCGGAGGCTCTAGTGGAACTTAAGGCTCCTCCCTGATGGCACCGAGGCGAGGAACTGCCAGCTGTCTGTCTCCTTCCTGCCTTGACCCAGAGC",
                        "ACCGCGGCTCCAGGGGCGAGAGGAGCCAGCCCTGTCTCTCTCTCCTGGCCCTGGGGGCGCCCGCCGCGAGGGCGGGGCCGGGATTGTGCTGATTCTGGCCTCCCTGGGCCGGAGGCTCTAGTGGAACTTAAGGCTCCTCCCTGATTGCACCGAGGCGAGGAACTGCCAGCTGTCTGTCTCCTTCCTGCCTTGACCCAGAGC"
                       ]


sequences = []
for seq in sequences_unpadded:
    sequences.append("tatGGATCCTCGGTTCACGCAATG" + seq + "AGTTGATCCGGTCCTGGATCCataa")
    sequences.append("ttatGGATCCTCGGTTCACGCAATG" + seq + "AGTTGATCCGGTCCTGGATCCata")

def oh_encode(sequence):
    sequence = sequence.upper()
    # One-hot encoding
    one_hot = np.zeros((len(sequence), 4))
    for i, char in enumerate(sequence):
        if char == 'A':
            one_hot[i, 0] = 1
        elif char == 'C':
            one_hot[i, 1] = 1
        elif char == 'G':
            one_hot[i, 2] = 1
        elif char == 'T':
            one_hot[i, 3] = 1
    return one_hot.T

sequences = [oh_encode(seq) for seq in sequences]
sequences = torch.tensor(sequences, dtype=torch.float32).to(device)

option1= []
option2= []
with torch.no_grad():
    outputs = model(sequences)
    # outputs = torch.sigmoid(outputs)
    # outputs = (outputs > 0.5).float()
    for i in range(len(outputs)//2):
        print(outputs[2*i], outputs[2*i+1])
        option1.append(-outputs[2*i].cpu().numpy()[0])
        option2.append(-outputs[2*i+1].cpu().numpy()[0])

option1 = np.array(option1)
option2 = np.array(option2)

import matplotlib.pyplot as plt

print(option1.shape)
print(option2.shape)

plt.figure(figsize=(8, 6))
plt.bar(np.arange(len(option1)), option1, color='blue')
plt.xticks(np.arange(len(option1)), ["WT Seq", "Neutral Mutation", "\nMutation with Decrease", "Mutation with Increase"])
plt.ylabel('Predicted Enhancer Activity (arbitrary units)')
plt.title('Predicted Enhancer Activity for Mutated Sequences')
plt.savefig('option1.png')
plt.tight_layout()
plt.clf()


plt.bar(np.arange(len(option2)), option2, color='red')
plt.savefig('option2.png')
plt.clf()