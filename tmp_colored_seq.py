import matplotlib.pyplot as plt

import numpy as np



save_loc = "temp_figs/.colored_seq.png"

DNA_seq = "TGGCCATGGAGTCGCAGGTCAACTAGTGACATT"


oh_encode = np.zeros((len(DNA_seq), 4))

for i, char in enumerate(DNA_seq):
    if char == 'A':
        oh_encode[i, 0] = 1
    elif char == 'C':
        oh_encode[i, 1] = 1
    elif char == 'G':
        oh_encode[i, 2] = 1
    elif char == 'T':
        oh_encode[i, 3] = 1


colors = ['#CC0000', '#0000CC', "#DDB300", '#008000']

#convert to RGB
colors_rgb = [tuple(int(c[i:i+2], 16) for i in (1, 3, 5)) for c in colors]

print(colors_rgb)

oh_encode = oh_encode.reshape(oh_encode.shape[0], 4, 1)
colors_rgb = np.array(colors_rgb).reshape(1, 4, 3)

oh_colored = oh_encode * colors_rgb + (1-oh_encode) * np.array([255, 255, 255]).reshape(1, 1, 3)


plt.figure(figsize=(8, 2*len(DNA_seq)))
plt.imshow(oh_colored, aspect='auto', interpolation='none')
plt.grid(True, color='black', linewidth=4)
plt.xticks(np.arange(4) + 0.5,['', '', '', ''])
plt.yticks(np.arange(len(DNA_seq)) + 0.5, ['']*(len(DNA_seq)))
plt.savefig(save_loc, bbox_inches='tight', pad_inches=0.1)
plt.show()
plt.close()