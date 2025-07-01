import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pickle

from models import SignalPredictor1D
from tqdm import tqdm


graph_points = []
# load points from the file
with open("temp_figs/.coverage_vs_f1_score.txt", "r") as f:
    lines = f.readlines()[1:]  # skip header
    for line in lines:
        coverage, f1, train_size = line.strip().split("\t")
        graph_points.append((coverage, float(f1), int(train_size)))


# plot coverage vs f1 score as a bar graph on the positive and one color, and the length of the training set as a bar graph flipped, and in the log scale
fig, ax1 = plt.subplots(4, 1, figsize=(7, 10), sharex=True, dpi=300)
ax1[0].bar([i for i in range(len(graph_points))], [x[1] for x in graph_points], color='b', alpha=0.5)
ax1[0].set_ylabel("F1 Score")
ax1[0].set_ylim(0, 1)
ax1[0].set_title("Coverage vs F1 Score")
ax1[0].set_xticks(np.arange(len(graph_points)), [x[0] for x in graph_points])
ax1[1].bar([i for i in range(len(graph_points))], [x[2] for x in graph_points], color='r', alpha=0.5)
ax1[1].set_ylabel("Training Set Size")
ax1[1].set_ylim(1, max([x[2] * 10 for x in graph_points]))
ax1[1].set_yscale('log')
ax1[1].set_xlabel("Coverage")
ax1[1].set_title("Coverage vs Training Set Size (Log10)")
ax1[1].set_xticks(np.arange(len(graph_points)), [x[0] for x in graph_points])
ax1[1].set_xticklabels([x[0] for x in graph_points])
ax1[2].set_title("Coverage vs Training Set Size")
ax1[2].bar([i for i in range(len(graph_points))], [x[2] for x in graph_points], color='r', alpha=0.5)
ax1[2].set_ylim(max([x[2] * 0.9 for x in graph_points]), max([x[2] * 1.01 for x in graph_points]))
ax1[2].spines['bottom'].set_visible(False)
ax1[3].spines['top'].set_visible(False)
ax1[3].bar([i for i in range(len(graph_points))], [x[2] for x in graph_points], color='r', alpha=0.5)
ax1[3].set_ylabel("Training Set Size")
ax1[3].set_ylim(1, max([x[2] * 1.1 for x in graph_points[1:]]))
ax1[3].set_xlabel("Coverage")
ax1[3].set_xticks(np.arange(len(graph_points)), [x[0] for x in graph_points])
ax1[3].set_xticklabels([x[0] for x in graph_points])

fig.text(0.22, 0.265, "(...)", ha='center', va='center', fontsize=14)

ax1[2].yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=False))
ax1[3].yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=False))
ax1[2].ticklabel_format(style='plain', axis='y')
ax1[3].ticklabel_format(style='plain', axis='y')

plt.tight_layout()
plt.savefig("temp_figs/.coverage_vs_f1_score.png")
plt.show()


print("Plot saved!!!")