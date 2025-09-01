import numpy as np
import matplotlib.pyplot as plt
import os

from tangermeme.plot import plot_logo

dataloc = "explainability_final/"
# interest_files = ["E25E102_191164107_saturation_mutation.npy", "E25E10R3_191165846_saturation_mutation.npy", "E22P1A3_191162606_saturation_mutation.npy"]
# enhancer_name = "ANK"
# interest_files = ["E22P1D10_205105342_saturation_mutation.npy", "E22P3B3_191161116_saturation_mutation.npy", "E22P3B3R2_191212610_saturation_mutation.npy"]
# enhancer_name = "Chr20"
interest_files = ["E25E102_191164107_deeplift_shap.npy", "E25E10R3_191165846_deeplift_shap.npy", "E22P1A3_191162606_deeplift_shap.npy"]
enhancer_name = "ANK_DeepLift"
# interest_files = ["E22P1D10_205105342_deeplift_shap.npy", "E22P3B3_191161116_deeplift_shap.npy", "E22P3B3R2_191212610_deeplift_shap.npy"]
# enhancer_name = "Chr20_DeepLift"

data = []
for file in interest_files:
    data.append(np.load(os.path.join(dataloc, file)))

fig, ax = plt.subplots(4, 1, figsize=(25, 12))
ax = ax.flatten()
for i, d in enumerate(data):
    # ax[i].imshow(d, aspect='auto', cmap='viridis')
    plot_logo(d, ax=ax[i])
    enh_name = interest_files[i].split('_')[0]
    ax[i].set_title(f'{enh_name} Saturation Mutagenesis Attribution Map', fontsize=20)
    ax[i].set_xlabel('Position', fontsize=12)
    ax[i].set_ylabel('Mutation', fontsize=14)
    ax[i].set_xticks(np.arange(0, d.shape[1], step=25))
    ax[i].set_xticklabels(np.arange(0, d.shape[1], step=25) - 25)
    print(f"Completed plotting for {i}")


# normalize the data
data = [d / np.max(np.abs(d)) for d in data]
data = np.mean(data, axis=0)
# Plot the average of the mutations
plot_logo(data, ax=ax[3])
ax[3].set_title('Average Attribution Map', fontsize=20)
ax[3].set_xlabel('Position', fontsize=12)
ax[3].set_ylabel('Mutation', fontsize=14)
ax[3].set_xticks(np.arange(0, data.shape[1], step=25))
ax[3].set_xticklabels(np.arange(0, data.shape[1], step=25) - 25)

plt.tight_layout()
plt.savefig(f"sat_mut_average//sat_mut_average_{enhancer_name}.png")