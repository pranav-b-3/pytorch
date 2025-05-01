import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Load results
# df = pd.read_csv("alpha_beta_accuracy_results.csv")
df = pd.read_csv("alpha_beta_accuracy_results_ENERGY.csv")

# Pivot accuracy
heatmap_data = df.pivot(index="beta", columns="alpha", values="accuracy")

# Ensure (0,0) exists in the data
if 0.0 in heatmap_data.index and 0.0 in heatmap_data.columns:
    heatmap_data.loc[0.0, 0.0] = np.nan  # Black out the (0,0) cell

# Create mask: True where you want to hide cells
mask = heatmap_data.isna()
cmap = matplotlib.cm.get_cmap("RdYlGn").copy()
cmap.set_bad(color='black')
# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    heatmap_data,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap=cmap,    
    center=0.6,
    vmin=0.0,
    vmax=1.0,
    linewidths=0.5,
    cbar_kws={"label": "Accuracy"},
    square=True
)

plt.title("Accuracy Heatmap across Alpha/Beta")
plt.xlabel("Alpha")
plt.ylabel("Beta")
plt.tight_layout()
plt.savefig("alpha_beta_accuracy_heatmap.png")
plt.show()