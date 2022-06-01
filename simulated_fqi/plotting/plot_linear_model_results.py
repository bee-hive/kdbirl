import seaborn as sns
import matplotlib.pyplot as plt
import json
import scipy
import numpy as np
import pandas as pd
import matplotlib

font = {"size": 15}
matplotlib.rc("font", **font)
# matplotlib.rcParams["text.usetex"] = True

##### Class imbalance

with open("../linear_model_comparison.json") as f:
    results = json.load(f)

results_df = pd.DataFrame(results)
results_df = pd.melt(results_df)

plt.figure(figsize=(7, 5))
sns.boxplot(data=results_df, x="variable", y="value", order=["linear_model", "nfqi"])
plt.xticks(np.arange(2), ["Linear model", "Neural network"])
plt.xlabel("")
plt.ylabel("Successful steps")
plt.title(r"Approximation functions $f$")
plt.tight_layout()
plt.savefig("../plots/linear_model_comparison.png")
plt.show()
import ipdb

ipdb.set_trace()
