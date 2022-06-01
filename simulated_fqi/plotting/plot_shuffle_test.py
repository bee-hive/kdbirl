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


with open("../shuffle_test.json") as f:
    results = json.load(f)


plt.figure(figsize=(7, 5))
plt.title("Structureless test")
sns.distplot(results["fg"], label="Foreground")
sns.distplot(results["bg"], label="Background")
plt.legend()
plt.xlabel("Steps Survived")
plt.tight_layout()
plt.savefig("../plots/shuffle_test.png")
plt.show()

import ipdb

ipdb.set_trace()
