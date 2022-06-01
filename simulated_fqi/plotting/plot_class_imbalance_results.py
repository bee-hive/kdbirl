import seaborn as sns
import matplotlib.pyplot as plt
import json
import scipy
import numpy as np
import pandas as pd
import matplotlib

font = {"size": 20}
matplotlib.rc("font", **font)
# matplotlib.rcParams["text.usetex"] = True


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, h


##### Class imbalance

with open("../class_imbalance_cfqi.json") as f:
    results = json.load(f)

cfqi_bg_success = []
fqi_joint_bg_success = []
fg_only_bg_success = []

cfqi_fg_success = []
fqi_joint_fg_success = []
fg_only_fg_success = []

cfqi_bg_errs = []
fqi_joint_bg_errs = []
fg_only_bg_errs = []

cfqi_fg_errs = []
fqi_joint_fg_errs = []
fg_only_fg_errs = []

fracs = results.keys()

n_reps = 10
n_reps_per_condition = n_reps // 2

bg_performance_array = np.zeros((n_reps_per_condition, len(fracs)))
fg_performance_array = np.zeros((n_reps_per_condition, len(fracs)))

method_list = []
condition_list = []
fraction_list = []
steps_list = []

for frac_ii, i in enumerate(fracs):
    cfqi_bg_perf = []
    fqi_joint_bg_perf = []
    fg_only_bg_perf = []

    cfqi_fg_perf = []
    fqi_joint_fg_perf = []
    fg_only_fg_perf = []
    for key in results[str(i)]["cfqi"]:
        cfqi_bg_perf.extend(results[str(i)]["cfqi"][key][:n_reps_per_condition])
        cfqi_fg_perf.extend(results[str(i)]["cfqi"][key][n_reps_per_condition:])

        method_list.extend(["CFQI" for _ in range(n_reps)])
        condition_list.extend(["Background" for _ in range(n_reps_per_condition)])
        condition_list.extend(["Foreground" for _ in range(n_reps_per_condition)])
        fraction_list.extend([round(float(i), 1) for _ in range(n_reps)])
        steps_list.extend(results[str(i)]["cfqi"][key])

    for key in results[str(i)]["fqi_joint"]:
        fqi_joint_bg_perf.extend(
            results[str(i)]["fqi_joint"][key][:n_reps_per_condition]
        )
        fqi_joint_fg_perf.extend(
            results[str(i)]["fqi_joint"][key][n_reps_per_condition:]
        )

        method_list.extend(["FQI (joint)" for _ in range(n_reps)])
        condition_list.extend(["Background" for _ in range(n_reps_per_condition)])
        condition_list.extend(["Foreground" for _ in range(n_reps_per_condition)])
        fraction_list.extend([round(float(i), 1) for _ in range(n_reps)])
        steps_list.extend(results[str(i)]["fqi_joint"][key])

    for key in results[str(i)]["fg_only"]:
        fg_only_bg_perf.extend(results[str(i)]["fg_only"][key][:n_reps_per_condition])
        fg_only_fg_perf.extend(results[str(i)]["fg_only"][key][n_reps_per_condition:])

        method_list.extend(["FQI (FG only)" for _ in range(n_reps)])
        condition_list.extend(["Background" for _ in range(n_reps_per_condition)])
        condition_list.extend(["Foreground" for _ in range(n_reps_per_condition)])
        fraction_list.extend([round(float(i), 1) for _ in range(n_reps)])
        steps_list.extend(results[str(i)]["fg_only"][key])

    cfqi_bg_success.append(np.mean(cfqi_bg_perf))
    fqi_joint_bg_success.append(np.mean(fqi_joint_bg_perf))
    fg_only_bg_success.append(np.mean(fg_only_bg_perf))

    cfqi_fg_success.append(np.mean(cfqi_fg_perf))
    fqi_joint_fg_success.append(np.mean(fqi_joint_fg_perf))
    fg_only_fg_success.append(np.mean(fg_only_fg_perf))

    m, h = mean_confidence_interval(cfqi_bg_perf)
    cfqi_bg_errs.append(h)
    m, h = mean_confidence_interval(fqi_joint_bg_perf)
    fqi_joint_bg_errs.append(h)
    m, h = mean_confidence_interval(fg_only_bg_perf)
    fg_only_bg_errs.append(h)

    m, h = mean_confidence_interval(cfqi_fg_perf)
    cfqi_fg_errs.append(h)
    m, h = mean_confidence_interval(fqi_joint_fg_perf)
    fqi_joint_fg_errs.append(h)
    m, h = mean_confidence_interval(fg_only_fg_perf)
    fg_only_fg_errs.append(h)

results_df = pd.DataFrame(
    {
        "Method": method_list,
        "Dataset": condition_list,
        "fraction_fg": fraction_list,
        "steps": steps_list,
    }
)

plt.figure(figsize=(12, 6))
sns.lineplot(
    data=results_df,
    x="fraction_fg",
    y="steps",
    hue="Method",
    style="Dataset",
    err_style="bars",
    ci=95,
)
plt.xlabel("Fraction foreground samples")
plt.ylabel("Steps survived")
plt.title("Performance with imbalanced datasets")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)  # , fontsize=15)
plt.tight_layout()
plt.savefig("../plots/class_imbalance.pdf")
plt.show()
import ipdb

ipdb.set_trace()

# x = [k for k in range(len(fracs))]
# # import ipdb; ipdb.set_trace()
# plt.figure(figsize=(10, 7))
# # sns.scatterplot(x, cfqi_bg_success)
# plt.errorbar(x, cfqi_bg_success ,yerr=cfqi_bg_errs, linestyle="-", c="green", label='CFQI, BG')
# # sns.scatterplot(x, fqi_joint_bg_success)
# plt.errorbar(x, fqi_joint_bg_success ,yerr=fqi_joint_bg_errs, linestyle="-", c="red", label='FQI (joint), BG')
# # sns.scatterplot(x, fg_only_bg_success)
# plt.errorbar(x, fg_only_bg_success ,yerr=fg_only_bg_errs, linestyle="-", c="blue", label='FQI (FG only), BG')

# # sns.scatterplot(x, cfqi_fg_success)
# plt.errorbar(x, cfqi_fg_success ,yerr=cfqi_fg_errs, linestyle="--", c="green", label='CFQI, FG')
# # sns.scatterplot(x, fqi_joint_fg_success)
# plt.errorbar(x, fqi_joint_fg_success ,yerr=fqi_joint_fg_errs, linestyle="--", c="red", label='FQI (joint), FG')
# # sns.scatterplot(x, fg_only_fg_success)
# plt.errorbar(x, fg_only_fg_success ,yerr=fg_only_fg_errs, linestyle="--", c="blue", label='FQI (FG only), FG')

# plt.title("Class imbalance")
# plt.legend()
# plt.xlabel("Fraction foreground samples")
# plt.ylabel("Steps Survived")
# plt.tight_layout()
# plt.xticks(np.arange(len(fracs)), labels=[round(float(x), 1) for x in fracs])
# plt.savefig("./plots/class_imbalance.png")
# plt.show()


# import ipdb; ipdb.set_trace()

# results_df = pd.DataFrame(results)
# results_df = pd.melt(results_df)

# plt.figure(figsize=(7, 5))
# sns.boxplot(data=results_df, x="variable", y="value", order=["fg_only", "fqi_joint", "cfqi"])
# plt.xticks(np.arange(3), ["FQI (FG only)", "FQI (Joint)", "CFQI"])
# plt.xlabel("")
# plt.ylabel("Number of successful steps")
# plt.tight_layout()
# # plt.savefig("")
# plt.title("Class imbalance test")
# plt.tight_layout()
# plt.savefig("./plots/class_imbalance.png")
# plt.show()
# import ipdb; ipdb.set_trace()
