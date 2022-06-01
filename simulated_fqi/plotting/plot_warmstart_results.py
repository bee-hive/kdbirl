import seaborn as sns
import matplotlib.pyplot as plt
import json
import scipy
import numpy as np
import pandas as pd
import matplotlib


# with open('warm_start_force0.json') as f:
#   results = json.load(f)

# fqi_results = []
# cfqi_results = []
# ws_results = []
# for alg in ['cfqi', 'fqi', 'warm_start']:
#     for key in results[alg]:
#         if alg == 'fqi':
#             fqi_results.extend(results[alg][key])
#         elif alg == 'cfqi':
#             cfqi_results.extend(results[alg][key])
#         else:
#             ws_results.extend(results[alg][key])
# # sns.distplot(fqi_results, label='FQI')
# # sns.distplot(cfqi_results, label='CFQI')
# # sns.distplot(ws_results, label='Warm Start')
# # plt.legend()
# # plt.xlabel("Steps survived")
# # plt.title("Force left = 0")
# # plt.show()

# results_df = pd.DataFrame({'fqi': fqi_results, 'warm_start': ws_results, 'cfqi': cfqi_results})
# results_df = pd.melt(results_df)

# sns.boxplot(data=results_df, x="variable", y="value")
# plt.xticks(np.arange(3), ["FQI", "Warm start", "CFQI"])
# plt.xlabel("")
# plt.ylabel("Steps survived")
# plt.title("Force left=0")
# plt.show()


with open("../warm_start_forcerange.json") as f:
    results = json.load(f)
# import ipdb; ipdb.set_trace()


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, h


c_success = []
f_success = []
w_success = []
c_errs = []
f_errs = []
w_errs = []
for i in range(11):
    i = str(i)
    cfqi_perf = []
    fqi_perf = []
    ws_perf = []
    for key in results[i]["fqi"]:
        fqi_perf.extend(results[i]["fqi"][key])
    for key in results[i]["cfqi"]:
        cfqi_perf.extend(results[i]["cfqi"][key])
    for key in results[i]["warm_start"]:
        ws_perf.extend(results[i]["warm_start"][key])
    c_success.append(np.mean(cfqi_perf))
    f_success.append(np.mean(fqi_perf))
    w_success.append(np.mean(ws_perf))
    m, h = mean_confidence_interval(cfqi_perf)
    c_errs.append(h)
    m, h = mean_confidence_interval(fqi_perf)
    f_errs.append(h)
    m, h = mean_confidence_interval(ws_perf)
    w_errs.append(h)
x = [k for k in range(11)]
# sns.scatterplot(x, c_success, label='CFQI')
plt.errorbar(x, c_success, yerr=c_errs, label="CFQI")  # , linestyle="None")
# sns.scatterplot(x, f_success, label='FQI')
plt.errorbar(x, f_success, yerr=f_errs, label="FQI")  # , linestyle="None")
# sns.scatterplot(x, w_success, label='Warm Start')
plt.errorbar(x, w_success, yerr=w_errs, label="Warm Start")  # , linestyle="None")
plt.legend()
plt.title("Performance of CFQI, FQI, Warm Start when force on cart is modified")
plt.xlabel("Force Left")
plt.ylabel("Steps Survived")
plt.show()
import ipdb

ipdb.set_trace()
