import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pyparsing import line

FONTSIZE = 15
params = {
    "backend": "ps",
    #   'text.latex.preamble': ['\\usepackage{gensymb}'],
    "axes.labelsize": FONTSIZE,  # fontsize for x and y labels (was 10)
    "axes.titlesize": FONTSIZE,
    #   'lines.linewidth': 0.5,
    #   'axes.linewidth': 0.0,
    # "text.fontsize": FONTSIZE,  # was 10
    "legend.fontsize": FONTSIZE-1,  # was 10
    "xtick.labelsize": FONTSIZE,
    "ytick.labelsize": FONTSIZE,
    #   'lines.markersize': 2,
    "text.usetex": True,
    # "figure.figsize": [8, 5],
    "font.family": "serif",
}

matplotlib.rcParams.update(params)
Pi = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])
no_no_tpr = np.array([0, 0.09, 0.25, 0.39, 0.55, 0.67])
yes_no_tpr = np.array([0, 0.12, 0.24, 0.35, 0.49, 0.60])
no_yes_tpr = np.array([0, None, 0.26, None, None, 0.64])
yes_yes_tpr = np.array([0, None, 0.24, None, None, 0.63])

Pi = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])
no_no_fpr = np.array([0, 0.02, 0.05, 0.9, 0.15, 0.25])
yes_no_fpr = np.array([0, 0.02, 0.04, 0.06, 0.1, 0.17])
no_yes_fpr = np.array([0, None, 0.05, None, None, 0.20])
yes_yes_fpr = np.array([0, None, 0.02, None, None, 0.08])


import matplotlib.pyplot as plt
import numpy as np

# Data
Pi = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])
no_no_tpr = np.array([0, 0.09, 0.25, 0.39, 0.55, 0.67])
yes_no_tpr = np.array([0, 0.12, 0.24, 0.35, 0.49, 0.60])
no_yes_tpr = np.array([0, np.nan, 0.26, np.nan, np.nan, 0.64])
yes_yes_tpr = np.array([0, np.nan, 0.24, np.nan, np.nan, 0.63])

no_no_fpr = np.array([0, 0.02, 0.05, 0.09, 0.15, 0.25])
yes_no_fpr = np.array([0, 0.02, 0.04, 0.06, 0.1, 0.17])
no_yes_fpr = np.array([0, np.nan, 0.05, np.nan, np.nan, 0.20])
yes_yes_fpr = np.array([0, np.nan, 0.02, np.nan, np.nan, 0.08])


# trim 0 entry from all arrays
Pi = Pi[1:]
no_no_tpr = no_no_tpr[1:]
yes_no_tpr = yes_no_tpr[1:]
no_yes_tpr = no_yes_tpr[1:]
yes_yes_tpr = yes_yes_tpr[1:]

no_no_fpr = no_no_fpr[1:]
yes_no_fpr = yes_no_fpr[1:]
no_yes_fpr = no_yes_fpr[1:]
yes_yes_fpr = yes_yes_fpr[1:]


# filter out nans
non_nan_Pi = Pi[~np.isnan(no_yes_tpr)]

no_yes_tpr = no_yes_tpr[~np.isnan(no_yes_tpr)]
yes_yes_tpr = yes_yes_tpr[~np.isnan(yes_yes_tpr)]
no_yes_fpr = no_yes_fpr[~np.isnan(no_yes_fpr)]
yes_yes_fpr = yes_yes_fpr[~np.isnan(yes_yes_fpr)]


fig = plt.figure(figsize=(6, 5))
# TPR subplot
(line1,) = plt.plot(no_no_fpr, no_no_tpr, marker="x")
(line2,) = plt.plot(yes_no_fpr, yes_no_tpr, marker="x")
(line3,) = plt.plot(no_yes_fpr, no_yes_tpr, marker="^", linestyle="--")
(line4,) = plt.plot(yes_yes_fpr, yes_yes_tpr, marker="^", linestyle="--")


# Annotate each value with its corresponding Pi value
for i, txt in enumerate(Pi):
    plt.annotate(f'{Pi[i]:.1f}', (no_no_fpr[i], no_no_tpr[i]), textcoords="offset points", xytext=(0,4), ha='center')
    plt.annotate(f'{Pi[i]:.1f}', (yes_no_fpr[i], yes_no_tpr[i]), textcoords="offset points", xytext=(0,4), ha='center')

for i, txt in enumerate(non_nan_Pi):
    plt.annotate(f'{non_nan_Pi[i]:.1f}', (no_yes_fpr[i], no_yes_tpr[i]), textcoords="offset points", xytext=(0,4), ha='center')
    plt.annotate(f'{non_nan_Pi[i]:.1f}', (yes_yes_fpr[i], yes_yes_tpr[i]), textcoords="offset points", xytext=(0,4), ha='center')

plt.ylabel("\\texttt{IDK} recall")
plt.xlabel("\\texttt{IDK} error rate")

# fig.set_ylabel("\\texttt{IDK} recall")
# fig.set_xlabel("\\texttt{IDK} error rate")
# ax1.set_yticks(np.arange(0.1, 0.8, 0.1))
line_labels = [
    "No $\\mathcal{L}_\\texttt{FP-reg}$, fixed $\\lambda$",
    "With $\\mathcal{L}_\\texttt{FP-reg}$, fixed $\\lambda$",
    "No $\\mathcal{L}_\\texttt{FP-reg}$, adaptive $\\lambda$",
    "With $\\mathcal{L}_\\texttt{FP-reg}$, adaptive $\\lambda$",
]

# Create a single legend
lines_labels = [
    (line1, "No $\\mathcal{L}_\\texttt{FP-reg}$, fixed $\\lambda$"),
    (line2, "With $\\mathcal{L}_\\texttt{FP-reg}$, fixed $\\lambda$"),
    (line3, "No $\\mathcal{L}_\\texttt{FP-reg}$, adaptive $\\lambda$"),
    (line4, "With $\\mathcal{L}_\\texttt{FP-reg}$, adaptive $\\lambda$"),
]

fig.legend(
    [line1, line2, line3, line4],
    [label for (_, label) in lines_labels],
    loc="upper center",
    bbox_to_anchor=(0.51, 1.0),
    ncol=2,
    frameon=False,
)
plt.tight_layout(rect=[0, 0, 0.97, 0.87])

# plt.figlegend(
#     [line1, line2, line3, line4, line5, line6, line7, line8],
#     line_labels,
#     frameon=False,
#     loc="center left",
#     bbox_to_anchor=(1.02, 0),
#     ncol=1,
#     fancybox=True,
#     shadow=True,
# )
# plt.tight_layout(rect=[0, 0, 0.65, 1])
# try:
#     os.remove("tpr_.pdf")
# except FileNotFoundError:
#     pass
plt.savefig("tpr_vs_fpr.pdf")


# plt.legend(frameon=False)
# plt.xlabel("Value of $\Pi$")
# plt.ylabel("")
# plt.tight_layout()
# plt.show()
# plt.savefig("scaling_laws.pdf")
