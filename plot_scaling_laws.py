import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pylab import f

FONTSIZE = 12
params = {
    "backend": "ps",
    #   'text.latex.preamble': ['\\usepackage{gensymb}'],
    "axes.labelsize": FONTSIZE,  # fontsize for x and y labels (was 10)
    "axes.titlesize": FONTSIZE,
    #   'lines.linewidth': 0.5,
    #   'axes.linewidth': 0.0,
    # "text.fontsize": FONTSIZE,  # was 10
    "legend.fontsize": FONTSIZE,  # was 10
    "xtick.labelsize": FONTSIZE,
    "ytick.labelsize": FONTSIZE,
    #   'lines.markersize': 2,
    "text.usetex": True,
      'figure.figsize': [8,3],
    "font.family": "serif",
}

matplotlib.rcParams.update(params)
size = np.array([70, 160, 410, 1000, 1400, 2800, 7000])
precision = np.array([78.2, 75.5, 70.8, 72.2, 72.5, 73.9, 76.4])
recall = np.array([8.1, 10.9, 20.0, 26.4, 28.3, 34.9, 44.6])
f1 = np.array([14.6, 19.0, 31.2, 38.6, 40.7, 47.4, 56.3])

colors = ["red", "blue", "darkviolet"]

plt.plot(size, precision, label="Precision", marker="^", color=colors[0])
plt.plot(size, recall, label="Recall", marker="^", color=colors[1])
plt.plot(size, f1, label="F1-score", marker="^", color=colors[2])


plt.xscale("log")
plt.xticks(size, labels=[f"{s}m" if s < 1000 else f"{s/1000}B" for s in size])

plt.legend(frameon=False)
plt.xlabel("Model Size (log scale)")
plt.ylabel("Avg. Performance (\%)")
plt.tight_layout()
plt.show()
plt.savefig("scaling_laws.pdf")
