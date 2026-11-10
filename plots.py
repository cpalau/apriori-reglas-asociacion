"""Generate the two SVG figures for the Apriori post from the real Groceries data."""
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

OUT = "figuras"
INK = "#111111"
GRAY = "#8a8a8a"
RED = "#e03131"
LIGHT = "#d9d9d9"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

tx = [[" ".join(i.split()) for i in l.strip().split(",")] for l in open("datos/groceries.csv") if l.strip()]
te = TransactionEncoder()
df = pd.DataFrame(te.fit(tx).transform(tx), columns=te.columns_)

# ---- Figure 1: how many itemsets and rules survive each threshold ----
supports = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
itemsets, rules = [], []
for s in supports:
    fi = apriori(df, min_support=s, use_colnames=True)
    r = association_rules(fi, metric="confidence", min_threshold=0.2)
    itemsets.append(len(fi))
    rules.append(len(r))
    print(s, len(fi), len(r))

fig, ax = plt.subplots(figsize=(9, 5.25), dpi=100)
ax.plot(supports, itemsets, "o-", color=INK, lw=2, ms=6, label="conjuntos frecuentes")
ax.plot(supports, rules, "s-", color=RED, lw=2, ms=6, label="reglas (confianza ≥ 20 %)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("umbral de soporte mínimo")
ax.set_ylabel("número de resultados")
ax.set_xticks(supports)
ax.set_xticklabels([f"{s*100:g} %" for s in supports])
ax.set_yticks([10, 100, 1000, 10000])
ax.set_yticklabels(["10", "100", "1.000", "10.000"])
ax.grid(True, which="major", color=LIGHT, lw=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper right")
ax.set_title("Bajar el umbral multiplica los resultados", fontsize=13, color=INK, loc="left", pad=12)
fig.tight_layout()
os.makedirs(OUT, exist_ok=True)
fig.savefig(f"{OUT}/umbral-soporte.svg", format="svg", facecolor="white")
plt.close(fig)

# ---- Figure 2: support vs confidence, coloured by lift ----
fi = apriori(df, min_support=0.01, use_colnames=True)
r = association_rules(fi, metric="confidence", min_threshold=0.2)
r["ant"] = r.antecedents.apply(lambda x: ", ".join(sorted(x)))
r["con"] = r.consequents.apply(lambda x: ", ".join(sorted(x)))

cmap = LinearSegmentedColormap.from_list("lift", ["#e9e9e9", "#f0a3a3", RED])
fig, ax = plt.subplots(figsize=(9, 5.25), dpi=100)
sc = ax.scatter(r.support * 100, r.confidence * 100, c=r.lift, cmap=cmap,
                s=28, edgecolors="none", vmin=1.0, vmax=r.lift.max())
ax.set_xlabel("soporte (%)")
ax.set_ylabel("confianza (%)")
ax.grid(True, color=LIGHT, lw=0.8)
ax.set_axisbelow(True)
cb = fig.colorbar(sc, ax=ax, pad=0.02)
cb.set_label("lift")
cb.outline.set_visible(False)

ax.set_xlim(0, 8.2)
ax.set_ylim(18, 62)
ax.set_title("Cada punto es una regla: 234 en total", fontsize=13, color=INK, loc="left", pad=12)
fig.tight_layout()
fig.savefig(f"{OUT}/soporte-confianza-lift.svg", format="svg", facecolor="white")
plt.close(fig)
print("ok")
