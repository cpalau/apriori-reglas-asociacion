"""Demo de Apriori sobre el dataset Groceries (9.835 tickets reales)."""
import itertools
import sys

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from traduccion import traducir

tx = [traducir([" ".join(i.split()) for i in l.strip().split(",")]) for l in open("datos/groceries.csv") if l.strip()]
te = TransactionEncoder()
df = pd.DataFrame(te.fit(tx).transform(tx), columns=te.columns_)

mode = sys.argv[1] if len(sys.argv) > 1 else "resumen"


def miles(n):
    return f"{n:,}".replace(",", ".")


if mode == "resumen":
    print(f"{miles(len(tx))} tickets · {df.shape[1]} productos distintos")
    print(f"{miles(df.size)} celdas y solo {str(round(100 * df.values.mean(), 1)).replace('.', ',')} % ocupadas")
elif mode == "frecuentes":
    fi = apriori(df, min_support=0.01, use_colnames=True)
    fi["k"] = fi["itemsets"].apply(len)
    L = {k: set(frozenset(x) for x in fi[fi.k == k].itemsets) for k in (1, 2, 3)}
    cand = {1: df.shape[1], 2: len(L[1]) * (len(L[1]) - 1) // 2}
    for k in (3, 4):
        c = set()
        base = sorted(L[k - 1], key=lambda s: sorted(s))
        for a, b in itertools.combinations(base, 2):
            u = a | b
            if len(u) == k and all(frozenset(x) in L[k - 1] for x in itertools.combinations(u, k - 1)):
                c.add(u)
        cand[k] = len(c)
        if k == 4:
            L[4] = set()
    print("nivel   candidatos   frecuentes")
    for k in (1, 2, 3, 4):
        print(f"  k={k}  {miles(cand[k]):>10}  {miles(len(L[k])):>11}")
    print(f"\ntotal examinado: {miles(sum(cand.values()))}")
elif mode == "umbral":
    print("soporte   conjuntos   reglas")
    for s in (0.05, 0.02, 0.01, 0.005):
        fi = apriori(df, min_support=s, use_colnames=True)
        r = association_rules(fi, metric="confidence", min_threshold=0.2)
        print(f"  {str(round(s*100,1)).replace('.', ','):>4} %  {miles(len(fi)):>10}  {miles(len(r)):>8}")
    print("\nbajar el umbral multiplica las reglas: 6 → 892")
elif mode == "reglas":
    fi = apriori(df, min_support=0.01, use_colnames=True)
    r = association_rules(fi, metric="confidence", min_threshold=0.2)
    r["ant"] = r.antecedents.apply(lambda x: ", ".join(sorted(x)))
    r["con"] = r.consequents.apply(lambda x: ", ".join(sorted(x)))
    print(f"{len(r)} reglas con soporte ≥ 1 % y confianza ≥ 20 %\n")
    for _, x in r.sort_values(["lift", "confidence"], ascending=False).head(8).iterrows():
        print(f"  {x.ant} → {x.con}")
        print(f"     sup {x.support*100:.2f} %   conf {x.confidence*100:.1f} %   lift {x.lift:.2f}".replace(".", ",").replace("sup ", "sup "))
