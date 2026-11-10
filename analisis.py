"""Todas las cifras del post, a partir del CSV real.

Uso:
    uv run --with pandas --with mlxtend python analisis.py groceries.csv
"""
import itertools
import sys
import time

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from traduccion import traducir

ruta = sys.argv[1] if len(sys.argv) > 1 else "datos/groceries.csv"
tickets = [traducir([" ".join(i.split()) for i in l.strip().split(",")]) for l in open(ruta) if l.strip()]

codificador = TransactionEncoder()
tabla = pd.DataFrame(codificador.fit(tickets).transform(tickets), columns=codificador.columns_)
print(f"{len(tickets)} tickets, {tabla.shape[1]} categorías, densidad {100 * tabla.values.mean():.2f} %")
print(f"conjuntos posibles: 2^{tabla.shape[1]} = {2 ** tabla.shape[1]:.3e}")
print(f"pares posibles: {tabla.shape[1] * (tabla.shape[1] - 1) // 2}")

print("\nbarrido de umbrales")
for s in (0.05, 0.02, 0.01, 0.005, 0.002, 0.001):
    t = time.time()
    frecuentes = apriori(tabla, min_support=s, use_colnames=True)
    reglas = association_rules(frecuentes, metric="confidence", min_threshold=0.2)
    print(f"  {s*100:5.1f} %  {len(frecuentes):6d} conjuntos  {len(reglas):6d} reglas  {time.time()-t:5.2f} s")

frecuentes = apriori(tabla, min_support=0.01, use_colnames=True)
frecuentes["k"] = frecuentes["itemsets"].apply(len)
print("\npor tamaño:", {int(k): int(v) for k, v in frecuentes.groupby("k").size().items()})

# candidatos por nivel, con la regla de combinación y poda de Apriori
L = {k: {frozenset(x) for x in frecuentes[frecuentes.k == k].itemsets} for k in (1, 2, 3)}
candidatos = {1: tabla.shape[1], 2: len(L[1]) * (len(L[1]) - 1) // 2}
for k in (3, 4):
    c = set()
    base = sorted(L[k - 1], key=lambda s: sorted(s))
    for a, b in itertools.combinations(base, 2):
        u = a | b
        if len(u) == k and all(frozenset(x) in L[k - 1] for x in itertools.combinations(u, k - 1)):
            c.add(u)
    candidatos[k] = len(c)
print("candidatos:", candidatos, "total", sum(candidatos.values()))

reglas = association_rules(frecuentes, metric="confidence", min_threshold=0.2)
reglas["ant"] = reglas.antecedents.apply(lambda x: ", ".join(sorted(x)))
reglas["con"] = reglas.consequents.apply(lambda x: ", ".join(sorted(x)))
print(f"\n{len(reglas)} reglas; lift>1: {(reglas.lift > 1).sum()}, lift>1,5: {(reglas.lift > 1.5).sum()}")
print(reglas.sort_values(["lift", "confidence"], ascending=False).head(10)[["ant", "con", "support", "confidence", "lift"]])
