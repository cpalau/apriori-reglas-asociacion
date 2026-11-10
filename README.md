# El algoritmo Apriori: qué se compra junto

Datos y código del artículo
[El algoritmo Apriori: qué se compra junto](https://cristianpalau.com/apriori-reglas-asociacion/).

El ejemplo son **9.835 tickets reales** de un supermercado, con **169 categorías**
de producto, y el algoritmo que encuentra qué se compra junto: soporte,
confianza y lift.

## Contenido

| Fichero | Qué es |
| --- | --- |
| `datos/groceries.csv` | El dataset original, tal cual se descarga: 9.835 líneas, una por ticket, productos separados por comas y en inglés. |
| `datos/groceries-es.csv` | El mismo dataset con las 169 categorías traducidas al español, una por ticket y separadas por `;`. Es el que se usa en el artículo. |
| `datos/traduccion.json` | El mapa de las 169 categorías: inglés → español. |
| `traduccion.py` | El mismo mapa, en Python, para no depender del JSON. |
| `analisis.py` | Barrido de umbrales, niveles de candidatos y reglas. Todas las cifras del post. |
| `demo.py` | El guion que se grabó en el cast del artículo (resumen, niveles, reglas y umbral). |
| `mano.py` | Los ocho tickets del ejemplo, calculados a mano. |
| `plots.py` | Los dos gráficos de matplotlib que acompañan al artículo. |

## Cómo reproducirlo

Necesitas Python 3.10 o superior y [`uv`](https://docs.astral.sh/uv/):

```bash
uv run --with pandas --with mlxtend python analisis.py
```

Salida esperada (con `min_support=0.01` y `min_confidence=0.2`):

```text
9835 tickets, 169 categorías, densidad 2.61 %
conjuntos posibles: 2^169 = 7.483e+50
pares posibles: 14196

barrido de umbrales
    5.0 %      31 conjuntos       6 reglas   0.01 s
    2.0 %     122 conjuntos      73 reglas   0.02 s
    1.0 %     333 conjuntos     234 reglas   0.04 s
    0.5 %    1001 conjuntos     892 reglas   0.10 s
    0.2 %    4223 conjuntos    5499 reglas   0.49 s
    0.1 %   13492 conjuntos   24871 reglas   2.01 s

por tamaño: {1: 88, 2: 213, 3: 32}
candidatos: {1: 169, 2: 3828, 3: 576, 4: 6} total 4579

234 reglas; lift>1: 231, lift>1,5: 163
```

Los tiempos son de una máquina concreta (Ryzen 7 8745HS, Python 3.13,
mlxtend 0.25) y varían; el resto de cifras no.

El mínimo ejemplo, si prefieres empezar por ahí:

```python
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

tickets = [linea.strip().split(";") for linea in open("datos/groceries-es.csv")]

codificador = TransactionEncoder()
tabla = pd.DataFrame(codificador.fit(tickets).transform(tickets), columns=codificador.columns_)

frecuentes = apriori(tabla, min_support=0.01, use_colnames=True)
reglas = association_rules(frecuentes, metric="confidence", min_threshold=0.2)
print(len(frecuentes), "conjuntos y", len(reglas), "reglas")
# 333 conjuntos y 234 reglas
```

## Origen de los datos

El dataset se llama `Groceries` y acompaña a la librería de R
[`arules`](https://github.com/mhahsler/arules): **un mes (30 días) de
transacciones reales de un supermercado local típico**, agregadas a 169
categorías. Lo publicaron Michael Hahsler, Kurt Hornik y Thomas Reutterer.

- Documentación original:
  <https://search.r-project.org/CRAN/refmans/arules/html/Groceries.html>
- CSV usado aquí (copia en inglés, sin tocar):
  <https://github.com/stedy/Machine-Learning-with-R-datasets>

Si usas el dataset en un trabajo, cita a sus autores:

```bibtex
@article{hahsler2007arules,
  title   = {Introduction to arules -- A computational environment for mining association rules and frequent item sets},
  author  = {Hahsler, Michael and Gr{\"u}n, Bettina and Hornik, Kurt},
  journal = {Journal of Statistical Software},
  year    = {2005}
}
```

## Licencia

- El **código** de este repositorio es MIT (ver `LICENSE`).
- Los **datos** pertenecen a sus autores originales y se incluyen para que el
  artículo se pueda reproducir. Se redistribuyen con la misma licencia del
  paquete `arules` (GPL-2 o posterior); si los reutilizas, cita la fuente.
- La traducción de las etiquetas es propia y se publica con la misma licencia
  que el código.
