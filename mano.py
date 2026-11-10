"""Los ocho tickets del ejemplo, calculados a mano."""
T = [
    {"pan", "leche", "mantequilla"},
    {"pan", "leche"},
    {"pan", "cerveza"},
    {"leche", "pañales"},
    {"pan", "leche", "pañales"},
    {"cerveza", "pañales"},
    {"leche", "cerveza"},
    {"pan", "leche", "cerveza"},
]
N = len(T)


def soporte(items):
    return sum(1 for t in T if set(items) <= t) / N


def regla(a, b):
    s = soporte(a + b)
    c = s / soporte(a)
    return s, c, c / soporte(b)


for a, b in [
    (["pan"], ["leche"]),
    (["leche"], ["pan"]),
    (["mantequilla"], ["pan"]),
    (["cerveza"], ["pan"]),
    (["pañales"], ["leche"]),
]:
    s, c, l = regla(a, b)
    print(f"{{{a[0]}}} -> {{{b[0]}}}: sup {s:.3f}  conf {c:.3f}  lift {l:.2f}")
