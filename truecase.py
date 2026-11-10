"""Truecaser por frequencia (linha de base A).

Aprende, no texto de treino, a forma mais comum de cada palavra no meio da frase.
O inicio de frase e sempre capitalizado. Palavra desconhecida vira minuscula.
"""
import collections, json, re

TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*|[^A-Za-z]+")
STARTS = re.compile(r"[.!?]\s*$")


def tokens(text):
    return TOKEN.findall(text)


def is_start(toks, i):
    """True se a palavra toks[i] abre uma frase."""
    prev = "".join(toks[:i])
    return not re.search(r"[A-Za-z]", prev) or bool(STARTS.search(prev))


class Truecaser:
    def __init__(self):
        self.forms = collections.defaultdict(collections.Counter)

    def fit(self, texts):
        for t in texts:
            toks = tokens(t)
            for i, w in enumerate(toks):
                if w[0].isalpha() and not is_start(toks, i):
                    self.forms[w.lower()][w] += 1
        return self

    def __call__(self, text):
        toks = tokens(text)
        out = []
        for i, w in enumerate(toks):
            if not w[0].isalpha():
                out.append(w)
                continue
            lw = w.lower()
            form = self.forms[lw].most_common(1)[0][0] if lw in self.forms else lw
            if is_start(toks, i):
                form = form[0].upper() + form[1:]
            out.append(form)
        return "".join(out)


def rule(t):
    """Regra atual do leitor (com o 'I' corrigido): tudo minusculo, maiuscula apos . ! ?"""
    t = t.lower()
    t = re.sub(r"\bi\b", "I", t)
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), t)


def word_acc(preds, golds):
    ok = n = 0
    for p, g in zip(preds, golds):
        a = [w for w in tokens(p) if w[0].isalpha()]
        b = [w for w in tokens(g) if w[0].isalpha()]
        if len(a) != len(b):
            continue
        ok += sum(x == y for x, y in zip(a, b))
        n += len(b)
    return ok / n, n


if __name__ == "__main__":
    S = json.load(open("splits.json", encoding="utf-8"))
    tr = [p["en"] for p in S["train"]]
    tc = Truecaser().fit(tr)
    for s in ("val", "test"):
        g = [p["en"] for p in S[s]]
        up = [x.upper() for x in g]
        for nome, f in (("regra_atual", rule), ("frequencia", tc)):
            a, n = word_acc([f(x) for x in up], g)
            print(s, nome, round(a, 4), "palavras", n)

    # validacao cruzada por episodio (5 grupos), para nao depender de 3 episodios
    rows = [p for s in S for p in S[s]]
    eps = sorted({p["ep"] for p in rows})
    folds = [eps[i::5] for i in range(5)]
    res = {"regra_atual": [0, 0], "frequencia": [0, 0]}
    for f in folds:
        trn = [p["en"] for p in rows if p["ep"] not in f]
        tst = [p["en"] for p in rows if p["ep"] in f]
        t = Truecaser().fit(trn)
        up = [x.upper() for x in tst]
        for nome, fn in (("regra_atual", rule), ("frequencia", t)):
            a, n = word_acc([fn(x) for x in up], tst)
            res[nome][0] += a * n
            res[nome][1] += n
    for k, (ok, n) in res.items():
        print("CV 5 grupos", k, round(ok / n, 4), "palavras", n)

    # erros mais comuns do truecaser no teste
    err = collections.Counter()
    for p in S["test"]:
        pr = tokens(tc(p["en"].upper()))
        gd = tokens(p["en"])
        if len(pr) == len(gd):
            for a, b in zip(pr, gd):
                if a != b and a[0].isalpha():
                    err[(a, b)] += 1
    print("erros frequentes (previsto -> correto):", err.most_common(12))
