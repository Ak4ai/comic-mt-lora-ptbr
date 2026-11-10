"""Gera 5 divisoes por grupos de episodios (sem repetir frase entre treino e teste)."""
import json, re, random
S = json.load(open("splits.json", encoding="utf-8"))
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
boiler = re.compile(r"(patron|patreon|peppercarrot|pepper carrot (is|and)|universe of hereva|hereva created|www |http|creative commons|cc by|david revoy|donat|support|translation|proofread|beta[- ]?read|licen[sc]e|^fin$|^end$|\bepisode \d)", re.I)
rows = [p for s in S for p in S[s] if len(norm(p["en"])) > 1 and len(norm(p["pt"])) > 1 and not boiler.search(norm(p["en"]))]
eps = sorted({p["ep"] for p in rows})
random.Random(7).shuffle(eps)
folds = [eps[i::5] for i in range(5)]
for i in range(5):
    test_eps, val_eps = set(folds[i]), set(folds[(i + 1) % 5])
    tr, va, te = [], [], []
    for p in rows:
        (te if p["ep"] in test_eps else va if p["ep"] in val_eps else tr).append(p)
    seen, tr2 = set(), []
    for p in tr:
        k = (norm(p["en"]), norm(p["pt"]))
        if k not in seen: seen.add(k); tr2.append(p)
    trk = {norm(p["en"]) for p in tr2}
    def dedup(rs):
        seen, out = set(), []
        for p in rs:
            k = norm(p["en"])
            if k in trk or k in seen: continue
            seen.add(k); out.append(p)
        return out
    out = {"train": tr2, "val": dedup(va), "test": dedup(te)}
    json.dump(out, open(f"splits_cv{i}.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(i, {k: len(v) for k, v in out.items()}, "eps teste:", len(test_eps))
