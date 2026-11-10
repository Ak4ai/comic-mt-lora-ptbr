"""Divisao para o modelo FINAL (o que sera publicado): usa todos os dados limpos.

Treina em tudo, menos em 4 episodios guardados so para parar o treino na epoca certa (parada antecipada).
ATENCAO: esses episodios guardados NAO sao um teste independente. A qualidade esperada do modelo vem da
validacao cruzada por episodio (resultado_cv*.json), em que o modelo nunca viu o episodio avaliado.

Saida: splits_final.json (no formato que finetune.py le).
"""
import json
import random
import re

S = json.load(open("splits.json", encoding="utf-8"))
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
creditos = re.compile(
    r"(patron|patreon|peppercarrot|pepper carrot (is|and)|universe of hereva|hereva created|www |http|creative commons|"
    r"cc by|david revoy|donat|support|translation|proofread|beta[- ]?read|licen[sc]e|^fin$|^end$|\bepisode \d)", re.I)
MAX_FALA = 200

rows, vistos = [], set()
for p in [p for s in S for p in S[s]]:
    e, t = norm(p["en"]), norm(p["pt"])
    if len(e) <= 1 or len(t) <= 1 or creditos.search(e) or len(p["en"]) > MAX_FALA or (e, t) in vistos:
        continue
    vistos.add((e, t))
    rows.append(p)

eps = sorted({p["ep"] for p in rows})
random.Random(11).shuffle(eps)
val_eps = set(eps[:4])
train = [p for p in rows if p["ep"] not in val_eps]
train_en = {norm(p["en"]) for p in train}
val, vistos_val = [], set()
for p in rows:
    k = norm(p["en"])
    if p["ep"] in val_eps and k not in train_en and k not in vistos_val:
        vistos_val.add(k)
        val.append(p)

json.dump({"train": train, "val": val, "test": val}, open("splits_final.json", "w", encoding="utf-8"), ensure_ascii=False)
print(f"treino: {len(train)} falas | validacao (parada antecipada): {len(val)} falas de {len(val_eps)} episodios")
