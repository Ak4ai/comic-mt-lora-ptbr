"""Gera os dados da analise: o corpus limpo e as traducoes por fala (5 grupos de episodios).

Para cada grupo (dobra) da validacao cruzada, traduz as falas de TESTE (episodios que o modelo
nunca viu) de cinco formas e guarda a nota de cada fala (chrF):

  base_normal    modelo original, texto como no quadrinho
  tuned_normal   modelo ajustado (LoRA), texto como no quadrinho
  base_upper     modelo original, texto TODO EM MAIUSCULAS (como sai de muito OCR)
  base_rule      modelo original, apos a regra simples de maiusculas do leitor
  base_truecase  modelo original, apos o truecaser por frequencia (treinado so nos episodios de treino)

Uso (da pasta do projeto): python visualizacao/gerar_dados.py
"""
import csv
import json
import os
import re
import sys
import time

import sacrebleu
import torch
from peft import PeftModel
from transformers import MarianMTModel, MarianTokenizer

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RAIZ)
sys.path.insert(0, RAIZ)
from truecase import Truecaser, rule  # noqa: E402

MODELO = "Helsinki-NLP/opus-mt-tc-big-en-pt"
PREFIXO = ">>pob<< "  # portugues do Brasil
SAIDA = os.path.join("visualizacao", "dados")
os.makedirs(SAIDA, exist_ok=True)
DOBRAS = range(5)
MAX_FALA = 200  # falas mais longas que isso sao descartadas (listas de apoiadores, notas)

norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
CREDITOS = re.compile(
    r"(patron|patreon|peppercarrot|pepper carrot (is|and)|universe of hereva|hereva created|www |http|creative commons|"
    r"cc by|david revoy|donat|support|translation|proofread|beta[- ]?read|licen[sc]e|^fin$|^end$|\bepisode \d)", re.I)

# ---------- corpus limpo (sem creditos, sem linhas vazias, sem pares repetidos) ----------
pares = json.load(open("pairs.json", encoding="utf-8"))
vistos, corpus = set(), []
for p in pares:
    en, pt = norm(p["en"]), norm(p["pt"])
    # listas de apoiadores e notas de tradutores (centenas de caracteres) nao sao falas de personagens
    if len(en) <= 1 or len(pt) <= 1 or CREDITOS.search(en) or len(p["en"]) > MAX_FALA:
        continue
    if (en, pt) in vistos:
        continue
    vistos.add((en, pt))
    corpus.append(p)
with open(os.path.join(SAIDA, "corpus_limpo.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["episodio", "en", "pt"])
    for p in corpus:
        w.writerow([p["ep"], p["en"], p["pt"]])
print(f"corpus: {len(pares)} pares brutos -> {len(corpus)} apos a limpeza", flush=True)

# ---------- modelos ----------
tok = MarianTokenizer.from_pretrained(MODELO)
base = MarianMTModel.from_pretrained(MODELO, dtype=torch.bfloat16).to("cuda").eval()
modelo = PeftModel.from_pretrained(base, "adapter_cv0", adapter_name="cv0")
for i in list(DOBRAS)[1:]:
    modelo.load_adapter(f"adapter_cv{i}", adapter_name=f"cv{i}")
modelo.eval()


def traduzir(textos, bs=32):
    saida = []
    for i in range(0, len(textos), bs):
        enc = tok([PREFIXO + t for t in textos[i:i + bs]], return_tensors="pt", padding=True,
                  truncation=True, max_length=128).to("cuda")
        with torch.no_grad():
            gen = modelo.generate(**enc, num_beams=4, max_new_tokens=128)
        saida += tok.batch_decode(gen, skip_special_tokens=True)
    return saida


def nota(hip, ref):
    return round(sacrebleu.sentence_chrf(hip, [ref]).score, 2)


linhas = []
t0 = time.time()
for i in DOBRAS:
    d = json.load(open(f"splits_cv{i}.json", encoding="utf-8"))
    teste, treino = [p for p in d["test"] if len(p["en"]) <= MAX_FALA], d["train"]
    tc = Truecaser().fit([p["en"] for p in treino])
    en = [p["en"] for p in teste]
    ref = [p["pt"] for p in teste]
    modelo.set_adapter(f"cv{i}")
    with modelo.disable_adapter():
        base_normal = traduzir(en)
        base_upper = traduzir([t.upper() for t in en])
        base_rule = traduzir([rule(t.upper()) for t in en])
        base_tc = traduzir([tc(t.upper()) for t in en])
    tuned = traduzir(en)
    for k, p in enumerate(teste):
        linhas.append({
            "dobra": i, "episodio": p["ep"], "en": p["en"], "pt_ref": p["pt"],
            "hip_base": base_normal[k], "hip_ajustado": tuned[k], "hip_maiusculas": base_upper[k],
            "hip_regra": base_rule[k], "hip_truecaser": base_tc[k],
            "chrf_base": nota(base_normal[k], ref[k]), "chrf_ajustado": nota(tuned[k], ref[k]),
            "chrf_maiusculas": nota(base_upper[k], ref[k]), "chrf_regra": nota(base_rule[k], ref[k]),
            "chrf_truecaser": nota(base_tc[k], ref[k]),
        })
    print(f"dobra {i}: {len(teste)} falas em {round(time.time() - t0)} s", flush=True)

with open(os.path.join(SAIDA, "traducoes_por_fala.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
    w.writeheader()
    w.writerows(linhas)
print(f"salvo: {len(linhas)} falas", flush=True)
