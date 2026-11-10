"""Ajuste fino (LoRA) do OPUS-MT EN->PT-BR em dialogo de quadrinho (Pepper&Carrot).

Treina nos episodios de treino, escolhe a melhor epoca pela perda de validacao e
compara, no teste (episodios nunca vistos), o modelo base contra o ajustado.
Base e ajustado sao avaliados no mesmo dtype (bf16) e com a mesma decodificacao.

Uso: python finetune.py [--max-minutes 60] [--epochs 8] [--smoke]
"""
import argparse, json, os, random, time

import sacrebleu
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import MarianMTModel, MarianTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Helsinki-NLP/opus-mt-tc-big-en-pt")
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--patience", type=int, default=2)
ap.add_argument("--max-minutes", type=float, default=60)
ap.add_argument("--out", default="adapter")
ap.add_argument("--smoke", action="store_true", help="teste rapido: poucos passos e poucas frases")
ap.add_argument("--seed", type=int, default=13)
args = ap.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
dev = "cuda"
DT = torch.bfloat16
PREFIX = ">>pob<< "

S = json.load(open(os.environ.get("SPLITS", "splits_clean.json"), encoding="utf-8"))
train, val, test = S["train"], S["val"], S["test"]
if args.smoke:
    train, val, test = train[:48], val[:16], test[:16]

tok = MarianTokenizer.from_pretrained(args.model)
base = MarianMTModel.from_pretrained(args.model, dtype=DT).to(dev).eval()


def translate(model, rows, bs=32):
    out = []
    for i in range(0, len(rows), bs):
        enc = tok([PREFIX + p["en"] for p in rows[i:i + bs]], return_tensors="pt", padding=True,
                  truncation=True, max_length=128).to(dev)
        with torch.no_grad():
            gen = model.generate(**enc, num_beams=4, max_new_tokens=128)
        out += tok.batch_decode(gen, skip_special_tokens=True)
    return out


def score(hyp, rows):
    ref = [p["pt"] for p in rows]
    return {"chrF": round(sacrebleu.corpus_chrf(hyp, [ref]).score, 2),
            "BLEU": round(sacrebleu.corpus_bleu(hyp, [ref]).score, 2)}


results = {"modelo_base": args.model, "n_treino": len(train), "n_val": len(val), "n_teste": len(test),
           "config": {k: getattr(args, k) for k in ("epochs", "bs", "lr", "rank", "patience", "seed")}}
t0 = time.time()
results["base_teste"] = score(translate(base, test), test)
results["base_val"] = score(translate(base, val), val)
print("base", results["base_teste"], results["base_val"], round(time.time() - t0), "s", flush=True)


def batches(rows, bs, shuffle):
    idx = list(range(len(rows)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield [rows[j] for j in idx[i:i + bs]]


def encode(batch):
    enc = tok([PREFIX + p["en"] for p in batch], text_target=[p["pt"] for p in batch], return_tensors="pt",
              padding=True, truncation=True, max_length=128)
    enc["labels"][enc["labels"] == tok.pad_token_id] = -100
    return {k: v.to(dev) for k, v in enc.items()}


cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.1, task_type="SEQ_2_SEQ_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"])
model = get_peft_model(base, cfg)
model.print_trainable_parameters()
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
steps_total = args.epochs * ((len(train) + args.bs - 1) // args.bs)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / steps_total))


def val_loss():
    model.eval()
    tot = n = 0
    with torch.no_grad():
        for b in batches(val, 32, False):
            enc = encode(b)
            loss = model(**enc).loss
            k = int((enc["labels"] != -100).sum())
            tot += float(loss) * k
            n += k
    return tot / n


best, bad, hist, step = float("inf"), 0, [], 0
hist.append({"epoca": 0, "val_loss": round(val_loss(), 4)})
print(hist[-1], flush=True)
stop_reason = "epocas completas"
for ep in range(1, args.epochs + 1):
    model.train()
    run = []
    for b in batches(train, args.bs, True):
        loss = model(**encode(b)).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        run.append(float(loss)); step += 1
        if (time.time() - t0) / 60 > args.max_minutes:
            break
    vl = val_loss()
    hist.append({"epoca": ep, "train_loss": round(sum(run) / len(run), 4), "val_loss": round(vl, 4)})
    print(hist[-1], flush=True)
    if vl < best:
        best, bad = vl, 0
        model.save_pretrained(args.out)
    else:
        bad += 1
        if bad >= args.patience:
            stop_reason = f"parada antecipada (sem melhora da perda de validacao por {args.patience} epocas)"
            break
    if (time.time() - t0) / 60 > args.max_minutes:
        stop_reason = f"limite de {args.max_minutes} min"
        break

results["historico"] = hist
results["parada"] = stop_reason
del model, opt
torch.cuda.empty_cache()
tuned = PeftModel.from_pretrained(MarianMTModel.from_pretrained(args.model, dtype=DT).to(dev), args.out).eval()
hyp_t = translate(tuned, test)
results["ajustado_teste"] = score(hyp_t, test)
results["ajustado_val"] = score(translate(tuned, val), val)
results["amostras"] = [{"en": p["en"], "ref": p["pt"], "ajustado": h} for p, h in zip(test[:10], hyp_t[:10])]
results["minutos"] = round((time.time() - t0) / 60, 1)
name = os.environ.get("RESULT") or ("resultado_finetune_smoke.json" if args.smoke else "resultado_finetune.json")
json.dump(results, open(name, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("base   teste", results["base_teste"], "\najust. teste", results["ajustado_teste"], "\nfim", results["minutos"], "min")
