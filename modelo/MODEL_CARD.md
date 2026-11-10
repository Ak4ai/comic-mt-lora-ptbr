---
language:
  - en
  - pt
license: cc-by-4.0
library_name: peft
base_model: Helsinki-NLP/opus-mt-tc-big-en-pt
pipeline_tag: translation
tags:
  - translation
  - lora
  - peft
  - marian
  - comics
  - portuguese
---

# OPUS-MT EN→PT-BR ajustado (LoRA) para diálogos de quadrinhos

Adaptador **LoRA** (4,3 M parâmetros treináveis, 17 MB) sobre o tradutor
[`Helsinki-NLP/opus-mt-tc-big-en-pt`](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-pt), ajustado em falas do webcomic
*Pepper&Carrot* (inglês → português do Brasil). Código, dados e análise: https://github.com/Ak4ai/comic-mt-lora-ptbr

## Como usar

```python
import torch
from peft import PeftModel
from transformers import MarianMTModel, MarianTokenizer

base_id = "Helsinki-NLP/opus-mt-tc-big-en-pt"
tok = MarianTokenizer.from_pretrained(base_id)
modelo = MarianMTModel.from_pretrained(base_id)
modelo = PeftModel.from_pretrained(modelo, "ID_DESTE_REPOSITORIO").eval()

frases = ["I completely forgot to turn off my security system!"]
# o prefixo >>pob<< escolhe o português do Brasil (>>por<< seria o de Portugal)
enc = tok([">>pob<< " + f for f in frases], return_tensors="pt", padding=True)
with torch.no_grad():
    saida = modelo.generate(**enc, num_beams=4, max_new_tokens=128)
print(tok.batch_decode(saida, skip_special_tokens=True))
```

## Resultados (validação cruzada por episódio)

Cada episódio aparece uma vez no teste, e o modelo **nunca vê o episódio que está traduzindo** (5 grupos de episódios).
Métrica: chrF (e BLEU) em relação à tradução humana.

| Grupo de episódios | Falas de teste | chrF original → ajustado | BLEU original → ajustado |
|---|---|---|---|
| 1 | 249 | 30,3 → 32,9 (+2,6) | 13,9 → 18,0 |
| 2 | 430 | 49,4 → 55,3 (+5,8) | 30,0 → 39,0 |
| 3 | 242 | 38,2 → 43,1 (+4,9) | 20,3 → 28,9 |
| 4 | 288 | 57,9 → 63,2 (+5,3) | 35,7 → 45,5 |
| 5 | 339 | 49,5 → 54,0 (+4,6) | 31,8 → 40,6 |
| **Média** | | **45,1 → 49,7 (+4,6)** | **26,3 → 34,4 (+8,1)** |

Em nota **média por fala** (1.538 falas de teste, dados limpos): 56,5 → 64,3 (+7,8), com ganho positivo em **todos os 30 episódios**
que têm 10 falas de teste ou mais (de +2,5 a +14,2). O ganho **não** depende de o modelo original ser fraco no episódio (r = 0,02).

## Treinamento

- **Dados:** 1.503 pares de falas EN/PT (Pepper&Carrot, CC BY 4.0), depois de remover créditos, linhas vazias, pares repetidos e falas com
  mais de 200 caracteres. 4 episódios ficam de fora só para a parada antecipada.
- **LoRA:** r = 16, alpha = 32, dropout = 0,1, em `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1` e `fc2`.
- **Otimização:** AdamW, taxa 2e-4 com aquecimento e decaimento linear, lotes de 16, bf16, parada antecipada (melhor época: 2).
- Hardware: uma GPU de 6 GB, cerca de 5 minutos.

## Limitações

- **Um único quadrinho**, traduzido por poucos voluntários: o ganho vem em parte de **aprender os nomes dos personagens e o estilo** dos
  tradutores, e não deve se estender automaticamente a outros textos.
- **Falas quase repetidas** entre episódios ("Hmm...", onomatopeias) podem favorecer o modelo ajustado. Os resultados acima removem
  falas idênticas entre treino e teste; o primeiro experimento tinha 19 a 28% de falas repetidas (créditos) e estava inflado.
- **Este modelo final foi treinado em todos os dados.** A qualidade esperada vem da validação cruzada acima, não de um teste separado
  deste modelo específico.
- chrF/BLEU não medem se o sentido está certo. A referência humana tem estilo próprio.
- Não sabemos se o modelo base viu este quadrinho no pré-treino.

## Licença e atribuição

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Atribuição a:
- **David Revoy** (*Pepper&Carrot*, https://www.peppercarrot.com) e aos tradutores do português do Brasil, Alexandre E. Almeida e Frederico Batista;
- **Helsinki-NLP / Language Technology Research Group at the University of Helsinki**, autores do modelo base (CC BY 4.0).
