# Ajuste fino (LoRA) do tradutor OPUS-MT EN→PT-BR para diálogos de quadrinhos

*English: LoRA fine-tuning of OPUS-MT (EN→PT-BR) on comic-book dialogue (Pepper&Carrot), evaluated with episode-grouped 5-fold cross-validation, plus a visual analysis of the corpus. Code is MIT; data and the adapter are CC BY 4.0.*

Projeto de IA aplicada que nasceu de um [leitor de quadrinhos com OCR e tradução local](https://github.com/Ak4ai/suwayomi-leitor):
o tradutor usado lá erra em falas de quadrinho, então medimos o erro, ajustamos o modelo e avaliamos com cuidado se melhorou de verdade.

## Perguntas e respostas

| Pergunta | Resposta (com dados limpos e episódios nunca vistos) |
|---|---|
| O ajuste fino melhora a tradução? | **Sim.** chrF médio por fala 56,5 → **64,3 (+7,8)**; positivo nos **30 de 30 episódios** com 10 falas de teste ou mais (de +2,5 a +14,2) |
| E medido no corpus inteiro, em 5 grupos de episódios? | chrF 45,1 → **49,7 (+4,6)** e BLEU 26,3 → 34,4 (+8,1), com ganho em todos os 5 grupos (de +2,6 a +5,8) |
| Ajuda mais onde o modelo original é pior? | **Não.** Correlação entre a nota do original e o ganho: r = 0,02 |
| Texto em CAIXA ALTA (comum em OCR) atrapalha? | **Muito:** 56,5 → 37,7 (−33%). Uma regra simples recupera 62% da perda e um truecaser por frequência 67%, diferença de +1,0 ponto (IC 95% pareado de +0,55 a +1,49) |
| O português fica mais longo que o inglês? | Não: razão mediana 1,00; só 13,7% das falas ficam mais de 20% maiores |

Detalhes, gráficos interativos e limitações: [`visualizacao/`](visualizacao/) (notebook, relatório HTML e dados).

## O achado que mais ensina: vazamento de dados

O primeiro resultado parecia ótimo (chrF de 60,9 → 69,2 no teste). Não estava: **19 a 28% das falas de teste eram idênticas a falas do treino**
(créditos e pedidos de apoio repetidos em vários episódios, "FIN", onomatopeias). Depois de remover créditos, linhas vazias e qualquer fala de
teste igual a uma do treino, o ganho caiu para um valor menor e **honesto**, medido com validação cruzada **por episódio**.
Todos os números acima vêm dessa versão limpa.

## Dados

- **Corpus:** [Pepper&Carrot](https://www.peppercarrot.com), de **David Revoy** (CC BY 4.0). Os textos vêm dos SVGs do repositório
  [peppercarrot/webcomics](https://framagit.org/peppercarrot/webcomics), em inglês e português do Brasil (tradução de Alexandre E. Almeida e
  Frederico Batista). Atribuição obrigatória em qualquer reuso.
- `pairs.json`: 2.442 pares de falas alinhados (EN/PT) em 37 episódios, antes da limpeza. Foram extraídos 2.467; as 25 falas com mais de 200 caracteres
  (listas de apoiadores do quadrinho e notas de tradutores, que não são falas de personagens) **não foram republicadas**.
- `splits.json`: o mesmo corpus já dividido por episódio (usado como ponto de partida das divisões abaixo).
- `splits_cv0..4.json`: as 5 divisões da validação cruzada (treino/validação/teste por grupos de episódios).
- `splits_final.json`: divisão do modelo final (treino em tudo, 4 episódios só para a parada antecipada).

## Modelo

Adaptador LoRA (4,3 M de parâmetros treináveis, 17 MB) sobre [`Helsinki-NLP/opus-mt-tc-big-en-pt`](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-pt).
Ficha completa, uso e limitações em [`modelo/MODEL_CARD.md`](modelo/MODEL_CARD.md). Os pesos ficam na
[release do GitHub](https://github.com/Ak4ai/comic-mt-lora-ptbr/releases). Para publicar o mesmo adaptador no Hugging Face: `hf auth login` e `python publicar_hf.py`.

## Como reproduzir

```bash
pip install -r requirements.txt
python cv_splits.py                  # gera as 5 divisões da validação cruzada
bash run_cv.sh                       # treina e avalia um adaptador por grupo (precisa de GPU)
python make_final_split.py           # divisão do modelo final
SPLITS=splits_final.json RESULT=resultado_final.json python finetune.py --out adapter_final
python visualizacao/gerar_dados.py   # traduções por fala (precisa dos adaptadores da validação cruzada)
```

Os resultados da validação cruzada estão em `resultado_cv0..4.json` (calculados antes de essas 10 falas longas, que ficavam no treino, serem removidas dos arquivos publicados; o efeito é desprezível). O treino leva cerca de 2 a 5 minutos por adaptador numa GPU de 6 GB.

| Arquivo | O que é |
|---|---|
| `finetune.py` | ajuste fino LoRA, parada antecipada e avaliação (chrF/BLEU) do original contra o ajustado |
| `cv_splits.py`, `make_final_split.py` | divisões por episódio, sem fala repetida entre treino e teste |
| `truecase.py` | truecaser por frequência (corrige a caixa alta do OCR) |
| `run_cv.sh` | roda a validação cruzada inteira |
| `visualizacao/` | análise visual: notebook Plotly, relatório HTML e dados por fala |

## Limitações

Um único quadrinho e poucos tradutores; parte do ganho é aprender os nomes e o estilo desse quadrinho; falas quase repetidas ainda podem
favorecer o modelo ajustado; o chrF por fala é uma aproximação e os intervalos de confiança ignoram a correlação entre falas do mesmo
episódio; a caixa alta foi simulada (`.upper()`), não é saída real de OCR; o modelo final foi treinado em todos os dados, então a qualidade
esperada vem da validação cruzada, não de um teste separado dele.

## Licença

Código: [MIT](LICENSE). Dados (falas do Pepper&Carrot) e adaptador treinado: CC BY 4.0, com atribuição a David Revoy, aos tradutores e aos autores
do modelo base (Helsinki-NLP). Este projeto não é afiliado ao Pepper&Carrot.
