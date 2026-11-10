# Maiúsculas, tradução e ajuste fino: análise visual do corpus Pepper&Carrot

Atividade de visualização de dados (Ciência de Dados). Pergunta: **a caixa alta atrapalha a tradução automática de quadrinhos, e o
quanto um ajuste fino do tradutor melhora o resultado?**

## Resultado

- **`visualizacao.html`**: o relatório interativo (5 gráficos Plotly, texto e descobertas). Arquivo único e autocontido: abre em
  qualquer navegador, **sem internet** (o plotly.js vai dentro).
- **`analise_pepper_carrot.ipynb`**: o mesmo, como notebook Jupyter, com o código de cada gráfico.

## Dataset e fonte

- **Corpus:** [Pepper&Carrot](https://www.peppercarrot.com), de David Revoy. Textos extraídos dos SVGs do repositório
  [peppercarrot/webcomics](https://framagit.org/peppercarrot/webcomics), em inglês e português do Brasil (tradução de Alexandre E.
  Almeida e Frederico Batista). Licença **CC BY 4.0**: atribuição a David Revoy e aos tradutores.
- **Tradutor:** [Helsinki-NLP/opus-mt-tc-big-en-pt](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-pt) (CC BY 4.0), original e com
  ajuste fino LoRA, avaliado por validação cruzada em 5 grupos de episódios (o modelo nunca vê o episódio que traduz).
- **Nota:** chrF por fala (0 a 100).

## Arquivos

| Arquivo | O que é |
|---|---|
| `dados/corpus_limpo.csv` | 1.676 pares de falas EN/PT depois da limpeza (sem créditos, linhas vazias, pares repetidos ou falas com mais de 200 caracteres) |
| `dados/traducoes_por_fala.csv` | 1.538 falas de teste, com a tradução e a nota de cada uma em 5 variantes (original, ajustado, caixa alta, regra simples, truecaser) |
| `gerar_dados.py` | gera os dois CSVs (precisa de GPU e dos adaptadores LoRA, que não estão neste repositório) |
| `montar_notebook.py` | monta o notebook a partir dos CSVs |
| `exportar_html.py` | exporta o notebook executado para o HTML único e autocontido |
| `truecase.py` (raiz do repositório) | o truecaser por frequência usado em `gerar_dados.py` |

## Reproduzir

```bash
pip install pandas plotly nbformat nbconvert ipykernel markdown
python montar_notebook.py
jupyter nbconvert --to notebook --execute --inplace analise_pepper_carrot.ipynb
python exportar_html.py
```

Os CSVs já estão prontos, então só o notebook precisa ser reexecutado; `gerar_dados.py` só é necessário para refazer as traduções.

## Limitações

Um único quadrinho e poucos tradutores; falas quase repetidas entre episódios; parte do ganho do ajuste fino é aprender os nomes e o estilo
do quadrinho; chrF por fala é uma aproximação; os intervalos de confiança ignoram a correlação entre falas do mesmo episódio; a caixa alta
foi simulada (`.upper()`), não é saída real de OCR. Detalhes no final do relatório.
