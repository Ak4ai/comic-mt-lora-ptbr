"""Monta o notebook analise_pepper_carrot.ipynb.

Os numeros dos textos (markdown) saem dos proprios dados, calculados aqui; as celulas de codigo do notebook
refazem o mesmo calculo e mostram os valores, para quem le poder conferir.
Uso (da pasta do projeto): python visualizacao/montar_notebook.py
"""
import os

import nbformat as nbf
import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(AQUI, "dados", "traducoes_por_fala.csv"))
co = pd.read_csv(os.path.join(AQUI, "dados", "corpus_limpo.csv"))


def fm(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")


def boot(x, n=2000, seed=1):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


# ---------- numeros usados no texto ----------
N_BRUTO = 2467
N_LIMPO = len(co)
N_EPS = co.episodio.nunique()
por_ep_n = co.episodio.value_counts()
co["n_en"], co["n_pt"] = co.en.str.len(), co.pt.str.len()
r = (co.n_pt / co.n_en)[co.n_en > 8]
mai = boot(df.chrf_maiusculas); reg = boot(df.chrf_regra); tcs = boot(df.chrf_truecaser)
nor = boot(df.chrf_base); aju = boot(df.chrf_ajustado)
perda = nor[0] - mai[0]
rec_regra = (reg[0] - mai[0]) / perda
rec_tc = (tcs[0] - mai[0]) / perda
dif_tc = boot(df.chrf_truecaser - df.chrf_regra, seed=2)
gan = boot(df.chrf_ajustado - df.chrf_base, seed=3)
d = df.chrf_ajustado - df.chrf_base
melhor, pior, igual = (d > 0).mean(), (d < 0).mean(), (d == 0).mean()
g = df.groupby("episodio").agg(n=("en", "size"), base=("chrf_base", "mean"), aj=("chrf_ajustado", "mean"))
g["ganho"] = g.aj - g.base
g10 = g[g.n >= 10]
corr = np.corrcoef(g10.base, g10.ganho)[0, 1]
pior_ep = g10.ganho.idxmin(); melhor_ep = g10.ganho.idxmax()
nome = lambda e: e.split("_", 1)[1].replace("-", " ")

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = []

cells.append(md(f"""# Maiúsculas, tradução e o que um ajuste fino realmente melhora
### Uma análise visual do corpus *Pepper&Carrot* (inglês → português do Brasil)

**Atividade prática de Ciência de Dados** · Henrique de Freitas Araújo · CEFET-MG, Campus Divinópolis · novembro de 2026

**Resumo.** Textos de quadrinhos costumam aparecer em CAIXA ALTA depois do OCR, e isso atrapalha a tradução automática. Usando
um corpus aberto de falas de quadrinho com tradução humana, medimos (1) como é esse corpus, (2) quanto a caixa alta prejudica a
tradução e quanto cada correção recupera, e (3) quanto um ajuste fino (LoRA) do tradutor melhora o resultado em episódios que ele
nunca viu. Os gráficos são interativos (passe o mouse).

| Item | Detalhe |
|---|---|
| **Ferramentas** | Python, pandas e **Plotly** (gráficos interativos), em um notebook Jupyter |
| **Dataset** | Pepper&Carrot (David Revoy), CC BY 4.0, {N_LIMPO} pares de falas EN→PT em {N_EPS} episódios |
| **Código e dados** | `gerar_dados.py` (traduções por fala), `dados/*.csv` (corpus e resultados), este notebook |"""))

cells.append(md(f"""## 1. Dataset e fonte

- **Corpus:** o webcomic aberto [Pepper&Carrot](https://www.peppercarrot.com), de **David Revoy**, que publica as páginas em vários idiomas.
  Os textos de cada balão ficam nos arquivos SVG do repositório da comunidade (`peppercarrot/webcomics`, em
  [framagit.org](https://framagit.org/peppercarrot/webcomics)). Usamos o inglês (`en`) e o **português do Brasil** (`pt`, traduzido por
  Alexandre E. Almeida e Frederico Batista).
- **Licença:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Atribuição: David Revoy / Pepper&Carrot e os tradutores acima.
- **Preparação (pequenos ajustes):** extraímos o texto de cada balão dos SVGs e alinhamos inglês e português por página.
  Dos **{N_BRUTO}** pares brutos, ficaram **{N_LIMPO}** depois de remover créditos e pedidos de apoio que se repetem em vários episódios,
  linhas vazias, pares idênticos e as raras falas com mais de 200 caracteres (listas de apoiadores e notas de tradutores, que não são falas de personagens).
- **Resultados de tradução (gerados por nós):** o tradutor *OPUS-MT* `opus-mt-tc-big-en-pt` (Helsinki-NLP, CC BY 4.0), original e com
  ajuste fino LoRA, testado com **validação cruzada em 5 grupos de episódios** (cada episódio aparece uma vez no teste, e o modelo
  nunca viu o episódio que está traduzindo). Nota: **chrF por fala** (0 a 100; mede a semelhança de trechos de caracteres com a
  tradução humana)."""))

cells.append(md("""## 2. Perguntas

1. Como é o corpus? Quantas falas por episódio, e o português fica mais longo que o inglês (o que importa para caber no balão)?
2. A **caixa alta** atrapalha a tradução? Quanto uma **regra simples** e um **truecaser** (modelo de frequências) recuperam?
3. O **ajuste fino** melhora a tradução? Em todos os episódios, ou só em alguns? Melhora mais onde o tradutor original é pior?"""))

cells.append(code('''import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "plotly_mimetype+notebook_connected"   # gráficos interativos (guarda a figura inteira na saída da célula)
pio.templates.default = "plotly_white"
AZUL, VERDE, VERMELHO, CINZA, LARANJA = "#2563EB", "#22A559", "#D64545", "#8A8F98", "#E08A1E"

corpus = pd.read_csv("dados/corpus_limpo.csv")          # falas EN/PT limpas
df = pd.read_csv("dados/traducoes_por_fala.csv")        # traduções por fala (episódios de teste de cada grupo)
corpus["n_en"], corpus["n_pt"] = corpus.en.str.len(), corpus.pt.str.len()
corpus["ep_curto"] = corpus.episodio.str.extract(r"^(ep\\d+)")[0]
corpus["titulo"] = corpus.episodio.str.replace(r"^ep\\d+_", "", regex=True).str.replace("-", " ")

def ic95(x, n=2000, seed=1):
    """média e intervalo de confiança de 95% por bootstrap (reamostragem das falas)"""
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)

print(f"corpus: {len(corpus)} falas em {corpus.episodio.nunique()} episódios | traduções avaliadas: {len(df)} falas")'''))

# ---------------- Grafico 1 ----------------
cells.append(md("""## 3. O corpus

### Gráfico 1 — Falas por episódio (barras)"""))
cells.append(code('''por_ep = corpus.groupby(["ep_curto", "titulo"]).size().reset_index(name="falas")
por_ep["amostra"] = np.where(por_ep.falas < 15, "menos de 15 falas", "15 falas ou mais")
fig = px.bar(por_ep, x="ep_curto", y="falas", color="amostra", hover_data={"titulo": True, "ep_curto": False},
             color_discrete_map={"15 falas ou mais": AZUL, "menos de 15 falas": CINZA},
             labels={"ep_curto": "episódio", "falas": "falas no corpus limpo", "titulo": "título", "amostra": ""},
             title="Quantas falas cada episódio tem")
fig.update_layout(xaxis_tickangle=-60, legend=dict(orientation="h", y=1.08))
fig.show()
print(por_ep.falas.describe().round(1).to_string())'''))
cells.append(md(f"""**Leitura.** O corpus é desigual: cada episódio tem de **{por_ep_n.min()}** a **{por_ep_n.max()}** falas (mediana
**{int(por_ep_n.median())}**). Episódios com poucas falas (em cinza) dão notas instáveis, por isso, nas análises por episódio,
só usamos os que têm **10 falas de teste ou mais**."""))

# ---------------- Grafico 2 ----------------
cells.append(md("""### Gráfico 2 — O português fica mais longo que o inglês? (dispersão interativa)"""))
cells.append(code('''amostra = corpus[corpus.n_en > 8].copy()                   # ignora falas muito curtas ("Oh!", "Hm...")
amostra["razao"] = amostra.n_pt / amostra.n_en
amostra["en_curto"] = amostra.en.str.slice(0, 80)
amostra["pt_curto"] = amostra.pt.str.slice(0, 80)
fig = px.scatter(amostra, x="n_en", y="n_pt", opacity=0.45, marginal_x="histogram", marginal_y="histogram",
                 hover_data={"en_curto": True, "pt_curto": True, "razao": ":.2f", "n_en": False, "n_pt": False, "titulo": True},
                 labels={"n_en": "caracteres em inglês", "n_pt": "caracteres em português", "en_curto": "EN", "pt_curto": "PT"},
                 title="Tamanho de cada fala: inglês contra português (passe o mouse para ler a fala)")
lim = float(max(amostra.n_en.max(), amostra.n_pt.max()))
fig.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim, line=dict(color=VERMELHO, dash="dash"), row=1, col=1)
fig.update_traces(marker_color=AZUL, selector=dict(type="scatter"))
fig.show()
print(f"razão PT/EN: mediana {amostra.razao.median():.2f} | média {amostra.razao.mean():.2f} | "
      f"PT mais de 20% maior em {(amostra.razao > 1.2).mean():.1%} das falas")'''))
cells.append(md(f"""**Leitura.** A linha tracejada é "mesmo tamanho nas duas línguas". A nuvem se concentra sobre ela: a razão entre
português e inglês tem **mediana {fm(r.median(), 2)}** e média {fm(r.mean(), 2)}, e só **{fm((r > 1.2).mean() * 100)}%** das falas ficam mais de 20%
maiores em português. Os tradutores mantêm o tamanho das falas, o que importa quando o texto precisa caber em um balão."""))

# ---------------- Grafico 3 ----------------
cells.append(md("""## 4. A caixa alta atrapalha a tradução?

### Gráfico 3 — Nota por fala em cada variante do texto (barras com intervalo de confiança)

Simulamos o texto como ele costuma sair do OCR de quadrinhos (**TUDO EM MAIÚSCULAS**) e comparamos quatro versões do mesmo texto:
a caixa alta crua, a **regra simples** (tudo minúsculo e maiúscula depois de `. ! ?`), o **truecaser** (aprende, nos episódios de
treino, a forma mais comum de cada palavra, como "Pepper" e "I") e o texto original (o teto)."""))
cells.append(code('''variantes = [("Caixa alta", "chrf_maiusculas", VERMELHO), ("Regra simples", "chrf_regra", LARANJA),
             ("Truecaser", "chrf_truecaser", AZUL), ("Texto normal (teto)", "chrf_base", VERDE)]
res = [(nome, *ic95(df[col]), cor) for nome, col, cor in variantes]
med = np.array([r[1] for r in res]); lo = np.array([r[2] for r in res]); hi = np.array([r[3] for r in res])
fig = go.Figure(go.Bar(x=[r[0] for r in res], y=med, marker_color=[r[4] for r in res],
                       error_y=dict(type="data", symmetric=False, array=hi - med, arrayminus=med - lo),
                       text=[f"{m:.1f}" for m in med], textposition="outside",
                       hovertemplate="%{x}<br>nota média: %{y:.1f}<extra></extra>"))
fig.update_layout(title="Qualidade da tradução (chrF médio por fala, com IC 95%)", yaxis_title="chrF médio por fala",
                  yaxis_range=[0, 70], showlegend=False)
fig.show()

perda = med[3] - med[0]
print(f"perda da caixa alta: {perda:.1f} pontos ({perda / med[3]:.0%} da nota)")
print(f"recuperado pela regra: {(med[1] - med[0]) / perda:.0%} | pelo truecaser: {(med[2] - med[0]) / perda:.0%}")
dif = ic95(df.chrf_truecaser - df.chrf_regra, seed=2)   # diferença PAREADA: mesma fala, duas versões
print(f"truecaser menos regra (pareado): {dif[0]:+.2f} pontos, IC 95% [{dif[1]:+.2f}, {dif[2]:+.2f}]")'''))
cells.append(md(f"""**Leitura.** A caixa alta derruba a nota média de **{fm(nor[0])}** para **{fm(mai[0])}**, uma perda de **{fm(perda)} pontos
({fm(perda / nor[0] * 100, 0)}%)**. A regra simples recupera **{fm(rec_regra * 100, 0)}%** dessa perda e o truecaser **{fm(rec_tc * 100, 0)}%**.
A diferença entre os dois é pequena: **{fm(dif_tc[0], 1)} ponto** (IC 95% pareado de {fm(dif_tc[1], 1)} a {fm(dif_tc[2], 1)}). {"O intervalo não inclui zero, mas o ganho é modesto." if dif_tc[1] > 0 else "O intervalo inclui zero: não dá para afirmar que o truecaser seja melhor que a regra."}"""))

# ---------------- Grafico 4 ----------------
cells.append(md("""## 5. O ajuste fino melhora a tradução?

### Gráfico 4 — Ganho do ajuste fino em cada episódio (barras ordenadas)

Para cada episódio, a diferença entre a nota média do tradutor **ajustado** e a do **original**, em episódios que o modelo nunca viu
(só episódios com 10 falas de teste ou mais)."""))
cells.append(code('''g = df.groupby("episodio").agg(falas=("en", "size"), original=("chrf_base", "mean"), ajustado=("chrf_ajustado", "mean")).reset_index()
g["ganho"] = g.ajustado - g.original
g["titulo"] = g.episodio.str.replace(r"^ep\\d+_", "", regex=True).str.replace("-", " ")
g["ep_curto"] = g.episodio.str.extract(r"^(ep\\d+)")[0]
g10 = g[g.falas >= 10].sort_values("ganho")
fig = px.bar(g10, x="ganho", y="ep_curto", orientation="h", color="ganho", color_continuous_scale=["#CFE0FF", AZUL],
             hover_data={"titulo": True, "falas": True, "original": ":.1f", "ajustado": ":.1f", "ganho": ":.1f", "ep_curto": False},
             labels={"ganho": "ganho do ajuste fino (pontos de chrF)", "ep_curto": "episódio"},
             title="Quanto o ajuste fino melhora cada episódio (passe o mouse)")
fig.update_layout(height=700, coloraxis_showscale=False, yaxis=dict(categoryorder="array", categoryarray=g10.ep_curto))
fig.show()

d = df.chrf_ajustado - df.chrf_base
print(f"nota média por fala: original {df.chrf_base.mean():.1f} -> ajustado {df.chrf_ajustado.mean():.1f}")
print(f"falas que melhoraram: {(d > 0).mean():.1%} | pioraram: {(d < 0).mean():.1%} | iguais: {(d == 0).mean():.1%}")
print(f"episódios (>=10 falas): {len(g10)} | com ganho positivo: {(g10.ganho > 0).sum()} | "
      f"mediana {g10.ganho.median():.1f} | de {g10.ganho.min():.1f} a {g10.ganho.max():.1f}")'''))
cells.append(md(f"""**Leitura.** O ajuste fino sobe a nota média por fala de **{fm(nor[0])}** para **{fm(aju[0])}** (**+{fm(gan[0])}**, IC 95% de
{fm(gan[1])} a {fm(gan[2])}). Em **{(g10.ganho > 0).sum()} dos {len(g10)} episódios** o ganho é positivo, mas o tamanho varia muito: da
menor melhora (**{nome(pior_ep)}**, +{fm(g10.ganho.min())}) à maior (**{nome(melhor_ep)}**, +{fm(g10.ganho.max())}). Por fala, {fm(melhor * 100, 0)}% melhoraram,
{fm(pior * 100, 0)}% pioraram e {fm(igual * 100, 0)}% ficaram iguais."""))

# ---------------- Grafico 5 ----------------
cells.append(md("""### Gráfico 5 — O ajuste ajuda mais onde o tradutor original é pior? (dispersão)"""))
cells.append(code('''fig = px.scatter(g10, x="original", y="ganho", size="falas", hover_name="titulo",
                 hover_data={"ep_curto": True, "falas": True, "original": ":.1f", "ajustado": ":.1f", "ganho": ":.1f"},
                 labels={"original": "nota do tradutor original no episódio (chrF)", "ganho": "ganho do ajuste fino (pontos)"},
                 title="Ganho do ajuste fino contra a qualidade do tradutor original (um ponto por episódio)")
a, b = np.polyfit(g10.original, g10.ganho, 1)
xs = np.array([g10.original.min(), g10.original.max()])
fig.add_trace(go.Scatter(x=xs, y=a * xs + b, mode="lines", line=dict(color=VERMELHO, dash="dash"), name="tendência"))
fig.update_traces(marker_color=AZUL, selector=dict(mode="markers"))
fig.show()
print(f"correlação entre a nota do original e o ganho: r = {np.corrcoef(g10.original, g10.ganho)[0, 1]:.2f}")'''))
cells.append(md(f"""**Leitura.** Se o ajuste ajudasse mais onde o original é fraco, os pontos desceriam da esquerda para a direita. Não é o que
acontece: a correlação é **r = {fm(corr, 2)}**, praticamente nula. O tamanho do ganho depende de **cada episódio** (de quais personagens
e expressões aparecem), não de o tradutor original estar mais ou menos bem."""))

# ---------------- Descobertas ----------------
cells.append(md(f"""## 6. Três descobertas

1. **A caixa alta é um problema real e grande.** Tira **{fm(perda)} pontos** ({fm(perda / nor[0] * 100, 0)}%) da nota média. Uma regra simples já recupera
   {fm(rec_regra * 100, 0)}% da perda; um truecaser treinado nos episódios de treino chega a {fm(rec_tc * 100, 0)}%, ganho de apenas {fm(dif_tc[0], 1)} ponto sobre a regra.
   Na prática, **corrigir a caixa alta antes de traduzir vale muito; o método de correção importa pouco**.
2. **O ajuste fino ajuda de forma consistente, mas desigual.** +{fm(gan[0])} pontos por fala em média, positivo em {(g10.ganho > 0).sum()} de {len(g10)} episódios,
   com ganho entre +{fm(g10.ganho.min())} e +{fm(g10.ganho.max())} e **sem relação** com a qualidade do tradutor original (r = {fm(corr, 2)}).
3. **O português não incha.** Neste corpus a razão de tamanho entre português e inglês tem mediana {fm(r.median(), 2)}; só {fm((r > 1.2).mean() * 100)}% das falas
   ficam mais de 20% maiores. Para encaixar texto traduzido em balões, a tradução humana mantém o tamanho.

### Por que escolhemos estas visualizações e ferramentas

- **Barras** (gráficos 1, 3 e 4) comparam quantidades entre categorias (episódios, variantes do texto); as **barras de erro** do gráfico 3 mostram
  a incerteza, para não confundir uma diferença pequena com uma diferença real.
- **Dispersão interativa** (gráficos 2 e 5) mostra a relação entre duas medidas, e o *hover* permite ler a fala de cada ponto.
- **Plotly em Jupyter:** interatividade sem servidor, exporta para um HTML único e mantém o código ao lado dos gráficos, o que ajuda quem
  quiser reproduzir a análise.

### Limitações dos dados (importante)

- **Um único quadrinho**, traduzido por poucos voluntários: os resultados não se estendem automaticamente a outros quadrinhos ou idiomas.
- **Falas repetidas.** Os créditos e os pares idênticos foram removidos, e nenhuma fala de teste é idêntica a uma de treino. Mas falas *quase*
  iguais ("Hmm...", onomatopeias) ainda existem entre os episódios e podem favorecer o modelo ajustado. Nosso primeiro resultado estava
  inflado exatamente por isso (19 a 28% das falas de teste eram créditos repetidos): vale a lição de **checar vazamento entre treino e teste**.
- **Parte do ganho é "aprender o quadrinho":** o ajuste fino aprende os nomes dos personagens e o estilo dos tradutores.
- **Métrica:** o chrF por fala é uma aproximação (não mede se o sentido está certo) e dá bastante peso a falas curtas. A média por fala é
  maior que o chrF calculado sobre o corpus inteiro, por isso os números aqui são mais altos que os do projeto original.
- **Intervalos de confiança** feitos reamostrando falas ignoram que falas do mesmo episódio se parecem; os intervalos reais são mais largos.
- A **caixa alta foi simulada** (`.upper()`), e não é saída real de OCR, que também erra letras.
- Não sabemos se o tradutor original viu este quadrinho no pré-treino.

## 7. Reprodução e entregáveis

| Item | Onde |
|---|---|
| Produto visual | este notebook e o HTML interativo exportado (`visualizacao.html`) |
| Dataset e fonte | [Pepper&Carrot](https://www.peppercarrot.com), repositório [peppercarrot/webcomics](https://framagit.org/peppercarrot/webcomics), CC BY 4.0 |
| Dados preparados | `dados/corpus_limpo.csv` e `dados/traducoes_por_fala.csv` |
| Código de preparação | `gerar_dados.py` (traduções por fala, no GPU), `montar_notebook.py` |
| Modelo usado | [Helsinki-NLP/opus-mt-tc-big-en-pt](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-pt) (CC BY 4.0), com ajuste fino LoRA |""" ))

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
destino = os.path.join(AQUI, "analise_pepper_carrot.ipynb")
nbf.write(nb, destino)
print("notebook salvo:", destino)
print(f"conferencia: caixa alta {mai[0]:.1f} | regra {reg[0]:.1f} | truecaser {tcs[0]:.1f} | normal {nor[0]:.1f} | ajustado {aju[0]:.1f}")
print(f"truecaser-regra {dif_tc[0]:+.2f} [{dif_tc[1]:+.2f}, {dif_tc[2]:+.2f}] | ganho {gan[0]:.2f} | corr {corr:.2f}")
