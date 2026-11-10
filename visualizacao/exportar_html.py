"""Exporta o notebook executado para UM arquivo HTML autocontido (funciona sem internet).

O plotly.js vai embutido uma unica vez; o texto vem das celulas markdown e os graficos, das saidas das celulas
(o notebook guarda cada figura inteira na saida). As celulas de codigo ficam de fora; os numeros impressos por elas
aparecem em um bloco recolhido ("Numeros calculados").

Uso (da pasta visualizacao): python exportar_html.py
"""
import html
import json
import os

import markdown
import nbformat
import plotly.io as pio

AQUI = os.path.dirname(os.path.abspath(__file__))
nb = nbformat.read(os.path.join(AQUI, "analise_pepper_carrot.ipynb"), as_version=4)
md = markdown.Markdown(extensions=["tables", "sane_lists"])

CSS = """
:root{color-scheme:light}
body{font:16px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color:#1f2430;margin:0;background:#fff}
main{max-width:980px;margin:0 auto;padding:32px 20px 64px}
h1{font-size:2rem;line-height:1.2;margin:0 0 .3em} h2{margin-top:2.2em;border-bottom:2px solid #e6e9f0;padding-bottom:.2em}
h3{margin-top:1.8em;color:#1d4ed8} a{color:#1d4ed8}
table{border-collapse:collapse;margin:1em 0;width:100%} th,td{border:1px solid #dfe3ec;padding:8px 10px;text-align:left;vertical-align:top}
th{background:#f4f6fb} code{background:#f1f3f9;padding:1px 5px;border-radius:4px;font-size:.92em}
.grafico{margin:1.2em 0;border:1px solid #e6e9f0;border-radius:10px;padding:6px;overflow:hidden}
details{margin:.4em 0 1.4em;color:#4b5263;font-size:.92rem} summary{cursor:pointer}
pre{background:#f6f7fb;padding:10px 12px;border-radius:8px;overflow:auto;font-size:.85rem}
p strong:first-child{color:#111}
"""

partes, primeiro, saidas = [], True, []


def esvaziar():
    global saidas
    if saidas:
        partes.append("<details><summary>Números calculados</summary><pre>" + html.escape("\n".join(saidas)) + "</pre></details>")
        saidas = []


for c in nb.cells:
    if c.cell_type == "markdown":
        esvaziar()
        md.reset()
        partes.append(md.convert(c.source))
    elif c.cell_type == "code":
        for o in c.outputs:
            if o.output_type in ("display_data", "execute_result") and "application/vnd.plotly.v1+json" in o.data:
                fig = pio.from_json(json.dumps(o.data["application/vnd.plotly.v1+json"]))
                div = pio.to_html(fig, full_html=False, include_plotlyjs=True if primeiro else False,
                                  config={"displaylogo": False, "responsive": True})
                partes.append('<div class="grafico">' + div + "</div>")
                primeiro = False
            elif o.output_type == "stream" and o.name == "stdout":
                saidas.append(o.text.strip())
esvaziar()

pagina = ('<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">'
          '<meta name="viewport" content="width=device-width,initial-scale=1">'
          "<title>Maiúsculas, tradução e ajuste fino · análise do corpus Pepper&amp;Carrot</title>"
          f"<style>{CSS}</style></head><body><main>" + "\n".join(partes) + "</main></body></html>")
destino = os.path.join(AQUI, "visualizacao.html")
with open(destino, "w", encoding="utf-8") as f:
    f.write(pagina)
print(f"salvo: {destino} ({os.path.getsize(destino) / 1048576:.1f} MB) | graficos: {pagina.count('class=\"plotly-graph-div')}")
