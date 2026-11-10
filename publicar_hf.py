"""Publica o adaptador final no Hugging Face Hub.

Antes, entre na sua conta NO TERMINAL (o token fica so na sua maquina):   hf auth login
Depois:                                                                    python publicar_hf.py [--repo usuario/nome]

Sobe adapter_config.json, adapter_model.safetensors e a ficha do modelo (modelo/MODEL_CARD.md, como README.md).
"""
import argparse
import os
import shutil
import tempfile

from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--repo", help="usuario/nome do repositorio (padrao: <seu usuario>/opus-mt-tc-big-en-pt-comics-lora)")
ap.add_argument("--adaptador", default="adapter_final")
args = ap.parse_args()

api = HfApi()
usuario = api.whoami()["name"]  # falha com mensagem clara se voce nao entrou na conta
repo = args.repo or f"{usuario}/opus-mt-tc-big-en-pt-comics-lora"

with tempfile.TemporaryDirectory() as tmp:
    for nome in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy(os.path.join(args.adaptador, nome), tmp)
    ficha = open(os.path.join("modelo", "MODEL_CARD.md"), encoding="utf-8").read().replace("ID_DESTE_REPOSITORIO", repo)
    open(os.path.join(tmp, "README.md"), "w", encoding="utf-8").write(ficha)
    api.create_repo(repo, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=tmp, repo_id=repo, repo_type="model", commit_message="Adaptador LoRA e ficha do modelo")
print(f"publicado: https://huggingface.co/{repo}")
