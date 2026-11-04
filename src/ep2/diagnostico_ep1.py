#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnóstico do EP1 para orientar o EP2, sem consultar conjuntos de teste.

Exemplo de execução a partir da raiz do repositório:
python src/ep2/diagnostico_ep1.py \
  --train data/raw/train.xlsx \
  --folds data/splits/folds.csv \
  --oof results/ensemble_majority_5_v3_norberto_oof.csv \
  --saida results/ep2/diagnostico

Dependências: pandas, numpy, openpyxl, scikit-learn.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

CLASSES = ["c1", "c234", "c5"]
IDS = {c: i for i, c in enumerate(CLASSES)}
TOKENS = re.compile(r"\b\w+\b", flags=re.UNICODE)


def ler_rotulo(value) -> str:
    if isinstance(value, str):
        s = value.strip()
        if s in IDS:
            return s
        if s in ("0", "1", "2"):
            return CLASSES[int(s)]
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1, 2):
        return CLASSES[int(value)]
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and value.is_integer() and int(value) in (0, 1, 2):
        return CLASSES[int(value)]
    raise ValueError(f"Rótulo inválido: {value!r}")


def coluna_predicao(oof: pd.DataFrame) -> str:
    for campo in ("y_pred_id", "y_pred", "pred_id", "pred", "prediction", "clarity_pred"):
        if campo in oof.columns:
            return campo
    raise ValueError(f"Não foi encontrada coluna de predição no OOF. Colunas: {list(oof.columns)}")


def carregar_dados(args) -> pd.DataFrame:
    train = pd.read_excel(args.train).reset_index(drop=True)
    folds = pd.read_csv(args.folds).reset_index(drop=True)
    oof = pd.read_csv(args.oof)

    if len(train) != args.expected_rows or len(folds) != len(train):
        raise ValueError(f"Esperadas {args.expected_rows} instâncias: treino={len(train)}, folds={len(folds)}")
    if not {"resp_text", "clarity"}.issubset(train.columns):
        raise ValueError("train.xlsx deve conter resp_text e clarity")
    if not {"row_index", "fold"}.issubset(folds.columns):
        raise ValueError("folds.csv deve conter row_index e fold")
    if not np.array_equal(folds["row_index"].to_numpy(), np.arange(len(train))):
        raise ValueError("folds.csv desalinhado em relação à ordem do treino")
    if sorted(folds["fold"].unique().tolist()) != [0, 1, 2, 3, 4]:
        raise ValueError("Folds esperados: 0,1,2,3,4")
    if train[["resp_text", "clarity"]].isna().any().any():
        raise ValueError("O treino contém valores nulos em resp_text/clarity")
    if "row_id" not in oof.columns:
        raise ValueError("OOF precisa de row_id para alinhamento seguro (não alinhar pela ordem das linhas)")
    if oof["row_id"].isna().any() or oof["row_id"].duplicated().any():
        raise ValueError("OOF possui row_id nulo ou duplicado")
    if len(oof) != len(train):
        raise ValueError(f"OOF tem {len(oof)} linhas; esperado {len(train)}")
    if set(pd.to_numeric(oof["row_id"], errors="raise").astype(int)) != set(range(len(train))):
        raise ValueError("OOF não cobre exatamente os row_id do treino")

    base = train[["resp_text", "clarity"]].copy()
    base["row_id"] = np.arange(len(base))
    base["fold"] = folds["fold"].astype(int)
    base["clarity"] = base["clarity"].apply(ler_rotulo)
    oof["row_id"] = pd.to_numeric(oof["row_id"], errors="raise").astype(int)

    campo = coluna_predicao(oof)
    pred = oof[["row_id", campo]].rename(columns={campo: "_pred_original"})
    if "y_true_id" in oof.columns:
        pred["_y_true_original"] = oof["y_true_id"].to_numpy()
    elif "y_true" in oof.columns:
        pred["_y_true_original"] = oof["y_true"].to_numpy()
    if "fold" in oof.columns:
        pred["_fold_original"] = oof["fold"].to_numpy()

    dados = base.merge(pred, on="row_id", how="left", validate="one_to_one", indicator=True)
    if not (dados["_merge"] == "both").all():
        raise ValueError("Alguns row_id estão ausentes do OOF")
    dados = dados.drop(columns="_merge")
    dados["y_pred"] = dados["_pred_original"].apply(ler_rotulo)
    if "_y_true_original" in dados:
        if not (dados["_y_true_original"].apply(ler_rotulo) == dados["clarity"]).all():
            raise ValueError("OOF tem y_true diferente dos rótulos de treino")
    if "_fold_original" in dados:
        if not (dados["_fold_original"].astype(int) == dados["fold"]).all():
            raise ValueError("OOF tem folds diferentes de folds.csv")

    if (dados.groupby("resp_text", sort=False)["fold"].nunique() > 1).any():
        raise ValueError("Existe texto duplicado repartido entre folds (potencial leakage)")
    dados["correct"] = dados["clarity"] == dados["y_pred"]
    return dados


def adicionar_caracteristicas(dados: pd.DataFrame) -> pd.DataFrame:
    d = dados.copy()
    s = d["resp_text"].astype(str)
    d["n_caracteres"] = s.str.len()
    d["n_palavras"] = s.apply(lambda x: len(TOKENS.findall(x)))
    d["n_paragrafos"] = s.apply(lambda x: len([p for p in re.split(r"\n\s*\n", x) if p.strip()]))
    d["n_frases_pontuadas"] = s.apply(lambda x: max(1, len(re.findall(r"[.!?]+", x))))
    d["contem_url"] = s.str.contains(r"(?:https?://|www\.)", flags=re.IGNORECASE, regex=True)
    d["contem_digitos"] = s.str.contains(r"\d", regex=True)
    d["contem_lista"] = s.str.contains(r"(?:^|\n)\s*(?:[-•*]|\d+[.)])\s+", regex=True)
    d["faixa_palavras"] = pd.cut(
        d["n_palavras"], bins=[-1, 25, 50, 100, 200, 400, np.inf],
        labels=["0–25", "26–50", "51–100", "101–200", "201–400", ">400"],
        ordered=True,
    )
    groups = d.groupby("resp_text", sort=False)
    d["tamanho_grupo"] = groups["clarity"].transform("size")
    d["rotulos_distintos_grupo"] = groups["clarity"].transform("nunique")
    d["tipo_duplicata"] = np.select(
        [d["tamanho_grupo"] == 1, d["rotulos_distintos_grupo"] == 1],
        ["unico", "duplicado_consistente"], default="duplicado_conflitante",
    )
    d["tipo_erro"] = np.where(
        d["correct"], "acerto",
        np.where((d["clarity"] == "c1") & (d["y_pred"] == "c5") |
                 (d["clarity"] == "c5") & (d["y_pred"] == "c1"),
                 "confusao_extremos", "confusao_adjacentes"),
    )
    return d


def agregar(d: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    a = d.groupby(keys, dropna=False, observed=True)["correct"].agg(["size", "sum", "mean"]).reset_index()
    a.columns = keys + ["n", "acertos", "acuracia"]
    a["erros"] = a["n"] - a["acertos"]
    return a


def gerar(d: pd.DataFrame, destino: Path, max_amostras: int) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    n = len(d)
    y = d["clarity"]
    p = d["y_pred"]
    metricas = {
        "linhas": n,
        "accuracy_oof": float(accuracy_score(y, p)),
        "macro_f1_oof": float(f1_score(y, p, labels=CLASSES, average="macro", zero_division=0)),
        "por_classe": classification_report(y, p, labels=CLASSES, output_dict=True, zero_division=0),
        "folds": [],
        "confusoes_adjacentes": int((d["tipo_erro"] == "confusao_adjacentes").sum()),
        "confusoes_extremos": int((d["tipo_erro"] == "confusao_extremos").sum()),
    }
    for fold, f in d.groupby("fold"):
        metricas["folds"].append({"fold": int(fold), "n": len(f),
                                  "accuracy": float(accuracy_score(f["clarity"], f["y_pred"]))})
    (destino / "metricas.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")

    m = confusion_matrix(y, p, labels=CLASSES)
    pd.DataFrame(m, index=["real_" + x for x in CLASSES], columns=["pred_" + x for x in CLASSES]).to_csv(destino / "matriz_confusao.csv", encoding="utf-8-sig")
    for nome, chaves in [
        ("por_classe", ["clarity"]),
        ("por_fold", ["fold"]),
        ("por_extensao", ["faixa_palavras"]),
        ("por_classe_e_extensao", ["clarity", "faixa_palavras"]),
        ("por_duplicatas", ["tipo_duplicata"]),
        ("por_classe_e_duplicatas", ["clarity", "tipo_duplicata"]),
        ("por_paragrafos", ["n_paragrafos"]),
        ("por_url", ["contem_url"]),
        ("por_digitos", ["contem_digitos"]),
        ("por_lista", ["contem_lista"]),
        ("por_tipo_erro", ["tipo_erro"]),
    ]:
        agregar(d, chaves).to_csv(destino / f"{nome}.csv", index=False, encoding="utf-8-sig")

    erros = d.loc[~d["correct"]].copy()
    # Amostragem determinística, distribuída entre pares real/predito.
    amostras = []
    for _, grupo in erros.groupby(["clarity", "y_pred"]):
        amostras.append(grupo.sample(n=min(max_amostras, len(grupo)), random_state=42))
    if amostras:
        selecionados = pd.concat(amostras).sort_values(["clarity", "y_pred", "row_id"]).copy()
        selecionados["trecho_texto"] = selecionados["resp_text"].str.slice(0, 650).str.replace(r"\s+", " ", regex=True)
        selecionados[["row_id", "fold", "clarity", "y_pred", "tipo_duplicata", "n_palavras", "n_paragrafos", "trecho_texto"]].to_csv(
            destino / "amostra_erros.csv", index=False, encoding="utf-8-sig"
        )

    grupos = d.loc[d["tipo_duplicata"] == "duplicado_conflitante"].groupby("resp_text", sort=False)
    # Somente agregados — evita exportar textos integrais nesta tabela.
    conflitos = grupos["clarity"].value_counts().unstack(fill_value=0).reindex(columns=CLASSES, fill_value=0)
    conflitos["n"] = conflitos[CLASSES].sum(axis=1)
    conflitos = conflitos.reset_index(drop=True).sort_values("n", ascending=False)
    conflitos.head(100).to_csv(destino / "conflitos_top100_anonimizados.csv", index=False, encoding="utf-8-sig")

    linhas = [
        "# Diagnóstico EP1 → EP2", "",
        "Avaliação realizada apenas sobre o treino e as predições OOF do EP1. Nenhum conjunto de teste foi consultado.", "",
        f"- Registros avaliados: **{n:,}**.",
        f"- Acurácia OOF: **{metricas['accuracy_oof']:.4%}**.",
        f"- Macro-F1 OOF: **{metricas['macro_f1_oof']:.4%}**.",
        f"- Erros adjacentes: **{metricas['confusoes_adjacentes']:,}**.",
        f"- Erros entre extremos (c1↔c5): **{metricas['confusoes_extremos']:,}**.", "",
        "## F1 por classe", "",
    ]
    for c in CLASSES:
        linhas.append(f"- {c}: {metricas['por_classe'][c]['f1-score']:.4%}")
    linhas += ["", "## Diagnóstico de comprimento", "", "| Palavras | N | Acurácia |", "|---|---:|---:|"]
    for rec in agregar(d, ["faixa_palavras"]).to_dict("records"):
        linhas.append(f"| {rec['faixa_palavras']} | {rec['n']} | {rec['acuracia']:.2%} |")
    linhas += ["", "## Duplicatas", "", "| Situação | N | Acurácia |", "|---|---:|---:|"]
    for rec in agregar(d, ["tipo_duplicata"]).to_dict("records"):
        linhas.append(f"| {rec['tipo_duplicata']} | {rec['n']} | {rec['acuracia']:.2%} |")
    linhas += [
        "", "## Como usar", "",
        "- Conferir `por_classe_e_extensao.csv` para verificar se o problema de textos longos se concentra em uma classe.",
        "- Conferir `amostra_erros.csv` e classificar qualitativamente os erros (evasão, falta de justificativa, resposta completa, ambiguidade etc.).",
        "- Conferir `por_classe_e_duplicatas.csv` para separar ruído de rótulos de falhas do modelo.",
        "- Somente após essa leitura escolher o novo experimento de treinamento.",
        "", "**Cuidado:** características textuais medidas aqui são correlatos, não causas estabelecidas da clareza. "
        "Não inferir conceitos semânticos (evasão, justificativa, completude) apenas por contagens de palavras.", "",
    ]
    (destino / "resumo.md").write_text("\n".join(linhas), encoding="utf-8")
    print("=" * 68)
    print("DIAGNÓSTICO EP1 — somente dados de treino/OOF")
    print(f"Accuracy = {metricas['accuracy_oof']:.6f} | Macro-F1 = {metricas['macro_f1_oof']:.6f}")
    print(f"Erros adjacentes = {metricas['confusoes_adjacentes']} | Extremos = {metricas['confusoes_extremos']}")
    print(f"Relatórios salvos em: {destino.resolve()}")
    if abs(metricas["accuracy_oof"] - 0.475661955) > 0.002:
        print("ATENÇÃO: acurácia difere da referência V3 de 0.475661955; confira se forneceu o OOF correto.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/raw/train.xlsx"))
    parser.add_argument("--folds", type=Path, default=Path("data/splits/folds.csv"))
    parser.add_argument("--oof", type=Path, default=Path("results/ensemble_majority_5_v3_norberto_oof.csv"))
    parser.add_argument("--saida", type=Path, default=Path("results/ep2/diagnostico"))
    parser.add_argument("--expected-rows", type=int, default=20092)
    parser.add_argument("--amostras-por-confusao", type=int, default=15)
    args = parser.parse_args()
    gerar(adicionar_caracteristicas(carregar_dados(args)), args.saida, args.amostras_por_confusao)


if __name__ == "__main__":
    main()
