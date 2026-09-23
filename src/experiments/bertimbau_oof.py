from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


# ============================================================
# CAMINHOS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT_DIR / "data" / "raw" / "train.xlsx"
FOLDS_PATH = ROOT_DIR / "data" / "splits" / "folds.csv"

RESULTS_DIR = ROOT_DIR / "results"

OOF_FOLDS_DIR = (
    RESULTS_DIR
    / "bertimbau_512_oof_folds"
)

OOF_PATH = (
    RESULTS_DIR
    / "bertimbau_512_oof.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "bertimbau_512_oof_summary.csv"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "output"
    / "bertimbau_512_oof"
)


# ============================================================
# MODELO / CONFIGURAÇÃO CONGELADA
# ============================================================

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

LABEL2ID = {
    "c1": 0,
    "c234": 1,
    "c5": 2,
}

ID2LABEL = {
    0: "c1",
    1: "c234",
    2: "c5",
}

MAX_LENGTH = 512

LEARNING_RATE = 3e-5
EPOCHS = 2

TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4

WEIGHT_DECAY = 0.01

FP16 = True
GRADIENT_CHECKPOINTING = False

FOLDS = [0, 1, 2, 3, 4]

SEED = 42


# ============================================================
# DATASET
# ============================================================

class BertimbauDataset(Dataset):

    def __init__(
        self,
        encodings: dict[str, torch.Tensor],
        labels: np.ndarray,
        indices: np.ndarray,
    ):
        self.encodings = encodings
        self.labels = labels
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(
        self,
        idx: int,
    ) -> dict[str, torch.Tensor]:

        original_idx = self.indices[idx]

        item = {
            key: value[original_idx]
            for key, value in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            self.labels[original_idx],
            dtype=torch.long,
        )

        return item


# ============================================================
# MÉTRICAS
# ============================================================

def softmax_numpy(
    logits: np.ndarray,
) -> np.ndarray:

    logits = (
        logits
        - logits.max(
            axis=1,
            keepdims=True,
        )
    )

    exp = np.exp(logits)

    return (
        exp
        / exp.sum(
            axis=1,
            keepdims=True,
        )
    )


def calcular_metricas(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float]:

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
    )

    return (
        float(accuracy),
        float(macro_f1),
    )


# ============================================================
# DADOS
# ============================================================

def carregar_dados() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    print("=" * 70)
    print("Carregando dados")
    print("=" * 70)

    train_df = (
        pd.read_excel(TRAIN_PATH)
        .reset_index(drop=True)
    )

    folds_df = (
        pd.read_csv(FOLDS_PATH)
        .reset_index(drop=True)
    )

    if len(train_df) != len(folds_df):
        raise ValueError(
            "train.xlsx e folds.csv possuem "
            "números diferentes de linhas: "
            f"{len(train_df)} != {len(folds_df)}"
        )

    if "resp_text" not in train_df.columns:
        raise ValueError(
            "Coluna 'resp_text' não encontrada."
        )

    if "clarity" not in train_df.columns:
        raise ValueError(
            "Coluna 'clarity' não encontrada."
        )

    if "fold" not in folds_df.columns:
        raise ValueError(
            "Coluna 'fold' não encontrada."
        )

    print(
        f"Instâncias: {len(train_df)}"
    )

    print(
        "Folds: "
        f"{sorted(folds_df['fold'].unique())}"
    )

    return train_df, folds_df


# ============================================================
# METADADOS DE DUPLICATAS
#
# IMPORTANTE:
# usados APENAS para análise posterior.
# Não são utilizados pelo modelo.
# ============================================================

def criar_metadados_duplicatas(
    train_df: pd.DataFrame,
) -> pd.DataFrame:

    df = train_df.copy()

    df["_text_key"] = (
        df["resp_text"]
        .fillna("")
        .astype(str)
    )

    tamanho_grupo = (
        df.groupby("_text_key")[
            "_text_key"
        ]
        .transform("size")
    )

    num_labels = (
        df.groupby("_text_key")[
            "clarity"
        ]
        .transform("nunique")
    )

    tabela_labels = pd.crosstab(
        df["_text_key"],
        df["clarity"],
    )

    tabela_labels = tabela_labels.reindex(
        columns=[
            "c1",
            "c234",
            "c5",
        ],
        fill_value=0,
    )

    df["duplicate_group_size"] = (
        tamanho_grupo
    )

    df["duplicate_num_labels"] = (
        num_labels
    )

    df["is_duplicate"] = (
        tamanho_grupo > 1
    )

    df["is_conflicting_duplicate"] = (
        (tamanho_grupo > 1)
        & (num_labels > 1)
    )

    for classe in [
        "c1",
        "c234",
        "c5",
    ]:

        mapping = (
            tabela_labels[classe]
            .to_dict()
        )

        df[
            f"group_{classe}_count"
        ] = (
            df["_text_key"]
            .map(mapping)
            .astype(int)
        )

    return df.drop(
        columns=["_text_key"]
    )


# ============================================================
# TOKENIZAÇÃO
# ============================================================

def tokenizar_corpus(
    textos: list[str],
) -> dict[str, torch.Tensor]:

    print()
    print("=" * 70)
    print("Tokenização")
    print("=" * 70)

    print(
        f"Modelo: {MODEL_NAME}"
    )

    print(
        f"max_length: {MAX_LENGTH}"
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    inicio = time.perf_counter()

    encodings = tokenizer(
        textos,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    tempo = (
        time.perf_counter()
        - inicio
    )

    print(
        "Tokenização concluída em "
        f"{tempo:.2f} s"
    )

    return encodings


# ============================================================
# RESUME
# ============================================================

def caminho_fold(
    fold: int,
) -> Path:

    return (
        OOF_FOLDS_DIR
        / f"fold_{fold}.csv"
    )


def fold_ja_concluido(
    fold: int,
    eval_indices: np.ndarray,
) -> bool:

    path = caminho_fold(fold)

    if not path.exists():
        return False

    try:

        df = pd.read_csv(path)

    except Exception:
        return False

    colunas = {
        "row_id",
        "fold",
        "max_length",
        "learning_rate",
        "epochs",
    }

    if not colunas.issubset(
        df.columns
    ):
        return False

    if len(df) != len(
        eval_indices
    ):
        return False

    if set(
        df["row_id"].astype(int)
    ) != set(
        eval_indices.astype(int)
    ):
        return False

    if not (
        df["fold"].astype(int)
        == fold
    ).all():
        return False

    if not (
        df["max_length"]
        == MAX_LENGTH
    ).all():
        return False

    if not np.allclose(
        df["learning_rate"],
        LEARNING_RATE,
    ):
        return False

    if not (
        df["epochs"]
        == EPOCHS
    ).all():
        return False

    return True


# ============================================================
# EXECUÇÃO DE UM FOLD
# ============================================================

def executar_fold(
    fold: int,
    train_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    encodings: dict[str, torch.Tensor],
    labels: np.ndarray,
    fold_values: np.ndarray,
) -> dict:

    print()
    print("=" * 70)
    print(f"FOLD {fold}")
    print("=" * 70)

    set_seed(SEED)

    train_indices = np.where(
        fold_values != fold
    )[0]

    eval_indices = np.where(
        fold_values == fold
    )[0]

    if fold_ja_concluido(
        fold,
        eval_indices,
    ):

        print(
            "OOF deste fold já existe "
            "e passou na validação. "
            "Pulando."
        )

        fold_df = pd.read_csv(
            caminho_fold(fold)
        )

        return {
            "fold": fold,
            "accuracy": (
                fold_df["correct"]
                .mean()
            ),
            "macro_f1": f1_score(
                fold_df["y_true_id"],
                fold_df["y_pred_id"],
                average="macro",
            ),
            "eval_size": len(
                fold_df
            ),
            "status": "reused",
        }

    train_dataset = BertimbauDataset(
        encodings=encodings,
        labels=labels,
        indices=train_indices,
    )

    eval_dataset = BertimbauDataset(
        encodings=encodings,
        labels=labels,
        indices=eval_indices,
    )

    print(
        f"Treino: {len(train_dataset)}"
    )

    print(
        f"Validação: {len(eval_dataset)}"
    )

    print(
        f"LR: {LEARNING_RATE}"
    )

    print(
        f"Épocas: {EPOCHS}"
    )

    print(
        f"max_length: {MAX_LENGTH}"
    )

    print(
        "Batch efetivo: "
        f"{TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=3,
            label2id=LABEL2ID,
            id2label=ID2LABEL,
        )
    )

    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()

    fold_output = (
        OUTPUT_DIR
        / f"fold_{fold}"
    )

    fold_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_args = TrainingArguments(
        output_dir=str(
            fold_output
        ),

        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,

        per_device_train_batch_size=(
            TRAIN_BATCH_SIZE
        ),

        per_device_eval_batch_size=(
            EVAL_BATCH_SIZE
        ),

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        eval_strategy="epoch",
        save_strategy="no",

        fp16=FP16,

        logging_strategy="steps",
        logging_steps=100,

        report_to="none",

        seed=SEED,
        data_seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    inicio = time.perf_counter()

    try:

        trainer.train()

        # --------------------------------------------
        # Predição OOF no conjunto de validação
        # --------------------------------------------

        prediction_output = (
            trainer.predict(
                eval_dataset
            )
        )

        logits = (
            prediction_output.predictions
        )

        if isinstance(
            logits,
            tuple,
        ):
            logits = logits[0]

        probabilities = (
            softmax_numpy(
                logits
            )
        )

        y_true = labels[
            eval_indices
        ]

        y_pred = np.argmax(
            probabilities,
            axis=1,
        )

        accuracy, macro_f1 = (
            calcular_metricas(
                y_true,
                y_pred,
            )
        )

        torch.cuda.synchronize()

        tempo = (
            time.perf_counter()
            - inicio
        )

        pico_vram = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

        # --------------------------------------------
        # Arquivo OOF do fold
        # --------------------------------------------

        oof = pd.DataFrame(
            {
                "row_id": eval_indices,
                "fold": fold,

                "y_true_id": y_true,
                "y_pred_id": y_pred,

                "y_true": [
                    ID2LABEL[int(x)]
                    for x in y_true
                ],

                "y_pred": [
                    ID2LABEL[int(x)]
                    for x in y_pred
                ],

                "prob_c1": (
                    probabilities[:, 0]
                ),

                "prob_c234": (
                    probabilities[:, 1]
                ),

                "prob_c5": (
                    probabilities[:, 2]
                ),

                "correct": (
                    y_true == y_pred
                ),
            }
        )

        # Texto original
        oof["resp_text"] = (
            train_df.loc[
                eval_indices,
                "resp_text",
            ]
            .fillna("")
            .astype(str)
            .to_numpy()
        )

        # Comprimentos simples
        oof["char_count"] = (
            oof["resp_text"]
            .str.len()
        )

        oof["word_count"] = (
            oof["resp_text"]
            .str.split()
            .str.len()
        )

        # --------------------------------------------
        # Metadados das duplicatas
        # APENAS para análise posterior
        # --------------------------------------------

        colunas_metadata = [
            "duplicate_group_size",
            "duplicate_num_labels",
            "is_duplicate",
            "is_conflicting_duplicate",
            "group_c1_count",
            "group_c234_count",
            "group_c5_count",
        ]

        for coluna in colunas_metadata:

            oof[coluna] = (
                metadata_df.loc[
                    eval_indices,
                    coluna,
                ]
                .to_numpy()
            )

        # Configuração do experimento
        oof["model_name"] = (
            MODEL_NAME
        )

        oof["max_length"] = (
            MAX_LENGTH
        )

        oof["learning_rate"] = (
            LEARNING_RATE
        )

        oof["epochs"] = (
            EPOCHS
        )

        oof[
            "effective_batch_size"
        ] = (
            TRAIN_BATCH_SIZE
            * GRADIENT_ACCUMULATION_STEPS
        )

        # --------------------------------------------
        # Valida antes de salvar
        # --------------------------------------------

        if len(oof) != len(
            eval_indices
        ):
            raise RuntimeError(
                "Número incorreto de "
                "previsões OOF."
            )

        if oof["row_id"].duplicated().any():
            raise RuntimeError(
                "row_id duplicado no OOF "
                f"do fold {fold}."
            )

        OOF_FOLDS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        oof.to_csv(
            caminho_fold(fold),
            index=False,
        )

        print()
        print("-" * 70)

        print(
            f"Fold {fold} concluído"
        )

        print("-" * 70)

        print(
            f"Accuracy: {accuracy:.6f}"
        )

        print(
            f"Macro-F1: {macro_f1:.6f}"
        )

        print(
            f"Tempo: {tempo / 60:.2f} min"
        )

        print(
            f"Pico VRAM: "
            f"{pico_vram:.2f} GB"
        )

        return {
            "fold": fold,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "eval_size": len(
                eval_indices
            ),
            "time_minutes": (
                tempo / 60
            ),
            "peak_vram_gb": (
                pico_vram
            ),
            "status": "ok",
        }

    finally:

        del trainer
        del model

        gc.collect()
        torch.cuda.empty_cache()


# ============================================================
# CONSOLIDAÇÃO FINAL
# ============================================================

def consolidar_oof(
    total_instancias: int,
) -> pd.DataFrame:

    arquivos = [
        caminho_fold(fold)
        for fold in FOLDS
    ]

    ausentes = [
        str(path)
        for path in arquivos
        if not path.exists()
    ]

    if ausentes:
        raise RuntimeError(
            "Arquivos OOF ausentes: "
            f"{ausentes}"
        )

    dfs = [
        pd.read_csv(path)
        for path in arquivos
    ]

    oof = pd.concat(
        dfs,
        ignore_index=True,
    )

    oof = (
        oof
        .sort_values("row_id")
        .reset_index(drop=True)
    )

    if len(oof) != total_instancias:
        raise RuntimeError(
            "OOF consolidado possui "
            f"{len(oof)} linhas, mas "
            f"eram esperadas "
            f"{total_instancias}."
        )

    if oof["row_id"].duplicated().any():
        raise RuntimeError(
            "Existem row_ids duplicados "
            "no OOF consolidado."
        )

    esperado = set(
        range(total_instancias)
    )

    encontrado = set(
        oof["row_id"]
        .astype(int)
        .tolist()
    )

    if encontrado != esperado:
        raise RuntimeError(
            "OOF não contém exatamente "
            "todas as instâncias."
        )

    oof.to_csv(
        OOF_PATH,
        index=False,
    )

    return oof


# ============================================================
# RESUMO
# ============================================================

def gerar_resumo(
    oof: pd.DataFrame,
) -> None:

    accuracy, macro_f1 = (
        calcular_metricas(
            oof["y_true_id"]
            .to_numpy(),
            oof["y_pred_id"]
            .to_numpy(),
        )
    )

    resumo = pd.DataFrame(
        [
            {
                "model": (
                    "BERTimbau"
                ),
                "max_length": (
                    MAX_LENGTH
                ),
                "learning_rate": (
                    LEARNING_RATE
                ),
                "epochs": (
                    EPOCHS
                ),
                "n_instances": (
                    len(oof)
                ),
                "accuracy_oof": (
                    accuracy
                ),
                "macro_f1_oof": (
                    macro_f1
                ),
                "n_duplicates": int(
                    oof[
                        "is_duplicate"
                    ].sum()
                ),
                "n_conflicting_duplicates": int(
                    oof[
                        "is_conflicting_duplicate"
                    ].sum()
                ),
            }
        ]
    )

    resumo.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print()
    print("=" * 70)
    print("OOF CONSOLIDADO")
    print("=" * 70)

    print(
        f"Instâncias: {len(oof)}"
    )

    print(
        f"Accuracy OOF: "
        f"{accuracy:.6f}"
    )

    print(
        f"Macro-F1 OOF: "
        f"{macro_f1:.6f}"
    )

    print()
    print(
        "Duplicatas (ocorrências): "
        f"{int(oof['is_duplicate'].sum())}"
    )

    print(
        "Ocorrências em grupos "
        "com conflito de rótulo: "
        f"{int(oof['is_conflicting_duplicate'].sum())}"
    )

    print()
    print(
        f"OOF salvo em:\n{OOF_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA não está disponível."
        )

    print("=" * 70)

    print(
        "BERTimbau — geração de OOF"
    )

    print("=" * 70)

    print(
        "GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        "Configuração congelada:"
    )

    print(
        f"max_length = {MAX_LENGTH}"
    )

    print(
        f"learning_rate = "
        f"{LEARNING_RATE}"
    )

    print(
        f"epochs = {EPOCHS}"
    )

    print(
        "batch efetivo = "
        f"{TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
    )

    train_df, folds_df = (
        carregar_dados()
    )

    metadata_df = (
        criar_metadados_duplicatas(
            train_df
        )
    )

    textos = (
        train_df["resp_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    labels_series = (
        train_df["clarity"]
        .map(LABEL2ID)
    )

    if labels_series.isna().any():

        invalidos = (
            train_df.loc[
                labels_series.isna(),
                "clarity",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Rótulos desconhecidos: "
            f"{invalidos}"
        )

    labels = (
        labels_series
        .astype(np.int64)
        .to_numpy()
    )

    fold_values = (
        folds_df["fold"]
        .astype(int)
        .to_numpy()
    )

    encodings = tokenizar_corpus(
        textos
    )

    resultados = []

    for fold in FOLDS:

        resultado = executar_fold(
            fold=fold,
            train_df=train_df,
            metadata_df=metadata_df,
            encodings=encodings,
            labels=labels,
            fold_values=fold_values,
        )

        resultados.append(
            resultado
        )

    resumo_folds = pd.DataFrame(
        resultados
    )

    print()
    print("=" * 70)
    print("RESULTADOS POR FOLD")
    print("=" * 70)

    print(
        resumo_folds.to_string(
            index=False
        )
    )

    oof = consolidar_oof(
        total_instancias=len(
            train_df
        )
    )

    gerar_resumo(oof)

    print()
    print("=" * 70)
    print("EXECUÇÃO FINALIZADA")
    print("=" * 70)


if __name__ == "__main__":
    main()