from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
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

RESULTS_PATH = (
    ROOT_DIR
    / "results"
    / "bertimbau_duplicate_soft_labels_256_cv.csv"
)

HISTORY_PATH = (
    ROOT_DIR
    / "results"
    / "bertimbau_duplicate_soft_labels_256_history.csv"
)

OOF_FOLDS_DIR = (
    ROOT_DIR
    / "results"
    / "bertimbau_duplicate_soft_labels_256_oof_folds"
)

OOF_PATH = (
    ROOT_DIR
    / "results"
    / "bertimbau_duplicate_soft_labels_256_oof.csv"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "output"
    / "bertimbau_duplicate_soft_labels_256"
)


# ============================================================
# MODELO / CONFIGURAÇÃO CONGELADA
#
# O objetivo deste experimento é alterar SOMENTE o alvo de
# treinamento nos grupos com o mesmo texto e rótulos
# conflitantes.
#
# Para textos únicos e duplicatas consistentes, o alvo continua
# sendo one-hot, portanto a loss é equivalente à CrossEntropy
# tradicional.
#
# Para duplicatas conflitantes, o alvo é a distribuição empírica
# dos rótulos observados NO TREINO DO FOLD.
# ============================================================

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

LABELS = ["c1", "c234", "c5"]

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

EXPERIMENT_ID = "duplicate_soft_labels_v1"

MAX_LENGTH = 256

LEARNING_RATE = 3e-5
EPOCHS = 2

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2

WEIGHT_DECAY = 0.01

FP16 = True
GRADIENT_CHECKPOINTING = False

FOLDS = [0, 1, 2, 3, 4]

SEED = 42


# ============================================================
# REFERÊNCIA: BERTimbau 256 SEM SOFT LABELS
# ============================================================

REFERENCE_ACCURACY = 0.462922
REFERENCE_MACRO_F1 = 0.458239


# ============================================================
# DATASET
# ============================================================

class SoftLabelDataset(Dataset):
    def __init__(
        self,
        encodings: dict[str, torch.Tensor],
        targets: np.ndarray,
        indices: np.ndarray,
    ):
        self.encodings = encodings
        self.targets = targets
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
            self.targets[original_idx],
            dtype=torch.float32,
        )

        return item


# ============================================================
# TRAINER COM CROSS-ENTROPY PARA SOFT LABELS
# ============================================================

class SoftLabelTrainer(Trainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        targets = inputs.pop("labels").float()

        outputs = model(**inputs)
        logits = outputs.logits

        # Calcula a loss em float32 por estabilidade numérica,
        # mesmo quando o forward está em FP16.
        log_probs = F.log_softmax(
            logits.float(),
            dim=-1,
        )

        loss_per_example = -(
            targets * log_probs
        ).sum(dim=-1)

        loss = loss_per_example.mean()

        if return_outputs:
            return loss, outputs

        return loss


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

    required = {
        "resp_text",
        "clarity",
    }

    missing = (
        required
        - set(train_df.columns)
    )

    if missing:
        raise ValueError(
            "Colunas ausentes em train.xlsx: "
            f"{missing}"
        )

    if "fold" not in folds_df.columns:
        raise ValueError(
            "Coluna 'fold' não encontrada "
            "em folds.csv."
        )

    unknown_labels = (
        set(train_df["clarity"].dropna().unique())
        - set(LABELS)
    )

    if unknown_labels:
        raise ValueError(
            "Rótulos desconhecidos: "
            f"{sorted(unknown_labels)}"
        )

    print(
        f"Instâncias: {len(train_df)}"
    )

    print(
        "Folds encontrados: "
        f"{sorted(folds_df['fold'].astype(int).unique())}"
    )

    return train_df, folds_df


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

    print(f"Modelo: {MODEL_NAME}")
    print(f"max_length: {MAX_LENGTH}")

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
# SOFT LABELS POR GRUPO
# ============================================================

def construir_targets_do_fold(
    train_df: pd.DataFrame,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
) -> tuple[
    np.ndarray,
    dict,
]:
    """
    Cria os targets usados neste fold.

    TREINO:
        Para cada resp_text, calcula a distribuição empírica de
        c1/c234/c5 usando EXCLUSIVAMENTE as linhas de treino do
        fold.

        Exemplo:
            texto X -> c1, c234, c234, c5

        target:
            [0.25, 0.50, 0.25]

        Textos únicos ou grupos consistentes continuam one-hot.

    VALIDAÇÃO:
        Sempre usa o rótulo verdadeiro individual em one-hot.
        Nenhuma informação da validação é utilizada para formar
        os targets de treino.
    """

    n = len(train_df)

    targets = np.zeros(
        (n, len(LABELS)),
        dtype=np.float32,
    )

    train_subset = (
        train_df.loc[
            train_indices,
            ["resp_text", "clarity"],
        ]
        .copy()
    )

    train_subset["_text_key"] = (
        train_subset["resp_text"]
        .fillna("")
        .astype(str)
    )

    counts = pd.crosstab(
        train_subset["_text_key"],
        train_subset["clarity"],
    ).reindex(
        columns=LABELS,
        fill_value=0,
    )

    probs = counts.div(
        counts.sum(axis=1),
        axis=0,
    )

    # Aplica o mesmo soft target a todas as ocorrências do mesmo
    # texto dentro do conjunto de treino.
    for class_id, label in enumerate(LABELS):
        mapping = probs[label].to_dict()

        targets[
            train_indices,
            class_id,
        ] = (
            train_subset["_text_key"]
            .map(mapping)
            .to_numpy(
                dtype=np.float32
            )
        )

    # Validação: alvo individual one-hot.
    eval_labels = (
        train_df.loc[
            eval_indices,
            "clarity",
        ]
        .map(LABEL2ID)
        .to_numpy(dtype=np.int64)
    )

    targets[
        eval_indices,
        :,
    ] = 0.0

    targets[
        eval_indices,
        eval_labels,
    ] = 1.0

    # Estatísticas do tratamento no TREINO deste fold.
    num_labels_per_group = (
        (counts > 0)
        .sum(axis=1)
    )

    conflicting_groups = counts[
        num_labels_per_group > 1
    ]

    group_sizes = counts.sum(
        axis=1
    )

    conflicting_group_sizes = (
        conflicting_groups.sum(
            axis=1
        )
    )

    n_conflicting_occurrences = int(
        conflicting_group_sizes.sum()
    )

    n_conflicting_groups = int(
        len(conflicting_groups)
    )

    n_duplicate_groups = int(
        (group_sizes > 1).sum()
    )

    n_unique_groups = int(
        len(counts)
    )

    # Confirma que todos os targets de treino somam 1.
    train_target_sums = (
        targets[
            train_indices
        ]
        .sum(axis=1)
    )

    if not np.allclose(
        train_target_sums,
        1.0,
        atol=1e-6,
    ):
        raise RuntimeError(
            "Há targets de treino que "
            "não somam 1."
        )

    stats = {
        "train_unique_text_groups": (
            n_unique_groups
        ),
        "train_duplicate_groups": (
            n_duplicate_groups
        ),
        "train_conflicting_groups": (
            n_conflicting_groups
        ),
        "train_conflicting_occurrences": (
            n_conflicting_occurrences
        ),
    }

    return targets, stats


# ============================================================
# MÉTRICAS
# ============================================================

def compute_metrics(
    eval_pred: EvalPrediction,
) -> dict[str, float]:
    logits = eval_pred.predictions

    if isinstance(
        logits,
        tuple,
    ):
        logits = logits[0]

    soft_targets = (
        eval_pred.label_ids
    )

    y_true = np.argmax(
        soft_targets,
        axis=1,
    )

    y_pred = np.argmax(
        logits,
        axis=1,
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
    )

    return {
        "accuracy": float(
            accuracy
        ),
        "macro_f1": float(
            macro_f1
        ),
    }


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


# ============================================================
# CSV / RESUME
# ============================================================

def carregar_resultados_existentes() -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(
        RESULTS_PATH
    )

    print()
    print(
        "Resultados existentes: "
        f"{len(df)} linha(s)"
    )

    return df


def fold_ja_concluido(
    resultados: pd.DataFrame,
    fold: int,
) -> bool:
    if resultados.empty:
        return False

    required = {
        "experiment_id",
        "fold",
        "status",
    }

    if not required.issubset(
        resultados.columns
    ):
        return False

    mask = (
        (
            resultados[
                "experiment_id"
            ]
            == EXPERIMENT_ID
        )
        & (
            resultados[
                "fold"
            ]
            == fold
        )
        & (
            resultados[
                "status"
            ]
            == "ok"
        )
    )

    return bool(
        mask.any()
    )


def salvar_resultado(
    resultado: dict,
) -> None:
    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    novo = pd.DataFrame(
        [resultado]
    )

    if RESULTS_PATH.exists():
        novo.to_csv(
            RESULTS_PATH,
            mode="a",
            header=False,
            index=False,
        )
    else:
        novo.to_csv(
            RESULTS_PATH,
            index=False,
        )


def salvar_historico(
    registros: list[dict],
) -> None:
    if not registros:
        return

    HISTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    novo = pd.DataFrame(
        registros
    )

    if HISTORY_PATH.exists():
        novo.to_csv(
            HISTORY_PATH,
            mode="a",
            header=False,
            index=False,
        )
    else:
        novo.to_csv(
            HISTORY_PATH,
            index=False,
        )


# ============================================================
# METADADOS DE DUPLICATAS PARA O OOF
#
# Somente diagnóstico. Não entram no modelo.
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

    group_size = (
        df.groupby(
            "_text_key"
        )["_text_key"]
        .transform("size")
    )

    group_num_labels = (
        df.groupby(
            "_text_key"
        )["clarity"]
        .transform("nunique")
    )

    df[
        "duplicate_group_size"
    ] = group_size

    df[
        "duplicate_num_labels"
    ] = group_num_labels

    df[
        "is_duplicate"
    ] = (
        group_size > 1
    )

    df[
        "is_conflicting_duplicate"
    ] = (
        (group_size > 1)
        & (group_num_labels > 1)
    )

    return df[
        [
            "duplicate_group_size",
            "duplicate_num_labels",
            "is_duplicate",
            "is_conflicting_duplicate",
        ]
    ].copy()


# ============================================================
# HISTÓRICO
# ============================================================

def extrair_historico(
    trainer: Trainer,
    fold: int,
) -> list[dict]:
    registros = []

    for item in (
        trainer.state.log_history
    ):
        if (
            "eval_accuracy"
            not in item
        ):
            continue

        registros.append(
            {
                "experiment_id": (
                    EXPERIMENT_ID
                ),
                "fold": fold,
                "epoch": item.get(
                    "epoch"
                ),
                "eval_loss": item.get(
                    "eval_loss"
                ),
                "accuracy": item.get(
                    "eval_accuracy"
                ),
                "macro_f1": item.get(
                    "eval_macro_f1"
                ),
            }
        )

    return registros


# ============================================================
# TREINO DE UM FOLD
# ============================================================

def executar_fold(
    fold: int,
    train_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    encodings: dict[str, torch.Tensor],
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

    targets, target_stats = (
        construir_targets_do_fold(
            train_df=train_df,
            train_indices=train_indices,
            eval_indices=eval_indices,
        )
    )

    train_dataset = (
        SoftLabelDataset(
            encodings=encodings,
            targets=targets,
            indices=train_indices,
        )
    )

    eval_dataset = (
        SoftLabelDataset(
            encodings=encodings,
            targets=targets,
            indices=eval_indices,
        )
    )

    print(
        f"Treino: {len(train_dataset)}"
    )

    print(
        f"Validação: {len(eval_dataset)}"
    )

    print(
        "Grupos conflitantes no treino: "
        f"{target_stats['train_conflicting_groups']}"
    )

    print(
        "Ocorrências conflitantes no treino: "
        f"{target_stats['train_conflicting_occurrences']}"
    )

    print(
        f"max_length: {MAX_LENGTH}"
    )

    print(
        f"LR: {LEARNING_RATE}"
    )

    print(
        f"epochs: {EPOCHS}"
    )

    print(
        "batch efetivo: "
        f"{TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=len(LABELS),
            label2id=LABEL2ID,
            id2label=ID2LABEL,
        )
    )

    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()

    fold_output_dir = (
        OUTPUT_DIR
        / f"fold_{fold}"
    )

    fold_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_args = (
        TrainingArguments(
            output_dir=str(
                fold_output_dir
            ),

            learning_rate=(
                LEARNING_RATE
            ),

            num_train_epochs=(
                EPOCHS
            ),

            weight_decay=(
                WEIGHT_DECAY
            ),

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
    )

    trainer = (
        SoftLabelTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=(
                compute_metrics
            ),
        )
    )

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    inicio = time.perf_counter()

    try:
        trainer.train()

        # Avaliação final explícita.
        eval_result = (
            trainer.evaluate(
                eval_dataset
            )
        )

        # OOF do fold.
        pred_output = (
            trainer.predict(
                eval_dataset
            )
        )

        logits = (
            pred_output.predictions
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

        y_true = np.argmax(
            targets[
                eval_indices
            ],
            axis=1,
        )

        y_pred = np.argmax(
            probabilities,
            axis=1,
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

        accuracy = float(
            eval_result[
                "eval_accuracy"
            ]
        )

        macro_f1 = float(
            eval_result[
                "eval_macro_f1"
            ]
        )

        eval_loss = float(
            eval_result[
                "eval_loss"
            ]
        )

        # --------------------------------------------
        # Histórico por época
        # --------------------------------------------

        salvar_historico(
            extrair_historico(
                trainer,
                fold,
            )
        )

        # --------------------------------------------
        # OOF do fold
        # --------------------------------------------

        oof = pd.DataFrame(
            {
                "row_id": (
                    eval_indices
                ),

                "fold": fold,

                "y_true_id": (
                    y_true
                ),

                "y_pred_id": (
                    y_pred
                ),

                "y_true": [
                    ID2LABEL[
                        int(x)
                    ]
                    for x in y_true
                ],

                "y_pred": [
                    ID2LABEL[
                        int(x)
                    ]
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

        oof["resp_text"] = (
            train_df.loc[
                eval_indices,
                "resp_text",
            ]
            .fillna("")
            .astype(str)
            .to_numpy()
        )

        for col in (
            metadata_df.columns
        ):
            oof[col] = (
                metadata_df.loc[
                    eval_indices,
                    col,
                ]
                .to_numpy()
            )

        oof[
            "experiment_id"
        ] = EXPERIMENT_ID

        oof[
            "max_length"
        ] = MAX_LENGTH

        OOF_FOLDS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        oof.to_csv(
            OOF_FOLDS_DIR
            / f"fold_{fold}.csv",
            index=False,
        )

        resultado = {
            "experiment_id": (
                EXPERIMENT_ID
            ),

            "method": (
                "empirical_group_soft_labels"
            ),

            "duplicate_weighting": (
                "none"
            ),

            "max_length": (
                MAX_LENGTH
            ),

            "learning_rate": (
                LEARNING_RATE
            ),

            "epochs": EPOCHS,

            "fold": fold,

            "train_size": (
                len(train_dataset)
            ),

            "eval_size": (
                len(eval_dataset)
            ),

            "train_batch_size": (
                TRAIN_BATCH_SIZE
            ),

            "eval_batch_size": (
                EVAL_BATCH_SIZE
            ),

            "gradient_accumulation_steps": (
                GRADIENT_ACCUMULATION_STEPS
            ),

            "effective_batch_size": (
                TRAIN_BATCH_SIZE
                * GRADIENT_ACCUMULATION_STEPS
            ),

            "weight_decay": (
                WEIGHT_DECAY
            ),

            "fp16": FP16,

            "gradient_checkpointing": (
                GRADIENT_CHECKPOINTING
            ),

            **target_stats,

            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "eval_loss": eval_loss,

            "time_minutes": (
                tempo / 60
            ),

            "peak_vram_gb": (
                pico_vram
            ),

            "status": "ok",
            "error": "",
        }

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
            f"Eval loss: {eval_loss:.6f}"
        )

        print(
            f"Tempo: {tempo / 60:.2f} min"
        )

        print(
            f"Pico VRAM: {pico_vram:.2f} GB"
        )

        return resultado

    except Exception as exc:
        tempo = (
            time.perf_counter()
            - inicio
        )

        return {
            "experiment_id": (
                EXPERIMENT_ID
            ),

            "method": (
                "empirical_group_soft_labels"
            ),

            "duplicate_weighting": (
                "none"
            ),

            "max_length": (
                MAX_LENGTH
            ),

            "learning_rate": (
                LEARNING_RATE
            ),

            "epochs": EPOCHS,

            "fold": fold,

            "train_size": (
                len(train_dataset)
            ),

            "eval_size": (
                len(eval_dataset)
            ),

            "train_batch_size": (
                TRAIN_BATCH_SIZE
            ),

            "eval_batch_size": (
                EVAL_BATCH_SIZE
            ),

            "gradient_accumulation_steps": (
                GRADIENT_ACCUMULATION_STEPS
            ),

            "effective_batch_size": (
                TRAIN_BATCH_SIZE
                * GRADIENT_ACCUMULATION_STEPS
            ),

            "weight_decay": (
                WEIGHT_DECAY
            ),

            "fp16": FP16,

            "gradient_checkpointing": (
                GRADIENT_CHECKPOINTING
            ),

            **target_stats,

            "accuracy": np.nan,
            "macro_f1": np.nan,
            "eval_loss": np.nan,

            "time_minutes": (
                tempo / 60
            ),

            "peak_vram_gb": np.nan,

            "status": "error",
            "error": repr(exc),
        }

    finally:
        del trainer
        del model

        gc.collect()
        torch.cuda.empty_cache()


# ============================================================
# CONSOLIDA OOF
# ============================================================

def consolidar_oof(
    total_instancias: int,
) -> None:
    paths = [
        OOF_FOLDS_DIR
        / f"fold_{fold}.csv"
        for fold in FOLDS
    ]

    if not all(
        path.exists()
        for path in paths
    ):
        return

    dfs = [
        pd.read_csv(path)
        for path in paths
    ]

    oof = pd.concat(
        dfs,
        ignore_index=True,
    )

    oof = (
        oof
        .sort_values(
            "row_id"
        )
        .reset_index(
            drop=True
        )
    )

    if len(oof) != total_instancias:
        raise RuntimeError(
            "OOF consolidado não possui "
            "o número esperado de linhas."
        )

    if (
        oof["row_id"]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "OOF consolidado possui "
            "row_id duplicado."
        )

    oof.to_csv(
        OOF_PATH,
        index=False,
    )


# ============================================================
# RESUMO FINAL
# ============================================================

def mostrar_resumo() -> None:
    if not RESULTS_PATH.exists():
        return

    df = pd.read_csv(
        RESULTS_PATH
    )

    ok = df[
        (
            df["experiment_id"]
            == EXPERIMENT_ID
        )
        & (
            df["status"]
            == "ok"
        )
    ].copy()

    if len(ok) != len(FOLDS):
        print()
        print(
            "Ainda não há cinco folds "
            "concluídos."
        )
        return

    acc_mean = float(
        ok["accuracy"].mean()
    )

    acc_std = float(
        ok["accuracy"].std(
            ddof=1
        )
    )

    f1_mean = float(
        ok["macro_f1"].mean()
    )

    f1_std = float(
        ok["macro_f1"].std(
            ddof=1
        )
    )

    print()
    print("=" * 70)
    print("RESULTADO CONSOLIDADO")
    print("=" * 70)

    print(
        "Duplicate-aware soft labels"
    )

    print(
        f"Accuracy = "
        f"{acc_mean:.4f} ± {acc_std:.4f}"
    )

    print(
        f"Macro-F1 = "
        f"{f1_mean:.4f} ± {f1_std:.4f}"
    )

    print()
    print(
        "Referência BERTimbau 256:"
    )

    print(
        f"Accuracy = "
        f"{REFERENCE_ACCURACY:.4f}"
    )

    print(
        f"Macro-F1 = "
        f"{REFERENCE_MACRO_F1:.4f}"
    )

    print()
    print(
        "Delta Accuracy = "
        f"{acc_mean - REFERENCE_ACCURACY:+.4f}"
    )

    print(
        "Delta Macro-F1 = "
        f"{f1_mean - REFERENCE_MACRO_F1:+.4f}"
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
        "BERTimbau — Duplicate-aware Soft Labels"
    )
    print("=" * 70)

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"Experimento: {EXPERIMENT_ID}"
    )

    print(
        f"max_length: {MAX_LENGTH}"
    )

    print(
        f"LR: {LEARNING_RATE}"
    )

    print(
        f"epochs: {EPOCHS}"
    )

    print(
        "batch efetivo: "
        f"{TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
    )

    train_df, folds_df = (
        carregar_dados()
    )

    textos = (
        train_df["resp_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    fold_values = (
        folds_df["fold"]
        .astype(int)
        .to_numpy()
    )

    metadata_df = (
        criar_metadados_duplicatas(
            train_df
        )
    )

    encodings = (
        tokenizar_corpus(
            textos
        )
    )

    resultados_existentes = (
        carregar_resultados_existentes()
    )

    for fold in FOLDS:
        if fold_ja_concluido(
            resultados_existentes,
            fold,
        ):
            print()
            print(
                f"Fold {fold} já concluído. "
                "Pulando."
            )
            continue

        resultado = (
            executar_fold(
                fold=fold,
                train_df=train_df,
                metadata_df=metadata_df,
                encodings=encodings,
                fold_values=fold_values,
            )
        )

        salvar_resultado(
            resultado
        )

        if (
            resultado["status"]
            != "ok"
        ):
            raise RuntimeError(
                f"Falha no fold {fold}: "
                f"{resultado['error']}"
            )

        resultados_existentes = (
            carregar_resultados_existentes()
        )

    consolidar_oof(
        total_instancias=len(
            train_df
        )
    )

    mostrar_resumo()

    print()
    print("=" * 70)
    print("EXECUÇÃO FINALIZADA")
    print("=" * 70)


if __name__ == "__main__":
    main()
