from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.modeling_outputs import SequenceClassifierOutput


# ============================================================
# CAMINHOS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT_DIR / "data" / "raw" / "train.xlsx"
FOLDS_PATH = ROOT_DIR / "data" / "splits" / "folds.csv"

RESULTS_PATH = (
    ROOT_DIR
    / "results"
    / "bertimbau_ordinal_256_cv.csv"
)

HISTORY_PATH = (
    ROOT_DIR
    / "results"
    / "bertimbau_ordinal_256_history.csv"
)

OOF_FOLDS_DIR = (
    ROOT_DIR
    / "results"
    / "bertimbau_ordinal_256_oof_folds"
)

OOF_PATH = (
    ROOT_DIR
    / "results"
    / "bertimbau_ordinal_256_oof.csv"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "output"
    / "bertimbau_ordinal_256"
)


# ============================================================
# CONFIGURAÇÃO EXPERIMENTAL
#
# Mantemos a configuração vencedora do BERTimbau em 256.
# Alteramos SOMENTE a formulação da tarefa:
#
# c1   -> [0, 0]
# c234 -> [1, 0]
# c5   -> [1, 1]
#
# Os dois logits representam:
#   P(y > c1)
#   P(y > c234)
#
# A cabeça usa um escore escalar + dois thresholds ordenados.
# Isso garante monotonicidade:
#   P(y > c1) >= P(y > c234)
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

EXPERIMENT_ID = "ordinal_thresholds_v1"

MAX_LENGTH = 256

LEARNING_RATE = 3e-5
EPOCHS = 2

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2

WEIGHT_DECAY = 0.01

FP16 = True

FOLDS = [0, 1, 2, 3, 4]

SEED = 42


# ============================================================
# REFERÊNCIA BERTIMBAU 256 MULTICLASS
# ============================================================

REFERENCE_ACCURACY = 0.462922
REFERENCE_MACRO_F1 = 0.458239


# ============================================================
# DATASET
# ============================================================

class OrdinalDataset(Dataset):
    def __init__(
        self,
        encodings: dict[str, torch.Tensor],
        ordinal_targets: np.ndarray,
        indices: np.ndarray,
    ):
        self.encodings = encodings
        self.ordinal_targets = ordinal_targets
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
            self.ordinal_targets[original_idx],
            dtype=torch.float32,
        )

        return item


# ============================================================
# MODELO ORDINAL
# ============================================================

class BertOrdinalClassifier(nn.Module):
    """
    BERT + escore ordinal escalar + dois thresholds ordenados.

    Se score = s e thresholds = t1 < t2:

        logit_1 = s - t1  -> P(y > c1)
        logit_2 = s - t2  -> P(y > c234)

    Como t1 < t2:
        logit_1 >= logit_2
    e portanto:
        sigmoid(logit_1) >= sigmoid(logit_2)

    Isso mantém consistência ordinal.
    """

    def __init__(
        self,
        model_name: str,
    ):
        super().__init__()

        self.config = AutoConfig.from_pretrained(
            model_name
        )

        self.bert = AutoModel.from_pretrained(
            model_name
        )

        hidden_size = (
            self.config.hidden_size
        )

        dropout_prob = getattr(
            self.config,
            "classifier_dropout",
            None,
        )

        if dropout_prob is None:
            dropout_prob = getattr(
                self.config,
                "hidden_dropout_prob",
                0.1,
            )

        self.dropout = nn.Dropout(
            dropout_prob
        )

        # Um escore latente de "clareza".
        self.score = nn.Linear(
            hidden_size,
            1,
        )

        # t1 livre.
        self.threshold_1 = nn.Parameter(
            torch.tensor(
                -0.5,
                dtype=torch.float32,
            )
        )

        # t2 = t1 + softplus(delta)
        # => t2 > t1 sempre.
        self.threshold_delta = nn.Parameter(
            torch.tensor(
                1.0,
                dtype=torch.float32,
            )
        )

    def get_thresholds(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t1 = self.threshold_1

        t2 = (
            t1
            + F.softplus(
                self.threshold_delta
            )
        )

        return t1, t2

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        if (
            getattr(
                outputs,
                "pooler_output",
                None,
            )
            is not None
        ):
            pooled = (
                outputs.pooler_output
            )
        else:
            pooled = (
                outputs.last_hidden_state[
                    :,
                    0,
                    :,
                ]
            )

        pooled = self.dropout(
            pooled
        )

        score = (
            self.score(
                pooled
            )
            .squeeze(-1)
        )

        t1, t2 = (
            self.get_thresholds()
        )

        logits = torch.stack(
            [
                score - t1,
                score - t2,
            ],
            dim=-1,
        )

        loss = None

        if labels is not None:
            loss = (
                F.binary_cross_entropy_with_logits(
                    logits.float(),
                    labels.float(),
                    reduction="mean",
                )
            )

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=getattr(
                outputs,
                "hidden_states",
                None,
            ),
            attentions=getattr(
                outputs,
                "attentions",
                None,
            ),
        )


# ============================================================
# TARGETS ORDINAIS
# ============================================================

def labels_para_targets_ordinais(
    class_ids: np.ndarray,
) -> np.ndarray:
    """
    class 0 (c1)   -> [0, 0]
    class 1 (c234) -> [1, 0]
    class 2 (c5)   -> [1, 1]
    """
    targets = np.zeros(
        (
            len(class_ids),
            2,
        ),
        dtype=np.float32,
    )

    targets[:, 0] = (
        class_ids > 0
    ).astype(np.float32)

    targets[:, 1] = (
        class_ids > 1
    ).astype(np.float32)

    return targets


def targets_ordinais_para_classes(
    targets: np.ndarray,
) -> np.ndarray:
    return (
        targets
        .sum(axis=1)
        .astype(np.int64)
    )


# ============================================================
# PROBABILIDADES DE CLASSE
# ============================================================

def ordinal_logits_para_probabilidades(
    logits: np.ndarray,
) -> np.ndarray:
    """
    q1 = P(y > c1)
    q2 = P(y > c234)

    Como q1 >= q2:
        P(c1)   = 1 - q1
        P(c234) = q1 - q2
        P(c5)   = q2
    """

    logits_tensor = torch.tensor(
        logits,
        dtype=torch.float32,
    )

    q = torch.sigmoid(
        logits_tensor
    ).numpy()

    q1 = q[:, 0]
    q2 = q[:, 1]

    probs = np.stack(
        [
            1.0 - q1,
            q1 - q2,
            q2,
        ],
        axis=1,
    )

    # Apenas proteção contra ruído numérico.
    probs = np.clip(
        probs,
        0.0,
        1.0,
    )

    probs = (
        probs
        / probs.sum(
            axis=1,
            keepdims=True,
        )
    )

    return probs


# ============================================================
# MÉTRICAS
# ============================================================

def compute_metrics(
    eval_pred: EvalPrediction,
) -> dict[str, float]:
    logits = (
        eval_pred.predictions
    )

    if isinstance(
        logits,
        tuple,
    ):
        logits = logits[0]

    ordinal_targets = (
        eval_pred.label_ids
    )

    y_true = (
        targets_ordinais_para_classes(
            ordinal_targets
        )
    )

    class_probs = (
        ordinal_logits_para_probabilidades(
            logits
        )
    )

    y_pred = np.argmax(
        class_probs,
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
            "Coluna 'fold' não encontrada."
        )

    unknown = (
        set(
            train_df[
                "clarity"
            ]
            .dropna()
            .unique()
        )
        - set(LABELS)
    )

    if unknown:
        raise ValueError(
            "Rótulos desconhecidos: "
            f"{sorted(unknown)}"
        )

    print(
        f"Instâncias: {len(train_df)}"
    )

    print(
        "Folds encontrados: "
        f"{sorted(folds_df['fold'].astype(int).unique())}"
    )

    return (
        train_df,
        folds_df,
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

    inicio = (
        time.perf_counter()
    )

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
# METADADOS DE DUPLICATAS
#
# Somente para diagnóstico do OOF.
# Não entram no treinamento.
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

    return pd.DataFrame(
        {
            "duplicate_group_size": (
                group_size
            ),
            "duplicate_num_labels": (
                group_num_labels
            ),
            "is_duplicate": (
                group_size > 1
            ),
            "is_conflicting_duplicate": (
                (group_size > 1)
                & (
                    group_num_labels > 1
                )
            ),
        }
    )


# ============================================================
# EXECUÇÃO DE UM FOLD
# ============================================================

def executar_fold(
    fold: int,
    train_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    encodings: dict[str, torch.Tensor],
    ordinal_targets: np.ndarray,
    fold_values: np.ndarray,
) -> dict:
    print()
    print("=" * 70)
    print(
        f"FOLD {fold}"
    )
    print("=" * 70)

    set_seed(SEED)

    train_indices = np.where(
        fold_values != fold
    )[0]

    eval_indices = np.where(
        fold_values == fold
    )[0]

    train_dataset = (
        OrdinalDataset(
            encodings=encodings,
            ordinal_targets=ordinal_targets,
            indices=train_indices,
        )
    )

    eval_dataset = (
        OrdinalDataset(
            encodings=encodings,
            ordinal_targets=ordinal_targets,
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
        BertOrdinalClassifier(
            MODEL_NAME
        )
    )

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    inicio = time.perf_counter()

    try:
        trainer.train()

        eval_result = (
            trainer.evaluate(
                eval_dataset
            )
        )

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

        class_probs = (
            ordinal_logits_para_probabilidades(
                logits
            )
        )

        y_true = (
            targets_ordinais_para_classes(
                ordinal_targets[
                    eval_indices
                ]
            )
        )

        y_pred = np.argmax(
            class_probs,
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

        t1, t2 = (
            model.get_thresholds()
        )

        t1_value = float(
            t1.detach()
            .cpu()
            .item()
        )

        t2_value = float(
            t2.detach()
            .cpu()
            .item()
        )

        # --------------------------------------------
        # Histórico
        # --------------------------------------------

        salvar_historico(
            extrair_historico(
                trainer,
                fold,
            )
        )

        # --------------------------------------------
        # OOF
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
                    class_probs[
                        :,
                        0,
                    ]
                ),

                "prob_c234": (
                    class_probs[
                        :,
                        1,
                    ]
                ),

                "prob_c5": (
                    class_probs[
                        :,
                        2,
                    ]
                ),

                "correct": (
                    y_true == y_pred
                ),
            }
        )

        oof[
            "resp_text"
        ] = (
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
            "threshold_1"
        ] = t1_value

        oof[
            "threshold_2"
        ] = t2_value

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
                "ordered_thresholds_bce"
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

            "threshold_1": (
                t1_value
            ),

            "threshold_2": (
                t2_value
            ),

            "accuracy": (
                accuracy
            ),

            "macro_f1": (
                macro_f1
            ),

            "eval_loss": (
                eval_loss
            ),

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
            f"Accuracy: "
            f"{accuracy:.6f}"
        )

        print(
            f"Macro-F1: "
            f"{macro_f1:.6f}"
        )

        print(
            f"Eval loss: "
            f"{eval_loss:.6f}"
        )

        print(
            "Thresholds: "
            f"{t1_value:.4f}, "
            f"{t2_value:.4f}"
        )

        print(
            f"Tempo: "
            f"{tempo / 60:.2f} min"
        )

        print(
            f"Pico VRAM: "
            f"{pico_vram:.2f} GB"
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
                "ordered_thresholds_bce"
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

            "threshold_1": np.nan,
            "threshold_2": np.nan,

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
# CONSOLIDAR OOF
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

    if (
        len(oof)
        != total_instancias
    ):
        raise RuntimeError(
            "OOF consolidado não possui "
            "o número esperado de linhas."
        )

    if (
        oof[
            "row_id"
        ]
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
# RESUMO
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
        "BERTimbau ordinal"
    )

    print(
        f"Accuracy = "
        f"{acc_mean:.4f} ± "
        f"{acc_std:.4f}"
    )

    print(
        f"Macro-F1 = "
        f"{f1_mean:.4f} ± "
        f"{f1_std:.4f}"
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
        "BERTimbau — Classificação Ordinal"
    )
    print("=" * 70)

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"Experimento: "
        f"{EXPERIMENT_ID}"
    )

    print(
        f"max_length: "
        f"{MAX_LENGTH}"
    )

    print(
        f"LR: "
        f"{LEARNING_RATE}"
    )

    print(
        f"epochs: "
        f"{EPOCHS}"
    )

    print(
        "batch efetivo: "
        f"{TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
    )

    train_df, folds_df = (
        carregar_dados()
    )

    textos = (
        train_df[
            "resp_text"
        ]
        .fillna("")
        .astype(str)
        .tolist()
    )

    class_ids = (
        train_df[
            "clarity"
        ]
        .map(
            LABEL2ID
        )
    )

    if (
        class_ids
        .isna()
        .any()
    ):
        raise ValueError(
            "Há rótulos inválidos."
        )

    class_ids = (
        class_ids
        .astype(
            np.int64
        )
        .to_numpy()
    )

    ordinal_targets = (
        labels_para_targets_ordinais(
            class_ids
        )
    )

    fold_values = (
        folds_df[
            "fold"
        ]
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
                ordinal_targets=(
                    ordinal_targets
                ),
                fold_values=(
                    fold_values
                ),
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
                f"Falha no fold "
                f"{fold}: "
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
    print(
        "EXECUÇÃO FINALIZADA"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
