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

RESULTS_PATH = (
    ROOT_DIR
    / "results"
    / "bertimbau_head_tail_256_cv.csv"
)

HISTORY_PATH = (
    ROOT_DIR
    / "results"
    / "bertimbau_head_tail_256_history.csv"
)

OOF_FOLDS_DIR = (
    ROOT_DIR
    / "results"
    / "bertimbau_head_tail_256_oof_folds"
)

OOF_PATH = (
    ROOT_DIR
    / "results"
    / "bertimbau_head_tail_256_oof.csv"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "output"
    / "bertimbau_head_tail_256"
)


# ============================================================
# CONFIGURAÇÃO
#
# Comparação direta contra BERTimbau 256 padrão.
#
# ÚNICA mudança experimental:
#
# padrão:
#     primeiros tokens até max_length
#
# head + tail:
#     início + final do documento,
#     mantendo o mesmo max_length total.
#
# Para BERT, max_length=256 inclui [CLS] e [SEP].
# Portanto, o orçamento de conteúdo será normalmente 254:
#
#     127 tokens do início
#     127 tokens do fim
#
# Textos que já cabem na janela não são modificados.
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

EXPERIMENT_ID = "head_tail_256_v1"

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
# REFERÊNCIAS
# ============================================================

REFERENCE_ACCURACY = 0.462922
REFERENCE_MACRO_F1 = 0.458239

BEST_512_ACCURACY = 0.464962
BEST_512_MACRO_F1 = 0.460948


# ============================================================
# DATASET
# ============================================================

class BertDataset(Dataset):
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
# DADOS
# ============================================================

def carregar_dados() -> tuple[pd.DataFrame, pd.DataFrame]:
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

    required = {"resp_text", "clarity"}

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
            "Coluna 'fold' não encontrada em folds.csv."
        )

    unknown = (
        set(train_df["clarity"].dropna().unique())
        - set(LABELS)
    )

    if unknown:
        raise ValueError(
            f"Rótulos desconhecidos: {sorted(unknown)}"
        )

    print(f"Instâncias: {len(train_df)}")
    print(
        "Folds encontrados: "
        f"{sorted(folds_df['fold'].astype(int).unique())}"
    )

    return train_df, folds_df


# ============================================================
# TOKENIZAÇÃO HEAD + TAIL
# ============================================================

def tokenizar_head_tail(
    textos: list[str],
) -> tuple[
    dict[str, torch.Tensor],
    np.ndarray,
    np.ndarray,
]:
    """
    Tokeniza primeiro SEM special tokens e SEM truncamento.

    Se o texto exceder o orçamento de conteúdo:
        pega metade do início + metade do final.

    Depois adiciona os special tokens do próprio tokenizer
    e faz padding até MAX_LENGTH.

    Retorna também:
        - número original de tokens por texto
        - indicador de aplicação de head+tail
    """

    print()
    print("=" * 70)
    print("Tokenização head + tail")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    n_special = tokenizer.num_special_tokens_to_add(
        pair=False
    )

    content_budget = (
        MAX_LENGTH
        - n_special
    )

    head_size = (
        content_budget // 2
    )

    tail_size = (
        content_budget
        - head_size
    )

    print(f"Modelo: {MODEL_NAME}")
    print(f"max_length: {MAX_LENGTH}")
    print(f"special tokens: {n_special}")
    print(f"orçamento de conteúdo: {content_budget}")
    print(f"head: {head_size}")
    print(f"tail: {tail_size}")

    inicio = time.perf_counter()

    # Tokenização completa, sem truncamento.
    raw = tokenizer(
        textos,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )

    original_lengths = np.fromiter(
        (
            len(ids)
            for ids in raw["input_ids"]
        ),
        dtype=np.int32,
        count=len(textos),
    )

    was_head_tail = (
        original_lengths
        > content_budget
    )

    all_input_ids = []
    all_attention_masks = []
    all_token_type_ids = []

    has_token_type_ids = (
        "token_type_ids"
        in tokenizer.model_input_names
    )

    for ids in raw["input_ids"]:
        if len(ids) > content_budget:
            selected_ids = (
                ids[:head_size]
                + ids[-tail_size:]
            )
        else:
            selected_ids = ids

        # Transformers 5.17.0 / BertTokenizer não expõe
        # tokenizer.prepare_for_model(). Como este experimento usa
        # apenas UMA sequência BERT, montamos a entrada manualmente:
        #
        #   [CLS] + selected_ids + [SEP] + padding
        #
        # Isso reproduz a estrutura esperada pelo BERT sem alterar
        # a hipótese experimental head + tail.

        cls_token_id = tokenizer.cls_token_id
        sep_token_id = tokenizer.sep_token_id
        pad_token_id = tokenizer.pad_token_id

        if cls_token_id is None:
            raise RuntimeError(
                "Tokenizer não possui cls_token_id."
            )

        if sep_token_id is None:
            raise RuntimeError(
                "Tokenizer não possui sep_token_id."
            )

        if pad_token_id is None:
            raise RuntimeError(
                "Tokenizer não possui pad_token_id."
            )

        input_ids = (
            [cls_token_id]
            + list(selected_ids)
            + [sep_token_id]
        )

        if len(input_ids) > MAX_LENGTH:
            raise RuntimeError(
                "Sequência head+tail excedeu MAX_LENGTH: "
                f"{len(input_ids)} > {MAX_LENGTH}."
            )

        attention_mask = [
            1
        ] * len(input_ids)

        padding_length = (
            MAX_LENGTH
            - len(input_ids)
        )

        if tokenizer.padding_side == "right":
            input_ids = (
                input_ids
                + [pad_token_id] * padding_length
            )

            attention_mask = (
                attention_mask
                + [0] * padding_length
            )

            if has_token_type_ids:
                token_type_ids = (
                    [0] * MAX_LENGTH
                )

        elif tokenizer.padding_side == "left":
            input_ids = (
                [pad_token_id] * padding_length
                + input_ids
            )

            attention_mask = (
                [0] * padding_length
                + attention_mask
            )

            if has_token_type_ids:
                token_type_ids = (
                    [0] * MAX_LENGTH
                )

        else:
            raise RuntimeError(
                "padding_side desconhecido: "
                f"{tokenizer.padding_side}"
            )

        if len(input_ids) != MAX_LENGTH:
            raise RuntimeError(
                "Tokenização produziu sequência "
                "com tamanho diferente de MAX_LENGTH."
            )

        if len(attention_mask) != MAX_LENGTH:
            raise RuntimeError(
                "attention_mask possui tamanho incorreto."
            )

        all_input_ids.append(
            input_ids
        )

        all_attention_masks.append(
            attention_mask
        )

        if has_token_type_ids:
            all_token_type_ids.append(
                token_type_ids
            )

    encodings = {
        "input_ids": torch.tensor(
            all_input_ids,
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            all_attention_masks,
            dtype=torch.long,
        ),
    }

    if has_token_type_ids:
        encodings["token_type_ids"] = (
            torch.tensor(
                all_token_type_ids,
                dtype=torch.long,
            )
        )

    tempo = (
        time.perf_counter()
        - inicio
    )

    n_head_tail = int(
        was_head_tail.sum()
    )

    print(
        f"Tokenização concluída em {tempo:.2f} s"
    )

    print(
        "Textos modificados por head+tail: "
        f"{n_head_tail}/{len(textos)} "
        f"({100 * n_head_tail / len(textos):.2f}%)"
    )

    print(
        "Textos que cabem integralmente: "
        f"{len(textos) - n_head_tail}"
    )

    return (
        encodings,
        original_lengths,
        was_head_tail,
    )


# ============================================================
# MÉTRICAS
# ============================================================

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
# RESUME
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
            resultados["fold"]
            == fold
        )
        & (
            resultados["status"]
            == "ok"
        )
    )

    return bool(mask.any())


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
    trainer: Trainer,
    fold: int,
) -> None:
    registros = []

    for item in trainer.state.log_history:
        if "eval_accuracy" not in item:
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
# METADADOS DE DUPLICATAS
# ============================================================

def criar_metadados_duplicatas(
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    text_key = (
        train_df["resp_text"]
        .fillna("")
        .astype(str)
    )

    temp = pd.DataFrame(
        {
            "text_key": text_key,
            "clarity": (
                train_df["clarity"]
            ),
        }
    )

    group_size = (
        temp.groupby(
            "text_key"
        )["text_key"]
        .transform("size")
    )

    group_num_labels = (
        temp.groupby(
            "text_key"
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
                & (group_num_labels > 1)
            ),
        }
    )


# ============================================================
# TREINO DE UM FOLD
# ============================================================

def executar_fold(
    fold: int,
    train_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    encodings: dict[str, torch.Tensor],
    labels: np.ndarray,
    fold_values: np.ndarray,
    original_lengths: np.ndarray,
    was_head_tail: np.ndarray,
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

    train_dataset = BertDataset(
        encodings=encodings,
        labels=labels,
        indices=train_indices,
    )

    eval_dataset = BertDataset(
        encodings=encodings,
        labels=labels,
        indices=eval_indices,
    )

    print(f"Treino: {len(train_dataset)}")
    print(f"Validação: {len(eval_dataset)}")
    print(f"max_length: {MAX_LENGTH}")
    print(f"LR: {LEARNING_RATE}")
    print(f"epochs: {EPOCHS}")
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

    # compute_metrics é necessário para as avaliações
    # automáticas ao fim de cada época.
    def compute_metrics(eval_pred):
        logits = eval_pred.predictions

        if isinstance(
            logits,
            tuple,
        ):
            logits = logits[0]

        y_true = eval_pred.label_ids

        y_pred = np.argmax(
            logits,
            axis=-1,
        )

        accuracy, macro_f1 = (
            calcular_metricas(
                y_true,
                y_pred,
            )
        )

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        }

    training_args = TrainingArguments(
        output_dir=str(
            fold_output_dir
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
        compute_metrics=compute_metrics,
    )

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    inicio = time.perf_counter()

    try:
        trainer.train()

        # Uma única predição final para métricas + OOF.
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

        eval_loss = float(
            prediction_output.metrics.get(
                "test_loss",
                np.nan,
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

        salvar_historico(
            trainer,
            fold,
        )

        # --------------------------------------------
        # OOF
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

                "original_token_count": (
                    original_lengths[
                        eval_indices
                    ]
                ),

                "head_tail_applied": (
                    was_head_tail[
                        eval_indices
                    ]
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

        for col in metadata_df.columns:
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
                "head_tail_balanced"
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
        print(f"Fold {fold} concluído")
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
            "experiment_id": EXPERIMENT_ID,
            "method": "head_tail_balanced",
            "max_length": MAX_LENGTH,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "fold": fold,
            "train_size": len(train_dataset),
            "eval_size": len(eval_dataset),
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "gradient_accumulation_steps": (
                GRADIENT_ACCUMULATION_STEPS
            ),
            "effective_batch_size": (
                TRAIN_BATCH_SIZE
                * GRADIENT_ACCUMULATION_STEPS
            ),
            "weight_decay": WEIGHT_DECAY,
            "fp16": FP16,
            "accuracy": np.nan,
            "macro_f1": np.nan,
            "eval_loss": np.nan,
            "time_minutes": tempo / 60,
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
# OOF CONSOLIDADO
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

    oof = pd.concat(
        [
            pd.read_csv(path)
            for path in paths
        ],
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
            "número incorreto de linhas."
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
            "Ainda não há cinco folds concluídos."
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

    print("BERTimbau head + tail")

    print(
        f"Accuracy = "
        f"{acc_mean:.4f} ± {acc_std:.4f}"
    )

    print(
        f"Macro-F1 = "
        f"{f1_mean:.4f} ± {f1_std:.4f}"
    )

    print()
    print("Referência BERTimbau 256 padrão:")

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
        "Delta Accuracy vs 256 = "
        f"{acc_mean - REFERENCE_ACCURACY:+.4f}"
    )

    print(
        "Delta Macro-F1 vs 256 = "
        f"{f1_mean - REFERENCE_MACRO_F1:+.4f}"
    )

    print()
    print("Melhor BERTimbau atual (512 padrão):")

    print(
        f"Accuracy = "
        f"{BEST_512_ACCURACY:.4f}"
    )

    print(
        f"Macro-F1 = "
        f"{BEST_512_MACRO_F1:.4f}"
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
        "BERTimbau — Head + Tail"
    )
    print("=" * 70)

    print(
        "GPU: "
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

    labels_series = (
        train_df["clarity"]
        .map(LABEL2ID)
    )

    if labels_series.isna().any():
        raise ValueError(
            "Há rótulos inválidos."
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

    metadata_df = (
        criar_metadados_duplicatas(
            train_df
        )
    )

    (
        encodings,
        original_lengths,
        was_head_tail,
    ) = tokenizar_head_tail(
        textos
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

        resultado = executar_fold(
            fold=fold,
            train_df=train_df,
            metadata_df=metadata_df,
            encodings=encodings,
            labels=labels,
            fold_values=fold_values,
            original_lengths=(
                original_lengths
            ),
            was_head_tail=(
                was_head_tail
            ),
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
