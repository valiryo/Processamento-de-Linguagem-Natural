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
    / "bertimbau_length_512_cv.csv"
)

HISTORY_PATH = (
    ROOT_DIR
    / "results"
    / "bertimbau_length_512_history.csv"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "output"
    / "bertimbau_length_512"
)


# ============================================================
# MODELO
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


# ============================================================
# CONFIGURAÇÃO EXPERIMENTAL
#
# Hiperparâmetros vencedores do Grid Search:
# LR=3e-5 e epochs=2.
#
# O objetivo aqui é alterar apenas max_length.
# ============================================================

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

def calcular_metricas(
    eval_pred: EvalPrediction,
) -> dict[str, float]:

    logits = eval_pred.predictions
    labels = eval_pred.label_ids

    predictions = np.argmax(
        logits,
        axis=-1,
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
    }


# ============================================================
# CARREGAMENTO DOS DADOS
# ============================================================

def carregar_dados() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    print("=" * 70)
    print("Carregando dados")
    print("=" * 70)

    train_df = pd.read_excel(
        TRAIN_PATH
    )

    folds_df = pd.read_csv(
        FOLDS_PATH
    )

    if len(train_df) != len(folds_df):
        raise ValueError(
            "train.xlsx e folds.csv possuem "
            "números diferentes de linhas: "
            f"{len(train_df)} != {len(folds_df)}"
        )

    colunas_necessarias = {
        "resp_text",
        "clarity",
    }

    faltantes = (
        colunas_necessarias
        - set(train_df.columns)
    )

    if faltantes:
        raise ValueError(
            "Colunas ausentes em train.xlsx: "
            f"{faltantes}"
        )

    if "fold" not in folds_df.columns:
        raise ValueError(
            "Coluna 'fold' não encontrada "
            "em folds.csv."
        )

    folds_encontrados = sorted(
        folds_df["fold"]
        .astype(int)
        .unique()
        .tolist()
    )

    print(
        f"Instâncias: {len(train_df)}"
    )

    print(
        "Folds encontrados: "
        f"{folds_encontrados}"
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
    print("Tokenização do corpus")
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

    if "status" not in resultados.columns:
        return False

    mask = (
        (resultados["fold"] == fold)
        & (resultados["status"] == "ok")
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
    historico: list[dict],
) -> None:

    if not historico:
        return

    HISTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    novo = pd.DataFrame(
        historico
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
# ÚLTIMA AVALIAÇÃO
# ============================================================

def obter_ultima_avaliacao(
    trainer: Trainer,
) -> dict:

    avaliacoes = [
        item
        for item in trainer.state.log_history
        if "eval_accuracy" in item
    ]

    if not avaliacoes:

        print(
            "Nenhuma avaliação encontrada "
            "no histórico. Avaliando agora..."
        )

        return trainer.evaluate()

    return avaliacoes[-1]


# ============================================================
# HISTÓRICO POR ÉPOCA
# ============================================================

def extrair_historico_avaliacao(
    trainer: Trainer,
    fold: int,
) -> list[dict]:

    registros = []

    for item in trainer.state.log_history:

        if "eval_accuracy" not in item:
            continue

        registros.append(
            {
                "max_length": MAX_LENGTH,
                "learning_rate": LEARNING_RATE,
                "epochs_config": EPOCHS,
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
# TREINAMENTO DE UM FOLD
# ============================================================

def executar_fold(
    fold: int,
    encodings: dict[str, torch.Tensor],
    labels: np.ndarray,
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

    effective_batch_size = (
        TRAIN_BATCH_SIZE
        * GRADIENT_ACCUMULATION_STEPS
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
        "learning_rate: "
        f"{LEARNING_RATE}"
    )

    print(
        f"epochs: {EPOCHS}"
    )

    print(
        "batch físico: "
        f"{TRAIN_BATCH_SIZE}"
    )

    print(
        "gradient accumulation: "
        f"{GRADIENT_ACCUMULATION_STEPS}"
    )

    print(
        "batch efetivo: "
        f"{effective_batch_size}"
    )

    # Novo modelo para cada fold.
    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=len(LABEL2ID),
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

    training_args = TrainingArguments(
        output_dir=str(
            fold_output_dir
        ),

        # -----------------------------
        # hiperparâmetros
        # -----------------------------
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,

        # -----------------------------
        # batch
        # -----------------------------
        per_device_train_batch_size=(
            TRAIN_BATCH_SIZE
        ),

        per_device_eval_batch_size=(
            EVAL_BATCH_SIZE
        ),

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        # -----------------------------
        # avaliação
        # -----------------------------
        eval_strategy="epoch",

        # -----------------------------
        # não salvar checkpoints
        # -----------------------------
        save_strategy="no",

        # -----------------------------
        # precisão
        # -----------------------------
        fp16=FP16,

        # -----------------------------
        # logs
        # -----------------------------
        logging_strategy="steps",
        logging_steps=100,

        report_to="none",

        # -----------------------------
        # reprodutibilidade
        # -----------------------------
        seed=SEED,
        data_seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=calcular_metricas,
    )

    # ========================================================
    # PREPARAÇÃO DA GPU
    # ========================================================

    gc.collect()

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()

    torch.cuda.synchronize()

    inicio = time.perf_counter()

    try:

        trainer.train()

        torch.cuda.synchronize()

        tempo_total = (
            time.perf_counter()
            - inicio
        )

        ultima_avaliacao = (
            obter_ultima_avaliacao(
                trainer
            )
        )

        accuracy = float(
            ultima_avaliacao[
                "eval_accuracy"
            ]
        )

        macro_f1 = float(
            ultima_avaliacao[
                "eval_macro_f1"
            ]
        )

        eval_loss = float(
            ultima_avaliacao[
                "eval_loss"
            ]
        )

        pico_vram = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

        pico_vram_reservada = (
            torch.cuda.max_memory_reserved()
            / 1024**3
        )

        historico = (
            extrair_historico_avaliacao(
                trainer=trainer,
                fold=fold,
            )
        )

        salvar_historico(
            historico
        )

        resultado = {
            "max_length": MAX_LENGTH,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "fold": fold,

            "train_size": len(
                train_dataset
            ),

            "eval_size": len(
                eval_dataset
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
                effective_batch_size
            ),

            "weight_decay": WEIGHT_DECAY,

            "fp16": FP16,

            "gradient_checkpointing": (
                GRADIENT_CHECKPOINTING
            ),

            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "eval_loss": eval_loss,

            "time_seconds": tempo_total,

            "time_minutes": (
                tempo_total / 60
            ),

            "peak_vram_gb": (
                pico_vram
            ),

            "peak_reserved_vram_gb": (
                pico_vram_reservada
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
            "Tempo: "
            f"{tempo_total / 60:.2f} min"
        )

        print(
            "Pico VRAM: "
            f"{pico_vram:.2f} GB"
        )

        return resultado

    except Exception as exc:

        tempo_total = (
            time.perf_counter()
            - inicio
        )

        return {
            "max_length": MAX_LENGTH,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "fold": fold,

            "train_size": len(
                train_dataset
            ),

            "eval_size": len(
                eval_dataset
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
                effective_batch_size
            ),

            "weight_decay": WEIGHT_DECAY,

            "fp16": FP16,

            "gradient_checkpointing": (
                GRADIENT_CHECKPOINTING
            ),

            "accuracy": np.nan,
            "macro_f1": np.nan,
            "eval_loss": np.nan,

            "time_seconds": tempo_total,

            "time_minutes": (
                tempo_total / 60
            ),

            "peak_vram_gb": np.nan,

            "peak_reserved_vram_gb": (
                np.nan
            ),

            "status": "error",

            "error": repr(
                exc
            ),
        }

    finally:

        del trainer
        del model

        gc.collect()

        torch.cuda.empty_cache()


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
        df["status"] == "ok"
    ].copy()

    if len(ok) != len(FOLDS):

        print()
        print(
            "Ainda não há 5 folds "
            "concluídos."
        )

        return

    accuracy_mean = (
        ok["accuracy"].mean()
    )

    accuracy_std = (
        ok["accuracy"].std(
            ddof=1
        )
    )

    macro_f1_mean = (
        ok["macro_f1"].mean()
    )

    macro_f1_std = (
        ok["macro_f1"].std(
            ddof=1
        )
    )

    print()
    print("=" * 70)
    print(
        "RESULTADO CONSOLIDADO"
    )
    print("=" * 70)

    print(
        f"max_length = {MAX_LENGTH}"
    )

    print(
        "Accuracy = "
        f"{accuracy_mean:.4f} "
        f"± {accuracy_std:.4f}"
    )

    print(
        "Macro-F1 = "
        f"{macro_f1_mean:.4f} "
        f"± {macro_f1_std:.4f}"
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
        "BERTimbau — CV max_length=512"
    )
    print("=" * 70)

    print(
        "GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        "Batch efetivo: "
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

    # Tokenização feita apenas uma vez.
    encodings = tokenizar_corpus(
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
            encodings=encodings,
            labels=labels,
            fold_values=fold_values,
        )

        # Salvar imediatamente.
        salvar_resultado(
            resultado
        )

        if resultado["status"] != "ok":

            raise RuntimeError(
                f"Falha no fold {fold}: "
                f"{resultado['error']}"
            )

        resultados_existentes = (
            carregar_resultados_existentes()
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