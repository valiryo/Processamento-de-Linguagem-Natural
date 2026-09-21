import gc
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.evaluation.folds import carregar_folds
from src.models.bertimbau import (
    criar_modelo,
    criar_tokenizer,
    preparar_dataset,
)


# ============================================================
# Configuração geral
# ============================================================
FOLDS = [0, 1, 2, 3, 4]

SEED = 42

MAX_LENGTH = 256

LEARNING_RATES = [
    1e-5,
    2e-5,
    3e-5,
]

EPOCHS = [
    1,
    2,
    3,
]

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8

GRADIENT_ACCUMULATION_STEPS = 2

WEIGHT_DECAY = 0.01

FP16 = True
GRADIENT_CHECKPOINTING = False


# ============================================================
# Caminhos
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "train.xlsx"
)

FOLDS_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "folds.csv"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "bertimbau_grid.csv"
)

CHECKPOINTS_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "bertimbau_grid"
)


# ============================================================
# Métricas
# ============================================================

def calcular_metricas(eval_pred):
    logits, labels = eval_pred

    predicoes = np.argmax(
        logits,
        axis=-1,
    )

    return {
        "accuracy": accuracy_score(
            labels,
            predicoes,
        ),
        "f1_macro": f1_score(
            labels,
            predicoes,
            average="macro",
        ),
    }


# ============================================================
# Controle dos experimentos
# ============================================================

def configuracao_id(
    learning_rate: float,
    epochs: int,
) -> str:
    return (
        f"lr_{learning_rate:.0e}"
        f"_epochs_{epochs}"
        f"_len_{MAX_LENGTH}"
    )


def resultado_ja_existe(
    resultados_existentes: pd.DataFrame,
    config_id: str,
    fold: int,
) -> bool:
    if resultados_existentes.empty:
        return False

    filtro = (
        (resultados_existentes["config_id"] == config_id)
        & (resultados_existentes["fold"] == fold)
        & (resultados_existentes["status"] == "ok")
    )

    return filtro.any()


def salvar_resultado(resultado: dict) -> None:
    novo = pd.DataFrame([resultado])

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


# ============================================================
# Execução
# ============================================================

def main():
    set_seed(SEED)

    CHECKPOINTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Dados
    # --------------------------------------------------------

    df = pd.read_excel(
        DATA_PATH,
        sheet_name="train",
    )

    df["resp_text"] = (
        df["resp_text"]
        .astype(str)
    )

    df = carregar_folds(
        df,
        FOLDS_PATH,
    )

    print(
        f"Dataset carregado: "
        f"{len(df)} instâncias"
    )

    # --------------------------------------------------------
    # Tokenização única
    # --------------------------------------------------------

    tokenizer = criar_tokenizer()

    dataset_completo = preparar_dataset(
        df["resp_text"],
        df["clarity"],
        tokenizer,
        max_length=MAX_LENGTH,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
    )

    # --------------------------------------------------------
    # Resultados anteriores
    # --------------------------------------------------------

    if RESULTS_PATH.exists():
        resultados_existentes = pd.read_csv(
            RESULTS_PATH
        )
    else:
        resultados_existentes = pd.DataFrame()

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    configuracoes = list(
        product(
            LEARNING_RATES,
            EPOCHS,
        )
    )

    total_configuracoes = len(configuracoes)

    print(
        f"\nConfigurações: "
        f"{total_configuracoes}"
    )

    print(
    f"Treinamentos possíveis: "
    f"{total_configuracoes * len(FOLDS)}"
    )

    # --------------------------------------------------------
    # Configurações × folds
    # --------------------------------------------------------

    for config_num, (
        learning_rate,
        epochs,
    ) in enumerate(
        configuracoes,
        start=1,
    ):
        config_id = configuracao_id(
            learning_rate,
            epochs,
        )

        print(
            "\n"
            "========================================"
        )

        print(
            f"Configuração "
            f"{config_num}/{total_configuracoes}"
        )

        print(config_id)

        print(
            "========================================"
        )

        for fold in FOLDS:
            fold = int(fold)

            if resultado_ja_existe(
                resultados_existentes,
                config_id,
                fold,
            ):
                print(
                    f"Fold {fold}: "
                    f"já concluído. Pulando."
                )
                continue

            print(
                f"\n--- Fold {fold} ---"
            )

            # ----------------------------------------------
            # Índices
            # ----------------------------------------------

            indices_treino = (
                df.index[
                    df["fold"] != fold
                ]
                .tolist()
            )

            indices_validacao = (
                df.index[
                    df["fold"] == fold
                ]
                .tolist()
            )

            dataset_treino = (
                dataset_completo.select(
                    indices_treino
                )
            )

            dataset_validacao = (
                dataset_completo.select(
                    indices_validacao
                )
            )

            # ----------------------------------------------
            # Limpeza
            # ----------------------------------------------

            gc.collect()

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # ----------------------------------------------
            # Novo modelo
            # ----------------------------------------------

            set_seed(SEED)

            modelo = criar_modelo()

            output_dir = (
                CHECKPOINTS_DIR
                / config_id
                / f"fold_{fold}"
            )

            training_args = TrainingArguments(
                output_dir=str(output_dir),

                learning_rate=learning_rate,
                num_train_epochs=epochs,

                per_device_train_batch_size=(
                    TRAIN_BATCH_SIZE
                ),

                per_device_eval_batch_size=(
                    EVAL_BATCH_SIZE
                ),

                gradient_accumulation_steps=(
                    GRADIENT_ACCUMULATION_STEPS
                ),

                weight_decay=WEIGHT_DECAY,

                fp16=FP16,

                gradient_checkpointing=(
                    GRADIENT_CHECKPOINTING
                ),

                eval_strategy="epoch",
                save_strategy="no",

                logging_strategy="epoch",

                report_to="none",

                seed=SEED,
                data_seed=SEED,
            )

            trainer = Trainer(
                model=modelo,
                args=training_args,

                train_dataset=dataset_treino,
                eval_dataset=dataset_validacao,

                data_collator=data_collator,
                processing_class=tokenizer,

                compute_metrics=calcular_metricas,
            )

            # ----------------------------------------------
            # Treino
            # ----------------------------------------------

            inicio = time.perf_counter()

            try:
                trainer.train()

                tempo_segundos = (
                    time.perf_counter()
                    - inicio
                )

                avaliacao = trainer.evaluate()

                memoria_gb = (
                    torch.cuda.max_memory_allocated()
                    / 1024**3
                )

                resultado = {
                    "config_id": config_id,
                    "learning_rate": learning_rate,
                    "epochs": epochs,
                    "max_length": MAX_LENGTH,

                    "fold": fold,

                    "n_treino": len(
                        dataset_treino
                    ),

                    "n_validacao": len(
                        dataset_validacao
                    ),

                    "accuracy": avaliacao[
                        "eval_accuracy"
                    ],

                    "f1_macro": avaliacao[
                        "eval_f1_macro"
                    ],

                    "eval_loss": avaliacao[
                        "eval_loss"
                    ],

                    "tempo_segundos": (
                        tempo_segundos
                    ),

                    "pico_vram_gb": (
                        memoria_gb
                    ),

                    "status": "ok",
                }

                salvar_resultado(
                    resultado
                )

                print(
                    f"Accuracy: "
                    f"{resultado['accuracy']:.4f}"
                )

                print(
                    f"Macro-F1: "
                    f"{resultado['f1_macro']:.4f}"
                )

                print(
                    f"Tempo: "
                    f"{tempo_segundos / 60:.2f} min"
                )

                print(
                    f"Pico VRAM: "
                    f"{memoria_gb:.2f} GB"
                )

            except torch.OutOfMemoryError:
                tempo_segundos = (
                    time.perf_counter()
                    - inicio
                )

                salvar_resultado({
                    "config_id": config_id,
                    "learning_rate": learning_rate,
                    "epochs": epochs,
                    "max_length": MAX_LENGTH,

                    "fold": fold,

                    "n_treino": len(
                        dataset_treino
                    ),

                    "n_validacao": len(
                        dataset_validacao
                    ),

                    "accuracy": np.nan,
                    "f1_macro": np.nan,
                    "eval_loss": np.nan,

                    "tempo_segundos": (
                        tempo_segundos
                    ),

                    "pico_vram_gb": (
                        torch.cuda.max_memory_allocated()
                        / 1024**3
                    ),

                    "status": "oom",
                })

                print(
                    "CUDA Out Of Memory."
                )

            finally:
                del trainer
                del modelo
                del dataset_treino
                del dataset_validacao

                gc.collect()
                torch.cuda.empty_cache()

    print(
        "\nGrid Search concluído."
    )


if __name__ == "__main__":
    main()