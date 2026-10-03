from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from transformers.modeling_outputs import (
    SequenceClassifierOutput,
)


# ============================================================
# CAMINHOS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    ROOT_DIR
    / "data"
    / "raw"
    / "train.xlsx"
)

TEST_PATH = (
    ROOT_DIR
    / "data"
    / "raw"
    / "test1.xlsx"
)

RESULTS_DIR = (
    ROOT_DIR
    / "results"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "bertimbau_ordinal_256_final.csv"
)

TRAIN_OUTPUT_DIR = (
    ROOT_DIR
    / "output"
    / "bertimbau_ordinal_256_final"
)


# ============================================================
# CONFIGURAÇÃO CONGELADA
# ============================================================

MODEL_NAME = (
    "neuralmind/"
    "bert-base-portuguese-cased"
)

LABELS = [
    "c1",
    "c234",
    "c5",
]

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

MAX_LENGTH = 256

LEARNING_RATE = 3e-5
EPOCHS = 2

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8

GRADIENT_ACCUMULATION_STEPS = 2

WEIGHT_DECAY = 0.01

FP16 = True

SEED = 42

EXPECTED_TRAIN_ROWS = 20_092
EXPECTED_TEST_ROWS = 900


# ============================================================
# DATASETS
# ============================================================

class OrdinalTrainDataset(Dataset):

    def __init__(
        self,
        encodings: dict[str, torch.Tensor],
        ordinal_targets: np.ndarray,
    ):
        self.encodings = encodings
        self.ordinal_targets = ordinal_targets

    def __len__(self) -> int:
        return len(self.ordinal_targets)

    def __getitem__(
        self,
        idx: int,
    ) -> dict[str, torch.Tensor]:

        item = {
            key: value[idx]
            for key, value in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            self.ordinal_targets[idx],
            dtype=torch.float32,
        )

        return item


class TestDataset(Dataset):

    def __init__(
        self,
        encodings: dict[str, torch.Tensor],
    ):
        self.encodings = encodings

    def __len__(self) -> int:

        first_key = next(
            iter(self.encodings)
        )

        return len(
            self.encodings[first_key]
        )

    def __getitem__(
        self,
        idx: int,
    ) -> dict[str, torch.Tensor]:

        return {
            key: value[idx]
            for key, value
            in self.encodings.items()
        }


# ============================================================
# MODELO ORDINAL HISTÓRICO
# ============================================================

class BertOrdinalClassifier(nn.Module):
    """
    BERT + escore ordinal escalar + dois thresholds ordenados.

    score = s

    t1 < t2

    logit_1 = s - t1 -> P(y > c1)
    logit_2 = s - t2 -> P(y > c234)

    Como t1 < t2:

        sigmoid(logit_1) >= sigmoid(logit_2)

    garantindo monotonicidade.
    """

    def __init__(
        self,
        model_name: str,
    ):
        super().__init__()

        self.config = (
            AutoConfig.from_pretrained(
                model_name
            )
        )

        self.bert = (
            AutoModel.from_pretrained(
                model_name
            )
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

        # Escore latente de clareza.
        self.score = nn.Linear(
            hidden_size,
            1,
        )

        # Threshold inicial 1.
        self.threshold_1 = nn.Parameter(
            torch.tensor(
                -0.5,
                dtype=torch.float32,
            )
        )

        # t2 = t1 + softplus(delta)
        # portanto t2 > t1.
        self.threshold_delta = nn.Parameter(
            torch.tensor(
                1.0,
                dtype=torch.float32,
            )
        )

    def get_thresholds(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

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
    c1   -> [0, 0]
    c234 -> [1, 0]
    c5   -> [1, 1]
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
    ).astype(
        np.float32
    )

    targets[:, 1] = (
        class_ids > 1
    ).astype(
        np.float32
    )

    return targets


# ============================================================
# PROBABILIDADES
# ============================================================

def logits_para_q(
    logits: np.ndarray,
) -> np.ndarray:
    """
    q1 = P(y > c1)
    q2 = P(y > c234)
    """

    logits_tensor = torch.tensor(
        logits,
        dtype=torch.float32,
    )

    return (
        torch.sigmoid(
            logits_tensor
        )
        .numpy()
    )


def ordinal_logits_para_probabilidades(
    logits: np.ndarray,
) -> np.ndarray:
    """
    Conversão histórica:

        q1 = P(y > c1)
        q2 = P(y > c234)

        P(c1)   = 1 - q1
        P(c234) = q1 - q2
        P(c5)   = q2
    """

    q = logits_para_q(
        logits
    )

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

    # Proteção contra ruído numérico,
    # exatamente como no experimento.
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
# REGRA ORDINAL CORRIGIDA
# ============================================================

def predicao_ordinal_corrigida(
    logits: np.ndarray,
) -> np.ndarray:
    """
    REGRA OFICIAL USADA PELO ENSEMBLE FINAL.

    classe =
        número de thresholds
        cuja probabilidade cumulativa > 0.5

    q1 = P(y > c1)
    q2 = P(y > c234)

    c1:
        q1 <= 0.5
        q2 <= 0.5
        -> classe 0

    c234:
        q1 > 0.5
        q2 <= 0.5
        -> classe 1

    c5:
        q1 > 0.5
        q2 > 0.5
        -> classe 2
    """

    q = logits_para_q(
        logits
    )

    q1 = q[:, 0]
    q2 = q[:, 1]

    y_pred = (
        (q1 > 0.5).astype(np.int64)
        + (q2 > 0.5).astype(np.int64)
    )

    return y_pred


# ============================================================
# DADOS
# ============================================================

def carregar_dados() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    print("=" * 78)
    print(
        "BERTIMBAU ORDINAL 256 FINAL "
        "— CARREGAMENTO"
    )
    print("=" * 78)

    if not TRAIN_PATH.exists():

        raise FileNotFoundError(
            f"train.xlsx não encontrado:\n"
            f"{TRAIN_PATH}"
        )

    if not TEST_PATH.exists():

        raise FileNotFoundError(
            f"test1.xlsx não encontrado:\n"
            f"{TEST_PATH}"
        )

    train_df = (
        pd.read_excel(
            TRAIN_PATH
        )
        .reset_index(
            drop=True
        )
    )

    test_df = (
        pd.read_excel(
            TEST_PATH
        )
        .reset_index(
            drop=True
        )
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
            "Colunas ausentes no treino: "
            f"{sorted(missing)}"
        )

    if "resp_text" not in test_df.columns:

        raise ValueError(
            "Coluna 'resp_text' ausente no teste."
        )

    if len(train_df) != EXPECTED_TRAIN_ROWS:

        raise ValueError(
            "Número incorreto de linhas no treino: "
            f"{len(train_df)}"
        )

    if len(test_df) != EXPECTED_TEST_ROWS:

        raise ValueError(
            "Número incorreto de linhas no teste: "
            f"{len(test_df)}"
        )

    train_df["resp_text"] = (
        train_df["resp_text"]
        .fillna("")
        .astype(str)
    )

    test_df["resp_text"] = (
        test_df["resp_text"]
        .fillna("")
        .astype(str)
    )

    if train_df["clarity"].isna().any():

        raise ValueError(
            "Existem labels ausentes no treino."
        )

    train_df["clarity"] = (
        train_df["clarity"]
        .astype(str)
        .str.strip()
    )

    unknown = (
        set(
            train_df[
                "clarity"
            ].unique()
        )
        - set(LABELS)
    )

    if unknown:

        raise ValueError(
            "Labels desconhecidos: "
            f"{sorted(unknown)}"
        )

    if (
        "clarity" in test_df.columns
        and test_df[
            "clarity"
        ].notna().any()
    ):

        raise ValueError(
            "test1.xlsx possui clarity preenchido."
        )

    print(
        f"Treino: {len(train_df)}"
    )

    print(
        f"Teste : {len(test_df)}"
    )

    return train_df, test_df


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA não está disponível."
        )

    print("=" * 78)
    print(
        "BERTIMBAU ORDINAL 256 — FINAL"
    )
    print("=" * 78)

    print(
        "GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print()
    print("Configuração congelada:")

    print(
        f"  modelo                = "
        f"{MODEL_NAME}"
    )

    print(
        f"  max_length            = "
        f"{MAX_LENGTH}"
    )

    print(
        f"  learning_rate         = "
        f"{LEARNING_RATE}"
    )

    print(
        f"  epochs                = "
        f"{EPOCHS}"
    )

    print(
        f"  batch físico          = "
        f"{TRAIN_BATCH_SIZE}"
    )

    print(
        "  gradient accumulation = "
        f"{GRADIENT_ACCUMULATION_STEPS}"
    )

    print(
        "  batch efetivo         = "
        f"{TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
    )

    print(
        f"  weight_decay          = "
        f"{WEIGHT_DECAY}"
    )

    print(
        f"  FP16                  = "
        f"{FP16}"
    )

    print(
        f"  seed                  = "
        f"{SEED}"
    )

    print(
        "  inferência            = "
        "thresholds > 0.5 (corrigida)"
    )

    # ========================================================
    # DADOS
    # ========================================================

    train_df, test_df = (
        carregar_dados()
    )

    class_ids = (
        train_df[
            "clarity"
        ]
        .map(
            LABEL2ID
        )
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

    # ========================================================
    # TOKENIZAÇÃO
    # ========================================================

    print()
    print("=" * 78)
    print("TOKENIZAÇÃO")
    print("=" * 78)

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_NAME
        )
    )

    train_texts = (
        train_df[
            "resp_text"
        ]
        .tolist()
    )

    test_texts = (
        test_df[
            "resp_text"
        ]
        .tolist()
    )

    print(
        "Tokenizando treino..."
    )

    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    print(
        "Tokenizando teste..."
    )

    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    train_dataset = (
        OrdinalTrainDataset(
            encodings=train_encodings,
            ordinal_targets=ordinal_targets,
        )
    )

    test_dataset = (
        TestDataset(
            encodings=test_encodings,
        )
    )

    # ========================================================
    # MODELO
    # ========================================================

    set_seed(SEED)

    model = (
        BertOrdinalClassifier(
            MODEL_NAME
        )
    )

    TRAIN_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_args = (
        TrainingArguments(

            output_dir=str(
                TRAIN_OUTPUT_DIR
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

            # Sem validação na fase final.
            eval_strategy="no",

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
    )

    # ========================================================
    # TREINAMENTO
    # ========================================================

    gc.collect()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    print()
    print("=" * 78)
    print("TREINAMENTO FINAL")
    print("=" * 78)

    inicio = (
        time.perf_counter()
    )

    try:

        trainer.train()

        torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            - inicio
        )

        peak_vram = (
            torch.cuda
            .max_memory_allocated()
            / 1024**3
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

        print()
        print(
            "Treinamento concluído."
        )

        print(
            f"Tempo: "
            f"{elapsed / 60:.2f} min"
        )

        print(
            f"Pico VRAM: "
            f"{peak_vram:.2f} GB"
        )

        print(
            f"Threshold 1 aprendido: "
            f"{t1_value:.6f}"
        )

        print(
            f"Threshold 2 aprendido: "
            f"{t2_value:.6f}"
        )

        # ====================================================
        # INFERÊNCIA
        # ====================================================

        print()
        print("=" * 78)
        print("INFERÊNCIA NO TESTE")
        print("=" * 78)

        pred_output = (
            trainer.predict(
                test_dataset
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

        logits = np.asarray(
            logits
        )

        # Probabilidades de classe, usadas
        # para auditoria e reconstrução.
        class_probs = (
            ordinal_logits_para_probabilidades(
                logits
            )
        )

        # REGRA CORRIGIDA usada pelo ensemble.
        y_pred_id = (
            predicao_ordinal_corrigida(
                logits
            )
        )

        y_pred = np.array(
            [
                ID2LABEL[int(x)]
                for x in y_pred_id
            ],
            dtype=object,
        )

        # ====================================================
        # OUTPUT
        # ====================================================

        output = pd.DataFrame(
            {
                "row_id": np.arange(
                    len(test_df),
                    dtype=np.int64,
                ),

                "y_pred_id": y_pred_id,

                "y_pred": y_pred,

                "prob_c1": (
                    class_probs[:, 0]
                ),

                "prob_c234": (
                    class_probs[:, 1]
                ),

                "prob_c5": (
                    class_probs[:, 2]
                ),

                "threshold_1": (
                    t1_value
                ),

                "threshold_2": (
                    t2_value
                ),
            }
        )

        # ====================================================
        # VALIDAÇÕES
        # ====================================================

        if len(output) != EXPECTED_TEST_ROWS:

            raise RuntimeError(
                "Número incorreto de previsões."
            )

        if (
            output[
                "row_id"
            ]
            .duplicated()
            .any()
        ):

            raise RuntimeError(
                "row_id duplicado."
            )

        if not np.array_equal(
            output[
                "row_id"
            ].to_numpy(),
            np.arange(
                EXPECTED_TEST_ROWS,
                dtype=np.int64,
            ),
        ):

            raise RuntimeError(
                "Ordem das linhas alterada."
            )

        if output.isna().any().any():

            raise RuntimeError(
                "Existem NaNs no resultado."
            )

        if not set(
            output[
                "y_pred_id"
            ].unique()
        ).issubset(
            {0, 1, 2}
        ):

            raise RuntimeError(
                "Classe ordinal inválida."
            )

        prob_cols = [
            "prob_c1",
            "prob_c234",
            "prob_c5",
        ]

        if not np.allclose(
            output[
                prob_cols
            ].sum(axis=1),
            1.0,
            atol=1e-6,
        ):

            raise RuntimeError(
                "Probabilidades não somam 1."
            )

        # ----------------------------------------------------
        # Confirma explicitamente a regra usada pelo V2.
        #
        # O script do ensemble histórico reconstruiu:
        #
        # pred =
        #   (1 - prob_c1 > 0.5)
        #   +
        #   (prob_c5 > 0.5)
        #
        # ----------------------------------------------------

        reconstructed = (
            (
                (
                    1.0
                    - output[
                        "prob_c1"
                    ].to_numpy()
                )
                > 0.5
            ).astype(np.int64)
            +
            (
                output[
                    "prob_c5"
                ].to_numpy()
                > 0.5
            ).astype(np.int64)
        )

        if not np.array_equal(
            reconstructed,
            output[
                "y_pred_id"
            ].to_numpy(),
        ):

            raise RuntimeError(
                "Predição não coincide com "
                "a reconstrução ordinal corrigida."
            )

        # ====================================================
        # SALVAMENTO
        # ====================================================

        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        output.to_csv(
            OUTPUT_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        print()
        print("=" * 78)
        print("PREVISÕES GERADAS")
        print("=" * 78)

        print(
            output[
                "y_pred"
            ]
            .value_counts()
            .reindex(
                LABELS,
                fill_value=0,
            )
            .to_string()
        )

        print()
        print(
            f"Total: {len(output)}"
        )

        print()
        print(
            "Arquivo salvo em:"
        )

        print(
            OUTPUT_PATH
        )

        print()
        print("=" * 78)
        print(
            "ORDINAL FINAL CONCLUÍDO"
        )
        print("=" * 78)

    finally:

        del trainer
        del model

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()