from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURAÇÃO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

RESULTS_DIR = ROOT_DIR / "results"

EXPECTED_N = 900

LABELS = (0, 1, 2)

ID2LABEL = {
    0: "c1",
    1: "c234",
    2: "c5",
}

C234_ID = 1


# ============================================================
# ARQUIVOS DOS COMPONENTES
# ============================================================

BERT512_PATH = (
    RESULTS_DIR
    / "bertimbau_512_final.csv"
)

SOFT256_PATH = (
    RESULTS_DIR
    / "bertimbau_duplicate_soft_labels_256_final.csv"
)

SVC_PATH = (
    RESULTS_DIR
    / "tfidf_word_char_linearsvc_final.csv"
)

BASELINE_PATH = (
    RESULTS_DIR
    / "baseline_final.csv"
)

ORDINAL_PATH = (
    RESULTS_DIR
    / "bertimbau_ordinal_256_final.csv"
)

NORBERTO_PATH = (
    RESULTS_DIR
    / "norberto_base_1024_final.csv"
)


# ============================================================
# SAÍDAS
# ============================================================

V2_OUTPUT_PATH = (
    RESULTS_DIR
    / "ensemble_majority_5_v2_final.csv"
)

V3_OUTPUT_PATH = (
    RESULTS_DIR
    / "ensemble_majority_5_v3_norberto_final.csv"
)


# ============================================================
# CARREGAMENTO / VALIDAÇÃO
# ============================================================

def carregar_componente(
    path: Path,
    name: str,
    require_probabilities: bool = False,
) -> pd.DataFrame:

    if not path.exists():

        raise FileNotFoundError(
            f"{name}: arquivo não encontrado:\n{path}"
        )

    df = (
        pd.read_csv(path)
        .reset_index(drop=True)
    )

    required = {
        "row_id",
        "y_pred_id",
    }

    if require_probabilities:

        required |= {
            "prob_c1",
            "prob_c234",
            "prob_c5",
        }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"{name}: colunas ausentes: "
            f"{sorted(missing)}"
        )

    if len(df) != EXPECTED_N:

        raise ValueError(
            f"{name}: {len(df)} linhas; "
            f"esperado {EXPECTED_N}."
        )

    if df["row_id"].duplicated().any():

        raise ValueError(
            f"{name}: row_id duplicado."
        )

    row_ids = (
        pd.to_numeric(
            df["row_id"],
            errors="raise",
        )
        .astype(int)
        .to_numpy()
    )

    expected_ids = np.arange(
        EXPECTED_N,
        dtype=int,
    )

    if not np.array_equal(
        row_ids,
        expected_ids,
    ):

        raise ValueError(
            f"{name}: row_id não corresponde "
            "exatamente a 0..899 na ordem esperada."
        )

    y_pred = (
        pd.to_numeric(
            df["y_pred_id"],
            errors="raise",
        )
        .astype(int)
        .to_numpy()
    )

    if not set(
        np.unique(y_pred)
    ).issubset(
        set(LABELS)
    ):

        raise ValueError(
            f"{name}: classe inválida."
        )

    if require_probabilities:

        prob_cols = [
            "prob_c1",
            "prob_c234",
            "prob_c5",
        ]

        probs = (
            df[prob_cols]
            .astype(float)
            .to_numpy()
        )

        if not np.isfinite(
            probs
        ).all():

            raise ValueError(
                f"{name}: probabilidades "
                "NaN/Inf."
            )

        if not np.allclose(
            probs.sum(axis=1),
            1.0,
            atol=1e-6,
        ):

            raise ValueError(
                f"{name}: probabilidades "
                "não somam 1."
            )

    return df


# ============================================================
# ENSEMBLE V2
# ============================================================

def majority_vote_v2(
    predictions: np.ndarray,
) -> np.ndarray:
    """
    predictions:
        shape [N, 5]

    Ordem / hierarquia congelada:

        0 BERTimbau 512
        1 soft-label 256
        2 LinearSVC
        3 baseline LR
        4 ordinal corrigido

    Regra:

    - vencedor único -> classe vencedora;

    - empate 2-2-1 -> somente as duas classes
      líderes são elegíveis;

    - percorre a hierarquia dos modelos acima
      e escolhe a primeira previsão pertencente
      às classes líderes.
    """

    output = np.empty(
        predictions.shape[0],
        dtype=np.int64,
    )

    for i, row in enumerate(
        predictions
    ):

        counts = np.bincount(
            row,
            minlength=3,
        )

        winners = np.flatnonzero(
            counts == counts.max()
        )

        # Vencedor único.
        if len(winners) == 1:

            output[i] = int(
                winners[0]
            )

            continue

        # Empate:
        # percorre a hierarquia fixa,
        # mas somente classes líderes
        # podem ser escolhidas.
        chosen = None

        for model_prediction in row:

            candidate = int(
                model_prediction
            )

            if candidate in winners:

                chosen = candidate
                break

        if chosen is None:

            raise RuntimeError(
                "Falha inesperada no "
                "desempate do V2."
            )

        output[i] = chosen

    return output


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 80)
    print("ENSEMBLE FINAL V3")
    print("=" * 80)

    # ========================================================
    # CARREGAR COMPONENTES
    # ========================================================

    bert512 = carregar_componente(
        BERT512_PATH,
        "BERTimbau 512",
    )

    soft256 = carregar_componente(
        SOFT256_PATH,
        "Soft-label 256",
    )

    svc = carregar_componente(
        SVC_PATH,
        "LinearSVC",
    )

    baseline = carregar_componente(
        BASELINE_PATH,
        "Baseline LR",
    )

    ordinal = carregar_componente(
        ORDINAL_PATH,
        "Ordinal corrigido",
    )

    norberto = carregar_componente(
        NORBERTO_PATH,
        "NorBERTo",
        require_probabilities=True,
    )

    print(
        "Todos os seis componentes "
        "foram carregados e validados."
    )

    # ========================================================
    # MATRIZ DOS CINCO VOTOS
    # ========================================================

    votes = np.column_stack(
        [
            bert512[
                "y_pred_id"
            ].astype(int),

            soft256[
                "y_pred_id"
            ].astype(int),

            svc[
                "y_pred_id"
            ].astype(int),

            baseline[
                "y_pred_id"
            ].astype(int),

            ordinal[
                "y_pred_id"
            ].astype(int),
        ]
    )

    if votes.shape != (
        EXPECTED_N,
        5,
    ):

        raise RuntimeError(
            f"Shape inesperado dos votos: "
            f"{votes.shape}"
        )

    # ========================================================
    # V2
    # ========================================================

    print()
    print("=" * 80)
    print("APLICANDO V2")
    print("=" * 80)

    pred_v2 = majority_vote_v2(
        votes
    )

    vote_patterns = []

    tie_leader_a = np.full(
        EXPECTED_N,
        -1,
        dtype=int,
    )

    tie_leader_b = np.full(
        EXPECTED_N,
        -1,
        dtype=int,
    )

    is_221_tie = np.zeros(
        EXPECTED_N,
        dtype=bool,
    )

    for i in range(
        EXPECTED_N
    ):

        counts = np.bincount(
            votes[i],
            minlength=3,
        )

        sorted_counts = (
            np.sort(
                counts
            )[::-1]
        )

        vote_patterns.append(
            "-".join(
                map(
                    str,
                    sorted_counts.tolist(),
                )
            )
        )

        leaders = np.flatnonzero(
            counts
            == counts.max()
        )

        if (
            len(leaders) == 2
            and
            sorted_counts.tolist()
            == [2, 2, 1]
        ):

            is_221_tie[i] = True

            tie_leader_a[i] = int(
                leaders[0]
            )

            tie_leader_b[i] = int(
                leaders[1]
            )

    # ========================================================
    # SALVAR V2 PARA AUDITORIA
    # ========================================================

    v2_output = pd.DataFrame(
        {
            "row_id": np.arange(
                EXPECTED_N,
                dtype=int,
            ),

            "pred_bert512": (
                votes[:, 0]
            ),

            "pred_soft256": (
                votes[:, 1]
            ),

            "pred_svc_word_char": (
                votes[:, 2]
            ),

            "pred_baseline": (
                votes[:, 3]
            ),

            "pred_ordinal256_corrected": (
                votes[:, 4]
            ),

            "vote_pattern": (
                vote_patterns
            ),

            "tie_leader_a": (
                tie_leader_a
            ),

            "tie_leader_b": (
                tie_leader_b
            ),

            "y_pred_id": (
                pred_v2
            ),

            "y_pred": [
                ID2LABEL[
                    int(value)
                ]
                for value
                in pred_v2
            ],
        }
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    v2_output.to_csv(
        V2_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # V3
    # ========================================================

    print()
    print("=" * 80)
    print("APLICANDO REGRA V3")
    print("=" * 80)

    norberto_probs = (
        norberto[
            [
                "prob_c1",
                "prob_c234",
                "prob_c5",
            ]
        ]
        .astype(float)
        .to_numpy()
    )

    final_pred = (
        pred_v2.copy()
    )

    targeted_tie = np.zeros(
        EXPECTED_N,
        dtype=bool,
    )

    changed_prediction = np.zeros(
        EXPECTED_N,
        dtype=bool,
    )

    for i in range(
        EXPECTED_N
    ):

        # V3 só considera empates
        # 2-2-1.
        if not is_221_tie[i]:

            continue

        leaders = np.array(
            [
                tie_leader_a[i],
                tie_leader_b[i],
            ],
            dtype=int,
        )

        # V3 só intervém se c234
        # está entre as duas líderes.
        if C234_ID not in leaders:

            continue

        targeted_tie[i] = True

        # IMPORTANTE:
        #
        # NorBERTo NÃO é um sexto voto.
        #
        # Comparamos as probabilidades
        # SOMENTE das duas classes líderes.
        chosen = int(
            leaders[
                np.argmax(
                    norberto_probs[
                        i,
                        leaders,
                    ]
                )
            ]
        )

        final_pred[i] = chosen

        changed_prediction[i] = (
            chosen
            != pred_v2[i]
        )

    # ========================================================
    # VALIDAÇÕES DA REGRA V3
    # ========================================================

    if not set(
        np.unique(
            final_pred
        )
    ).issubset(
        set(LABELS)
    ):

        raise RuntimeError(
            "V3 produziu classe inválida."
        )

    # Toda alteração V3 obrigatoriamente
    # precisa estar dentro dos empates-alvo.
    if np.any(
        changed_prediction
        & ~targeted_tie
    ):

        raise RuntimeError(
            "V3 alterou previsão fora "
            "de um empate-alvo."
        )

    # Fora dos empates alvo,
    # V3 deve ser idêntico ao V2.
    if not np.array_equal(
        final_pred[
            ~targeted_tie
        ],
        pred_v2[
            ~targeted_tie
        ],
    ):

        raise RuntimeError(
            "V3 divergiu do V2 fora "
            "dos empates-alvo."
        )

    # Nos empates-alvo, a classe escolhida
    # precisa ser uma das duas líderes.
    targeted_indices = np.flatnonzero(
        targeted_tie
    )

    for i in targeted_indices:

        leaders = {
            int(
                tie_leader_a[i]
            ),
            int(
                tie_leader_b[i]
            ),
        }

        if (
            int(
                final_pred[i]
            )
            not in leaders
        ):

            raise RuntimeError(
                f"row_id {i}: V3 escolheu "
                "classe fora das líderes."
            )

        if C234_ID not in leaders:

            raise RuntimeError(
                f"row_id {i}: marcado como "
                "empate-alvo sem c234."
            )

    # ========================================================
    # OUTPUT V3
    # ========================================================

    v3_output = pd.DataFrame(
        {
            "row_id": np.arange(
                EXPECTED_N,
                dtype=int,
            ),

            # ------------------------------
            # Cinco votos
            # ------------------------------

            "pred_bert512": (
                votes[:, 0]
            ),

            "pred_soft256": (
                votes[:, 1]
            ),

            "pred_svc_word_char": (
                votes[:, 2]
            ),

            "pred_baseline": (
                votes[:, 3]
            ),

            "pred_ordinal256_corrected": (
                votes[:, 4]
            ),

            # ------------------------------
            # NorBERTo
            # ------------------------------

            "pred_norberto": (
                norberto[
                    "y_pred_id"
                ]
                .astype(int)
                .to_numpy()
            ),

            "norberto_prob_c1": (
                norberto_probs[:, 0]
            ),

            "norberto_prob_c234": (
                norberto_probs[:, 1]
            ),

            "norberto_prob_c5": (
                norberto_probs[:, 2]
            ),

            # ------------------------------
            # Auditoria do voto
            # ------------------------------

            "vote_pattern": (
                vote_patterns
            ),

            "tie_leader_a": (
                tie_leader_a
            ),

            "tie_leader_b": (
                tie_leader_b
            ),

            "v3_targeted_tie": (
                targeted_tie
            ),

            "v3_changed_prediction": (
                changed_prediction
            ),

            # ------------------------------
            # V2
            # ------------------------------

            "y_pred_id_v2": (
                pred_v2
            ),

            "y_pred_v2": [
                ID2LABEL[
                    int(value)
                ]
                for value
                in pred_v2
            ],

            # ------------------------------
            # V3 FINAL
            # ------------------------------

            "y_pred_id": (
                final_pred
            ),

            "y_pred": [
                ID2LABEL[
                    int(value)
                ]
                for value
                in final_pred
            ],
        }
    )

    # ========================================================
    # VALIDAÇÃO FINAL
    # ========================================================

    if len(
        v3_output
    ) != EXPECTED_N:

        raise RuntimeError(
            "Número incorreto de "
            "predições finais."
        )

    if not np.array_equal(
        v3_output[
            "row_id"
        ].to_numpy(),
        np.arange(
            EXPECTED_N,
            dtype=int,
        ),
    ):

        raise RuntimeError(
            "row_id final desalinhado."
        )

    if (
        v3_output
        .isna()
        .any()
        .any()
    ):

        raise RuntimeError(
            "Há valores NaN "
            "no ensemble final."
        )

    v3_output.to_csv(
        V3_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # DIAGNÓSTICOS
    #
    # NÃO são usados para tomar decisões.
    # ========================================================

    total_221 = int(
        is_221_tie.sum()
    )

    total_targeted = int(
        targeted_tie.sum()
    )

    total_changed = int(
        changed_prediction.sum()
    )

    print(
        f"Empates 2-2-1 totais     : "
        f"{total_221}"
    )

    print(
        f"Empates alvo V3           : "
        f"{total_targeted}"
    )

    print(
        f"Predições alteradas V2->V3: "
        f"{total_changed}"
    )

    print()
    print("Distribuição V2:")

    print(
        pd.Series(
            [
                ID2LABEL[int(x)]
                for x in pred_v2
            ]
        )
        .value_counts()
        .reindex(
            [
                "c1",
                "c234",
                "c5",
            ],
            fill_value=0,
        )
        .to_string()
    )

    print()
    print("Distribuição V3:")

    print(
        v3_output[
            "y_pred"
        ]
        .value_counts()
        .reindex(
            [
                "c1",
                "c234",
                "c5",
            ],
            fill_value=0,
        )
        .to_string()
    )

    print()
    print("Arquivos salvos:")

    print(
        f"  V2: {V2_OUTPUT_PATH}"
    )

    print(
        f"  V3: {V3_OUTPUT_PATH}"
    )

    print()
    print("=" * 80)
    print(
        "ENSEMBLE V3 FINAL CONCLUÍDO"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()