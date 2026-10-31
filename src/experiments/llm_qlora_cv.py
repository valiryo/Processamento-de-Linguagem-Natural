"""
QLoRA de um LLM decoder pequeno como classificador ternário (c1/c234/c5).

Hipótese: um decoder com pré-treino diferente do BERTimbau/NorBERTo (e que lê
o texto longo quase inteiro) erra de forma diferente e é competitivo, o que é
o que o V3 aproveita. Pensado para GPU de 6 GB (GTX 1660): pesos em 4 bits,
LoRA, gradient checkpointing, FP16 (a 1660 não tem BF16).

Modos:
    --benchmark          fold 0, poucos passos: mede s/passo, VRAM, loss finita
                         e projeta o tempo de 1 fold, 5 folds e do ajuste final.
    (padrão)             CV nos folds de --folds (padrão: só o fold 0, piloto).
    --final              treina em 100% de train.xlsx e prevê test1 e test2 numa
                         única execução (sem retreinar duas vezes).

Funil (mesmo critério pré-declarado do Legal-BERTimbau, NÃO alterar depois):
    GO se accuracy no fold 0 >= 0,450  e  saldo em c234 contra o V3 >= +1.

Dependências extras (fora do requirements.txt):
    pip install peft bitsandbytes scipy
(bitsandbytes tem wheel oficial para Windows nas versões recentes.)

Uso (raiz do projeto):
    python src/experiments/llm_qlora_cv.py --benchmark
    python src/experiments/llm_qlora_cv.py                      # fold 0
    python src/experiments/llm_qlora_cv.py --folds 1 2 3 4      # só após GO
    python src/experiments/llm_qlora_cv.py --final

Saídas (results/), no formato do projeto:
    <tag>_oof_folds/fold_k.csv, <tag>_oof.csv, <tag>_cv.csv, <tag>_summary.csv
    <tag>_final.csv (test1), <tag>_final_test2.csv (test2)
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.folds import carregar_folds  # noqa: E402
from src.evaluation.stacking import (  # noqa: E402
    ID2LABEL,
    LABEL2ID,
    PROB_COLS,
    read_csv_bom,
    summarize,
)

DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

TRAIN_PATH = DATA_DIR / "train.xlsx"
FOLDS_PATH = PROJECT_ROOT / "data" / "splits" / "folds.csv"
TEST_PATHS = {
    "test1": DATA_DIR / "test1-rotulado-para-teste-estatistico.xlsx",
    "test2": DATA_DIR / "test2.xlsx",
}
V3_OOF = RESULTS_DIR / "ensemble_majority_5_v3_norberto_oof.csv"

# Critério go/no-go pré-declarado.
GO_MIN_ACCURACY = 0.450
GO_MIN_C234_BALANCE = 1


class FixedGASTrainer(Trainer):
    """Mesmo workaround do NorBERTo: força a normalização correta da loss
    com gradient accumulation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False


class NonFiniteLossCallback(TrainerCallback):
    """Aborta se loss/grad_norm virar NaN/Inf (risco real: FP16 em modelo
    treinado em BF16)."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        for key in ("loss", "grad_norm"):
            if logs and key in logs and not math.isfinite(float(logs[key])):
                raise RuntimeError(f"{key} não finito no passo {state.global_step}: {logs[key]}")
        return control


class EncodedDataset(Dataset):
    def __init__(self, input_ids: list[list[int]], labels: list[int] | None):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, i: int) -> dict:
        item = {"input_ids": self.input_ids[i], "attention_mask": [1] * len(self.input_ids[i])}
        if self.labels is not None:
            item["labels"] = self.labels[i]
        return item


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B",
                   help="Alternativas: Qwen/Qwen3-1.7B, Qwen/Qwen2.5-0.5B (mais rápido)")
    p.add_argument("--tag", default="qwen25_1p5b_qlora_512")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, nargs="+", default=[0])
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--benchmark-steps", type=int, default=30)
    p.add_argument("--final", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args()


def load_train() -> pd.DataFrame:
    df = pd.read_excel(TRAIN_PATH)
    df["resp_text"] = df["resp_text"].fillna("").astype(str)
    df["clarity"] = df["clarity"].astype(str).str.strip()
    df = carregar_folds(df, FOLDS_PATH)
    df["row_id"] = np.arange(len(df), dtype=int)
    df["y"] = df["clarity"].map(LABEL2ID).astype(int)
    return df


def encode(texts: list[str], tokenizer, max_length: int) -> list[list[int]]:
    return tokenizer(texts, truncation=True, max_length=max_length, padding=False)["input_ids"]


def build_model(args, tokenizer):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    set_seed(args.seed)  # seed imediatamente antes de criar a cabeça de classificação
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        quantization_config=quant,
        dtype=torch.float16,
        device_map={"": 0},
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        task_type="SEQ_CLS",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


def make_training_args(args, output_dir: Path, max_steps: int | None) -> TrainingArguments:
    kwargs = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=max_steps if max_steps else -1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        max_grad_norm=1.0,
        fp16=True,
        logging_steps=5 if max_steps else 50,
        save_strategy="no" if max_steps else "steps",
        save_steps=400,
        save_total_limit=1,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=0,
        group_by_length=True,
    )
    try:
        return TrainingArguments(**kwargs)
    except TypeError:
        kwargs.pop("group_by_length")
        print("AVISO: group_by_length não suportado nesta versão; seguindo sem ele (mais lento).")
        return TrainingArguments(**kwargs)


def train_and_predict(args, tokenizer, train_texts, train_y, pred_sets, output_dir, max_steps=None):
    """Treina e devolve {nome: probabilidades (n,3)} para cada conjunto de previsão."""
    model = build_model(args, tokenizer)
    train_ds = EncodedDataset(encode(train_texts, tokenizer, args.max_length), list(map(int, train_y)))

    trainer = FixedGASTrainer(
        model=model,
        args=make_training_args(args, output_dir, max_steps),
        train_dataset=train_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        processing_class=tokenizer,
        callbacks=[NonFiniteLossCallback()],
    )

    resume = None
    if not max_steps and not args.no_resume and output_dir.exists() and any(output_dir.glob("checkpoint-*")):
        resume = True
        print(f"Retomando de checkpoint em {output_dir}")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    trainer.train(resume_from_checkpoint=resume)
    train_seconds = time.perf_counter() - t0

    if max_steps:  # benchmark: sem previsão
        return trainer, train_seconds, {}

    out = {}
    for name, texts in pred_sets.items():
        ds = EncodedDataset(encode(texts, tokenizer, args.max_length), None)
        logits = trainer.predict(ds).predictions
        out[name] = softmax(np.asarray(logits, dtype=np.float64), axis=1)
    return trainer, train_seconds, out


def cleanup() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def go_no_go(oof: pd.DataFrame) -> None:
    """Compara com o V3 nas mesmas linhas, por row_id."""
    v3 = read_csv_bom(V3_OOF)[["row_id", "y_true_id", "y_pred_id_v3"]]
    m = oof.merge(v3, on="row_id", validate="one_to_one", suffixes=("", "_v3"))
    assert (m["y_true_id"] == m["y_true_id_v3"]).all(), "y_true diverge do V3"
    new_ok, v3_ok = m["y_pred_id"] == m["y_true_id"], m["y_pred_id_v3"] == m["y_true_id"]
    acc = float(new_ok.mean())
    print(f"\nGO/NO-GO  accuracy={acc:.4f}  (V3 nas mesmas linhas: {float(v3_ok.mean()):.4f})")
    balances = {}
    for cls_id, cls in ID2LABEL.items():
        sel = m["y_true_id"] == cls_id
        balances[cls] = int((new_ok & ~v3_ok & sel).sum() - (~new_ok & v3_ok & sel).sum())
        print(f"  saldo em {cls:<5}: {balances[cls]:+d}  (novo certo/V3 errado menos o inverso)")
    go = acc >= GO_MIN_ACCURACY and balances["c234"] >= GO_MIN_C234_BALANCE
    print(f"VEREDITO: {'GO' if go else 'NO-GO'} (acc >= {GO_MIN_ACCURACY} e saldo c234 >= +{GO_MIN_C234_BALANCE})")
    print("Resultado de fold 0 = piloto, não CV completa.")


def run_benchmark(args, df, tokenizer) -> None:
    train = df[df["fold"] != 0]
    steps = args.benchmark_steps
    _, seconds, _ = train_and_predict(
        args, tokenizer, train["resp_text"].tolist(), train["y"].to_numpy(), {},
        CHECKPOINTS_DIR / f"{args.tag}_bench", max_steps=steps,
    )
    eff = args.batch_size * args.gradient_accumulation
    per_step = seconds / steps
    steps_fold = math.ceil(len(train) / eff) * args.epochs
    steps_full = math.ceil(len(df) / eff) * args.epochs
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nBENCHMARK: {per_step:.2f} s/passo (batch efetivo {eff}), VRAM pico {peak:.2f} GB")
    print(f"  1 fold      ≈ {per_step * steps_fold / 3600:.1f} h")
    print(f"  5 folds     ≈ {per_step * steps_fold * 5 / 3600:.1f} h")
    print(f"  ajuste final (100%) ≈ {per_step * steps_full / 3600:.1f} h")
    print("Inclui o aquecimento inicial; a estimativa é ligeiramente pessimista.")


def run_cv(args, df, tokenizer) -> None:
    oof_dir = RESULTS_DIR / f"{args.tag}_oof_folds"
    oof_dir.mkdir(parents=True, exist_ok=True)

    for fold in args.folds:
        fold_path = oof_dir / f"fold_{fold}.csv"
        if fold_path.exists():
            print(f"fold {fold} já concluído; pulando.")
            continue
        train, val = df[df["fold"] != fold], df[df["fold"] == fold]
        _, seconds, probs = train_and_predict(
            args, tokenizer, train["resp_text"].tolist(), train["y"].to_numpy(),
            {"val": val["resp_text"].tolist()}, CHECKPOINTS_DIR / f"{args.tag}_fold{fold}",
        )
        p = probs["val"]
        pred = p.argmax(axis=1)
        out = pd.DataFrame(
            {
                "row_id": val["row_id"].to_numpy(),
                "fold": fold,
                "y_true_id": val["y"].to_numpy(),
                "y_pred_id": pred,
                "y_true": val["clarity"].to_numpy(),
                "y_pred": [ID2LABEL[i] for i in pred],
                PROB_COLS[0]: p[:, 0],
                PROB_COLS[1]: p[:, 1],
                PROB_COLS[2]: p[:, 2],
                "correct": pred == val["y"].to_numpy(),
                "resp_text": val["resp_text"].to_numpy(),
            }
        )
        out.to_csv(fold_path, index=False)
        m = summarize(val["y"].to_numpy(), pred)
        print(f"fold {fold}: {m}  ({seconds / 60:.1f} min)")
        pd.DataFrame(
            [{"model": args.model, "fold": fold, "max_length": args.max_length,
              "learning_rate": args.learning_rate, "epochs": args.epochs,
              "physical_batch": args.batch_size, "gradient_accumulation": args.gradient_accumulation,
              "effective_batch": args.batch_size * args.gradient_accumulation,
              "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "runtime_minutes": seconds / 60}]
        ).to_csv(RESULTS_DIR / f"{args.tag}_summary.csv", mode="a", index=False,
                 header=not (RESULTS_DIR / f"{args.tag}_summary.csv").exists())
        cleanup()

    done = sorted(oof_dir.glob("fold_*.csv"))
    oof = pd.concat([pd.read_csv(f) for f in done]).sort_values("row_id").reset_index(drop=True)
    oof.to_csv(RESULTS_DIR / f"{args.tag}_oof.csv", index=False)
    print(f"\nOOF parcial/total: {len(done)} fold(s), {len(oof)} linhas -> {args.tag}_oof.csv")
    if len(done) == 1:
        go_no_go(oof)
    else:
        print("OOF completo: compare com o V3 e rode stacking_oof_cv.py com este componente.")


def run_final(args, df, tokenizer) -> None:
    pred_sets = {}
    for split, path in TEST_PATHS.items():
        pred_sets[split] = pd.read_excel(path)["resp_text"].fillna("").astype(str).tolist()

    _, seconds, probs = train_and_predict(
        args, tokenizer, df["resp_text"].tolist(), df["y"].to_numpy(), pred_sets,
        CHECKPOINTS_DIR / f"{args.tag}_final",
    )
    for split, p in probs.items():
        pred = p.argmax(axis=1)
        out = pd.DataFrame(
            {
                "row_id": np.arange(len(p)),
                "y_pred_id": pred,
                "y_pred": [ID2LABEL[i] for i in pred],
                PROB_COLS[0]: p[:, 0], PROB_COLS[1]: p[:, 1], PROB_COLS[2]: p[:, 2],
            }
        )
        name = f"{args.tag}_final.csv" if split == "test1" else f"{args.tag}_final_test2.csv"
        out.to_csv(RESULTS_DIR / name, index=False)
        print(f"{split}: {len(out)} previsões -> {name}; distribuição {np.bincount(pred, minlength=3).tolist()}")
    print(f"Tempo de treino: {seconds / 3600:.2f} h")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA não disponível; este script exige GPU.")
    RESULTS_DIR.mkdir(exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    df = load_train()
    print(f"{args.model} | max_length={args.max_length} | LR={args.learning_rate} | "
          f"épocas={args.epochs} | batch efetivo={args.batch_size * args.gradient_accumulation}")

    if args.benchmark:
        run_benchmark(args, df, tokenizer)
    elif args.final:
        run_final(args, df, tokenizer)
    else:
        run_cv(args, df, tokenizer)


if __name__ == "__main__":
    main()
