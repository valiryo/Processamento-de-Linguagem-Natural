"""EP2: mistura condicional de especialistas com calibração separada por grupos.

Somente treino/OOF. Para cada fold externo, os modelos-base não viram nenhuma
linha dele. Duas metades por grupo se alternam entre calibração e avaliação do
gate. Não se treina o gate em OOFs dos outros folds (vazamento indireto).
As partições internas NÃO substituem os cinco folds históricos.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from discurso_features import FEATURE_NAMES, features

ROOT = Path(__file__).resolve().parents[2]
LABELS = ["c1", "c234", "c5"]
PROBS = ["prob_c1", "prob_c234", "prob_c5"]
VOTES = ["pred_bert512", "pred_soft256", "pred_svc_word_char", "pred_baseline", "pred_ordinal256_corrected"]
BASES = ["bertimbau_512", "bertimbau_duplicate_soft_labels_256", "bertimbau_ordinal_256", "norberto_base_1024"]
EXPERTS = BASES + ["svc_voto", "baseline_voto", "v3_voto"]
VARIANTS = ["constante", "incerteza", "discurso"]
CONFIG = {
    "seed": 42, "inner_splits": 2, "l2_gate": 0.02, "l2_bias": 0.02,
    "maxiter": 300, "hard_vote_probability": 0.6,
    "experts": EXPERTS, "variants": VARIANTS,
    "selection": "discurso somente se accuracy > V3, > incerteza e > constante e ganho em >=3/5 folds; senão V3",
    "protocol": "calibrar em metade do MESMO fold OOF e avaliar na outra, grupos por texto; alternar metades",
    "test_data_read": False,
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def metrics(y, p):
    return {"n": len(y), "correct": int((y == p).sum()),
            "accuracy": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0)),
            "recall": dict(zip(LABELS, recall_score(y, p, labels=[0, 1, 2], average=None, zero_division=0).tolist())),
            "confusion_matrix": confusion_matrix(y, p, labels=[0, 1, 2]).tolist()}


def validate_aligned(df, y, folds, name):
    if len(df) != len(y) or not np.array_equal(df.row_id.to_numpy(), np.arange(len(y))):
        raise ValueError(f"{name}: row_id ausente/duplicado/desalinhado")
    if not np.array_equal(df.fold.to_numpy(), folds) or not np.array_equal(df.y_true_id.to_numpy(), y):
        raise ValueError(f"{name}: folds/rótulos divergentes")


def load_train():
    train_path = ROOT / "data/raw/train.xlsx"
    folds_path = ROOT / "data/splits/folds.csv"
    v3_path = ROOT / "results/ensemble_majority_5_v3_norberto_oof.csv"
    train = pd.read_excel(train_path)
    fd = pd.read_csv(folds_path)
    v3 = pd.read_csv(v3_path).sort_values("row_id")
    if len(train) != 20092 or len(fd) != len(train) or train[["resp_text", "clarity"]].isna().any().any():
        raise ValueError("Treino/folds inválidos")
    numeric_texts = int(train.resp_text.map(lambda x: not isinstance(x, str)).sum())
    train['resp_text'] = train.resp_text.astype(str)
    if not np.array_equal(fd.row_index.to_numpy(), np.arange(len(train))):
        raise ValueError("Ordem do folds.csv inválida")
    y = train.clarity.map(dict(zip(LABELS, range(3)))).to_numpy()
    if not np.isin(y, [0, 1, 2]).all():
        raise ValueError("Classe desconhecida")
    y = y.astype(int)
    folds = fd.fold.to_numpy(int)
    if set(folds) != set(range(5)):
        raise ValueError("Folds devem ser 0..4")
    validate_aligned(v3, y, folds, "V3")
    if train.assign(fold=folds).groupby("resp_text").fold.nunique().max() != 1:
        raise ValueError("Textos idênticos atravessam folds")
    sources = [train_path, folds_path, v3_path, Path(__file__), Path(__file__).with_name("discurso_features.py")]
    arrays = []
    for name in BASES:
        path = ROOT / "results" / (name + "_oof.csv")
        d = pd.read_csv(path).sort_values("row_id")
        validate_aligned(d, y, folds, name)
        if "resp_text" in d and d.resp_text.tolist() != train.resp_text.tolist():
            raise ValueError(f"Textos desalinhados: {name}")
        prob = d[PROBS].to_numpy(float)
        if not np.isfinite(prob).all() or (prob < 0).any() or not np.allclose(prob.sum(1), 1, atol=1e-5):
            raise ValueError(f"Probabilidades inválidas: {name}")
        arrays.append(prob)
        sources.append(path)
    for col in ["pred_svc_word_char", "pred_baseline", "y_pred_id_v3"]:
        pred = v3[col].to_numpy(int)
        if not np.isin(pred, [0, 1, 2]).all():
            raise ValueError("Votos inválidos")
        q = np.full((len(y), 3), (1 - CONFIG["hard_vote_probability"]) / 2)
        q[np.arange(len(y)), pred] = CONFIG["hard_vote_probability"]
        arrays.append(q)
    # Reproduz a regra pelo código histórico, sem escrever em results/EP1.
    sys.path.insert(0, str(ROOT))
    from src.experiments.ensemble_v3_final import majority_vote_v2
    from src.experiments.ensemble_majority_5_v3_norberto import apply_v3_rule
    original = v3.copy()
    original["y_pred_id_ensemble"] = majority_vote_v2(v3[VOTES].to_numpy(int))
    original["y_true_id_ensemble"] = y
    reproduced = apply_v3_rule(original).y_pred_id_v3.to_numpy(int)
    if not np.array_equal(reproduced, v3.y_pred_id_v3.to_numpy(int)):
        raise ValueError("V3 não reproduzido pela regra histórica")
    sources += [ROOT / "src/experiments/ensemble_v3_final.py", ROOT / "src/experiments/ensemble_majority_5_v3_norberto.py"]
    tensor = np.stack(arrays, axis=1)
    grouped = train.groupby("resp_text").clarity.agg(["size", "nunique"])
    audit = {"n": len(y), "numeric_texts_cast_to_string": numeric_texts, "unique_texts": len(grouped), "duplicate_groups": int((grouped['size'] > 1).sum()),
             "conflicting_groups": int((grouped['nunique'] > 1).sum()),
             "class_counts": train.clarity.value_counts().to_dict(),
             "v3": metrics(y, reproduced), "v3_reproduced": True,
             "folds": {str(f): metrics(y[folds == f], reproduced[folds == f]) for f in range(5)},
             "sources_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources},
             "versions": {x: importlib.metadata.version(x) for x in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'openpyxl']},
             "test_labels_read": False}
    return train.resp_text.to_numpy(), y, folds, v3, tensor, audit


def uncertainty(tensor, votes):
    neural = np.clip(tensor[:, :4], 1e-9, 1)
    ent = -(neural * np.log(neural)).sum(2) / np.log(3)
    sorted_p = np.sort(neural, axis=2)
    margin = sorted_p[:, :, -1] - sorted_p[:, :, -2]
    fractions = np.column_stack([(votes == c).mean(1) for c in range(3)])
    names = [f"entropy_{b}" for b in BASES] + [f"margin_{b}" for b in BASES] + [f"vote_fraction_{c}" for c in LABELS]
    return np.column_stack([ent, margin, fractions]), names


class ConditionalMixture:
    """Softmax gate, mixture of fixed probabilities, regularized class intercepts."""
    def fit(self, x, prob, y):
        self.scaler = StandardScaler().fit(x) if x.shape[1] else None
        z = self.design(x)
        k = prob.shape[1]
        shape = (z.shape[1], k)
        n = len(y)
        def objective(theta):
            w = theta[:-3].reshape(shape)
            bias = theta[-3:]
            weights = softmax(z @ w, axis=1)
            mix = np.clip(np.einsum('nk,nkc->nc', weights, prob), 1e-12, 1)
            pred = softmax(np.log(mix) + bias, axis=1)
            loss = -np.log(pred[np.arange(n), y]).mean()
            loss += .5 * CONFIG['l2_gate'] * (w * w).sum() + .5 * CONFIG['l2_bias'] * (bias * bias).sum()
            delta = pred.copy()
            delta[np.arange(n), y] -= 1
            delta /= n
            dmix = delta / mix
            dw = np.einsum('nc,nkc->nk', dmix, prob)
            dlogit = weights * (dw - (dw * weights).sum(1, keepdims=True))
            gradw = z.T @ dlogit + CONFIG['l2_gate'] * w
            gradb = delta.sum(0) + CONFIG['l2_bias'] * bias
            return loss, np.r_[gradw.ravel(), gradb]
        result = minimize(objective, np.zeros(np.prod(shape) + 3), jac=True, method='L-BFGS-B',
                          options={'maxiter': CONFIG['maxiter'], 'ftol': 1e-10})
        if not result.success:
            raise RuntimeError(f"Otimização não convergiu: {result.message}")
        self.w = result.x[:-3].reshape(shape)
        self.bias = result.x[-3:]
        self.iterations = int(result.nit)
        return self

    def design(self, x):
        z = self.scaler.transform(x) if self.scaler is not None else x
        return np.column_stack([np.ones(len(x)), z])

    def predict_proba(self, x, prob):
        weights = softmax(self.design(x) @ self.w, axis=1)
        mix = np.einsum('nk,nkc->nc', weights, prob)
        return softmax(np.log(np.clip(mix, 1e-12, 1)) + self.bias, axis=1)


def run(args):
    started = perf_counter()
    texts, y, folds, v3, tensor, audit = load_train()
    args.out.mkdir(parents=True, exist_ok=True)
    dump(args.out / "auditoria.json", audit)
    print(f"INTEGRIDADE OK: {len(y)} linhas; {audit['unique_texts']} textos; V3={audit['v3']['accuracy']:.8f}", flush=True)
    if args.check_data:
        return
    manifest = {"configuration": CONFIG, "sources_sha256": audit['sources_sha256']}
    path = args.out / "protocolo_pre_especificado.json"
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != manifest:
        raise ValueError("Configuração/fontes mudaram; use outro diretório de saída")
    dump(path, manifest)
    cache = args.out / "features.npy"
    if cache.exists():
        disc = np.load(cache)
    else:
        rows = []
        for i, t in enumerate(texts):
            rows.append(features(t))
            if (i + 1) % 2500 == 0:
                print(f"Atributos {i+1}/{len(y)}", flush=True)
        disc = np.stack(rows)
        np.save(cache, disc)
    if disc.shape != (len(y), len(FEATURE_NAMES)) or not np.isfinite(disc).all():
        raise ValueError("Cache de atributos inválido")
    unc, unc_names = uncertainty(tensor, v3[VOTES].to_numpy())
    matrices = {"constante": np.empty((len(y), 0)), "incerteza": unc, "discurso": np.column_stack([unc, disc])}
    names = {"constante": [], "incerteza": unc_names, "discurso": unc_names + FEATURE_NAMES}
    selected = [args.fold] if args.fold is not None else list(range(5))
    for fold in selected:
        output = args.out / f"fold{fold}.csv"
        if output.exists():
            print(f"Retomada: fold {fold} já concluído", flush=True)
            continue
        idx = np.flatnonzero(folds == fold)
        oof = pd.DataFrame({'row_id': idx, 'fold': fold, 'y_true_id': y[idx], 'pred_v3': v3.y_pred_id_v3.to_numpy()[idx]})
        splitter = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
        provenance = []
        for half, (cal, val) in enumerate(splitter.split(idx, y[idx], texts[idx])):
            train_ids, eval_ids = idx[cal], idx[val]
            if set(texts[train_ids]) & set(texts[eval_ids]):
                raise ValueError("Colisão de grupos na calibração")
            oof.loc[val, 'half'] = half
            record = {'fold': fold, 'half': half, 'calibration_row_ids': train_ids.tolist(), 'evaluation_row_ids': eval_ids.tolist(), 'models': {}}
            for variant in VARIANTS:
                x = matrices[variant]
                model = ConditionalMixture().fit(x[train_ids], tensor[train_ids], y[train_ids])
                p = model.predict_proba(x[eval_ids], tensor[eval_ids])
                oof.loc[val, 'pred_' + variant] = p.argmax(1)
                for c, label in enumerate(LABELS):
                    oof.loc[val, f'{variant}_{label}'] = p[:, c]
                record['models'][variant] = {'iterations': model.iterations, 'features': names[variant],
                    'weights': model.w.tolist(), 'bias': model.bias.tolist(),
                    'scaler_mean': None if model.scaler is None else model.scaler.mean_.tolist(),
                    'scaler_scale': None if model.scaler is None else model.scaler.scale_.tolist()}
            provenance.append(record)
        for c in ['half'] + ['pred_' + v for v in VARIANTS]:
            oof[c] = oof[c].astype(int)
        dump(args.out / f"fold{fold}_calibracao.json", provenance)
        oof.to_csv(output, index=False)
        print(f"Fold {fold}: " + ' | '.join(f"{v}={accuracy_score(oof.y_true_id, oof['pred_'+v]):.4%}" for v in ['v3'] + VARIANTS), flush=True)
    files = [args.out / f'fold{f}.csv' for f in range(5)]
    if not all(p.exists() for p in files):
        print("Piloto concluído; ainda não há OOF completo.", flush=True)
        return
    oof = pd.concat([pd.read_csv(p) for p in files]).sort_values('row_id').reset_index(drop=True)
    if not np.array_equal(oof.row_id, np.arange(len(y))) or not np.array_equal(oof.y_true_id, y) or not np.array_equal(oof.fold, folds):
        raise ValueError("OOF agregado inválido")
    summary = {'protocol': CONFIG, 'seconds': round(perf_counter() - started, 2), 'models': {}, 'folds': []}
    table = []
    for model in ['v3'] + VARIANTS:
        p = oof['pred_' + model].to_numpy(int)
        m = metrics(y, p)
        original = oof.pred_v3.to_numpy()
        m.update({'corrected': int(((original != y) & (p == y)).sum()), 'broken': int(((original == y) & (p != y)).sum())})
        summary['models'][model] = m
        table.append({'model': model, **{k: m[k] for k in ['accuracy', 'macro_f1', 'corrected', 'broken']}, 'recall_c234': m['recall']['c234']})
        for f in range(5):
            mask = folds == f
            summary['folds'].append({'fold': f, 'model': model, **metrics(y[mask], p[mask])})
    ms = summary['models']
    wins = sum(r['accuracy'] > audit['folds'][str(r['fold'])]['accuracy'] for r in summary['folds'] if r['model'] == 'discurso')
    promote = all(ms['discurso']['accuracy'] > ms[b]['accuracy'] for b in ['v3', 'incerteza', 'constante']) and wins >= 3
    summary['decision'] = {'promote_discourse': promote, 'fold_wins': wins, 'recommended': 'discurso' if promote else 'v3',
                           'warning': 'Desenvolvimento exploratório; seleção e modelos-base já utilizaram estes folds historicamente. Não é teste confirmatório.'}
    dump(args.out / 'resumo.json', summary)
    oof.to_csv(args.out / 'oof_completo.csv', index=False)
    pd.DataFrame(table).to_csv(args.out / 'comparacao.csv', index=False)
    pd.DataFrame(summary['folds']).drop(columns=['recall', 'confusion_matrix']).to_csv(args.out / 'por_fold.csv', index=False)
    # Exemplos determinísticos, apenas do treino. Não confundir com análise humana.
    errors = oof.loc[(oof.pred_discurso != y) | (oof.pred_discurso != oof.pred_v3)].copy()
    errors['texto_trecho'] = [texts[i][:650] for i in errors.row_id]
    errors.groupby(['y_true_id', 'pred_discurso'], group_keys=False).head(5).to_csv(args.out / 'exemplos_erros.csv', index=False)
    print(pd.DataFrame(table).to_string(index=False), flush=True)
    print(json.dumps(summary['decision'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-data', action='store_true')
    parser.add_argument('--fold', type=int, choices=range(5))
    parser.add_argument('--out', type=Path, default=ROOT / 'results/ep2/gating_discursivo')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args)
