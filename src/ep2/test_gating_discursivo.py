"""Checks matemáticos, de isolamento e de persistência do gate EP2."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.optimize import check_grad
from scipy.optimize import minimize as scipy_minimize
from threadpoolctl import threadpool_limits

import gating_discursivo as gating
import candidato_incerteza as candidate


class GateTests(unittest.TestCase):
    def test_gradient_against_finite_differences(self):
        rng = np.random.default_rng(15)
        x = rng.normal(size=(31, 4))
        p = rng.dirichlet([2, 2, 2], size=(31, 3))
        y = rng.integers(0, 3, size=31)
        errors = []
        def verified_minimize(fun, x0, **kwargs):
            point = rng.normal(0, .1, size=x0.shape)
            error = check_grad(lambda z: fun(z)[0], lambda z: fun(z)[1], point)
            errors.append(error)
            return scipy_minimize(fun, x0, **kwargs)
        with patch.object(gating, 'minimize', verified_minimize):
            model = gating.ConditionalMixture().fit(x, p, y)
        self.assertLess(max(errors), 1e-6)
        prob = model.predict_proba(x, p)
        np.testing.assert_allclose(prob.sum(1), 1, atol=1e-12)
        self.assertTrue(np.isfinite(prob).all())

    def test_group_isolation_and_complete_evaluation(self):
        train = pd.read_excel(gating.ROOT / 'data/raw/train.xlsx').resp_text.astype(str).to_numpy()
        fd = pd.read_csv(gating.ROOT / 'data/splits/folds.csv').fold.to_numpy()
        evaluated = []
        for fold in range(5):
            records = json.loads((gating.ROOT / f'results/ep2/gating_discursivo/fold{fold}_calibracao.json').read_text(encoding='utf-8'))
            for rec in records:
                cal, val = rec['calibration_row_ids'], rec['evaluation_row_ids']
                self.assertFalse(set(cal) & set(val))
                self.assertFalse(set(train[cal]) & set(train[val]))
                self.assertEqual(set(cal) | set(val), set(np.flatnonzero(fd == fold)))
                self.assertTrue((fd[cal] == fold).all())
                self.assertTrue((fd[val] == fold).all())
                evaluated.extend(val)
        self.assertEqual(sorted(evaluated), list(range(20092)))

    def test_no_inference_before_freeze(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(candidate, 'FREEZE', Path(folder) / 'absent.json'):
                with patch.object(pd, 'read_excel', side_effect=AssertionError('Teste aberto antes do gate')):
                    with self.assertRaisesRegex(ValueError, 'não congelado'):
                        candidate.infer()

    def test_serialization_and_candidate_integrity(self):
        candidate.validate_candidate()
        model = json.loads(candidate.MODEL.read_text(encoding='utf-8'))
        rng = np.random.default_rng(42)
        x = rng.normal(size=(10, 11))
        p = rng.dirichlet([2, 2, 2], size=(10, 7))
        result = candidate.predict_serialized(model, x, p)
        again = candidate.predict_serialized(json.loads(json.dumps(model)), x, p)
        np.testing.assert_array_equal(result, again)
        np.testing.assert_allclose(result.sum(1), 1, atol=1e-12)

    def test_future_inference_export_with_synthetic_inputs_only(self):
        """Exercita 900+897 linhas artificiais; não abre os arquivos oficiais."""
        import inferir_v3_ep2 as historical
        from entrega_ep2 import carregar_planilha
        model_text = candidate.MODEL.read_text(encoding='utf-8')
        rng = np.random.default_rng(9)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            comp = root / 'components'
            comp.mkdir()
            for tag, n in [('test1', 900), ('test2', 897)]:
                pd.DataFrame({'resp_text': [f'Texto sintético {tag} {i}' for i in range(n)],
                              'clarity': [None] * n}).to_excel(root / f'{tag}.xlsx', index=False)
            for base in gating.BASES:
                probs = rng.dirichlet([2, 2, 2], size=1797)
                pd.DataFrame(probs, columns=gating.PROBS).to_csv(comp / f'{base}_final.csv', index=False)
            votes = pd.DataFrame(rng.integers(0, 3, size=(1797, 5)), columns=gating.VOTES)
            votes['y_pred_id'] = rng.integers(0, 3, size=1797)
            votes.to_csv(comp / 'ensemble.csv', index=False)
            model = root / 'model.json'
            model.write_text(model_text, encoding='utf-8')
            manifest = root / 'candidate.json'
            manifest.write_text('{}', encoding='utf-8')
            freeze = root / 'freeze.json'
            freeze.write_text(json.dumps({'candidate_sha256': gating.sha(manifest), 'model_sha256': gating.sha(model)}), encoding='utf-8')
            with patch.multiple(candidate, OUT=root, MODEL=model, MANIFEST=manifest, FREEZE=freeze):
                with patch.object(candidate, 'validate_candidate', return_value={}):
                    with patch.multiple(historical, COMP=comp, ORDER=['ensemble'], PARTS={'ensemble': ('unused.py', 'ensemble.csv')},
                                        ORIGINAL_TEST1=root / 'test1.xlsx', TEST2=root / 'test2.xlsx', MANIFEST=manifest):
                        with patch.object(historical, 'load_manifest', return_value={}):
                            with patch.object(historical, 'check_cache', return_value=True):
                                candidate.infer()
            first = pd.read_csv(root / 'final/predicoes_test1.csv')
            second = pd.read_csv(root / 'final/predicoes_test2.csv')
            self.assertEqual(len(first), 900)
            self.assertEqual(len(second), 897)
            exported, _, _ = carregar_planilha(root / 'final/test2_rotulado_GATE.xlsx', 897, rotulado=True)
            self.assertEqual([r[0] for r in exported], [f'Texto sintético test2 {i}' for i in range(897)])
            self.assertEqual([r[1] for r in exported], second.y_pred.tolist())


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        unittest.main()
