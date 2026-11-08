"""Cria uma cópia limpa com fontes, dados oficiais e folds; não treina modelos."""
from pathlib import Path
import argparse
import shutil

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destino', type=Path, required=True)
    a = p.parse_args()
    from reproduzir_ep2 import check_files
    check_files(data_only=True)
    target = a.destino.resolve()
    target.relative_to(ROOT / 'results/ep2')
    if target.exists():
        raise ValueError('Destino deve ser novo; nenhum arquivo será apagado.')
    target.mkdir(parents=True)
    shutil.copytree(ROOT / 'src', target / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for source in list(ROOT.glob('requirements*.txt')) + [ROOT / 'README_ENTREGA.md', ROOT / 'REPRODUCAO_EP2.md']:
        shutil.copy2(source, target / source.name)
    for relative in ['data/raw/train.xlsx', 'data/raw/test1.xlsx', 'data/raw/test2.xlsx', 'data/splits/folds.csv']:
        output = target / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, output)
    print(f'Cópia limpa preparada: {target}. Nenhum OOF, modelo treinado, cache ou gabarito de teste foi copiado.')


if __name__ == '__main__':
    main()
