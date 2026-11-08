"""Baixa revisões explícitas para cache isolado; não treina nem altera modelos."""
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[2]
REVISIONS = {
    'neuralmind/bert-base-portuguese-cased': '94d69c95f98f7d5b2a8700c420230ae10def0baa',
    'Itau-Unibanco/NorBERTo-base': 'db73446f89c96044863ea05a39f680524b84bccb',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conferir', action='store_true', help='Não baixa; confere o cache já preparado')
    args = parser.parse_args()
    cache = ROOT / 'results/ep2/hf_cache'
    from huggingface_hub import snapshot_download
    for repo, revision in REVISIONS.items():
        folder = Path(snapshot_download(repo, revision=revision, cache_dir=cache, local_files_only=args.conferir))
        if folder.name != revision:
            raise ValueError(f'Revisão inesperada: {folder}')
        ref = cache / ('models--' + repo.replace('/', '--')) / 'refs/main'
        if args.conferir:
            if ref.read_text().strip() != revision:
                raise ValueError(f'Referência main divergente: {repo}')
        else:
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text(revision, encoding='utf-8')
        print(f'{repo}: {revision}')
    print('Configure HF_HUB_CACHE para este cache e HF_HUB_OFFLINE=1 antes dos scripts históricos.')


if __name__ == '__main__':
    main()
