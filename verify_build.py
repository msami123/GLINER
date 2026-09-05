"""Compare two full builds byte-for-byte without modifying their data."""
import argparse

from project_utils import file_hash, resolve_path, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rebuild',help='Directory produced by a second prepare_data.py run')
    args=parser.parse_args()
    files={}
    for path in sorted(resolve_path('data/v2').glob('*')):
        if path.suffix not in {'.json','.jsonl'}: continue
        rebuilt=resolve_path(args.rebuild)/path.name
        first,second=file_hash(path),file_hash(rebuilt)
        if first!=second: raise ValueError(f'Non-reproducible output: {path.name}')
        files[path.name]={'sha256':first,'bytes':path.stat().st_size}
    write_json('reports/build_verification.json',{'status':'pass','identical_full_rebuild':True,
               'files':files,'gpu_training_performed':False})
    print(f'PASS: {len(files)} outputs byte-identical across two complete source scans.')


if __name__=='__main__': main()
