"""Offline validation of raw spans, negative targets and split isolation."""
from __future__ import annotations

import argparse
from collections import Counter

from data_contract import LABEL, group_id, sample_id, token_spans
from project_utils import load_config, load_jsonl, resolve_path, write_json


def validate_row(row, label=LABEL):
    text = row['text']
    assert text.strip(), 'empty input'
    assert row['sample_id'] == sample_id(text), 'sample hash mismatch'
    assert row['group_id'] == group_id(text), 'template hash mismatch'
    assert row['ner_labels'] == [label], 'explicit label required, including negatives'
    spans = []
    for entity in row['entities']:
        start, end = entity['start'], entity['end']
        assert 0 <= start < end <= len(text), 'invalid character span'
        assert text[start:end] == entity['text'], 'not an original substring'
        assert entity['label'] == label, 'unexpected label'
        spans.append((start, end))
    assert len(spans) == len(set(spans)), 'duplicate spans'
    tokens, ner = token_spans(text, spans)
    assert tokens == row['tokenized_text'], 'GLiNER tokenizer mismatch'
    assert ner == row['ner'], 'character/token annotation mismatch'
    assert len(tokens) <= 384, 'input exceeds max_len'
    assert all(e-s+1 <= 12 for s,e,_ in ner), 'entity exceeds max_width'
    assert all(a[1] < b[0] for a,b in zip(ner,ner[1:])), 'overlapping spans'
    if not ner:
        assert row['annotation_status'] in {'verified_structure_negative','agent_reviewed'}, 'unknown converted to negative'
    return row


def validate_split(path, expected_label=LABEL):
    rows = load_jsonl(path)
    if not rows: raise ValueError(f'Empty split: {path}')
    for index,row in enumerate(rows,1):
        try: validate_row(row,expected_label)
        except (AssertionError,KeyError,ValueError,TypeError) as exc:
            raise ValueError(f'{path} row {index}: {exc}') from exc
    ids = {r['sample_id'] for r in rows}
    if len(ids) != len(rows): raise ValueError(f'Duplicate normalized texts in {path}')
    report = {
        'examples':len(rows), 'positive_examples':sum(bool(r['ner']) for r in rows),
        'negative_examples':sum(not r['ner'] for r in rows),
        'spans':sum(len(r['ner']) for r in rows), 'duplicates':0,
        'groups':len({r['group_id'] for r in rows}),
        'banks':dict(Counter(b for r in rows for b in r['bank_ids'])),
        'bank_unknown':sum(not r['bank_ids'] for r in rows),
        'annotation_status':dict(Counter(r['annotation_status'] for r in rows)),
        'max_tokens':max(len(r['tokenized_text']) for r in rows),
    }
    return report, ids


def validate_all(config):
    reports, ids, groups = {}, {}, {}
    for name,path in config['data'].items():
        reports[name],ids[name] = validate_split(resolve_path(path),config['label'])
        groups[name] = {r['group_id'] for r in load_jsonl(path)}
    for i,left in enumerate(ids):
        for right in list(ids)[i+1:]:
            if ids[left] & ids[right]: raise ValueError(f'Exact text leakage: {left}/{right}')
            if groups[left] & groups[right]: raise ValueError(f'Template leakage: {left}/{right}')
    reports['exact_text_overlap'] = reports['template_group_overlap'] = 0
    return reports


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config.json')
    args=parser.parse_args()
    report=validate_all(load_config(args.config))
    write_json('reports/data_validation.json',report)
    for name,values in report.items(): print(f'{name}: {values}')
    print('PASS: raw spans valid; no normalized-text or template-group leakage.')


if __name__=='__main__': main()
