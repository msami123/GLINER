"""Measure extraction on ORIGINAL issue strings, including false extractions."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone

from project_utils import best_model, file_hash, load_config, load_jsonl, resolve_path, write_json


def prediction_span(entity, text):
    start,end=int(entity['start']),int(entity['end'])
    if not (0<=start<end<=len(text)) or text[start:end]!=entity['text']:
        raise ValueError('Prediction offsets do not match the raw issue')
    return start,end,str(entity['label'])


def metrics(counts):
    tp,fp,fn=counts['tp'],counts['fp'],counts['fn']
    precision=tp/(tp+fp) if tp+fp else 0.0
    recall=tp/(tp+fn) if tp+fn else 0.0
    return {
        'examples':counts['examples'],'tp':tp,'fp':fp,'fn':fn,
        'precision':precision,'recall':recall,
        'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
        'exact_document_accuracy':counts['exact']/counts['examples'] if counts['examples'] else None,
        'negative_examples':counts['negatives'],
        'negative_false_extractions':counts['negative_fp'],
        'negative_false_extraction_rate':counts['negative_fp']/counts['negatives'] if counts['negatives'] else None,
    }


def score_dataset(model, rows, label, threshold=0.5, batch_size=4):
    total=Counter()
    banks,rules=defaultdict(Counter),defaultdict(Counter)
    mistakes=[]
    for offset in range(0,len(rows),batch_size):
        batch=rows[offset:offset+batch_size]
        # Crucially, never rebuild text with ' '.join(tokens).
        results=model.inference([r['text'] for r in batch],[label],threshold=threshold,
                                flat_ner=True,batch_size=batch_size)
        if len(results)!=len(batch): raise ValueError('Missing batch predictions')
        for row,entities in zip(batch,results):
            gold={(e['start'],e['end'],e['label']) for e in row['entities']}
            predicted={prediction_span(e,row['text']) for e in entities}
            c=Counter(tp=len(gold & predicted),fp=len(predicted-gold),fn=len(gold-predicted),
                      examples=1,exact=int(gold==predicted),negatives=int(not gold),
                      negative_fp=int(not gold and bool(predicted)))
            total.update(c)
            for bank in row.get('bank_ids') or ['unknown']: banks[bank].update(c)
            rules[row['annotation_rule']].update(c)
            if gold!=predicted and len(mistakes)<100:
                mistakes.append({'sample_id':row['sample_id'],'text':row['text'],
                                 'gold':row['entities'],'predicted':entities})
    return {**metrics(total),'threshold':threshold,'by_bank':{b:metrics(c) for b,c in banks.items()},
            'by_rule':{r:metrics(c) for r,c in rules.items()},'mistake_samples':mistakes,
            'limitation':'Provisional rule/source/assistant labels, not independent human gold.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config.json')
    parser.add_argument('--split',choices=['validation','test','sanity'],default='validation')
    choice=parser.add_mutually_exclusive_group()
    choice.add_argument('--model',help='Local checkpoint directory')
    choice.add_argument('--base-model',action='store_true',help='Evaluate the unchanged pretrained model')
    parser.add_argument('--threshold',type=float)
    args=parser.parse_args()
    import torch
    from gliner import GLiNER
    config=load_config(args.config)
    model_path=config['model_name'] if args.base_model else str(resolve_path(args.model) if args.model else best_model(config['output_dir']))
    threshold=args.threshold if args.threshold is not None else config['evaluation']['threshold']
    if not 0<threshold<1: raise ValueError('threshold must be between 0 and 1')
    model=GLiNER.from_pretrained(model_path,map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.inference_mode():
        report=score_dataset(model,load_jsonl(config['data'][args.split]),config['label'],threshold,config['evaluation']['batch_size'])
    report.update(model=model_path,split=args.split,data_sha256=file_hash(config['data'][args.split]))
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path=f'reports/{args.split}_{stamp}.json'
    write_json(path,report)
    print(f"F1={report['f1']:.4f}; precision={report['precision']:.4f}; recall={report['recall']:.4f}")
    print(f"Negative false extraction rate={report['negative_false_extraction_rate']}; report: {resolve_path(path)}")


if __name__=='__main__': main()
