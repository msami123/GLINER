"""CUDA fine-tuning with raw-text validation, best model selection and resume."""
from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version

from data_contract import training_view
from evaluate import score_dataset
from project_utils import file_hash, latest_checkpoint, load_config, load_jsonl, resolve_path, write_json
from validate_data import validate_all


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config.json')
    parser.add_argument('--output',help='Use a separate directory for a new experiment')
    parser.add_argument('--resume',help='latest, or a checkpoint path inside this run')
    parser.add_argument('--smoke-test',action='store_true',help='Five steps on 32 mixed examples, separate output')
    args=parser.parse_args()
    import torch
    from gliner import GLiNER
    from gliner.training import Trainer
    from transformers import TrainerCallback, set_seed
    if version('gliner')!='0.2.28' or version('transformers')!='4.57.6':
        raise SystemExit('Install the pinned requirements; this trainer targets GLiNER 0.2.28 / Transformers 4.57.6.')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA not detected. Run check_environment.py; do not start long training on CPU.')
    config=load_config(args.config)
    report=validate_all(config)
    training=dict(config['training'])
    seed=training['seed']
    set_seed(seed)
    train_rows=load_jsonl(config['data']['train'])
    validation_rows=load_jsonl(config['data']['validation'])
    output=resolve_path(args.output or ('models/smoke-v2' if args.smoke_test else config['output_dir']))
    if args.smoke_test:
        train_rows=[r for r in train_rows if r['ner']][:24]+[r for r in train_rows if not r['ner']][:8]
        validation_rows=[r for r in validation_rows if r['ner']][:24]+[r for r in validation_rows if not r['ner']][:8]
        training.update(max_steps=5,save_steps=5,logging_steps=1,warmup_steps=0,gradient_accumulation_steps=1)
    resume=latest_checkpoint(output) if args.resume=='latest' else resolve_path(args.resume) if args.resume else None
    if resume and (resume.parent.resolve()!=output.resolve() or not (resume/'trainer_state.json').exists()):
        raise SystemExit('Resume must point to a training checkpoint inside this output directory.')
    hashes={s:file_hash(config['data'][s]) for s in ('train','validation')}
    manifest_path=output/'run_manifest.json'
    signature={'model_name':config['model_name'],'label':config['label'],'data_sha256':hashes,
               'training':training,'evaluation':config['evaluation'],'smoke_test':args.smoke_test}
    if resume:
        previous=json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous['signature']!=signature:
            raise SystemExit('Data/config changed. Use a NEW output directory; exact resume requires the original settings.')
    elif output.exists() and any(output.iterdir()):
        raise SystemExit(f'Output is not empty: {output}. Use --resume latest or a NEW --output.')
    output.mkdir(parents=True,exist_ok=True)
    use_bf16=bool(training['bf16'] and torch.cuda.is_bf16_supported())
    manifest={'started_at':datetime.now(timezone.utc).isoformat(),'signature':signature,
              'python':sys.version,'platform':platform.platform(),'gpu':torch.cuda.get_device_name(0),
              'versions':{p:version(p) for p in ('torch','gliner','transformers','accelerate')},
              'bf16':use_bf16,'train_examples':len(train_rows),'validation_examples':len(validation_rows)}
    if not resume: write_json(manifest_path,manifest)
    model=GLiNER.from_pretrained(str(resume) if resume else config['model_name']).to(dtype=torch.float32)
    for row in train_rows[:32]+validation_rows[:32]:
        actual=[t[0] for t in model.data_processor.words_splitter(row['text'])]
        if actual!=row['tokenized_text']: raise ValueError('Runtime splitter differs from prepared data')
    if model.config.max_width<12 or model.config.max_len<report['train']['max_tokens']:
        raise ValueError('Model span/input limits do not support this dataset')

    class OOMFlag(logging.Handler):
        happened=False
        def emit(self,record):
            if 'Skipping batch due to' in record.getMessage(): self.happened=True

    flag=OOMFlag()
    source_logger=logging.getLogger('gliner.training.trainer')
    source_logger.addHandler(flag)

    class StrictTrainer(Trainer):
        def _load_from_checkpoint(self,checkpoint,model=None):
            # GLiNER saves the INNER model keys, whereas HF's generic loader
            # targets the OUTER wrapper. GLiNER.from_pretrained above already
            # restored the weights correctly. Trainer.train still restores
            # optimizer, scheduler, RNG and global step from this checkpoint.
            if resume is None or resolve_path(checkpoint).resolve()!=resume.resolve():
                raise ValueError('Checkpoint was not preloaded through GLiNER')

        def training_step(self,*a,**kw):
            flag.happened=False
            loss=super().training_step(*a,**kw)
            if flag.happened:
                raise RuntimeError('Training aborted after GPU memory exhaustion. Lower batch size in a NEW run; do not accept skipped batches.')
            if not torch.isfinite(loss).all(): raise RuntimeError('Non-finite training loss; run aborted.')
            return loss

    class RawValidation(TrainerCallback):
        def __init__(self):
            self.best=-1.0
            self.last_step=-1
            state_path=output/'best_state.json'
            if resume and state_path.exists():
                self.best=json.loads(state_path.read_text(encoding='utf-8'))['f1']

        def measure(self,state,current_model):
            if self.last_step==state.global_step: return
            was_training=current_model.training
            current_model.eval()
            try:
                with torch.inference_mode():
                    result=score_dataset(current_model,validation_rows,config['label'],
                                         config['evaluation']['threshold'],config['evaluation']['batch_size'])
            finally: current_model.train(was_training)
            result.update(step=state.global_step,split='validation',data_sha256=hashes['validation'])
            write_json(output/f'validation-step-{state.global_step}.json',result)
            print(f"Validation at step {state.global_step}: F1={result['f1']:.4f}, negative false extraction rate={result['negative_false_extraction_rate']}",flush=True)
            if result['f1']>self.best:
                current_model.save_pretrained(str(output/'best'),safe_serialization=True)
                self.best=result['f1']
                write_json(output/'best_state.json',{'f1':self.best,'step':state.global_step,
                                                    'threshold':config['evaluation']['threshold']})
            self.last_step=state.global_step

        def on_save(self,args,state,control,model=None,**kwargs): self.measure(state,model)
        def on_train_end(self,args,state,control,model=None,**kwargs): self.measure(state,model)

    keyword_map={'train_batch_size':'per_device_train_batch_size','eval_batch_size':'per_device_eval_batch_size',
                 'scheduler_type':'lr_scheduler_type'}
    kwargs={keyword_map.get(k,k):v for k,v in training.items() if k!='bf16'}
    train_args=model.create_training_args(output_dir=str(output),bf16=use_bf16,
        eval_strategy='no',save_strategy='steps',dataloader_num_workers=0,
        report_to='none',remove_unused_columns=False,save_safetensors=True,**kwargs)
    collator=model._create_data_collator()
    for expect_positive in (False,True):
        example=next(r for r in train_rows if bool(r['ner'])==expect_positive)
        labels=collator([training_view(example)])['labels']
        if labels.numel()==0 or bool(torch.count_nonzero(labels))!=expect_positive:
            raise ValueError('Runtime collator failed positive/empty-target preflight')
    trainer=StrictTrainer(model=model,args=train_args,
        train_dataset=[training_view(r) for r in train_rows],
        data_collator=collator,
        tokenizer=model.data_processor.transformer_tokenizer,callbacks=[RawValidation()])
    print(f'Training: {len(train_rows)} examples, {len(validation_rows)} validation, output {output}',flush=True)
    try:
        trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        trainer.save_model(str(output/'final'))
        trainer.save_state()
        write_json(output/'training_complete.json',{'completed_at':datetime.now(timezone.utc).isoformat(),
                   'global_step':trainer.state.global_step,'best_state':json.loads((output/'best_state.json').read_text())})
    finally: source_logger.removeHandler(flag)
    print(f'Complete. Use {output / "best"} for inference; checkpoints are for exact resume.')


if __name__=='__main__': main()
