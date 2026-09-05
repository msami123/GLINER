import copy
import json
import tempfile
import unittest
from pathlib import Path

from data_contract import LABEL, TOKEN_RE, group_id, token_spans, training_view
from evaluate import score_dataset
from prepare_data import ROOT, all_occurrences, build_record, extract, iter_jsonl
from project_utils import best_model, latest_checkpoint, load_config
from validate_data import validate_all, validate_row


def record(text,signals):
    spans=[p for signal in signals for p in all_occurrences(text,signal)]
    return build_record(text,spans,'agent_reviewed','test',set(),'test')


class DatasetTests(unittest.TestCase):
    def test_real_merchant_channel_suffix(self):
        text='Online Purchase from Mcdonalds Riyadh Mada Riyadh'
        spans,_,_=extract(text)
        self.assertEqual([text[s:e] for s,e in spans],['Mcdonalds'])

    def test_real_location_only(self):
        spans,status,_=extract('CITY:DAMMAM مدى:**9434 11588850 DAMMAM')
        self.assertEqual(spans,[])
        self.assertEqual(status,'verified_structure_negative')

    def test_real_full_name_internal_city(self):
        text='(6475579200313002-605506020273) RIYADH CARE HOSPITAL RIYADH SA'
        spans,_,_=extract(text)
        self.assertEqual([text[s:e] for s,e in spans],['RIYADH CARE HOSPITAL'])

    def test_real_apple_is_not_apple_pay(self):
        text='4092013488    : APPLE.COM/BILL CORK IE'
        spans,_,_=extract(text)
        self.assertEqual([text[s:e] for s,e in spans],['APPLE.COM/BILL'])

    def test_withdrawal_action_without_channel(self):
        text='ATM Cash Withdrawal'
        spans,_,_=extract(text)
        self.assertEqual([text[s:e] for s,e in spans],['Cash Withdrawal'])

    def test_preserves_specific_commission(self):
        text='Cash Withdrawal Commission for Card ProductCIIA for currency SAR'
        spans,_,_=extract(text)
        self.assertEqual([text[s:e] for s,e in spans],['Cash Withdrawal Commission'])

    def test_unknown_is_not_negative(self):
        self.assertIsNone(extract('الباقي')[0])
        with self.assertRaises(ValueError):
            build_record('something',[],'rule_labeled','test',set(),'test')

    def test_reference_person_not_merchant(self):
        self.assertIsNone(extract('9117022600056377/عبدالمحسن مهاوش الشمري',['عبدالمحسن مهاوش الشمري'])[0])

    def test_long_identifier_not_merchant(self):
        self.assertIsNone(extract('SMRT8181B596C2DE40EF817FE',['SMRT8181B596C2DE40EF817FE'])[0])

    def test_hyphenated_word_is_single_token(self):
        self.assertEqual(TOKEN_RE.findall('Pik-SA H&M'),['Pik-SA','H','&','M'])
        with self.assertRaises(ValueError): token_spans('Pik-SA',[(0,3)])

    def test_references_and_case_same_group(self):
        self.assertEqual(group_id('CITY:DAMMAM مدى:**9434 11588850 DAMMAM'),
                         group_id('city:dammam مدى:**2544 11588848 dammam'))

    def test_negative_has_class_label(self):
        row=record(': 300002512',[])
        self.assertEqual(training_view(row),{'tokenized_text':[':', '300002512'],'ner':[],'ner_labels':[LABEL]})
        validate_row(row)

    def test_annotation_mismatch_rejected(self):
        row=record('ATM Cash Withdrawal',['Cash Withdrawal'])
        row['entities'][0]['end']-=1
        with self.assertRaises(AssertionError): validate_row(row)

    def test_all_reviewed_examples(self):
        for override in json.loads((ROOT/'annotation_overrides.json').read_text(encoding='utf-8')):
            validate_row(record(override['text'],override['signals']))

    def test_all_splits(self):
        report=validate_all(load_config())
        self.assertEqual(report['template_group_overlap'],0)
        self.assertGreater(report['train']['negative_examples'],0)
        self.assertEqual(len(report['train']['banks']),8)

    def test_evaluation_receives_original_text(self):
        rows=[record('4092013488    : APPLE.COM/BILL CORK IE',['APPLE.COM/BILL']),record(': 300002512',[])]
        class FakeModel:
            def inference(self,texts,labels,**kwargs):
                self.texts=texts
                return [[{'start':0,'end':1,'text':':','label':LABEL}] if text==': 300002512' else rows[0]['entities'] for text in texts]
        model=FakeModel()
        report=score_dataset(model,rows,LABEL)
        self.assertEqual(model.texts,[r['text'] for r in rows])
        self.assertEqual((report['tp'],report['fp'],report['fn']),(1,1,0))
        self.assertEqual(report['negative_false_extraction_rate'],1.0)

    def test_checkpoint_choice(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'checkpoint-5').mkdir()
            (root/'checkpoint-10').mkdir()
            (root/'best').mkdir()
            (root/'best'/'gliner_config.json').touch()
            self.assertEqual(best_model(root),root/'best')
            self.assertEqual(latest_checkpoint(root),root/'checkpoint-10')


if __name__=='__main__': unittest.main()
