"""Build v2 using explicit narrative boundaries and traceable source annotations.

No old-engine classification/confidence/merchant_used/context keywords are labels.
Unresolved texts are quarantined, not converted to empty targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from data_contract import LABEL, TOKEN_RE, group_id, normalize, sample_id, token_spans

ROOT = Path(__file__).resolve().parent
CITIES = '''RIYADH|JEDDAH|JIDDAH|JEDDA|JED|MAKKAH|MECCA|MADINAH|MADINA|MADENAH|MEDINA|DAMMAM|DAMAM|KHOBAR|DHAHRAN|ALDHHRAN|TAIF|TABUK|TABOUK|BURAIDAH|BURAYDAH|KHAMIS MUSHAIT|KHAMIS|HAIL|NAJRAN|JAZAN|JIZAN|YANBU|JUBAIL|JUBAYL|ABHA|QATIF|QATEEF|HOFUF|HAFOUF|MUBARRAZ|AL AHSA|AL-AHSA|AHSA|ALKHOBAR|AL KHOBAR|AL-KHOBAR|ARTAWEYYAH|MAHAIL ASIR|RAFHA|MAJARDAH|SAKAKA|ARAR|BISHA|KHARJ|QASSIM|QASIM|HAFAR AL BATIN|HAFR AL BATIN|HAFR ALBATIN|UNAIZAH|UNIZAH|ONAIZAH|RABIGH|RABEGH|JOMOUM|SHARURAH|THUWAL|KHAFJI|RAS TANURA|SAFWA|QURAYYAT|RUFAIDAH|ALMAA|AL MAA|KHULAIS|TURAIF|DAWADMI|LAYLA|AL BAHA|ALBAHA|SABYA|ABU ARISH|SAMTAH|AL DUWADIMI|AMSTERDAM|LONDON|TALLINN|DUBLIN|CORK|MOUNTAIN VIEW'''.split('|')
CITIES += ['RIY', 'BAH']
CITY_ALT = '|'.join(re.escape(x) for x in sorted(CITIES, key=len, reverse=True))
CITY_END = re.compile(r'(?<!\w)(?:' + CITY_ALT + r')\s*$', re.I)
CHANNEL_RE = re.compile(r'(?<!\w)(?:apple\s+pay(?:\s+app)?|google\s+pay(?:\s+app)?|samsung\s+pay(?:\s+app)?|mada(?:\s+(?:pay|atheer))?|atm|pos)(?!\w)', re.I)
METADATA_RE = re.compile(r'\b(?:CITY\s*:|CHNL\s*:|MCC[- :]|CARD\s+NO|BEN\s+ID|DEP\s+ID|VALUE\s+DATE|REFERENCE|RATE\s+COMMISSION|ACCOUNT\s+\d)|تاريخ الاستحقاق|رقم المرجع|رقم الايبان', re.I)
ACTION_PHRASES = [
    'Cash Withdrawal Commission', 'Cash Withdrawal Reversal', 'ATM Withdrawal Reversal',
    'Cash Withdrawal Fee', 'Cash Withdrawal', 'Cash Deposit', 'Credit Profit',
    'Salary Deposit', 'Payroll Deposit', 'PAYROLL', 'Bill Payment',
    'Credit Card Settlement', 'Advance payment for Card', 'Account to Card Transfer',
    'Card Payment', 'Purchase Refund', 'Refund Transaction', 'Cash Transfer',
    'Incoming internal transfer', 'Outgoing internal transfer', 'Internal transfer',
    'Instant Incoming transfer', 'Instant outgoing transfer', 'Incoming transfer',
    'Outgoing transfer', 'Local transfer', 'Money transfer', 'ACCOUNT TO ACCOUNT TRANSFER',
    'INCOMING SARIE PAYMENT', 'INCOMING SARIE TRANSFER', 'OUTGOING SARIE TRANSFER',
    'VAT on Cash Transactions', 'VAT on Wallet Fee', 'Card Delivery Fee',
    'Disclaimer Letter Charges', 'CASH ADVANCE', 'Payment Received',
    'عكس عملية سحب نقدي', 'عملية سحب نقدي', 'إيداع راتب', 'ايداع راتب',
    'إيداع نقدي', 'ايداع نقدي', 'حوالة داخلية', 'حوالة صادرة', 'حوالة واردة',
    'التحويل الوارد', 'تحويل صادر', 'تحويل وارد', 'تحويل محلي', 'تحويل دولي',
]
PURPOSES = [
    'تحويل لأفراد الأسرة أو الأصدقاء', 'تحويل الى الاهل والاصدقاء',
    'Transfer to family and friends', 'تحويل إلى العمالة المنزلية',
    'تسديد بطاقة الائتمان لعميل آخر', 'تحويل إلى شركات الصرافة',
    'تحويل إلى جهات حكومية', 'مصاريف سفر', 'مصاريف دراسية', 'مصاريف الحج و العمرة',
    'استثمارات محلية', 'استثمارات أجنبية', 'شراء بضاعة', 'مصاريف تشغيلية',
    'بدلات ومكافآت', 'توزيع أرباح', 'إيداعات شركات تحت التأسيس',
]
ACTION_RE = re.compile(r'(?<!\w)(?:' + '|'.join(re.escape(x) for x in sorted(set(ACTION_PHRASES + PURPOSES), key=len, reverse=True)) + r')(?!\w)', re.I)
TRANSFER_RE = re.compile(r'\b(?:transfer|withdrawal|payroll|deposit)\b|حوالة|تحويل|سحب|إيداع|ايداع', re.I)
PREFIX_RE = re.compile(r'^(?:POS\s+Purchase(?:\s+(?:Intl|International))?(?:\s*-?\s*(?:Apple Pay|Mada Pay|Mada Atheer))?|Local Internet purchase|Internet Purchase Transaction|Internet Purchase Intl|Online Purchase from|Online Purchase)\s*', re.I)
BOUNDARY_PATTERNS = [
    ('alinma_account_purchase', re.compile(r'\bPurchase\s+from\s+account\b.*?\bSAR\s+from\s+(?P<merchant>.+?)\s+in\s+\d{2}-\d{2}-\d{4}', re.I)),
    ('alinma_card_purchase', re.compile(r'\bpurchase\s+was\s+made\b.*?\bat\s+(?P<merchant>.+?)\s+On\s+\d', re.I)),
    ('alinma_visa_purchase', re.compile(r'\bVisa\s+Purchase\s+Transaction\b.*?\bfrom\s+account\b.*?\bfrom\s+(?P<merchant>.+?)\s+on\s+\d', re.I)),
    ('alinma_arabic_purchase', re.compile(r'تم الشراء.*?\bفي\s+(?P<merchant>.+?)\s+على\s+\d')),
    ('value_date_merchant', re.compile(r'^Value Date Reference\s+(?P<merchant>.+?)\s+Rate Commission\s*$', re.I)),
    ('parenthesized_reference', re.compile(r'^\([A-Za-z0-9 -]*\d[A-Za-z0-9 -]*\)\s*(?P<merchant>.+)$')),
    ('online_purchase', re.compile(r'^Online Purchase from\s+(?P<merchant>.+)$', re.I)),
    ('numeric_colon_merchant', re.compile(r'^\d{6,}\s*:\s*(?P<merchant>.+)$')),
    ('tilde_merchant', re.compile(r'^(?P<merchant>[^~]+)~~')),
]
BAD_EXACT = {'sa', 'sar', 'usd', 'eur', 'gbp', 'visa', 'mastercard', 'purchase',
             'bill', 'ben id', 'dep id', 'payment type', 'reference', 'city', 'mcc',
             'cash', 'none', 'null', 'nan', 'unknown', 'digital channel'}


def iter_jsonl(path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def clean_bounds(text, start, end):
    # Only boundary trimming: preserve original characters and internal city names.
    while start < end and text[start].isspace(): start += 1
    while end > start and text[end - 1].isspace(): end -= 1
    for _ in range(10):
        fragment = text[start:end]
        m = re.search(r'\s+(?:using\s+)?(?:Apple Pay(?: App)?|Mada(?: Pay| Atheer)?)\b.*$', fragment, re.I)
        if not m:
            m = re.search(r'\s+(?:SA|IE|GB|NL|US|AE|UK|BH|SG|SAR|\d{5,})\s*$', fragment, re.I)
        if not m:
            m = CITY_END.search(fragment)
        if not m or m.start() == 0:
            break
        end = start + m.start()
        while end > start and text[end - 1].isspace(): end -= 1
    return start, end


def valid_merchant(text):
    return (len(text) >= 2 and any(c.isalpha() for c in text)
            and normalize(text) not in BAD_EXACT and not CHANNEL_RE.search(text)
            and normalize(text) not in {'co', 'company', 'est', 'ltd', 'and'}
            and not text.endswith('__') and not re.match(r'^and\s', text, re.I)
            and not METADATA_RE.search(text) and not re.search(r'\d{5,}', text)
            and not re.search(r'\b(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*\d)[A-Z0-9]{16,}\b', text, re.I)
            and len(TOKEN_RE.findall(text)) <= 12 and len(text) <= 110)


def negative_reason(text):
    if re.fullmatch(r'[\d\W_]+', text):
        return 'numbers_and_symbols_only'
    rest = CHANNEL_RE.sub(' ', text)
    if not any(c.isalpha() for c in rest):
        return 'payment_channels_only'
    if re.match(r'^CITY\s*:', text, re.I):
        rest = re.sub(r'\b(?:' + CITY_ALT + r'|CITY|ARE|BHR|JOR|MADA|SA|SAR)\b|مدى', ' ', text, flags=re.I)
        if not any(c.isalpha() for c in rest):
            return 'location_card_metadata_only'
    return None


def all_occurrences(text, value):
    parts = [re.escape(x) for x in value.split()]
    pattern = re.compile(r'(?<!\w)' + r'\s+'.join(parts) + r'(?!\w)', re.I)
    return [(m.start(), m.end()) for m in pattern.finditer(text)]


def extract(text, merchant_fields=()):
    negative = negative_reason(text)
    if negative:
        return [], 'verified_structure_negative', negative
    actions = [(m.start(), m.end()) for m in ACTION_RE.finditer(text)]
    if actions:
        # Do not erase an unrecognized reversal/fee modifier by labeling only a withdrawal.
        for start, end in actions:
            if normalize(text[start:end]) in {'cash withdrawal', 'عملية سحب نقدي'}:
                nearby = text[max(0, start - 18):min(len(text), end + 20)]
                if re.search(r'revers|commission|fee|\bvat\b|عكس|رسوم', nearby, re.I):
                    return None, 'review', 'unresolved_action_modifier'
        return actions, 'rule_labeled', 'explicit_financial_action'
    if TRANSFER_RE.search(text):
        return None, 'review', 'unresolved_transfer_or_deposit'
    # SNB bilingual merchant prefix, with a literal field boundary.
    city = re.search(r'\bCITY\s*:', text, re.I)
    if city and city.start() > 0 and not text.upper().startswith('FROM '):
        start, end = 0, city.start()
        prefix = PREFIX_RE.match(text[:end])
        if prefix: start = prefix.end()
        start, end = clean_bounds(text, start, end)
        value = text[start:end]
        if valid_merchant(value):
            return [(start, end)], 'rule_labeled', 'merchant_before_city_field'
    # Riyad narratives repeat a merchant phrase at the end of the payload.
    marker = re.search(r'MCHNAME:\s*', text, re.I)
    if marker:
        payload = re.split(r'\bSaudi\s+Arabia\b|\bECOM#', text[marker.end():], maxsplit=1, flags=re.I)[0]
        words = payload.split()
        for size in range(min(12, len(words) // 2), 0, -1):
            first, second = words[-2 * size:-size], words[-size:]
            if normalize(' '.join(first)) == normalize(' '.join(second)):
                value = ' '.join(second)
                if valid_merchant(value):
                    return all_occurrences(text, value), 'rule_labeled', 'repeated_merchant_field'
        return None, 'review', 'unresolved_mchname'
    for name, pattern in BOUNDARY_PATTERNS:
        m = pattern.search(text)
        if not m: continue
        start, end = clean_bounds(text, *m.span('merchant'))
        value = text[start:end]
        if valid_merchant(value):
            return all_occurrences(text, value), 'rule_labeled', name
        return None, 'review', 'invalid_' + name
    # A reference/person narrative is not a merchant label. Only use a source
    # field when the ENTIRE issue agrees, not a name embedded in a transfer.
    options = []
    for field in merchant_fields:
        field = str(field or '').strip()
        if not valid_merchant(field): continue
        if normalize(field) != normalize(text): continue
        for start, end in all_occurrences(text, field):
            start, end = clean_bounds(text, start, end)
            value = text[start:end]
            if valid_merchant(value): options.append((start, end))
    if options:
        # Prefer a complete name; nested source aliases do not create conflicting labels.
        longest = max(options, key=lambda x: x[1] - x[0])
        return all_occurrences(text, text[longest[0]:longest[1]]), 'source_field_labeled', 'merchant_name_field'
    return None, 'review', 'no_reliable_boundary'


def build_record(text, spans, provenance, reason, banks, source):
    if not spans and provenance not in {'verified_structure_negative', 'agent_reviewed'}:
        raise ValueError('An unresolved positive must not become a negative example')
    tokens, ner = token_spans(text, spans)
    if len(tokens) > 384 or any(e - s + 1 > 12 for s, e, _ in ner):
        raise ValueError('Model token/span limit exceeded')
    if any(a[1] >= b[0] for a, b in zip(ner, ner[1:])):
        raise ValueError('Overlapping entity spans')
    return {
        'sample_id': sample_id(text), 'group_id': group_id(text), 'text': text,
        'tokenized_text': tokens, 'ner': ner, 'ner_labels': [LABEL],
        'entities': [{'start': s, 'end': e, 'text': text[s:e], 'label': LABEL} for s, e in sorted(set(spans))],
        'annotation_status': provenance, 'annotation_rule': reason,
        'bank_ids': sorted(banks), 'source': source,
    }


def stats(rows):
    return {
        'examples': len(rows), 'positives': sum(bool(r['ner']) for r in rows),
        'negatives': sum(not r['ner'] for r in rows),
        'spans': sum(len(r['ner']) for r in rows),
        'groups': len({r['group_id'] for r in rows}),
        'primary_banks': dict(Counter(r['bank_ids'][0] if r['bank_ids'] else 'unknown' for r in rows)),
        'sources': dict(Counter(r['source'] for r in rows)),
        'annotation_status': dict(Counter(r['annotation_status'] for r in rows)),
        'rules': dict(Counter(r['annotation_rule'] for r in rows)),
    }


def write_jsonl(path, rows):
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detailed', type=Path, required=True)
    parser.add_argument('--issues', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/v2')
    parser.add_argument('--max-positive', type=int, default=14000)
    parser.add_argument('--max-negative', type=int, default=1800)
    args = parser.parse_args()
    details = json.loads(args.detailed.read_text(encoding='utf-8'))
    if isinstance(details, dict): details = details['items']
    by_text = defaultdict(list)
    for r in details:
        if str(r.get('TransactionInformation') or '').strip():
            by_text[normalize(r['TransactionInformation'])].append(r)
    overrides = json.loads((ROOT / 'annotation_overrides.json').read_text(encoding='utf-8'))
    override_map = {x['text']: x for x in overrides}
    groups = defaultdict(list)
    seen = set()
    review = []
    counters = Counter()
    sanity = []
    found_overrides = set()

    def accept(text, source):
        if not text.strip():
            counters['empty_text'] += 1
            return
        sid = sample_id(text)
        if sid in seen:
            counters['duplicate_normalized_text'] += 1
            return
        seen.add(sid)
        matches = by_text.get(normalize(text), [])
        banks = {str(r['bank_id']) for r in matches if r.get('bank_id') is not None}
        merchant_fields = [r.get('MerchantName') for r in matches]
        override = override_map.get(text)
        if override:
            found_overrides.add(text)
            spans = []
            for value in override['signals']:
                occurrences = all_occurrences(text, value)
                if not occurrences: raise ValueError(f'Override not in raw text: {value}')
                spans.extend(occurrences)
            record = build_record(text, spans, 'agent_reviewed', 'explicit_review_correction', banks, source)
            record['review_note'] = override['note']
            if override.get('use_as') == 'training_candidate':
                groups[record['group_id']].append(record)
                counters['reviewed_training_candidates'] += 1
            else:
                sanity.append(record)
                counters['sanity_examples_found'] += 1
            return
        spans, provenance, reason = extract(text, merchant_fields)
        if spans is not None:
            try:
                record = build_record(text, spans, provenance, reason, banks, source)
            except ValueError:
                spans, reason = None, 'unaligned_or_oversize_span'
        if spans is None:
            counters['quarantined_' + reason] += 1
            # Deterministic representative queue, not a hidden negative dataset.
            if len(review) < 1500 or sid < review[-1]['sample_id']:
                review.append({'sample_id': sid, 'text': text, 'bank_ids': sorted(banks), 'source': source, 'reason': reason, 'annotation_status': 'unreviewed'})
                if len(review) >= 3000:
                    review.sort(key=lambda r: r['sample_id'])
                    del review[1500:]
            return
        counters['eligible_' + provenance] += 1
        bucket = groups[record['group_id']]
        bucket.append(record)
        bucket.sort(key=lambda r: r['sample_id'])
        del bucket[3 if record['ner'] else 12:]

    # Prefer original detailed raw text for its known bank metadata; then expand from all issues.
    for row in details: accept(str(row.get('TransactionInformation') or ''), 'detailed')
    for index, row in enumerate(iter_jsonl(args.issues), start=1):
        accept(str(row.get('issue') or ''), 'issues')
        if index % 100000 == 0: print(f'Scanned {index:,} issue rows', flush=True)
    if len(found_overrides) != len(overrides):
        missing = set(override_map) - found_overrides
        raise ValueError(f'Review examples missing from source: {missing}')
    blocked_groups = {r['group_id'] for r in sanity}
    pools = defaultdict(list)
    for gid, rows in sorted(groups.items()):
        if gid in blocked_groups: continue
        rep = rows[0]
        # Prefer actual known-bank examples when normalized templates coincide.
        banks = sorted({b for r in rows for b in r['bank_ids']})
        stratum = (bool(rep['ner']), banks[0] if banks else 'unknown', rep['annotation_rule'])
        pools[stratum].append(rows)
    # Round-robin over strata protects smaller known banks and action/negative formats.
    selected = []
    count = Counter()
    positions = Counter()
    while True:
        changed = False
        for key in sorted(pools):
            positive = key[0]
            limit = args.max_positive if positive else args.max_negative
            if count[positive] >= limit or positions[key] >= len(pools[key]): continue
            rows = pools[key][positions[key]]
            positions[key] += 1
            chosen = rows[:limit - count[positive]]
            selected.append((key, chosen))
            count[positive] += len(chosen)
            changed = True
        if not changed: break
    split_pools = defaultdict(list)
    for key, rows in selected: split_pools[key].append(rows)
    splits = {s: [] for s in ('train', 'validation', 'test')}
    for key, chunks in sorted(split_pools.items()):
        chunks.sort(key=lambda rows: rows[0]['group_id'])
        targets = {'test': max(1, round(len(chunks)*.1)) if len(chunks)>=3 else 0,
                   'validation': max(1, round(len(chunks)*.1)) if len(chunks)>=3 else 0}
        for i, rows in enumerate(chunks):
            split = 'test' if i < targets['test'] else 'validation' if i < sum(targets.values()) else 'train'
            splits[split].extend(rows)
    for name, rows in splits.items():
        rows.sort(key=lambda r: r['sample_id'])
        write_jsonl(args.output / f'{name}.jsonl', rows)
    write_jsonl(args.output / 'sanity.jsonl', sorted(sanity, key=lambda r: r['sample_id']))
    review.sort(key=lambda r: r['sample_id'])
    write_jsonl(args.output / 'review_queue.jsonl', review[:1500])
    summary = {
        'version': 2, 'label': LABEL, 'status': 'rule_labeled_pilot_not_human_gold',
        'source_files': {p.name: {'sha256': file_hash(p), 'bytes': p.stat().st_size} for p in (args.detailed, args.issues)},
        'scanned_detailed_rows': len(details), 'scanned_issue_rows': index,
        'counters': dict(counters), 'splits': {k: stats(v) for k,v in splits.items()},
        'sanity': stats(sanity), 'review_queue_examples': min(1500, len(review)),
        'split_policy': 'whole normalized numeric/reference-template groups; reviewed sanity groups excluded; known bank metadata only',
        'limits': {'max_positive': args.max_positive, 'max_negative': args.max_negative, 'max_width': 12},
        'limitations': ['Rule/source-field labels require independent human evaluation before production.', 'Unknown bank IDs are not inferred.', 'Sanity labels were reviewed by the assistant, not a human annotator.', 'Generalization to unseen merchants needs a separate independently reviewed benchmark.'],
    }
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: stats(v) for k,v in splits.items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
