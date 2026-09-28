# Classification-signal training data

This package contains **silver (rule-proposed) annotations**, not human-certified
gold labels. It is an offline preparation of the saved
`transaction-issues-production-1000.json` export. It does not call the API, verify
today's production state, or train a model. The original source is unchanged.

Version 4 trains one GLiNER label, `classification_signal`. Internal audit types
distinguish merchants, people and standalone category phrases, but those types
are not model labels. Public bodies, charities, educational institutions and
named photography/other businesses are eligible. Do not mix dataset versions.

## What to load

- `train.json`, `validation.json`, `test.json`: JSON arrays for GLiNER.
- Equivalent `.jsonl` files: one example per line; load **one format only**.
- `*.audit.jsonl`: original text, exact character spans, rules, bank, source row,
  source chunk, group ID and split, in the same order as the model file.
- `quality_report.json`: all counts, bank/label coverage, exclusions and checks.
- `manifest.json`: source hash, code hashes, file hashes, configuration and seed.
- `source_accounting.jsonl`: disposition of every source row. A source row index
  is zero-based in the saved JSON array, not an API transaction identifier.
- `review.jsonl`: incomplete/ambiguous annotations. **Do not train these as empty
  `ner` examples.** `proposed_spans` are suggestions only.
- `review_clusters.jsonl`: unresolved rows grouped by bank, reason and normalized
  template, ranked by frequency with representative samples for the next rule pass.
- `annotation_sample.jsonl` and `review_sample.jsonl`: stratified inspection packs.

The model sees only tokens from `issue`. Bank names, IDs and split/rule metadata
are audit fields, not extra model features. Names, reference numbers and other
financial information remain in the local original text: keep this package
private and do not commit or upload it to a public dataset service.

## Label contract

| Label | Include | Exclude |
| --- | --- | --- |
| `classification_signal` | A merchant, explicit beneficiary/remitter/depositor, named organization or standalone category-defining phrase | Routing-bank metadata, city, card/account/reference, payment method and generic bank action wrappers beside a reliable name |

When a reliable name exists, it is the only target: transfer/SADAD/refund/fee
wrappers beside it are noise. `Payroll deposit` is an explicit reviewed
exception and is kept instead of its employer/routing organization. A separated
trailing `CO.` is removed from a merchant span. Actual spelling otherwise stays
unchanged; names are not translated or corrected. Ambiguous boundaries go to
review rather than being stripped by guessing. Internal semantic types remain
available in audit files only.

Direction markers establish the role described in the text, not independently
verified account ownership. The source lacks structured transaction direction and
account-owner IDs, so ambiguous layouts are deliberately withheld.

## GLiNER format and tokenization

Each model example has exactly these fields:

```json
{
  "tokenized_text": ["رسوم", "تحويل", "محلي"],
  "ner": [[0, 2, "classification_signal"]],
  "ner_labels": ["classification_signal"]
}
```

`ner` indices are **zero-based and inclusive at both ends**. In audit files,
`start`/`end` are Python character offsets with an **exclusive end**;
`start_token`/`end_token` are inclusive. Empty `ner: []` is used only for reviewed
no-target layouts, trusted metadata or an explicitly empty merchant field.
`ner_labels` is supplied even for negatives. See the
[official GLiNER training format](https://urchade.github.io/GLiNER/training.html).

Word tokenization for this version is
`(?<=\d)TO(?=[A-Za-z])|(?<=\d)FR(?=[A-Za-z])|[A-Za-z]+|[\u0600-\u06FF]+|\d+|_+|[^\w\s]|[^\W\d_]+`.
Punctuation, digits and
Arabic/Latin script transitions are separate. This makes exact targets such as
`Zain` inside `044-Zain-182138999` and `محمد` inside `TOمحمد` representable. The
inference wrapper must use the same tokenizer; do not re-tokenize these rows
without recomputing spans.

Before training, compare `quality_report.json -> checks` with the chosen model's
`max_width`, `max_len` and encoder/subword limit. Word-token count is not subword
count. Long rows/spans are withheld according to the manifest limits rather than
silently truncated. If your model has a narrower span window, rebuild with
`--max-span-width` set to its supported value. Do not just increase a loaded
checkpoint's limits without checking architecture support.

No GLiNER package or checkpoint was installed, downloaded or trained to create
this package. Loading/forward-pass compatibility must be smoke-tested with the
specific pinned model before a full training job.

## Selection and evaluation

1. Each source row is classified as supported and complete, reliable negative,
   or review. Detectable incomplete entity annotations never become negatives.
2. Exact text duplicates merge with source provenance; conflicting annotations
   for the same text are all withheld for review.
3. Group normalized templates (references/digits/punctuation ignored), plus
   shared normalized merchant/person surfaces **across banks**. All connected
   rows stay in the same split. This is deliberately name-disjoint evaluation.
4. Cap repetition per group/bank and the number of rows per bank. Original rows
   are not deleted; every exclusion is recorded. Negative groups have a separate
   cap. No oversampling or synthetic negatives are used to force a ratio.
5. Assign whole groups, targeting 80/10/10 within bank/label strata. Rare or very
   large groups can prevent exact ratios and may leave small strata with no
   holdout. Read actual counts, not nominal ratios.

Automated validation checks exact spans, token boundaries, accounting and
cross-split overlap under the documented grouping rules. This does not prove
semantic correctness or detect every possible near duplicate. Rule-generated
test labels share training annotation bias: independently review and adjudicate
the validation/test labels before reporting model quality. Do not tune extraction
rules on a held-out human gold test after it is frozen.

Coverage is not uniform: clipped/glued names, unclear direction, international
layouts and freeform notes are common review cases. A small natural negative
fraction is intentional when diverse, trustworthy negatives are scarce. Expand
coverage through adjudication, not by treating parser failures as no-entity rows.

## Reproduce

From the repository root:

```sh
python merchant_categorization_service/scripts/prepare_transaction_training_data.py \
  --input transaction-issues-production-1000.json \
  --output-dir training_data/classification_signal_silver_v4 \
  --max-per-group 25 \
  --max-per-bank 25000 \
  --max-negative-per-group 1000
```

Use a new directory for a different version. `--overwrite` only allows rebuilding
a directory recognized by this script's manifest. Keep the seed/config/source and
code fixed for deterministic selected rows and splits; build timestamps differ.

Tests (from `merchant_categorization_service`):

```sh
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_prepare_transaction_training_data.py \
  tests/test_transaction_labels_rajhi_alinma.py \
  tests/test_transaction_labels_snb_riyad.py \
  tests/test_transaction_labels_other_banks.py
```
