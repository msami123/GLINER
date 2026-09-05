"""Shared text/span contract. Matches GLiNER 0.2.28's default word splitter."""
from __future__ import annotations

import hashlib
import re

LABEL = 'classification_signal'
TOKEN_RE = re.compile(r'\w+(?:[-_]\w+)*|\S')
MIXED_REF_RE = re.compile(r'\b(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{8,}\b')


def normalize(text):
    return ' '.join(text.casefold().split())


def sample_id(text):
    return hashlib.sha256(normalize(text).encode('utf-8')).hexdigest()


def template_key(text):
    # Used only for grouping; never changes the model input text.
    text = MIXED_REF_RE.sub('<REF>', normalize(text))
    return re.sub(r'\d+', '<N>', text)


def group_id(text):
    return hashlib.sha256(template_key(text).encode('utf-8')).hexdigest()


def token_spans(text, spans):
    tokens = list(TOKEN_RE.finditer(text))
    starts = {m.start(): i for i, m in enumerate(tokens)}
    ends = {m.end(): i for i, m in enumerate(tokens)}
    ner = []
    for start, end in sorted(set(spans)):
        if start not in starts or end not in ends:
            raise ValueError(f'Unaligned span: {text[start:end]!r}')
        ner.append([starts[start], ends[end], LABEL])
    return [m.group() for m in tokens], ner


def training_view(row):
    # Provenance, bank IDs and text metadata never become model features.
    return {k: row[k] for k in ('tokenized_text', 'ner', 'ner_labels')}
