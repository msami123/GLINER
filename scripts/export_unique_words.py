"""Export unique words from the training dataset's issue field to CSV."""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('Big Data for Training.jsonl'))
    parser.add_argument('--output', type=Path, default=Path('unique_words.csv'))
    args = parser.parse_args()
    counts = Counter()
    records = 0
    # Unicode letters and numbers, excluding underscores and punctuation.
    pattern = re.compile(r'[^\W_]+', re.UNICODE)
    with args.input.open(encoding='utf-8-sig') as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            issue = record.get('issue', '')
            if not isinstance(issue, str):
                raise ValueError(f'Expected a string issue in record {records + 1}')
            counts.update(pattern.findall(issue.casefold()))
            records += 1
    with args.output.open('w', encoding='utf-8-sig', newline='') as target:
        writer = csv.writer(target)
        writer.writerow(['word', 'count'])
        writer.writerows(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
    with args.output.open(encoding='utf-8-sig', newline='') as target:
        reader = csv.DictReader(target)
        rows = list(reader)
    assert len(rows) == len(counts)
    assert sum(int(row['count']) for row in rows) == sum(counts.values())
    print(f'Read {records:,} records; exported {len(counts):,} unique words to {args.output}')


if __name__ == '__main__':
    main()
