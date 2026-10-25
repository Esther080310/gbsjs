"""Lossless, record-block based normalization of CSMAR's quoted exports.

No row is skipped. Malformed records without a unique repair retain their complete
UTF-8 source block in quarantine. Schema constraints are used only for repairs.
"""
from __future__ import annotations
import base64
import csv
import gzip
import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

csv.field_size_limit(128 * 1024 * 1024)
START = re.compile(rb'^"[0-9]{6}",')
DECIMAL = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z')
DATE = re.compile(r'\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?)?\Z')
TRACE = ['__source_member', '__source_record_index', '__line_start', '__line_end',
         '__raw_sha256', '__parse_method', '__content_sha256']


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for part in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def content_sha256(values):
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()


@contextmanager
def gzip_text(path, mode='w'):
    if mode == 'r':
        with gzip.open(path, 'rt', encoding='utf-8', newline='') as stream:
            yield stream
    else:
        with open(path, 'wb') as raw:
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=6) as compressed:
                with io.TextIOWrapper(compressed, encoding='utf-8', newline='') as stream:
                    yield stream


def iter_blocks(stream):
    """Return (member-local ordinal, physical first/last line, original bytes)."""
    pieces, first, ordinal, last = [], None, 0, 1
    for line_no, line in enumerate(stream, 2):
        last = line_no
        if START.match(line):
            if pieces:
                ordinal += 1
                yield ordinal, first, line_no - 1, b''.join(pieces)
            first, pieces = line_no, [line]
        else:
            if not pieces:
                raise ValueError(f'Unframed bytes before first record at line {line_no}')
            pieces.append(line)
    if pieces:
        yield ordinal + 1, first, last, b''.join(pieces)


def _valid(value, kind, field):
    if field == 'Stkcd':
        return bool(re.fullmatch(r'\d{6}', value))
    if not value:
        return True
    if kind.lower() in ('decimal', 'numeric', 'int', 'integer', 'float'):
        return bool(DECIMAL.fullmatch(value))
    if kind.lower() == 'datetime':
        # CSMAR explicitly permits 00 month/day for incomplete dates.
        return bool(DATE.fullmatch(value))
    return True


def unique_partition(parts, fields, types):
    """Count paths (capped at two), merging only unconstrained text values.

    Reinsert the exact literal delimiter for joined fragments. Never silently
    choose a text column when several schema-consistent partitions exist.
    """
    count_fields, count_parts = len(fields), len(parts)
    @lru_cache(None)
    def solve(i, j):
        if i == count_fields:
            return (1, ()) if j == count_parts else (0, None)
        remaining = count_fields - i - 1
        max_end = count_parts - remaining
        constrained = types[i].lower() in ('decimal', 'numeric', 'int', 'integer', 'float', 'datetime') or fields[i] == 'Stkcd'
        choices = range(j + 1, min(j + 1, max_end) + 1) if constrained else range(j + 1, max_end + 1)
        total, path = 0, None
        for end in choices:
            value = '\",\"'.join(parts[j:end])
            if not _valid(value, types[i], fields[i]):
                continue
            n, tail = solve(i + 1, end)
            if n:
                if total == 0 and n == 1:
                    path = (value,) + tail
                total = min(2, total + n)
                if total == 2:
                    return 2, None
        return total, path
    return solve(0, 0)


def parse_block(raw, fields, types):
    text = raw.decode('utf-8')
    try:
        reader = csv.reader(io.StringIO(text, newline=''), strict=True)
        values = next(reader)
        if len(values) == len(fields) and next(reader, None) is None:
            return values, 'strict_csv', None
    except (csv.Error, StopIteration):
        pass
    body = text[:-2] if text.endswith('\r\n') else text[:-1] if text.endswith(('\n', '\r')) else text
    if not (body.startswith('"') and body.endswith('"')):
        return None, None, {'reason': 'missing_outer_quotes'}
    parts = body[1:-1].split('\",\"')
    if len(parts) == len(fields):
        return parts, 'literal_delimited', None
    if len(parts) > len(fields):
        count, values = unique_partition(parts, fields, types)
        if count == 1:
            return list(values), 'unique_schema_partition', None
        return None, None, {'reason': 'ambiguous_schema_partition' if count == 2 else 'no_schema_partition', 'literal_fields': len(parts), 'candidate_count_capped_at_2': count}
    return None, None, {'reason': 'insufficient_literal_fields', 'literal_fields': len(parts)}


def _json_line(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')


def verify_output(path, fields, expected_rows, expected_empty, expected_zero, expected_trace_digest):
    """Re-read emitted CSV with strict parsing and constant-memory provenance checks."""
    empties, zeros = [0] * len(fields), [0] * len(fields)
    count, previous_member, previous_index = 0, None, 0
    seen_members, trace_digest = set(), hashlib.sha256()
    with gzip_text(path, 'r') as stream:
        reader = csv.reader(stream, strict=True)
        if next(reader) != fields + TRACE:
            raise ValueError('Normalized header mismatch')
        for row in reader:
            if len(row) != len(fields) + len(TRACE):
                raise ValueError('Normalized row width mismatch')
            values, trace = row[:-len(TRACE)], row[-len(TRACE):]
            member, index, line_start, line_end, raw_hash, method, content_hash = trace
            if member != previous_member:
                if member in seen_members:
                    raise ValueError('Source member repeated non-contiguously')
                seen_members.add(member)
                previous_member, previous_index = member, 0
            if int(index) <= previous_index or int(line_start) > int(line_end):
                raise ValueError('Invalid or duplicate provenance')
            previous_index = int(index)
            if content_sha256(values) != content_hash or not re.fullmatch('[0-9a-f]{64}', raw_hash):
                raise ValueError('Content/provenance hash invalid')
            trace_digest.update(json.dumps(trace, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
            for i, value in enumerate(values):
                empties[i] += value == ''
                zeros[i] += value == '0'
            count += 1
    if (count, empties, zeros, trace_digest.hexdigest()) != (expected_rows, expected_empty, expected_zero, expected_trace_digest):
        raise ValueError('Re-read counts, empty strings, literal zeros or provenance changed')
    return {'rows': count, 'widths_valid': True, 'provenance_unique': True, 'content_hashes_verified': True,
            'empty_strings_unchanged': True, 'literal_zeros_unchanged': True, 'trace_digest': expected_trace_digest}



def verify_quarantine(path, tables):
    counts = Counter()
    with gzip_text(path, 'r') as stream:
        for line in stream:
            item = json.loads(line)
            raw = base64.b64decode(item['raw_block_base64'], validate=True)
            if raw.decode('utf-8') != item['raw_block'] or hashlib.sha256(raw).hexdigest() != item['raw_sha256']:
                raise ValueError('Quarantine source block changed')
            counts[item['table']] += 1
    for table in tables:
        if counts[table['table']] != table['counts'].get('quarantine', 0):
            raise ValueError('Quarantine count mismatch')
    if set(counts) - {table['table'] for table in tables}:
        raise ValueError('Unexpected quarantined table')
    return {'rows': sum(counts.values()), 'complete_raw_blocks_verified': True}


def normalize_all(root, output_dir=None):
    root = Path(root)
    out = Path(output_dir) if output_dir else root / 'processed/20261016-v1'
    normalized = out / 'normalized'
    normalized.mkdir(parents=True, exist_ok=True)
    ledger = json.loads((root / '下载台账_20261016.json').read_text(encoding='utf-8'))
    dictionaries = {}
    with (root / '字段字典汇总.csv').open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            dictionaries.setdefault(row['table'], {})[row['field']] = row['data_type']
    manifest = {'version': '20261016-v1', 'source_ledger_sha256': sha256_file(root / '下载台账_20261016.json'),
                'dictionary_sha256': sha256_file(root / '字段字典汇总.csv'),
                'policy': {'duplicates_removed': False, 'values_coerced': False, 'date_currency_amount_transformed': False,
                           'record_boundary': 'physical line starts with quoted six-digit Stkcd and comma',
                           'repair_order': ['strict_csv', 'literal_delimited', 'unique_schema_partition'],
                           'ambiguous_records': 'complete original bytes base64 plus UTF-8 text in quarantine',
                           'hash_encoding': 'SHA256 UTF8 JSON string array ensure_ascii=False compact separators',
                           'gzip_mtime': 0}, 'tables': []}
    with gzip_text(out / 'format_quarantine.jsonl.gz') as quarantine, (out / 'format_repair_examples.jsonl').open('w', encoding='utf-8') as examples:
        for entry in ledger['downloaded']:
            table, fields = entry['table'], entry['headers']
            types = [dictionaries[table][field] for field in fields]
            source = root / 'raw/20261016' / f'{table}_all_available.zip'
            before = sha256_file(source)
            if before != entry['sha256']:
                raise ValueError(f'{table}: source hash differs from ledger')
            counts, members, ambiguities = Counter(), [], []
            empties, zeros = [0] * len(fields), [0] * len(fields)
            trace_digest = hashlib.sha256()
            dest = normalized / f'{table}.csv.gz'
            with zipfile.ZipFile(source) as archive, gzip_text(dest) as target:
                writer = csv.writer(target, lineterminator='\n', quoting=csv.QUOTE_ALL)
                writer.writerow(fields + TRACE)
                names = [name for name in archive.namelist() if name.lower().endswith('.csv')]
                if len(names) != len(set(names)):
                    raise ValueError('Duplicate archive member names')
                expected_parts = {part['filename']: part['record_starts'] for part in entry['csv_parts']}
                if {Path(n).name for n in names} != set(expected_parts) or len(names) != len(expected_parts):
                    raise ValueError(f'{table}: archive members differ from ledger')
                for name in names:
                    member_count, member_accepted, member_quarantine = 0, 0, 0
                    with archive.open(name) as stream:
                        header = next(csv.reader([stream.readline().decode('utf-8-sig')], strict=True))
                        if header != fields:
                            raise ValueError(f'{table}/{name}: header mismatch')
                        for index, first, last, raw in iter_blocks(stream):
                            member_count += 1
                            raw_hash = hashlib.sha256(raw).hexdigest()
                            values, method, failure = parse_block(raw, fields, types)
                            origin = {'table': table, 'source_member': name, 'source_record_index': index,
                                      'line_start': first, 'line_end': last, 'raw_sha256': raw_hash}
                            if values is None:
                                counts['quarantine'] += 1
                                member_quarantine += 1
                                issue = {**origin, **failure}
                                ambiguities.append(issue)
                                _json_line(quarantine, {**issue, 'raw_block': raw.decode('utf-8'), 'raw_block_base64': base64.b64encode(raw).decode('ascii')})
                                continue
                            counts[method] += 1
                            counts['accepted'] += 1
                            member_accepted += 1
                            trace = [name, str(index), str(first), str(last), raw_hash, method, content_sha256(values)]
                            writer.writerow(values + trace)
                            trace_digest.update(json.dumps(trace, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
                            for i, value in enumerate(values):
                                empties[i] += value == ''
                                zeros[i] += value == '0'
                            if method != 'strict_csv' and counts[method] <= 5:
                                _json_line(examples, {**origin, 'parse_method': method, 'values': dict(zip(fields, values)), 'raw_block': raw.decode('utf-8')})
                    if member_count != expected_parts[Path(name).name]:
                        raise ValueError(f'{table}/{name}: record count differs from ledger')
                    members.append({'member': name, 'record_starts': member_count, 'accepted': member_accepted, 'quarantine': member_quarantine})
            expected = entry['server_export_rows']
            if counts['accepted'] + counts['quarantine'] != expected:
                raise ValueError(f'{table}: total count mismatch')
            verification = verify_output(dest, fields, counts['accepted'], empties, zeros, trace_digest.hexdigest())
            after = sha256_file(source)
            if before != after:
                raise ValueError(f'{table}: source modified')
            result = {'table': table, 'expected': expected, 'fields': len(fields), 'counts': dict(counts), 'members': members,
                      'source_sha256_before': before, 'source_sha256_after': after, 'source_matches_ledger': True,
                      'normalized_file': str(dest), 'normalized_sha256': sha256_file(dest),
                      'empty_strings_by_field': dict(zip(fields, empties)), 'literal_zero_by_field': dict(zip(fields, zeros)),
                      'verification': verification, 'ambiguities': ambiguities}
            manifest['tables'].append(result)
            print(json.dumps({'table': table, 'counts': dict(counts), 'verified': True}), flush=True)
    manifest['expected_total'] = sum(t['expected'] for t in manifest['tables'])
    manifest['accepted_total'] = sum(t['counts'].get('accepted', 0) for t in manifest['tables'])
    manifest['quarantine_total'] = sum(t['counts'].get('quarantine', 0) for t in manifest['tables'])
    manifest['quarantine_verification'] = verify_quarantine(out / 'format_quarantine.jsonl.gz', manifest['tables'])
    manifest['artifacts'] = {name: sha256_file(out / name) for name in ('format_quarantine.jsonl.gz', 'format_repair_examples.jsonl')}
    (out / 'format_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return manifest
