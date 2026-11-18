"""Portable table_query v1 contract and exact stdlib reference backend."""
import csv
from fractions import Fraction
import io
import json
import operator
import os
from pathlib import PurePosixPath
import re
import stat

VERSION = 'table-query-v1'
MAX_BYTES, MAX_ROWS, MAX_GROUPS = 16 * 1024 * 1024, 100000, 1000
COMPARE = {name: getattr(operator, name) for name in ('eq', 'ne', 'lt', 'le', 'gt', 'ge')}
ARITH = {name: getattr(operator, name) for name in ('add', 'sub', 'mul', 'truediv')}


class QueryError(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise QueryError(message)


def fields(obj, required, optional=()):
    need(isinstance(obj, dict) and set(required) <= obj.keys()
         and obj.keys() <= set(required) | set(optional), 'Missing or unsupported fields')


def number(value):
    # Exact decimal/exponent or canonical fraction. Never infer a code as numeric.
    need(isinstance(value, str) and len(value) <= 100
         and re.fullmatch(r'[+-]?(?:\d+/[1-9]\d*|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d{1,3})?)', value),
         'Number must be a finite decimal or fraction string, <=100 characters')
    return Fraction(value)


def load_csv(workspace, name, delimiter):
    need(isinstance(name, str), 'table must be a relative CSV path')
    pure = PurePosixPath(name)
    need(not pure.is_absolute() and '..' not in pure.parts and str(pure) == name
         and '\\' not in name and pure.suffix.lower() == '.csv', 'Invalid CSV path')
    # Open each component without following links; reject devices and FIFOs.
    fd = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in pure.parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        file_fd = os.open(pure.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    finally:
        os.close(fd)
    with os.fdopen(file_fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        need(stat.S_ISREG(info.st_mode) and info.st_size <= MAX_BYTES, 'Input file/size limit')
        raw = stream.read(MAX_BYTES + 1)
    need(len(raw) <= MAX_BYTES, 'Input byte limit')
    reader = csv.reader(io.StringIO(raw.decode('utf-8-sig'), newline=''), delimiter=delimiter, strict=True)
    columns = next(reader, [])
    need(0 < len(columns) <= 64 and all(columns) and len(set(columns)) == len(columns),
         'Empty, duplicate or excessive CSV columns')
    rows = []
    for cells in reader:
        need(len(cells) == len(columns), 'Ragged CSV record')
        need(len(rows) < MAX_ROWS, 'Row limit')
        rows.append(dict(zip(columns, cells)))
    return columns, rows


def validate(q, columns):
    fields(q, ('tool', 'table', 'schema', 'aggregates'),
           ('version', 'delimiter', 'filters', 'derive', 'group_by', 'sort', 'limit', 'missing'))
    need(q['tool'] == 'table_query' and q.get('version', VERSION) == VERSION, 'Unknown tool/version')
    schema = q['schema']
    need(isinstance(schema, dict) and set(schema) <= set(columns)
         and all(t in ('text', 'string', 'number') for t in schema.values()), 'Invalid schema')
    # Accept the common JSON spelling at the boundary; backends only see text/number.
    types = {c: 'text' if schema.get(c, 'text') == 'string' else schema.get(c, 'text')
             for c in columns}
    need(q.get('missing', 'error') in ('error', 'exclude'), 'Invalid missing policy')
    need(type(q.get('limit', 20)) is int and 1 <= q.get('limit', 20) <= 100, 'limit must be 1..100')
    filters, derive = q.get('filters', []), q.get('derive', [])
    need(isinstance(filters, list) and len(filters) <= 8, 'At most 8 filters')
    for f in filters:
        fields(f, ('column', 'op', 'value'))
        need(f['column'] in types and f['op'] in COMPARE, 'Invalid filter')
        need(isinstance(f['value'], str) and f['value'] != '', 'Filter value must be nonempty string')
        if types[f['column']] == 'number':
            number(f['value'])
    need(isinstance(derive, list) and len(derive) <= 4, 'At most 4 derived columns')
    for d in derive:
        fields(d, ('as', 'op', 'left', 'right'))
        need(isinstance(d['as'], str) and re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]{0,31}', d['as'])
             and d['as'] not in types, 'Derived column collision/invalid name')
        need(d['op'] in ARITH and types.get(d['left']) == types.get(d['right']) == 'number',
             'Derived operands must name numeric columns')
        types[d['as']] = 'number'
    groups = q.get('group_by', [])
    need(isinstance(groups, list) and len(groups) <= 2 and all(isinstance(c, str) for c in groups)
         and len(set(groups)) == len(groups) and all(c in types for c in groups), 'Invalid grouping')
    aggs = q['aggregates']
    need(isinstance(aggs, list) and 1 <= len(aggs) <= 4, 'Use 1..4 aggregates')
    aliases = set(groups)
    for a in aggs:
        need(isinstance(a, dict), 'Aggregate must be an object')
        fields(a, ('as', 'op') if a.get('op') == 'count' else ('as', 'op', 'column'))
        need(isinstance(a['as'], str) and re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]{0,31}', a['as'])
             and a['as'] not in aliases, 'Invalid/repeated aggregate alias')
        aliases.add(a['as'])
        need(a['op'] in ('count', 'sum', 'mean', 'min', 'max'), 'Unknown aggregate')
        if a['op'] != 'count':
            need(types.get(a['column']) == 'number', 'Aggregate requires numeric column')
    sorts = q.get('sort', [])
    need(isinstance(sorts, list) and len(sorts) <= 3, 'At most 3 sort keys')
    for s in sorts:
        fields(s, ('column',), ('descending',))
        need(s['column'] in aliases and type(s.get('descending', False)) is bool, 'Invalid sort key')
    return types


def python_backend(rows, q, types):
    """Backend boundary: raw text rows + validated query + column types -> exact rows/counts."""
    groups, aggs = q.get('group_by', []), q['aggregates']
    selected = []
    for raw in rows:
        cache = {}
        def cell(c):
            if c not in cache:
                v = raw[c]
                cache[c] = None if v == '' else number(v) if types[c] == 'number' else v
            return cache[c]
        verdicts = []
        for f in q.get('filters', []):
            v = cell(f['column'])
            target = number(f['value']) if types[f['column']] == 'number' else f['value']
            verdicts.append(None if v is None else COMPARE[f['op']](v, target))
        if False in verdicts:
            continue
        if None in verdicts:
            need(q.get('missing', 'error') == 'exclude', 'Missing filter value')
            continue
        for d in q.get('derive', []):
            a, b = cell(d['left']), cell(d['right'])
            cache[d['as']] = None if a is None or b is None else ARITH[d['op']](a, b)
        needed = groups + [a['column'] for a in aggs if a['op'] != 'count']
        values = {c: cell(c) for c in needed}
        if any(v is None for v in values.values()):
            need(q.get('missing', 'error') == 'exclude', 'Missing group or aggregate value')
            continue
        selected.append(values)
    buckets = {} if groups else {(): []}
    for r in selected:
        key = tuple(r[c] for c in groups)
        buckets.setdefault(key, []).append(r)
        need(len(buckets) <= MAX_GROUPS, 'Group limit')
    result = []
    for key, records in sorted(buckets.items()):
        item = dict(zip(groups, key))
        for a in aggs:
            op = a['op']
            vals = [r[a['column']] for r in records] if op != 'count' else []
            if op == 'count': value = Fraction(len(records))
            elif op == 'sum': value = sum(vals, Fraction(0))
            else:
                need(bool(vals), 'Undefined aggregate over empty selection')
                value = sum(vals, Fraction(0)) / len(vals) if op == 'mean' else min(vals) if op == 'min' else max(vals)
            item[a['as']] = value
        result.append(item)
    for s in reversed(q.get('sort', [])):
        result.sort(key=lambda r: r[s['column']], reverse=s.get('descending', False))
    return result, len(selected)


BACKENDS = {'python': python_backend}


def execute(request, workspace, *, backend='python'):
    """One backend-independent response; never submit an answer or invoke an LLM."""
    try:
        need(backend in BACKENDS, 'Backend unavailable')
        need(isinstance(request, dict), 'Expected a query object')
        delimiter = request.get('delimiter', ',')
        need(delimiter in (',', ';', '\t'), 'Unsupported delimiter')
        columns, rows = load_csv(workspace, request.get('table'), delimiter)
        types = validate(request, columns)
        output, matched = BACKENDS[backend](rows, request, types)
        limit = request.get('limit', 20)
        serial = [{k: str(v) if isinstance(v, Fraction) else v for k, v in r.items()} for r in output[:limit]]
        result = dict(version=VERSION, ok=True, result=serial, scanned_rows=len(rows), matched_rows=matched,
                      total_groups=len(output), truncated=len(output) > limit)
        need(len(json.dumps(result).encode()) <= 16384, 'Output byte limit; no partial result')
        return result
    except (QueryError, OSError, UnicodeError, csv.Error, ValueError, TypeError, KeyError,
            ZeroDivisionError, OverflowError) as exc:
        return dict(version=VERSION, ok=False, result=None,
                    error=str(exc) if isinstance(exc, QueryError) else type(exc).__name__)
