"""P2: bounded evidence for explicit public operations (stdlib only).

One narrow full-sentence text grammar complements the original JSON contract.
Found establishes data availability, never correctness of a submitted answer.
"""
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import time

CONTRACT_VERSION = 'public-sufficiency-v1'
VERSION = 'public-sufficiency-v3-reviewed'
MODES = ('off', 'observe', 'skip')
PHASES = ('pre_api', 'post_tool')


@dataclass(frozen=True)
class Limits:
    bytes: int = 65536
    files: int = 8
    rows: int = 1000
    seconds: float = .25
    instruction_bytes: int = 16000

    def __post_init__(self):
        for key in ('bytes', 'files', 'rows', 'instruction_bytes'):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError('Positive integer limits required')
        if not math.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError('Positive time limit required')


class Incomplete(Exception):
    pass


class Unsupported(Exception):
    pass


def _unknown(value):
    return value is None or isinstance(value, str) and not value.strip()


def _scalar(value):
    return value is None or type(value) in (str, bool, int, float) and (
        not isinstance(value, float) or math.isfinite(value))


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def _json(text):
    return json.loads(text, object_pairs_hook=_pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def _contract(instruction, limits):
    if len(instruction.encode('utf-8')) > limits.instruction_bytes:
        raise Incomplete('instruction_bytes')
    try:
        q = _json(instruction)
    except (ValueError, RecursionError):
        raise Unsupported('Instruction is not an exact supported JSON contract') from None
    common = {'p2_contract', 'operation', 'sources'}
    if not isinstance(q, dict) or q.get('p2_contract') != CONTRACT_VERSION:
        raise Unsupported('No explicit public operation contract')
    sources = q.get('sources')
    if not isinstance(sources, list) or not sources or any(type(s) is not str for s in sources):
        raise Unsupported('Explicit nonempty source list required')
    if len(set(sources)) != len(sources):
        raise Unsupported('Duplicate source names')
    if q.get('operation') == 'select':
        if set(q) != common | {'where', 'require', 'derive_product'}:
            raise Unsupported('Unsupported select keys')
        if (not isinstance(q['where'], dict) or not isinstance(q['require'], list)
                or not q['require'] or not isinstance(q['derive_product'], dict)):
            raise Unsupported('Invalid select shape')
        if any(type(k) is not str or not k for k in [*q['where'], *q['require'], *q['derive_product']]):
            raise Unsupported('Nonempty field names required')
        if any(not _scalar(v) or _unknown(v) for v in q['where'].values()):
            raise Unsupported('Only known scalar equality predicates supported')
        for field, operands in q['derive_product'].items():
            if (field not in q['require'] or not isinstance(operands, list) or len(operands) != 2
                    or any(type(k) is not str or not k or k == field for k in operands)):
                raise Unsupported('Only one-level product fallback is supported')
    elif q.get('operation') == 'follow':
        if (set(q) != common | {'start', 'relations'} or type(q['start']) is not str or not q['start']
                or not isinstance(q['relations'], list) or not 1 <= len(q['relations']) <= 8
                or any(type(v) is not str or not v for v in q['relations'])):
            raise Unsupported('Only exact functional chains of 1..8 relations supported')
    else:
        raise Unsupported('Unsupported operation')
    return q


def _open_public(workspace, name):
    """Same workspace containment/link policy as core.with_preview; pin every component.

    No glob, arbitrary Python, environment, network, or directory enumeration.
    O_NONBLOCK prevents a public FIFO from blocking before the regular-file check.
    """
    parts = Path(name).parts
    denied = {'answer.txt', 'secrets', 'secrets.json', 'credentials', 'credentials.json',
              'private', 'expected', 'labels.json', 'token.json'}
    if (not parts or Path(name).is_absolute() or any(p in ('.', '..') or p.startswith('.')
            or p.lower() in denied for p in parts) or Path(name).suffix.lower() not in {'.csv', '.json'}):
        raise ValueError('Public source path rejected')
    fd = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            os.close(file_fd)
            raise ValueError('Only regular public files supported')
        return os.fdopen(file_fd, 'rb')
    finally:
        os.close(fd)


def check(workspace, instruction, files, *, limits=Limits(), clock=time.monotonic, allow_text=True):
    """Only these public inputs are consumed. Labels/IDs/reward cannot affect evidence.

    Each source is a finite closed table for this operation. Incomplete sources
    invalidate absence proofs; no source prefix is interpreted as a complete table.
    """
    started = clock()
    report = dict(version=VERSION, limits=asdict(limits), scope=[], bytes_read=0,
                  rows_read=0, evidence=[], complete=False, result='Unknown', reason='',
                  necessary_condition=None, plan_built=False, plan=None,
                  interpretation=None, reason_code='unsupported_contract')

    def tick():
        if clock() - started >= limits.seconds:
            raise Incomplete('seconds')

    def finish(result, reason, evidence=(), condition=None):
        tick()
        report.update(result=result, reason=reason, evidence=list(evidence), necessary_condition=condition)

    try:
        tick()
        text_plan = None
        try:
            q = _contract(instruction, limits)
            report.update(plan_built=True, plan=q, reason_code='explicit_contract')
        except Unsupported:
            if not allow_text:
                raise
            from .sufficiency_text import interpret
            parsed = interpret(instruction, files)
            report.update(interpretation=parsed.basis, reason_code=parsed.reason_code)
            if parsed.plan is None:
                raise
            text_plan = parsed.plan
            report['plan'] = text_plan.public_dict()
            q = dict(operation=text_plan.operation, sources=text_plan.sources)
        sources = sorted(q['sources'])
        if len(sources) > limits.files:
            raise Incomplete('files')
        manifest = {f['path'] for f in files}
        rows = []
        for name in sources:
            tick()
            if name not in manifest:
                # Missing files may be mount/read failures, not a factual absence proof.
                raise OSError('Named source is not in public manifest')
            scope = dict(path=name, bytes_read=0, rows_read=0, complete=False)
            report['scope'].append(scope)
            with _open_public(workspace, name) as stream:
                before = os.fstat(stream.fileno())
                if before.st_nlink != 1:
                    raise ValueError('Hardlinked public source rejected')
                size = before.st_size
                scope['size_bytes'] = size
                remaining = limits.bytes - report['bytes_read']
                data = stream.read(min(size, remaining))
                report['bytes_read'] += len(data)
                scope['bytes_read'] = len(data)
                scope['sha256_read'] = hashlib.sha256(data).hexdigest()
                tick()
                if size > remaining or len(data) != size:
                    raise Incomplete('bytes')
                after = os.fstat(stream.fileno())
                if (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (size, before.st_mtime_ns, before.st_ctime_ns):
                    raise OSError('Source changed during reading')
            text = data.decode('utf-8')
            if name.lower().endswith('.csv'):
                parser = csv.DictReader(io.StringIO(text, newline=''), strict=True)
                fields = parser.fieldnames
                if not fields or len(set(fields)) != len(fields) or any(not f for f in fields):
                    raise ValueError('Missing/duplicate CSV headers')
                scope['fields'] = fields
                records = parser
            else:
                records = _json(text)
                if not isinstance(records, list):
                    raise ValueError('JSON source must be an array of objects')
            for i, row in enumerate(records, 1):
                tick()
                if report['rows_read'] >= limits.rows:
                    raise Incomplete('rows')
                report['rows_read'] += 1
                scope['rows_read'] += 1
                if (not isinstance(row, dict) or any(type(k) is not str or not _scalar(v) for k, v in row.items())
                        or None in row or (name.lower().endswith('.csv') and any(v is None for v in row.values()))):
                    raise ValueError('Expected flat records with matching CSV width')
                rows.append((row, dict(path=name, row=i, fields=sorted(row))))
            scope['complete'] = True
        tick()
        report['complete'] = True
        if text_plan is not None:
            from .sufficiency_text import verify
            report.update(verify(text_plan, rows, report['scope'], tick))
            report['interpretation']['schema_validated'] = report['plan_built']
        elif q['operation'] == 'select':
            selected, ambiguous = [], []
            for row, ref in rows:
                tick()
                # A known false conjunct excludes a row even if another cell is null.
                if any(not _unknown(row.get(k)) and (type(row[k]) != type(v) or row[k] != v)
                       for k, v in q['where'].items()):
                    continue
                if any(_unknown(row.get(k)) for k in q['where']):
                    ambiguous.append(ref)
                else:
                    selected.append((row, ref))
            if ambiguous:
                finish('Unknown', 'Unknown predicate cells may change selection', ambiguous)
            else:
                evidence, missing, uncertain_products = [], [], []
                for row, ref in selected:
                    for field in q['require']:
                        tick()
                        if not _unknown(row.get(field)):
                            evidence.append(dict(ref, required_field=field, via='direct'))
                            continue
                        operands = q['derive_product'].get(field, [])
                        known = {}
                        for key in operands:
                            value = row.get(key)
                            if _unknown(value):
                                continue
                            try:
                                number = Decimal(str(value))
                            except InvalidOperation:
                                raise ValueError('Product operands must be numeric') from None
                            if type(value) is bool or not number.is_finite():
                                raise ValueError('Finite numeric product required')
                            known[key] = number
                        exact_zero = any(v == 0 and type(row[k]) in (int, str) for k, v in known.items())
                        if operands and exact_zero:
                            evidence.append(dict(ref, required_field=field, via='product_zero', operands=operands))
                            continue
                        if operands and len(known) < len(set(operands)) and any(v == 0 for v in known.values()):
                            # json.loads may round a tiny nonzero float to 0.0.
                            # Without lexical provenance it cannot certify 0*x.
                            uncertain_products.append(dict(ref, required_field=field, operands=operands))
                            continue
                        if operands and all(not _unknown(row.get(k)) for k in operands):
                            try:
                                values = [float(row[k]) for k in operands]
                            except (ValueError, TypeError):
                                raise ValueError('Product operands must be numeric') from None
                            if any(type(row[k]) is bool for k in operands) or not math.isfinite(math.prod(values)):
                                raise ValueError('Finite numeric product required')
                            evidence.append(dict(ref, required_field=field, via='product', operands=operands))
                        else:
                            missing.append(dict(ref, required_field=field, absent_or_null=True,
                                                unavailable_fallback=[k for k in operands if _unknown(row.get(k))]))
                if uncertain_products:
                    finish('Unknown', 'Incomplete product has floating zero without exact lexical provenance', uncertain_products)
                elif missing:
                    finish('Missing', 'Required selected value and all declared derivations unavailable', missing,
                           'Each selected row needs the required value or its declared product inputs')
                else:
                    finish('Found', 'All selected values available; empty selection is valid', evidence)
        else:
            current, evidence = q['start'], []
            for relation in q['relations']:
                tick()
                uncertain = []
                matches = []
                for row, ref in rows:
                    tick()
                    if set(row) != {'subject', 'predicate', 'object'}:
                        raise ValueError('Fact records require subject/predicate/object')
                    if any(not _unknown(row[k]) and row[k] != v for k, v in
                           (('subject', current), ('predicate', relation))):
                        continue
                    if any(_unknown(row[k]) for k in ('subject', 'predicate', 'object')):
                        uncertain.append(ref)
                    elif any(type(row[k]) is not str for k in row):
                        raise ValueError('Fact identifiers must be strings')
                    else:
                        matches.append((row['object'], ref))
                if uncertain or len({v for v, _ in matches}) > 1:
                    finish('Unknown', 'Relation is unknown or nonfunctional', uncertain or [r for _, r in matches])
                    break
                if not matches:
                    finish('Missing', 'No required edge in the complete declared fact tables',
                           [*evidence, dict(subject=current, predicate=relation, absent=True)],
                           dict(subject=current, predicate=relation, needs='one known object'))
                    break
                evidence.extend(dict(ref, subject=current, predicate=relation, next=value)
                                for value, ref in matches)
                current = matches[0][0]
            else:
                finish('Found', 'Every required functional edge is present', evidence)
    except Unsupported as exc:
        report.update(result='Unknown', reason=str(exc))
    except Incomplete as exc:
        report.update(result='Unknown', reason='Limit reached: ' + str(exc), complete=False,
                      evidence=[], necessary_condition=None, reason_code='limit_' + str(exc))
    except (OSError, ValueError, csv.Error, RecursionError, OverflowError) as exc:
        report.update(result='Error', reason='Public read/format failure: ' + type(exc).__name__,
                      complete=False, evidence=[], necessary_condition=None, reason_code='read_or_format_error')
    report['duration_seconds'] = max(0, clock() - started)
    return report


def diagnostic(workspace, task, mode, phase, *, limits=Limits()):
    if mode not in MODES or phase not in PHASES:
        raise ValueError('Unknown sufficiency mode/phase')
    report = check(workspace, task['instruction'], task['files'], limits=limits)
    report.update(mode=mode, phase=phase, action='continue',
                  first_model_call_already_paid=phase == 'post_tool')
    if mode == 'skip' and phase == 'pre_api' and report['result'] == 'Missing' and report['complete']:
        report['action'] = 'skip'
    return report
