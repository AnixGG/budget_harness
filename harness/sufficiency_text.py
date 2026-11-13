"""Conservative full-sentence grammar for one CSV operation, not general NLP.

No data reads, IDs, labels, generator names, models, or corpus-specific paths.
Field and path captures remain case-sensitive; only grammar words ignore case.
"""
from dataclasses import asdict, dataclass, replace
from decimal import Decimal, InvalidOperation
import re

ADAPTER_VERSION = 'latest-sum-text-v1'
FIELD = r'[A-Za-z_][A-Za-z_0-9]*'
PATH = r'[A-Za-z_0-9][A-Za-z_0-9./-]*\.csv'


@dataclass(frozen=True)
class LatestSumPlan:
    group: str
    order: str
    value: str
    sources: tuple[str, ...]
    product: tuple[str, ...]
    value_may_be_plural: bool
    operation: str = 'latest_sum'

    def public_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Interpretation:
    plan: LatestSumPlan | None
    reason_code: str
    basis: dict


GRAMMARS = (
    ('en_retain',
     rf'For each (?P<group>{FIELD}) (?:retain only|keep only) the (?:record|row) with '
     rf'(?:the )?(?:largest|highest|greatest) (?P<order>{FIELD}); '
     rf'(?P=order)s are distinct within an? (?P=group)\. '
     rf'Sum the retained (?P<value>{FIELD})\. Return an integer\.'),
    ('en_group',
     rf'Group rows by (?P<group>{FIELD})\. Keep the row with the highest (?P<order>{FIELD}) '
     rf'in each group\. (?P=order) is unique within each group\. '
     rf'Return the integer sum of (?P<value>{FIELD}) from those rows\.'),
    ('ru_retain',
     rf'Для каждого (?P<group>{FIELD}) оставьте запись с максимальной (?P<order>{FIELD}); '
     rf'(?:версии|значения (?P=order)) внутри (?P=group) различны\. '
     rf'Сложите (?P<value>{FIELD}) оставшихся записей и верните целое число\.'),
    ('ru_group',
     rf'Сгруппируйте строки по (?P<group>{FIELD})\. В каждой группе выберите строку с '
     rf'наибольшим (?P<order>{FIELD})\. (?P=order) уникален внутри группы\. '
     rf'Верните целую сумму (?P<value>{FIELD}) выбранных строк\.'),
)
FORMAT_SUFFIXES = (
    ' CSV files have headers. Blank cells mean unknown values, never implicit zero.',
    ' В CSV есть заголовки. Пустые ячейки означают неизвестные значения, не нули.',
)


def interpret(instruction, files):
    """Build a typed candidate; source schema validation is done by the checker.

    Full matching rejects added filters, alternative ordering, omitted uniqueness,
    commentary and contradictory instructions. A rejection never implies Missing.
    """
    text = ' '.join(instruction.split())
    basis = dict(adapter_version=ADAPTER_VERSION, full_instruction_match=False,
                 schema_validated=False)
    for suffix in FORMAT_SUFFIXES:
        if text.endswith(suffix):
            text = text[:-len(suffix)]
            basis['format_clause'] = suffix.strip()
            break
    source = None
    prefix = re.match(rf'(?:Read |Используйте )(?P<source>{PATH})\. ', text, re.IGNORECASE)
    if prefix:
        source = prefix['source']
        basis['source_clause'] = prefix[0].strip()
        text = text[prefix.end():]
    product = ()
    derived = re.search(
        rf' (?:Missing (?P<value>{FIELD}) can be computed as (?P<a>{FIELD})\*(?P<b>{FIELD})|'
        rf'Неизвестное (?P<ruvalue>{FIELD}) вычисляется как (?P<rua>{FIELD})\*(?P<rub>{FIELD}))\.$',
        text, re.IGNORECASE)
    if derived:
        product = (derived['a'] or derived['rua'], derived['b'] or derived['rub'])
        basis['derivation_clause'] = derived[0].strip()
        text = text[:derived.start()]
    for rule, pattern in GRAMMARS:
        pattern = re.sub(r'\(\?P=(\w+)\)', r'(?-i:(?P=\1))', pattern)
        match = re.fullmatch(pattern, text, re.IGNORECASE)
        if not match:
            continue
        group, order, value = (match[k] for k in ('group', 'order', 'value'))
        if len({group, order, value}) != 3 or (derived and (derived['value'] or derived['ruvalue']) != value):
            return Interpretation(None, 'ambiguous_fields_or_derivation', basis)
        if product and any(k in (group, order, value) for k in product):
            return Interpretation(None, 'unsupported_derivation', basis)
        paths = sorted(f['path'] for f in files if f['path'].lower().endswith('.csv'))
        if source is not None and source not in paths:
            return Interpretation(None, 'named_source_unavailable', basis)
        if not paths:
            return Interpretation(None, 'no_csv_sources', basis)
        if source is None and any(f['path'] not in paths and f['path'] != 'answer.txt' for f in files):
            return Interpretation(None, 'uninterpreted_non_csv_sources', basis)
        basis.update(full_instruction_match=True, rule=rule, operation_clause=text,
                     source_selection='explicit' if source else 'unique complete CSV schema',
                     captured_fields=dict(group=group, order=order, value=value),
                     closed_operation='sum value of unique maximum-order row per group')
        return Interpretation(LatestSumPlan(group, order, value, (source,) if source else tuple(paths),
                                           product, rule == 'en_retain'), 'text_candidate', basis)
    return Interpretation(None, 'unsupported_text_operation', basis)


def verify(plan, rows, scopes, tick):
    """Validate source binding and data dependencies; never compute an answer."""
    def result(state, code, reason, *, bound=None, evidence=(), condition=None):
        tick()
        return dict(result=state, reason_code=code, reason=reason, plan_built=bound is not None,
                    plan=bound.public_dict() if bound else plan.public_dict(),
                    evidence=list(evidence), necessary_condition=condition)

    candidates = []
    for scope in scopes:
        tick()
        fields = set(scope.get('fields', []))
        # English plural "retained amounts" can designate amount, but never
        # choose between columns amount and amounts when both are present.
        values = {plan.value}
        if plan.value_may_be_plural and plan.value.endswith('s'):
            values.add(plan.value[:-1])
        present = values & fields
        if {plan.group, plan.order} <= fields and present:
            if len(present) != 1:
                return result('Unknown', 'ambiguous_value_column', 'Plural field reference matches multiple columns')
            candidates.append(replace(plan, value=next(iter(present)), sources=(scope['path'],)))
    if len(candidates) != 1:
        code = 'ambiguous_csv_sources' if candidates else 'unverified_csv_schema'
        return result('Unknown', code, 'Need exactly one complete CSV with the requested group/order/value fields')
    bound = candidates[0]
    if len({bound.group, bound.order, bound.value}) != 3:
        return result('Unknown', 'ambiguous_resolved_fields', 'Resolved value field overlaps group/order')
    relevant = [(row, ref) for row, ref in rows if ref['path'] == bound.sources[0]]
    groups = {}
    seen = set()

    def unknown(value):
        return value is None or not value.strip()

    def number(value):
        try:
            value = Decimal(value)
        except (InvalidOperation, ValueError):
            raise ValueError('Expected a finite numeric CSV value') from None
        if not value.is_finite():
            raise ValueError('Expected a finite numeric CSV value')
        return value

    for row, ref in relevant:
        tick()
        if unknown(row.get(bound.group)) or unknown(row.get(bound.order)):
            return result('Unknown', 'unknown_group_or_order', 'Unknown key/order may change retained rows',
                          bound=bound, evidence=[ref])
        key, order = row[bound.group], number(row[bound.order])
        if (key, order) in seen:
            return result('Unknown', 'nonunique_order', 'Ordering values are not unique within a group',
                          bound=bound, evidence=[ref])
        seen.add((key, order))
        if key not in groups or order > groups[key][0]:
            groups[key] = order, row, ref
    evidence, missing, undeclared = [], [], []
    for key in sorted(groups):
        tick()
        _, row, ref = groups[key]
        if not unknown(row.get(bound.value)):
            number(row[bound.value])
            evidence.append(dict(ref, required_field=bound.value, via='direct',
                                 group_field=bound.group, order_field=bound.order))
            continue
        if bound.product:
            # An explicitly declared product also permits the mathematical zero
            # rule: a missing other numeric factor does not make 0*x unknown.
            known = {k: number(row[k]) for k in bound.product if not unknown(row.get(k))}
            if len(known) == len(set(bound.product)) or any(v == 0 for v in known.values()):
                evidence.append(dict(ref, required_field=bound.value,
                                     via='product_zero' if len(known) < len(set(bound.product)) else 'product',
                                     operands=list(bound.product)))
                continue
        # Additional known fields could support a derivation this grammar does
        # not understand. Do not use a direct null as an absence proof then.
        modeled = {bound.group, bound.order, bound.value, *bound.product}
        extra = [k for k, v in row.items() if k not in modeled and not unknown(v)]
        if extra:
            undeclared.append(dict(ref, unmodeled_fields=extra))
            continue
        missing.append(dict(ref, required_field=bound.value, absent_or_null=True,
                            unavailable_fallback=[k for k in bound.product if unknown(row.get(k))]))
    if undeclared:
        return result('Unknown', 'unmodeled_derivation_possible',
                      'Additional known fields may provide an undeclared derivation',
                      bound=bound, evidence=undeclared)
    if missing and any(s['path'] != bound.sources[0] and s['rows_read'] for s in scopes):
        return result('Unknown', 'unmodeled_sources', 'Other nonempty tables may support an unmodeled derivation',
                      bound=bound, evidence=missing)
    if missing:
        return result('Missing', 'required_retained_value_unavailable',
                      'Retained value and declared derivations unavailable in the complete selected CSV',
                      bound=bound, evidence=missing,
                      condition=dict(needs='known value or declared product for every retained row',
                                     field=bound.value, group=bound.group, order=bound.order))
    return result('Found', 'retained_values_available',
                  'All retained values available; empty table has a defined empty sum',
                  bound=bound, evidence=evidence)
