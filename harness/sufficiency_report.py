"""Reproducible offline P2 fixture report; never imports or calls a provider."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile

from corpus_lab.p2_fixtures import ROOT, materialize
from .core import digest, public_task
from .sufficiency import VERSION, Limits, check


def evaluate(fixtures=ROOT):
    fixtures = Path(fixtures).resolve()
    public = json.loads((fixtures/'public.json').read_text())
    private = json.loads((fixtures/'private.json').read_text())
    if set(public) != set(private):
        raise ValueError('Fixture/expectation mismatch')
    rows = []
    for name, case in public.items():
        with tempfile.TemporaryDirectory(prefix='p2-fixture-') as tmp:
            root = Path(tmp)
            materialize(root, case)
            task = public_task(root)
            kwargs = {}
            if case.get('simulated_clock'):
                ticks = iter(i * .01 for i in range(10000))
                kwargs['clock'] = lambda: next(ticks)
            report = check(root / 'workspace', task['instruction'], task['files'],
                           limits=Limits(**case.get('limits', {})), **kwargs)
        expected = private[name]['result']
        rows.append(dict(case=name, public_sha256=digest(case), expected=expected,
                         matches=report['result'] == expected, report=report,
                         timing='simulated limit fixture' if kwargs else 'local monotonic'))
    project = Path(__file__).resolve().parent.parent
    sources = ['harness/sufficiency.py', 'harness/sufficiency_text.py', 'harness/sufficiency_report.py',
               'harness/core.py', 'corpus_lab/p2_fixtures.py',
               (fixtures/'public.json').relative_to(project).as_posix(),
               (fixtures/'private.json').relative_to(project).as_posix()]
    return dict(schema_version='offline-public-sufficiency-v1', checker_version=VERSION,
                mode='observe', phase='pre_api', new_api_calls=0,
                source_sha256={p: hashlib.sha256((project/p).read_bytes()).hexdigest() for p in sources},
                input_sha256=hashlib.sha256((fixtures/'public.json').read_bytes()).hexdigest(),
                cases=rows, counts=dict(Counter(r['report']['result'] for r in rows)),
                matched=sum(r['matches'] for r in rows), total=len(rows),
                false_missing_on_found=sum(r['expected'] == 'Found' and r['report']['result'] == 'Missing' for r in rows),
                limitations=['Hand-specified local logic checks, not quality on unseen tasks.',
                             'No token savings, scores or real-provider cost measured.',
                             'Duration varies; compare evidence/results and source/input hashes.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, default=ROOT)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error('A new report path is required')
    report = evaluate(args.fixtures)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ('matched', 'total', 'counts', 'new_api_calls')}))
    return 0 if report['matched'] == report['total'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
