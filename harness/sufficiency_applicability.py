"""Public-only applicability report; no provider, grader, training or labels.

Role membership is reporting metadata read only after checking each task. It is
never passed to interpret/check. Unread files have only manifest path/size hashes.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .core import digest, public_task
from .sufficiency import VERSION, check
from .sufficiency_text import ADAPTER_VERSION


def summarize(rows):
    total = len(rows)
    built = sum(r['check']['plan_built'] for r in rows)
    return dict(tasks=total, plans_built=built, plan_fraction=built/total if total else None,
                results=dict(Counter(r['check']['result'] for r in rows)),
                recognized_operations=dict(Counter((r['check']['plan'] or {}).get('operation', 'unsupported') for r in rows)),
                unknown_reasons=dict(Counter(r['check']['reason_code'] for r in rows if r['check']['result'] == 'Unknown')))


def evaluate(inputs, *, project=None):
    project = Path(project or Path(__file__).resolve().parent.parent).resolve()
    rows = []
    seen = set()
    for source in inputs['sources']:
        root = (project/source['root']).resolve()
        if not root.is_relative_to(project) or not root.is_dir():
            raise ValueError('Existing public task root inside project required')
        paths = sorted(root.glob('*/task.json'))
        if not paths:
            raise ValueError('No public tasks at source root')
        for path in paths:
            relative = path.parent.relative_to(project).as_posix()
            if relative in seen:
                raise ValueError('Duplicate public task path')
            seen.add(relative)
            task = public_task(path.parent)
            # Intentionally no task_id/type/reward, role, label or membership here.
            report = check(path.parent/'workspace', task['instruction'], task['files'])
            baseline = check(path.parent/'workspace', task['instruction'], task['files'], allow_text=False)
            role = inputs.get('roles', {}).get(relative, source['default_role'])
            rows.append(dict(source=source['name'], path=relative, role=role,
                             task_type=task['task_type'], public_instruction_manifest_sha256=digest(task),
                             check=report, json_only_baseline=baseline))
    root = Path(__file__).resolve().parent.parent
    code = ('harness/sufficiency.py', 'harness/sufficiency_text.py',
            'harness/sufficiency_applicability.py', 'harness/core.py')
    groups = lambda key: {value: summarize([r for r in rows if r[key] == value]) for value in sorted({r[key] for r in rows})}
    return dict(schema_version='public-sufficiency-applicability-v1', checker_version=VERSION,
                adapter_version=ADAPTER_VERSION, mode='observe', new_api_calls=0,
                inputs_sha256=digest(inputs),
                source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in code},
                summary=summarize(rows), by_type=groups('task_type'), by_source=groups('source'),
                by_role=groups('role'), json_only_baseline=dict(Counter(r['json_only_baseline']['result'] for r in rows)),
                tasks=rows,
                limitations=['Applicability, not success, false-skip rate, or real token/cost savings.',
                             'Train-only public inputs selected the grammar. Disclosed holdout/original are regression only.',
                             'JSON-only baseline uses the same checker with text adaptation disabled, not historical quality estimates.',
                             'Unread unsupported files are hashed as path/size manifests only; checked scopes record content hashes.',
                             'Membership roles are report-only, never adapter inputs. No private labels or grader results read.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error('New report path required')
    inputs = json.loads(args.inputs.read_text())
    result = evaluate(inputs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps(result['summary'], ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
