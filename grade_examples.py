"""Offline local checks, including original coding tests. Never imported by a solver."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from uuid import uuid4

from harness.submission import read_answer, SubmissionError, test_hashes, CODING_TEST_CODE


def check_coding(directory, original, *, local=False, image='jiuwen-budget-harness:baseline'):
    """Run in a clean stage with trusted tests, never tests supplied by the solution."""
    workspace = directory / 'workspace'
    trusted = original / 'workspace'
    hashes = test_hashes(trusted)
    if not hashes:
        return None, {'reason': 'no_original_tests'}
    if test_hashes(workspace) != hashes:
        return False, {'reason': 'original_tests_changed', 'original_tests_unchanged': False}
    marker = 'CHECK_COMPLETED_' + uuid4().hex
    with tempfile.TemporaryDirectory(prefix='example-check-') as tmp:
        stage = Path(tmp)
        for path in workspace.rglob('*'):
            rel = path.relative_to(workspace)
            if '__pycache__' in rel.parts or path.suffix == '.pyc':
                continue
            if path.is_symlink():
                return False, {'reason': 'workspace_link'}
            if path.is_file():
                target = stage / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        for name in hashes:
            (stage / name).write_bytes((trusted / name).read_bytes())
        code = CODING_TEST_CODE.replace('raise SystemExit(0 if result.wasSuccessful() else 1)',
               f'print({marker!r}, flush=True)\n    raise SystemExit(0 if result.wasSuccessful() else 1)')
        name = 'example-grader-' + uuid4().hex
        command = ([sys.executable, '-I', '-B', '-c', code] if local else
                   ['docker', 'run', '--rm', '--name', name, '--network', 'none', '--read-only',
                    '--tmpfs', '/tmp:rw,size=64m', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
                    '--memory', '256m', '--cpus', '1', '--pids-limit', '32',
                    '-v', f'{stage}:/grade:ro', '--workdir', '/grade', '--entrypoint', 'python',
                    image, '-I', '-B', '-c', code])
        import signal
        with tempfile.TemporaryFile() as output:
            try:
                process = subprocess.Popen(command, cwd=stage, start_new_session=True,
                    env={'PATH': os.environ.get('PATH', ''), 'HOME': tmp}, stdout=output, stderr=subprocess.STDOUT)
            except OSError:
                return None, {'reason': 'grader_unavailable'}
            timeout = False
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                timeout = True
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                if not local:
                    subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=30)
            output.seek(0)
            log = output.read(32000).decode(errors='replace')
        details = {'original_tests_unchanged': True, 'exit_code': process.returncode,
                   'timed_out': timeout, 'stdout': log, 'reason': 'original_local_tests'}
        if timeout or (not local and process.returncode in {125, 126, 127}):
            return None, {**details, 'reason': 'grader_timeout_or_infrastructure'}
        return process.returncode == 0 and marker in log, details


def grade(root, labels=None, *, source=None, local=False, image='jiuwen-budget-harness:baseline', runs=None):
    results = []
    executions = {r['task_id']: r for r in runs or []}
    for path in sorted(Path(root).glob('*/task.json')):
        directory = path.parent
        # Source metadata and reward are authoritative when provided.
        original = Path(source) / directory.name if source else None
        task = json.loads(((original / 'task.json') if original else path).read_text())
        if labels:
            task.update(labels.get(directory.name, {}))
        answer_path = directory / 'answer.txt'
        exists = answer_path.exists() or answer_path.is_symlink()
        details = {}
        correct = None
        try:
            answer = read_answer(answer_path).strip() if exists else None
        except (SubmissionError, UnicodeError, OSError):
            answer, correct = None, False
            details = {'reason': 'invalid_answer_file'}
        check = task.get('check')
        if answer is not None:
            if task['task_type'] == 'coding':
                if answer != 'DONE':
                    correct, details = False, {'reason': 'coding_requires_DONE'}
                elif original:
                    correct, details = check_coding(directory, original, local=local, image=image)
                else:
                    details = {'reason': 'original_tests_not_supplied'}
            elif 'expected' in task and check in {'exact_int', 'exact_str'}:
                try:
                    correct = int(answer) == int(task['expected']) if check == 'exact_int' else answer == str(task['expected'])
                except ValueError:
                    correct = False
                details = {'reason': check}
        run = executions.get(task['task_id'])
        results.append({'task_id': task['task_id'], 'answer_exists': exists, 'answer_saved': exists,
                        'correctness': 'verified_correct' if correct is True else 'verified_incorrect' if correct is False else 'unknown',
                        'example_correct': correct, 'verification': details, 'check': check,
                        'execution': {k: run.get(k) for k in ('status', 'stop_reason', 'duration_seconds')} if run else None,
                        'cost': run.get('actual') if run else None,
                        'reward': task['reward'] if correct is True else 0})
    return {'scope': 'local example exact checks and original tests; not official hidden evaluation',
            'tasks': results, 'points': sum(r['reward'] for r in results),
            'verified_correct': sum(r['example_correct'] is True for r in results),
            'verified_incorrect': sum(r['example_correct'] is False for r in results),
            'unknown': sum(r['example_correct'] is None for r in results)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks-dir', required=True)
    parser.add_argument('--labels', help='Private labels; never mount into solver')
    parser.add_argument('--source', help='Trusted original example tasks, outside solver mount')
    parser.add_argument('--allow-local-exec', action='store_true')
    parser.add_argument('--image', default='jiuwen-budget-harness:baseline')
    parser.add_argument('--runs')
    parser.add_argument('--out')
    args = parser.parse_args()
    labels = json.loads(Path(args.labels).read_text()) if args.labels else None
    runs = [json.loads(s) for s in Path(args.runs).read_text().splitlines() if s.strip()] if args.runs else None
    report = grade(args.tasks_dir, labels, source=args.source, local=args.allow_local_exec, image=args.image, runs=runs)
    if args.out:
        from harness.core import atomic_json
        atomic_json(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
