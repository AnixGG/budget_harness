"""Submission protocol, independent of hidden answers and example labels."""
import hashlib
import shutil
import stat
from uuid import uuid4


class SubmissionError(ValueError):
    pass


def validate_answer(value, task_type):
    if not isinstance(value, str) or not value.strip() or len(value.encode()) > 64000:
        raise SubmissionError("Answer must be nonempty UTF-8 text, at most 64000 bytes")
    value = value.strip()
    if task_type == "coding":
        if value != "DONE":
            raise SubmissionError("Coding requires DONE after the provided tests pass")
    elif value.upper() == "DONE":
        raise SubmissionError("DONE is only for coding. Submit the actual answer value")
    return value


def fingerprint(path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    return (info.st_dev, info.st_ino, info.st_mtime_ns, info.st_ctime_ns, info.st_size)


class AnswerFiles:
    def __init__(self, directory, workspace, diagnostics):
        self.paths = [(workspace / "answer.txt", "workspace_file"), (directory / "answer.txt", "task_file")]
        self.previous = {p: fingerprint(p) for p, _ in self.paths}
        self.baseline = self.previous.copy()
        self.diagnostics = diagnostics
        self.backups = {}
        for path, source in self.paths:
            if self.baseline[path] is not None:
                if not path.is_file() and not path.is_symlink():
                    raise SubmissionError('Existing answer.txt must be a file or symlink')
                backup = diagnostics / 'before' / source
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, backup, follow_symlinks=False)
                self.backups[path] = backup

    def changed(self):
        for path, source in self.paths:
            current = fingerprint(path)
            previous, self.previous[path] = self.previous[path], current
            if current is not None and current != previous:
                yield path, source

    def reject(self, path):
        """Retain unaccepted changes for diagnosis and restore the pre-attempt file."""
        current = fingerprint(path)
        if current == self.baseline[path]:
            return None
        source = dict(self.paths)[path]
        retained = None
        if current is not None:
            retained = self.diagnostics / 'rejected' / (source + '-' + str(uuid4()))
            retained.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(retained))
        backup = self.backups.get(path)
        if backup is not None:
            shutil.copy2(backup, path, follow_symlinks=False)
        self.baseline[path] = self.previous[path] = fingerprint(path)
        return {'source': source, 'path': str(path),
                'diagnostic_path': str(retained) if retained else None,
                'restored_existing': backup is not None}


def read_answer(path):
    info = path.lstat()
    if path.parent.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise SubmissionError("answer.txt must be a regular file, not a link")
    if info.st_size > 64000:
        raise SubmissionError("answer.txt exceeds 64000 bytes")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeError as exc:
        raise SubmissionError("answer.txt must be UTF-8") from exc


def test_hashes(workspace):
    return {p.relative_to(workspace).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in workspace.rglob("test*.py") if p.is_file() and not p.is_symlink()}


def source_hashes(workspace):
    result={}
    for path in workspace.rglob('*.py'):
        if path.name.startswith('test') or '__pycache__' in path.parts:
            continue
        if path.is_symlink():
            raise SubmissionError('Implementation links cannot trigger automatic submission')
        if path.is_file():
            result[path.relative_to(workspace).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def verify_tests_unchanged(workspace, original):
    if not original:
        raise SubmissionError("No provided unittest tests; cannot verify coding completion")
    for name, expected in original.items():
        path = workspace / name
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SubmissionError("Provided tests were changed or removed; submission rejected")


CODING_TEST_CODE = """import sys, tempfile, unittest
sys.path.insert(0, '.')
with tempfile.TemporaryDirectory(prefix='jiuwen-test-cache-') as cache:
    # Test the current source even when an old .pyc has the same size and mtime.
    sys.pycache_prefix = cache
    sys.dont_write_bytecode = True
    suite = unittest.defaultTestLoader.discover('.', pattern='test*.py')
    if suite.countTestCases() == 0:
        print('No unittest tests discovered')
        raise SystemExit(2)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
"""
