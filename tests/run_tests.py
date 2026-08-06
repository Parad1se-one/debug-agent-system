from __future__ import annotations

import argparse
import fnmatch
import importlib.util
import inspect
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from typing import Any


class _MonkeyPatch:
    """Small pytest-compatible monkeypatch subset for the local runner."""

    def __init__(self) -> None:
        self._undo: list[tuple[str, Any, Any, bool]] = []

    def setenv(self, name: str, value: Any) -> None:
        self.setitem(os.environ, name, str(value))

    def delenv(self, name: str, raising: bool = True) -> None:
        self.delitem(os.environ, name, raising=raising)

    def setitem(self, mapping: Any, key: Any, value: Any) -> None:
        existed = key in mapping
        old = mapping.get(key) if existed else None
        self._undo.append(("item", mapping, key, (existed, old)))
        mapping[key] = value

    def delitem(self, mapping: Any, key: Any, raising: bool = True) -> None:
        existed = key in mapping
        if not existed:
            if raising:
                raise KeyError(key)
            return
        self._undo.append(("item", mapping, key, (True, mapping[key])))
        del mapping[key]

    def setattr(self, target: Any, name: str, value: Any, raising: bool = True) -> None:
        existed = hasattr(target, name)
        if not existed and raising:
            raise AttributeError(name)
        old = getattr(target, name, None)
        self._undo.append(("attr", target, name, (existed, old)))
        setattr(target, name, value)

    def delattr(self, target: Any, name: str, raising: bool = True) -> None:
        existed = hasattr(target, name)
        if not existed:
            if raising:
                raise AttributeError(name)
            return
        self._undo.append(("attr", target, name, (True, getattr(target, name))))
        delattr(target, name)

    def chdir(self, path: str | Path) -> None:
        old = Path.cwd()
        self._undo.append(("cwd", None, None, old))
        os.chdir(path)

    def syspath_prepend(self, path: str | Path) -> None:
        self._undo.append(("syspath", None, None, list(sys.path)))
        sys.path.insert(0, str(path))

    def undo(self) -> None:
        while self._undo:
            kind, target, key, state = self._undo.pop()
            if kind == "item":
                existed, old = state
                if existed:
                    target[key] = old
                else:
                    target.pop(key, None)
            elif kind == "attr":
                existed, old = state
                if existed:
                    setattr(target, key, old)
                elif hasattr(target, key):
                    delattr(target, key)
            elif kind == "cwd":
                os.chdir(state)
            elif kind == "syspath":
                sys.path[:] = state


def load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _test_paths(values: list[str]) -> list[Path]:
    root = Path(__file__).parent
    if not values:
        return sorted(root.glob('**/test_*.py'))
    paths: set[Path] = set()
    for value in values:
        candidate = Path(value)
        if candidate.is_dir():
            paths.update(candidate.glob('**/test_*.py'))
        elif candidate.is_file():
            paths.add(candidate)
        else:
            paths.update(root.glob(value))
    return sorted(paths)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'paths',
        nargs='*',
        help='optional test files, directories, or glob patterns relative to tests/',
    )
    parser.add_argument(
        '--name-pattern',
        action='append',
        default=[],
        help='optional shell-style test function/method pattern; may be repeated',
    )
    args = parser.parse_args(argv)
    failures = []
    count = 0
    paths = _test_paths(args.paths)
    if not paths:
        parser.error('no test files matched')
    for path in paths:
        mod = load(path)
        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            if not name.startswith('test_'):
                continue
            if args.name_pattern and not any(
                fnmatch.fnmatchcase(name, pattern) for pattern in args.name_pattern
            ):
                continue
            count += 1
            try:
                parameters = set(inspect.signature(fn).parameters)
                supported = {'tmp_path', 'monkeypatch'}
                if not parameters <= supported:
                    raise RuntimeError(
                        f'unsupported test fixtures: {", ".join(parameters)}'
                    )
                with ExitStack() as stack:
                    kwargs: dict[str, Any] = {}
                    if 'tmp_path' in parameters:
                        kwargs['tmp_path'] = Path(stack.enter_context(tempfile.TemporaryDirectory()))
                    if 'monkeypatch' in parameters:
                        monkeypatch = _MonkeyPatch()
                        stack.callback(monkeypatch.undo)
                        kwargs['monkeypatch'] = monkeypatch
                    fn(**kwargs)
            except Exception as exc:  # noqa: BLE001
                failures.append((str(path), name, type(exc).__name__, str(exc)))
        suite = unittest.defaultTestLoader.loadTestsFromModule(mod)
        if args.name_pattern:
            selected = unittest.TestSuite()
            for group in suite:
                for test in group:
                    method = getattr(test, '_testMethodName', '')
                    if any(fnmatch.fnmatchcase(method, pattern) for pattern in args.name_pattern):
                        selected.addTest(test)
            suite = selected
        if suite.countTestCases():
            result = unittest.TestResult()
            suite.run(result)
            count += result.testsRun
            for test, traceback in [*result.failures, *result.errors]:
                failures.append((str(path), str(test), "unittest", traceback))
    if failures:
        for item in failures:
            print('FAIL', *item)
        print(f'{len(failures)}/{count} failed')
        return 1
    print(f'{count} tests passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
