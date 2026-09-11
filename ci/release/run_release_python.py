#!/usr/bin/env python3
"""Run a release helper on Python 3.10 without weakening offline verification.

Ubuntu 22.04 supplies Python 3.10, while release_common imports the Python 3.11
``tomllib`` name. Builder images install ``tomli`` for helpers that parse TOML.
Host-only evidence packagers do not parse TOML, so a fail-closed placeholder is
enough when neither implementation exists.
"""

from __future__ import annotations

import runpy
import sys
import types


def _install_tomllib_compatibility() -> None:
    try:
        __import__("tomllib")
        return
    except ModuleNotFoundError:
        pass
    try:
        tomli = __import__("tomli")
    except ModuleNotFoundError:
        module = types.ModuleType("tomllib")

        class TOMLDecodeError(ValueError):
            """Compatibility exception used by release_common error paths."""

        def unavailable(*_args: object, **_kwargs: object) -> object:
            raise TOMLDecodeError(
                "TOML parsing requires Python 3.11+ or the pinned tomli package"
            )

        module.TOMLDecodeError = TOMLDecodeError
        module.load = unavailable
        module.loads = unavailable
        sys.modules["tomllib"] = module
    else:
        sys.modules["tomllib"] = tomli


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_release_python.py {-c CODE|SCRIPT} [ARG ...]")
    _install_tomllib_compatibility()
    script = sys.argv[1]
    if script == "-c":
        if len(sys.argv) < 3:
            raise SystemExit("run_release_python.py -c requires CODE")
        code = sys.argv[2]
        sys.argv = ["-c", *sys.argv[3:]]
        exec(compile(code, "<release-helper>", "exec"), {"__name__": "__main__"})
        return
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
