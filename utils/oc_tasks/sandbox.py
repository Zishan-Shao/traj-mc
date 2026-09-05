"""Run model-generated Python in a throwaway subprocess.

HumanEval and MBPP are scored by executing what the model wrote.  Upstream
(human-eval, OpenCompass) does this in-process behind ``reliability_guard`` and
a ``signal`` alarm; a generated ``while True`` or a C-level segfault then takes
the evaluator down with it.  One fresh interpreter per program is a few tens of
milliseconds and cannot corrupt the parent, so the harness can keep going.

Only the execution mechanism differs from upstream -- the program text handed
to it is assembled exactly as upstream assembles it.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import subprocess
import sys
import tempfile

# Ceilings for one generated program.  Address space stops a runaway
# allocation from evicting the model from host RAM; CPU time is a backstop for
# the wall-clock timeout, which a process ignoring SIGTERM could outlive.
_MEM_LIMIT_BYTES = 4 * 1024 ** 3
_PREAMBLE = """\
import resource, sys
resource.setrlimit(resource.RLIMIT_AS, ({mem}, {mem}))
resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))
sys.setrecursionlimit(10000)
"""


def run_program(program: str, timeout: float = 3.0) -> str:
    """Execute ``program``; return 'pass', 'timeout', or 'failed'."""
    source = _PREAMBLE.format(mem=_MEM_LIMIT_BYTES, cpu=int(timeout) + 1) + program
    with tempfile.TemporaryDirectory() as work_dir:
        path = os.path.join(work_dir, "candidate.py")
        with open(path, "w") as handle:
            handle.write(source)
        try:
            done = subprocess.run(
                [sys.executable, path],
                cwd=work_dir,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
                # A generated program that reads the environment (or imports
                # torch) should not inherit the parent's CUDA context.
                env={"PATH": os.environ.get("PATH", ""), "HOME": work_dir,
                     "TMPDIR": work_dir, "CUDA_VISIBLE_DEVICES": ""},
            )
        except subprocess.TimeoutExpired:
            return "timeout"
        except Exception:
            return "failed"
    return "pass" if done.returncode == 0 else "failed"


def run_programs(programs, timeout: float = 3.0, workers: int = 8):
    """Execute programs concurrently, preserving input order."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda p: run_program(p, timeout), programs))
