"""IFEval strict/loose instruction-following judge.

Vendored from OpenCompass (``opencompass/datasets/IFEval``), which is itself
Google Research's ``instruction_following_eval``.  Only the three
``import opencompass.datasets.IFEval.*`` lines were rewritten as relative
imports; the judging logic is byte-identical to the code that produced the
published LLaDA IFEval numbers.
"""

from .evaluation_main import (  # noqa: F401
    InputExample,
    test_instruction_following_loose,
    test_instruction_following_strict,
)
