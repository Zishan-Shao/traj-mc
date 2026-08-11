# Security policy

Please report security issues through the repository's private GitHub Security
Advisory flow rather than opening a public issue. Include affected files,
reproduction steps, and potential impact.

This project loads model implementations with `trust_remote_code=True`.
Only use model repositories and revisions you trust, review remote code before
execution, and pin revisions in production environments. Evaluation tasks that
execute generated code (for example HumanEval or MBPP) must run inside an
isolated sandbox.
