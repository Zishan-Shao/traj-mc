# Contributing

Contributions are welcome through issues and pull requests.

## Development setup

```bash
conda env create -f environment.yml
conda activate quant-dllm
```

For evaluation changes, also install:

```bash
python -m pip install -r requirements-eval.txt
```

## Pull requests

1. Keep model weights, datasets, caches, and generated checkpoints out of Git.
2. Do not add machine-specific absolute paths or credentials.
3. Preserve deterministic seeds when changing quantization or evaluation code.
4. Run the syntax and CLI checks below before opening a pull request.
5. Describe any expected metric change and include the exact command used.

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
bash -n run.sh eval_table1.sh scripts/setup_evaluation_repos.sh
python run_arb_llada.py --help
python run_arb_dream.py --help
```

## Reporting results

Report the model revision, calibration dataset and size, sequence length, group
size, ABMP ratio, random seed, task version, number of Monte Carlo samples, and
hardware. Do not commit raw evaluation logs, model weights, or generated
checkpoints.
