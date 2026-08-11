# Third-party components

Quant-dLLM builds on prior open-source research code. Please retain this file,
`NOTICE`, and the upstream license files when redistributing the repository.

## Core implementation

- [ARB-LLM](https://github.com/ZHITENGLI/ARB-LLM), Apache-2.0
- [BiLLM](https://github.com/Aaronhuang-778/BiLLM), MIT

## Optional evaluation repositories

- [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA)
- [HKUNLP/Dream](https://github.com/HKUNLP/Dream), Apache-2.0
- [OpenCompass](https://github.com/open-compass/opencompass), Apache-2.0

These repositories are not redistributed by Quant-dLLM. The setup script
clones them into ignored local directories and applies the loader patches under
`patches/`. Reproducible defaults are pinned to LLaDA commit
`b7e6c3565f194352da44b38dd12b6970597309f3` and Dream commit
`31f94a60d187e3fd481fee3bbc2c732eb94a879c`:

```bash
./scripts/setup_evaluation_repos.sh
```

Set `LLADA_REPO` and `DREAM_REPO` to use separate upstream checkouts:

```bash
export LLADA_REPO=/path/to/LLaDA
export DREAM_REPO=/path/to/Dream
```

## Models and datasets

No model weights or datasets are distributed with this repository. Users must
obtain them from their original providers and comply with the corresponding
licenses and terms.
