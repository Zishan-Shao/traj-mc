# Third-party notices

Parts of the evaluation adapters are derived from upstream projects:

- `eval/llada_harness.py` and `eval/llada_generate.py` derive
  from [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA), licensed under the
  MIT License. Its original notice is reproduced below.
- `eval/dream_harness.py` derives from
  [DreamLM/Dream](https://github.com/DreamLM/Dream), licensed under the Apache
  License 2.0. The file has been modified to load Traj-MC factors and to remove
  unrelated experiment paths.
- `baselines/quant_dllm/` is vendored from
  [ZTA2785/Quant-dLLM](https://github.com/ZTA2785/Quant-dLLM). Its Apache-2.0
  license and third-party notices are included inside that directory.
- `baselines/sink_aware_pruning/` is vendored from
  [VILA-Lab/Sink-Aware-Pruning](https://github.com/VILA-Lab/Sink-Aware-Pruning).
  Its MIT license is included inside that directory.
- `utils/oc_tasks/` reproduces the HumanEval, MBPP, IFEval and BBH protocols
  from [open-compass/opencompass](https://github.com/open-compass/opencompass),
  licensed under the Apache License 2.0: the few-shot prompts, the answer
  extractors, and the BBH chain-of-thought hints in `bbh_prompts/` (originally
  from [suzgunmirac/BIG-Bench-Hard](https://github.com/suzgunmirac/BIG-Bench-Hard),
  MIT). `utils/oc_tasks/ifeval/` is vendored verbatim from OpenCompass's
  `datasets/IFEval/`, which is in turn Google Research's
  `instruction_following_eval` (Apache-2.0); only its three intra-package
  imports were rewritten. The Apache-2.0 notice each carries is reproduced
  below.

## LLaDA MIT notice

MIT License

Copyright (c) 2025 NieShenRuc

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Apache License 2.0 notice (OpenCompass, Google Research IFEval)

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
these files except in compliance with the License. You may obtain a copy of the
License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
