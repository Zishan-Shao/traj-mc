# Third-party notices

Parts of this repository are derived from upstream projects:

- `utils/generate.py` and the LLaDA sampler logic in `utils/eval_benchmarks.py`
  derive from [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA), licensed under
  the MIT License. Its original notice is reproduced below.
- `utils/dream_eval_instruct/` is the evaluation toolkit released with
  [DreamLM/Dream](https://github.com/DreamLM/Dream) (`eval_instruct/`, Apache
  License 2.0), itself a
  fork of [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
  (MIT). It has been modified to load Traj-MC low-rank factors through a
  `weights_path` model argument (`lm_eval/models/diffllm.py`), and task configs
  marked `[trajmc]` were added under `lm_eval/tasks/trajmc_added/`.
- `utils/oc_tasks/` reproduces benchmark protocols (prompts and answer
  extractors) from [open-compass/opencompass](https://github.com/open-compass/opencompass),
  licensed under the Apache License 2.0. The Apache-2.0 notice is reproduced
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

## Apache License 2.0 notice (OpenCompass, Dream)

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
these files except in compliance with the License. You may obtain a copy of the
License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
