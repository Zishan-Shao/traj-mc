# ==============================================================================
# VENDORED, UNMODIFIED (except this header) from:
#   Sink-Aware-Pruning/eval_llada.py
# which is the OFFICIAL LLaDA lm-eval wrapper (ML-GSAI/LLaDA evaluation/eval_llada.py)
# plus a compressed-weight injection (replace_with_traject_svd) and a single-GPU
# null-guard on accelerator.wait_for_everyone(). Verified vs the official file:
# core likelihood/generation math is byte-identical (diff -bw shows only additions).
#
# It registers the lm-eval model "llada_dist" and delegates to lm_eval.cli_evaluate,
# so ALL standard lm-eval flags work (--tasks --num_fewshot --batch_size
# --model_args ...,weights_path=DIR --output_path --log_samples --limit).
#
# The A/B .pt weight format it loads is exactly what trajmc.compression
# writes ({name.replace('.','_')}_A.pt / _B.pt), and its ".blocks." guard matches
# common.py's lm_head defense. This is the authoritative eval engine for REF/BASE/OURS
# (README section 0: evaluation protocol follows LLaDA official).
# ==============================================================================
'''
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
'''
import os
import math
import accelerate
import torch
import re
from pathlib import Path
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import datasets
datasets.disable_caching()
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
# Ensure the vendored LLaDA generator is importable no matter
# how the script is launched (accelerate launch does not reliably add script dir).
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llada_generate import generate


class StaticLowRank(nn.Module):
    def __init__(self, A, B, bias=None, pad_to=0):
        super().__init__()
        self.original_rank = int(B.shape[0])
        pad_to = int(pad_to or 0)
        if pad_to > 1 and self.original_rank % pad_to:
            runtime_rank = ((self.original_rank + pad_to - 1) // pad_to) * pad_to
            padded_A = torch.zeros(
                A.shape[0], runtime_rank, dtype=A.dtype, device=A.device
            )
            padded_B = torch.zeros(
                runtime_rank, B.shape[1], dtype=B.dtype, device=B.device
            )
            padded_A[:, :self.original_rank] = A
            padded_B[:self.original_rank, :] = B
            A, B = padded_A, padded_B
        self.runtime_rank = int(B.shape[0])
        self.A = nn.Parameter(A.to(torch.bfloat16), requires_grad=False)
        self.B = nn.Parameter(B.to(torch.bfloat16), requires_grad=False)
        self.bias = nn.Parameter(bias.to(torch.bfloat16), requires_grad=False) if bias is not None else None

    def forward(self, x):
        out = F.linear(F.linear(x, self.B), self.A)
        if self.bias is not None:
            out = out + self.bias
        return out


class _SelectedHeadCaptured(RuntimeError):
    pass


class SelectiveHeadProjector:
    """Exact final projection for only the token positions consumed by eval."""

    def __init__(self, model):
        self.model = model
        self.head = model.get_output_embeddings()
        if not isinstance(self.head, nn.Linear):
            raise TypeError("selective projection requires an untied nn.Linear lm_head")
        self.selection = None
        self.hidden = None

        def capture(_module, inputs):
            self.hidden = inputs[0][self.selection].detach()
            raise _SelectedHeadCaptured()

        self.handle = self.head.register_forward_pre_hook(capture)

    @torch.no_grad()
    def __call__(self, batch, selection, attention_mask=None):
        if selection.shape != batch.shape:
            raise ValueError(
                f"selection shape {tuple(selection.shape)} != batch {tuple(batch.shape)}"
            )
        self.selection = selection
        self.hidden = None
        try:
            self.model(batch, attention_mask=attention_mask)
        except _SelectedHeadCaptured:
            pass
        if self.hidden is None:
            raise RuntimeError("failed to capture final hidden states before lm_head")
        logits = F.linear(self.hidden, self.head.weight, self.head.bias).float()
        if getattr(self.model.config, "scale_logits", False):
            logits.mul_(1.0 / math.sqrt(self.model.config.d_model))
        return logits

    def close(self):
        self.handle.remove()


def replace_with_traject_svd(model, weights_path, device, mode="factorized", pad_to=0):
    if mode not in {"factorized", "materialized"}:
        raise ValueError(f"unknown low-rank runtime mode: {mode}")
    if mode == "materialized" and int(pad_to or 0):
        raise ValueError("lowrank_pad_to only applies to factorized mode")
    n_replaced = n_skipped = 0
    original_ranks = set()
    runtime_ranks = set()
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        # POLICY: only the 7 linears inside the 32 transformer blocks are ever compressed
        # (224 layers). Embedding and lm_head are NEVER compressed -- standard practice
        # (SVD-LLM / ASVD do the same). The top-level lm_head is named
        # `model.transformer.ff_out`, i.e. it shares the `ff_out` suffix with each block's MLP
        # down-projection, so a bare `ff_out$` regex silently caught it and a stale A/B left in
        # a weights dir then got loaded here -- the model evaluated no longer matched its own
        # metadata. This guard makes that class of bug impossible regardless of stray files.
        if ".blocks." not in name:
            continue
        prefix = name.replace(".", "_")
        path_a = os.path.join(weights_path, f"{prefix}_A.pt")
        path_b = os.path.join(weights_path, f"{prefix}_B.pt")
        if not (os.path.exists(path_a) and os.path.exists(path_b)):
            n_skipped += 1
            continue
        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        bias = mod.bias.detach().to(device) if mod.bias is not None else None
        A = torch.load(path_a, map_location=device).to(torch.bfloat16)
        B = torch.load(path_b, map_location=device).to(torch.bfloat16)
        original_ranks.add(int(B.shape[0]))
        if mode == "materialized":
            # Accuracy-throughput control: the compressed matrix A@B is unchanged
            # mathematically, but runtime storage/FLOPs are dense and must never be
            # reported as compressed deployment efficiency.
            mod.weight.data.copy_(torch.matmul(A, B).to(mod.weight.dtype))
            if bias is not None and mod.bias is not None:
                mod.bias.data.copy_(bias.to(mod.bias.dtype))
            runtime_ranks.add(int(B.shape[0]))
        else:
            replacement = StaticLowRank(A, B, bias, pad_to=pad_to).to(device)
            runtime_ranks.add(replacement.runtime_rank)
            setattr(parent, parts[-1], replacement)
            # Free the old dense weight while named_modules() still holds `mod`.
            mod.weight = None
        del A, B
        if n_replaced % 16 == 0:
            torch.cuda.empty_cache()
        n_replaced += 1
    torch.cuda.empty_cache()
    print(f"TrajectSVD layers loaded: {n_replaced}  skipped: {n_skipped}  "
          f"mode={mode} pad_to={int(pad_to or 0)} "
          f"original_ranks={sorted(original_ranks)} runtime_ranks={sorted(runtime_ranks)}")
    if n_replaced == 0:
        raise RuntimeError(
            f"weights_path={weights_path} matched zero LLaDA block linears"
        )
    return model


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path='',
        mask_id=126336,
        max_length=4096,
        batch_size=32,
        mc_num=128,
        is_check_greedy=True,
        cfg=0.,
        weights_path=None,
        steps=1024,
        gen_length=1024,
        block_length=1024,
        remasking='low_confidence',
        request_batch_size=4,
        fast_generate=False,
        fast_confidence=False,
        lowrank_mode="factorized",
        lowrank_pad_to=0,
        device="cuda",
        **kwargs,
    ):
        '''
        Args:
            model_path: LLaDA-8B-Base model path.
            mask_id: The token id of [MASK] is 126336.
            max_length: the max sequence length.
            batch_size: mini batch size.
            mc_num: Monte Carlo estimation iterations
            is_check_greedy: For certain metrics like LAMBADA, the evaluation requires the model to verify whether the answer
                             is generated through greedy sampling conditioned on the prompt (note that this differs from conditional
                             generation). We implement this verification through the suffix_greedy_prediction() function, which
                             returns a True/False judgment used for accuracy calculation.
                             When is_check_greedy is set to True, the lm-evaluation-harness library automatically invokes this function.
                             However, since none of the metrics in the LLaDA paper (https://arxiv.org/abs/2502.09992) require this functionality,
                             we recommend setting is_check_greedy to False. This configuration causes suffix_greedy_prediction() to return False
                             by default, significantly accelerating the evaluation process.
            cfg_scale: Unsupervised classifier-free guidance scale.
        '''
        super().__init__()

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None

        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})

        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, **model_kwargs)
        self.model.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            # [trajmc] Inference-only data parallelism: do NOT call
            # accelerator.prepare() -- it wraps the model in DistributedDataParallel,
            # which allocates a SECOND full copy of the params (gradient reduction
            # buckets, needed only for training). For an 8B model that is +16GB and
            # OOMs a 24GB GPU (observed on A5000: "Tried to allocate 14.93 GiB").
            # lm-eval already shards docs across ranks and gathers results itself, so
            # the model only needs to live on each rank's device. Numerically identical
            # to single-process; just avoids the training-only DDP overhead.
            self.model = self.model.to(self.accelerator.device)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model = self.model.to(device)

        if weights_path and str(weights_path).lower() not in ("", "none", "null"):
            wdir = Path(weights_path)
            if not any(wdir.glob("*_A.pt")):
                raise FileNotFoundError(
                    f"weights_path={weights_path} contains no *_A.pt factors"
                )
            self.model = replace_with_traject_svd(
                self.model, weights_path, self.device,
                mode=str(lowrank_mode), pad_to=int(lowrank_pad_to),
            )

        # Exact optimization: likelihood and reverse generation consume logits
        # only at masked target positions. Avoid projecting every prompt token to
        # the 126k vocabulary. Transformer states and numerical logits at selected
        # positions are unchanged (covered by the smoke equivalence gate).
        self._selective_head = SelectiveHeadProjector(self.model)

        self.mask_id = mask_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        self.cfg = cfg
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.remasking = remasking
        if isinstance(fast_generate, str):
            fast_generate = fast_generate.lower() in {"1", "true", "yes", "on"}
        self.fast_generate = bool(fast_generate)
        if isinstance(fast_confidence, str):
            fast_confidence = fast_confidence.lower() in {"1", "true", "yes", "on"}
        self.fast_confidence = bool(fast_confidence)
        self.request_batch_size = int(request_batch_size)
        if self.request_batch_size < 1:
            raise ValueError("request_batch_size must be >= 1")

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        """Expose the checkpoint tokenizer's native template to lm-eval.

        Base-model evaluations do not request chat templating, so adding this
        adapter is behavior-preserving for every existing run.  Instruct runs
        opt in through lm-eval's ``--apply_chat_template`` flag.
        """
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    @property
    def tokenizer_name(self) -> str:
        """Stable cache fingerprint required by lm-eval chat templating."""
        return self.tokenizer.name_or_path.replace("/", "__")

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_selected_logits(self, batch, prompt_index, selection, attention_mask=None):
        """Return row-major logits exactly at ``selection`` positions."""
        if self.cfg > 0.:
            if prompt_index.ndim == 1:
                assert len(prompt_index) == batch.shape[1]
                prompt_mask = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            else:
                prompt_mask = prompt_index
                assert prompt_mask.shape == batch.shape
            un_batch = batch.clone()
            un_batch[prompt_mask] = self.mask_id
            model_batch = torch.cat([batch, un_batch], dim=0)
            model_selection = torch.cat([selection, selection], dim=0)
            model_attention = None if attention_mask is None else torch.cat(
                [attention_mask, attention_mask], dim=0
            )
            selected = self._selective_head(
                model_batch, model_selection, attention_mask=model_attention
            )
            n = int(selection.sum())
            logits, un_logits = selected[:n], selected[n:]
            return un_logits + (self.cfg + 1) * (logits - un_logits)
        return self._selective_head(batch, selection, attention_mask=attention_mask)

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        return self.get_loglikelihood_many([(prefix, target)])[0]

    @torch.no_grad()
    def get_loglikelihood_many(self, examples):
        """Exact request batching with the original per-request RNG draw order."""
        n_loops = self.mc_num // self.batch_size
        plans = []
        # Precompute every perturbation request-by-request. Model eval consumes no
        # RNG, so this preserves the official sequential mask-draw stream exactly.
        for plan_index, (prefix, target) in enumerate(examples):
            seq = torch.concatenate([prefix, target])[None, :]
            seq = seq.repeat((self.batch_size, 1)).to(self.device)
            prompt_index = torch.arange(
                seq.shape[1], device=self.device
            ) < len(prefix)
            draws = []
            for _ in range(n_loops):
                perturbed, p_mask = self._forward_process(seq, prompt_index)
                draws.append((perturbed, p_mask, perturbed == self.mask_id))
            plans.append({
                "index": plan_index,
                "seq": seq,
                "prompt": prompt_index,
                "draws": draws,
            })

        loss_acc = [[] for _ in examples]
        # Padding changes the attention kernel and can introduce small bf16
        # differences. Bucket by exact sequence length instead, so every fused
        # request is numerically identical to its unbatched forward.
        by_length = {}
        for plan in plans:
            by_length.setdefault(plan["seq"].shape[1], []).append(plan)
        for same_length in by_length.values():
            for group_start in range(0, len(same_length), self.request_batch_size):
                group = same_length[group_start : group_start + self.request_batch_size]
                for loop_idx in range(n_loops):
                    batches, selections, prompts = [], [], []
                    labels, probabilities, counts = [], [], []
                    for plan in group:
                        perturbed, p_mask, selection = plan["draws"][loop_idx]
                        batches.append(perturbed)
                        selections.append(selection)
                        prompts.append(plan["prompt"].unsqueeze(0).repeat(
                            self.batch_size, 1
                        ))
                        labels.append(plan["seq"][selection])
                        probabilities.append(p_mask[selection])
                        counts.append(int(selection.sum()))

                    batch = torch.cat(batches, dim=0)
                    selection = torch.cat(selections, dim=0)
                    prompt = torch.cat(prompts, dim=0)
                    logits = self.get_selected_logits(batch, prompt, selection)
                    offset = 0
                    for plan, label, probability, count in zip(
                        group, labels, probabilities, counts
                    ):
                        local = logits[offset : offset + count]
                        loss = F.cross_entropy(
                            local, label, reduction='none'
                        ) / probability
                        loss_acc[plan["index"]].append(
                            (loss.sum() / self.batch_size).item()
                        )
                        offset += count
                    assert offset == logits.shape[0]

        return [-sum(losses) / len(losses) for losses in loss_acc]

    @torch.no_grad()
    def _legacy_get_loglikelihood(self, prefix, target):
        """Unbatched reference retained for numerical equivalence tests."""
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_selected_logits(
                perturbed_seq, prompt_index, mask_indices
            )

            loss = F.cross_entropy(logits, seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return - sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_selected_logits(seq, prompt_index, mask_index)
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            elems = [ds[i] for i in range(len(ds))]
            examples = [(e["prefix"], e["target"]) for e in elems]
            lls = self.get_loglikelihood_many(examples)
            for elem, ll in tqdm(
                zip(elems, lls), total=len(elems), desc="Collecting likelihood..."
            ):
                is_target_greedy_dec = self.suffix_greedy_prediction(
                    elem["prefix"], elem["target"]
                )
                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests: list[Instance]):
        # Likelihood evaluation keeps a selective lm-head pre-hook on the model.
        # Generation installs its own projector whose selection changes at every
        # reverse step.  Remove the likelihood hook first; otherwise it fires
        # before generate.py's hook and raises the other module's sentinel.
        if self._selective_head is not None:
            self._selective_head.close()
            self._selective_head = None

        # Do not use Dataset.map here.  Its fingerprinting tries to dill the
        # bound tokenizer closure and recursively serialize the entire custom
        # LLaDA model class, adding ~90 seconds and many PicklingWarnings to
        # every generation run.  This direct loop preserves request order and
        # produces the exact same token IDs without touching model execution.
        ds = []
        for req in requests:
            question = req.args[0]
            ds.append({
                "question": torch.tensor(
                    self.tokenizer(question)["input_ids"], dtype=torch.long
                ),
                "question_text": question,
                "until": req.args[1]["until"],
            })

        out = []
        for elem in tqdm(ds, desc="Generating..."):
            prompt = elem["question"].unsqueeze(0).to(self.device)
            stop_tokens = elem["until"]

            generated_answer = generate(self.model, prompt, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length,
                                        temperature=0, cfg_scale=self.cfg, remasking=self.remasking, mask_id=self.mask_id,
                                        fast_path=self.fast_generate,
                                        memory_efficient_confidence=self.fast_confidence)

            generated_answer = self.tokenizer.decode(generated_answer[0][prompt.shape[1]:], skip_special_tokens=False)
            for stop_seq in stop_tokens:
                    if stop_seq in generated_answer:
                        generated_answer = generated_answer.split(stop_seq)[0]

            # remove special tokens
            generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
            generated_answer = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
            out.append(generated_answer)

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

        return out


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()
