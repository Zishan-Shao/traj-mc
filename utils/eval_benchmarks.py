"""
eval_benchmarks.py -- LLaDA generative benchmark evaluation, aligned with Table 2

All benchmarks use generation-based eval (no log-likelihood / mc_num).
Parameters match LLaDA-8B-Instruct EVAL.md exactly.

Usage:
  python eval_benchmarks.py --mode uncompressed --benchmarks all
  python eval_benchmarks.py --mode traject_svd --weights_path results/weights/llada/ours \
      --benchmarks mmlu,arc_c,gsm8k,math,gpqa
  python eval_benchmarks.py --mode uncompressed --benchmarks mmlu --limit 100
  python eval_benchmarks.py --benchmarks all --download_only
"""

import torch, argparse, os, sys, time, json, re, random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate import generate as llada_generate

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.path.join(REPO_ROOT, 'results')

# ── Benchmark configs (official EVAL.md) ─────────────────────
ALL_seqlenBENCHMARKS = ['mmlu', 'mmlu_pro', 'hellaswag', 'arc_c', 'gsm8k', 'math', 'gpqa']

BENCH_CONFIG = {
    'mmlu':      dict(gen_length=3,   block_length=3,   steps=3,   logits_eos_inf=False, confidence_eos_eot_inf=False, num_fewshot=5),
    'mmlu_pro':  dict(gen_length=256, block_length=256, steps=256, logits_eos_inf=False, confidence_eos_eot_inf=False, num_fewshot=5),
    'hellaswag': dict(gen_length=3,   block_length=3,   steps=3,   logits_eos_inf=False, confidence_eos_eot_inf=False, num_fewshot=0),
    'arc_c':     dict(gen_length=512, block_length=512, steps=512, logits_eos_inf=False, confidence_eos_eot_inf=False, num_fewshot=0),
    'gsm8k':     dict(gen_length=256, block_length=8,   steps=256, logits_eos_inf=False, confidence_eos_eot_inf=False, num_fewshot=4),
    'math':      dict(gen_length=512, block_length=512, steps=512, logits_eos_inf=False, confidence_eos_eot_inf=True,  num_fewshot=4),
    'gpqa':      dict(gen_length=64,  block_length=64,  steps=64,  logits_eos_inf=False, confidence_eos_eot_inf=True,  num_fewshot=5),
}

# Target accuracy from LLaDA Table 2 (OpenCompass column), for reference
TABLE2_TARGET = {
    'mmlu': 65.4, 'mmlu_pro': 36.6, 'hellaswag': 75.3,
    'arc_c': 89.2, 'gsm8k': 78.9, 'math': 29.6, 'gpqa': 32.3,
}

# ── args ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='uncompressed',
                    choices=['uncompressed', 'whitening', 'whitening_mixed', 'baseline_a', 'baseline_b',
                             'ids', 'ids_multi', 'joint_qkv', 'tsvd', 'traject_svd', 'traject_svd_zh',
                             'traject_svd_mixed', 'obs', 'obs_diff', 'obs_whitening', 'streaming_multistep'])
parser.add_argument('--model_path',   type=str, default='GSAI-ML/LLaDA-8B-Instruct')
parser.add_argument('--llada_path',   type=str, default=REPO_ROOT)
parser.add_argument('--weights_path', type=str,
                    default=os.path.join(RESULTS_ROOT, 'weights', 'llada', 'ours'))
parser.add_argument('--benchmarks',   type=str, default='all',
                    help='Comma-separated: mmlu,mmlu_pro,hellaswag,arc_c,gsm8k,math,gpqa  or "all"')
parser.add_argument('--limit',        type=int, default=None, help='Max samples per benchmark (None=full)')
parser.add_argument('--batch_size',   type=int, default=4,   help='Inference batch size (default 4)')
parser.add_argument('--output',       type=str, default=None, help='Output JSON path')
parser.add_argument('--resume',       action='store_true',
                    help='Skip benchmarks already in the latest checkpoint for this mode')
parser.add_argument('--download_only',action='store_true', help='Pre-cache all datasets then exit')
parser.add_argument('--shuffle_eval', action='store_true',
                    help='Shuffle MMLU test set before applying --limit (avoids subject-ordering bias)')
parser.add_argument('--sample_seed',  type=int, default=42,
                    help='RNG seed for --shuffle_eval (default 42)')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
MASK_ID = 126336

if args.benchmarks == 'all':
    BENCHMARKS = ALL_BENCHMARKS
else:
    BENCHMARKS = [b.strip() for b in args.benchmarks.split(',')]

# ── download-only mode ────────────────────────────────────────
if args.download_only:
    print("Pre-caching datasets (run this on a node with internet access)...")
    from datasets import load_dataset
    DATASET_SPECS = [
        ('cais/mmlu',             'all',           ['test', 'dev', 'validation']),
        ('TIGER-Lab/MMLU-Pro',    None,            ['test', 'validation']),
        ('Rowan/hellaswag',       None,            ['validation']),
        ('allenai/ai2_arc',       'ARC-Challenge', ['test']),
        ('gsm8k',                 'main',          ['test', 'train']),
    ]
    for path, name, splits in DATASET_SPECS:
        for split in splits:
            try:
                kw = {'path': path, 'split': split}
                if name:
                    kw['name'] = name
                ds = load_dataset(**kw)
                print(f"  OK  {path} {name or '':<15} {split} ({len(ds)} rows)")
            except Exception as e:
                print(f"  SKIP {path} {name or '':<15} {split}: {e}")

    # MATH: EleutherAI/hendrycks_math (7 categories, must be fetched individually)
    MATH_CATS = ['algebra', 'counting_and_probability', 'geometry',
                 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
    for split in ['test', 'train']:
        n_ok = 0
        for cat in MATH_CATS:
            try:
                load_dataset('EleutherAI/hendrycks_math', cat, split=split)
                n_ok += 1
            except Exception as e:
                print(f"  SKIP EleutherAI/hendrycks_math {cat} {split}: {e}")
        if n_ok == len(MATH_CATS):
            print(f"  OK  EleutherAI/hendrycks_math all_cats     {split} ({n_ok}/{len(MATH_CATS)} cats)")

    # GPQA: gated dataset — requires HF token
    hf_token = None
    token_path = os.path.expanduser('~/.cache/huggingface/token')
    if os.path.exists(token_path):
        hf_token = open(token_path).read().strip()
    for subset in ['gpqa_main', 'gpqa_extended']:
        try:
            ds = load_dataset('Idavidrein/gpqa', subset, split='train', token=hf_token)
            print(f"  OK  Idavidrein/gpqa {subset:<16} train ({len(ds)} rows)")
        except Exception as e:
            print(f"  SKIP Idavidrein/gpqa {subset}: {e}")
    print("Done. Re-run without --download_only to evaluate.")
    sys.exit(0)

# ── IDS model machinery (unchanged from original) ─────────────
_current_alpha = 1.0
_current_mask_ratio = 1.0

def set_alpha(a: float):
    global _current_alpha
    _current_alpha = float(a)

def set_mask_ratio(r: float):
    global _current_mask_ratio
    _current_mask_ratio = float(r)

SOFT_TEMP = 0.05

def get_soft_weights(mask_ratio):
    ANCHORS = [(1.00, 1), (0.67, 2), (0.33, 3), (0.00, 4)]
    dists = torch.tensor([(mask_ratio - r)**2 for r, _ in ANCHORS], dtype=torch.float32)
    return torch.softmax(-dists / SOFT_TEMP, dim=0)

class IDSLinear(nn.Module):
    def __init__(self, Am, Bm, At, Bt, bias=None):
        super().__init__()
        self.Am = Am.to(torch.bfloat16)
        self.Bm = Bm.to(torch.bfloat16)
        self.At = At.to(torch.bfloat16)
        self.Bt = Bt.to(torch.bfloat16)
        self.bias = bias.to(torch.bfloat16) if bias is not None else None

    def forward(self, x):
        a = _current_alpha
        out = a * F.linear(F.linear(x, self.Bm), self.Am) + \
              (1 - a) * F.linear(F.linear(x, self.Bt), self.At)
        if self.bias is not None:
            out = out + self.bias
        return out

class IDSLinearCPU(nn.Module):
    def __init__(self, Am, Bm, At, Bt, bias=None):
        super().__init__()
        self.Am_cpu = Am.to(torch.bfloat16).cpu()
        self.Bm_cpu = Bm.to(torch.bfloat16).cpu()
        self.At_cpu = At.to(torch.bfloat16).cpu()
        self.Bt_cpu = Bt.to(torch.bfloat16).cpu()
        self.has_bias = bias is not None
        if self.has_bias:
            self.bias_cpu = bias.to(torch.bfloat16).cpu()

    def forward(self, x):
        dev = x.device
        a = _current_alpha
        out = a * F.linear(F.linear(x, self.Bm_cpu.to(dev)), self.Am_cpu.to(dev)) + \
              (1 - a) * F.linear(F.linear(x, self.Bt_cpu.to(dev)), self.At_cpu.to(dev))
        if self.has_bias:
            out = out + self.bias_cpu.to(dev)
        return out

class MultiAnchorLinear(nn.Module):
    ANCHOR_IDS = [1, 2, 3, 4]

    def __init__(self, anchor_matrices, bias=None):
        super().__init__()
        self.cpu_weights = {}
        for aid, (A, B) in anchor_matrices.items():
            self.cpu_weights[aid] = (A.to(torch.bfloat16).cpu(), B.to(torch.bfloat16).cpu())
        self.has_bias = bias is not None
        if self.has_bias:
            self.bias_val = bias.to(torch.bfloat16).cpu()

    def forward(self, x):
        dev = x.device
        weights = get_soft_weights(_current_mask_ratio)
        out = None
        for i, aid in enumerate(self.ANCHOR_IDS):
            w = weights[i].item()
            if w < 1e-4:
                continue
            A, B = self.cpu_weights[aid]
            branch = F.linear(F.linear(x, B.to(dev)), A.to(dev))
            out = branch * w if out is None else out + branch * w
        if self.has_bias:
            out = out + self.bias_val.to(dev)
        return out

def get_parent_attr(model, name):
    parts = name.split('.')
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]

class StaticLowRank(nn.Module):
    def __init__(self, A, B, bias=None):
        super().__init__()
        self.A    = A.to(torch.bfloat16)
        self.B    = B.to(torch.bfloat16)
        self.bias = bias.to(torch.bfloat16) if bias is not None else None

    def forward(self, x):
        out = F.linear(F.linear(x, self.B), self.A)
        if self.bias is not None:
            out = out + self.bias
        return out

def _linear_names(model):
    return [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]

def replace_with_baseline_a(model, weights_path):
    n_replaced = n_skipped = 0
    for name in _linear_names(model):
        parent, attr = get_parent_attr(model, name)
        mod = getattr(parent, attr)
        if not isinstance(mod, nn.Linear):
            continue
        prefix = name.replace('.', '_')
        p_A = os.path.join(weights_path, f"{prefix}_At.pt")
        p_B = os.path.join(weights_path, f"{prefix}_Bt.pt")
        if not (os.path.exists(p_A) and os.path.exists(p_B)):
            n_skipped += 1
            continue
        bias = mod.bias.data.cpu() if mod.bias is not None else None
        del mod
        setattr(parent, attr, nn.Identity())
        torch.cuda.empty_cache()
        A = torch.load(p_A, map_location=device).to(torch.bfloat16)
        B = torch.load(p_B, map_location=device).to(torch.bfloat16)
        setattr(parent, attr, StaticLowRank(A, B, bias.to(device) if bias is not None else None))
        n_replaced += 1
        del A, B
        torch.cuda.empty_cache()
    print(f"  Whitening (SVD-LLM, whitening-only, no closed-form update) layers replaced: {n_replaced}  skipped: {n_skipped}")
    return model

def replace_with_static_ab(model, weights_path, label):
    n_replaced = n_skipped = 0
    for name in _linear_names(model):
        parent, attr = get_parent_attr(model, name)
        mod = getattr(parent, attr)
        if not isinstance(mod, nn.Linear):
            continue
        # POLICY: only the 7 linears inside the 32 transformer blocks are ever compressed
        # (224 layers). Embedding and lm_head are NEVER compressed. The top-level lm_head is
        # `model.transformer.ff_out` and shares the `ff_out` suffix with each block's MLP
        # down-projection, so a bare `ff_out$` regex caught it; a stale A/B then got loaded
        # here and the evaluated model silently diverged from its metadata. Guard against it.
        if '.blocks.' not in name:
            continue
        prefix = name.replace('.', '_')
        p_A = os.path.join(weights_path, f"{prefix}_A.pt")
        p_B = os.path.join(weights_path, f"{prefix}_B.pt")
        if not (os.path.exists(p_A) and os.path.exists(p_B)):
            n_skipped += 1
            continue
        bias = mod.bias.data.cpu() if mod.bias is not None else None
        del mod
        setattr(parent, attr, nn.Identity())
        torch.cuda.empty_cache()
        A = torch.load(p_A, map_location=device).to(torch.bfloat16)
        B = torch.load(p_B, map_location=device).to(torch.bfloat16)
        setattr(parent, attr, StaticLowRank(A, B, bias.to(device) if bias is not None else None))
        n_replaced += 1
        del A, B
        torch.cuda.empty_cache()
    print(f"  {label} layers replaced: {n_replaced}  skipped: {n_skipped}")
    return model

def replace_with_tsvd(model, weights_path):
    return replace_with_static_ab(model, weights_path, "TSVD")

def replace_with_obs(model, weights_path):
    weights_file = os.path.join(weights_path, 'weights.pt')
    sparse_weights = torch.load(weights_file, map_location=device)
    n_loaded = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name in sparse_weights:
            mod.weight.data = sparse_weights[name].to(torch.bfloat16).to(device)
            n_loaded += 1
    print(f"  OBS sparse weights loaded: {n_loaded} layers")
    return model

def replace_with_streaming_multistep(model, weights_path):
    return replace_with_static_ab(model, weights_path, "Streaming multi-step whitening")

def replace_with_baseline_b(model, weights_path):
    n_replaced = n_skipped = 0
    for name in _linear_names(model):
        parent, attr = get_parent_attr(model, name)
        mod = getattr(parent, attr)
        if not isinstance(mod, nn.Linear):
            continue
        prefix = name.replace('.', '_')
        p_A = os.path.join(weights_path, f"{prefix}_Am.pt")
        p_B = os.path.join(weights_path, f"{prefix}_Bm.pt")
        if not (os.path.exists(p_A) and os.path.exists(p_B)):
            n_skipped += 1
            continue
        bias = mod.bias.data.cpu() if mod.bias is not None else None
        del mod
        setattr(parent, attr, nn.Identity())
        torch.cuda.empty_cache()
        A = torch.load(p_A, map_location=device).to(torch.bfloat16)
        B = torch.load(p_B, map_location=device).to(torch.bfloat16)
        setattr(parent, attr, StaticLowRank(A, B, bias.to(device) if bias is not None else None))
        n_replaced += 1
        del A, B
        torch.cuda.empty_cache()
    print(f"  Baseline B layers replaced: {n_replaced}  skipped: {n_skipped}")
    return model

def replace_with_ids_multi(model, weights_path):
    n_replaced = n_skipped = 0
    for name in _linear_names(model):
        parent, attr = get_parent_attr(model, name)
        mod = getattr(parent, attr)
        if not isinstance(mod, nn.Linear):
            continue
        prefix = name.replace('.', '_')
        paths = {aid: (os.path.join(weights_path, f"{prefix}_A{aid}.pt"),
                       os.path.join(weights_path, f"{prefix}_B{aid}.pt"))
                 for aid in [1, 2, 3, 4]}
        if not all(os.path.exists(p) for pa, pb in paths.values() for p in (pa, pb)):
            n_skipped += 1
            continue
        bias = mod.bias.data.cpu() if mod.bias is not None else None
        del mod
        setattr(parent, attr, nn.Identity())
        torch.cuda.empty_cache()
        anchor_mats = {}
        for aid, (pa, pb) in paths.items():
            A = torch.load(pa, map_location='cpu').to(torch.bfloat16)
            B = torch.load(pb, map_location='cpu').to(torch.bfloat16)
            anchor_mats[aid] = (A, B)
        setattr(parent, attr, MultiAnchorLinear(anchor_mats, bias))
        n_replaced += 1
        del anchor_mats
        torch.cuda.empty_cache()
    print(f"  IDS 4-anchor layers replaced: {n_replaced}  skipped: {n_skipped}")
    return model

def replace_with_ids(model, weights_path):
    n_replaced = n_skipped = 0
    for name in _linear_names(model):
        parent, attr = get_parent_attr(model, name)
        mod = getattr(parent, attr)
        if not isinstance(mod, nn.Linear):
            continue
        prefix = name.replace('.', '_')
        paths = {
            'Am': os.path.join(weights_path, f"{prefix}_A1.pt"),
            'Bm': os.path.join(weights_path, f"{prefix}_B1.pt"),
            'At': os.path.join(weights_path, f"{prefix}_A4.pt"),
            'Bt': os.path.join(weights_path, f"{prefix}_B4.pt"),
        }
        if not all(os.path.exists(p) for p in paths.values()):
            n_skipped += 1
            continue
        bias = mod.bias.data.cpu() if mod.bias is not None else None
        del mod
        setattr(parent, attr, nn.Identity())
        torch.cuda.empty_cache()
        Am = torch.load(paths['Am'], map_location='cpu').to(torch.bfloat16)
        Bm = torch.load(paths['Bm'], map_location='cpu').to(torch.bfloat16)
        At = torch.load(paths['At'], map_location='cpu').to(torch.bfloat16)
        Bt = torch.load(paths['Bt'], map_location='cpu').to(torch.bfloat16)
        setattr(parent, attr, IDSLinearCPU(Am, Bm, At, Bt, bias))
        n_replaced += 1
        del Am, Bm, At, Bt
        torch.cuda.empty_cache()
    print(f"  IDS 2-anchor layers replaced: {n_replaced}  skipped: {n_skipped}")
    return model

class IDSJointLinear(nn.Module):
    def __init__(self, Am, Bm_ref, At, Bt_ref, bias=None):
        super().__init__()
        self.Am  = Am.to(torch.bfloat16)
        self.At  = At.to(torch.bfloat16)
        self._Bm = Bm_ref
        self._Bt = Bt_ref
        self.bias = bias.to(torch.bfloat16) if bias is not None else None

    def forward(self, x):
        a = _current_alpha
        out = a * F.linear(F.linear(x, self._Bm), self.Am) + \
              (1 - a) * F.linear(F.linear(x, self._Bt), self.At)
        if self.bias is not None:
            out = out + self.bias
        return out

def replace_with_joint_qkv(model, weights_path):
    block_prefixes = set()
    for name, mod in model.named_modules():
        for sfx in ['.q_proj', '.k_proj', '.v_proj']:
            if name.endswith(sfx) and isinstance(mod, nn.Linear):
                block_prefixes.add(name[:-len(sfx)])

    qkv_names = set()
    n_joint = n_ind = n_skip = 0

    for pfx in sorted(block_prefixes):
        safe = pfx.replace('.', '_')
        fp = {k: os.path.join(weights_path, f"{safe}_qkv_{k}.pt")
              for k in ['Aq1','Ak1','Av1','Aq4','Ak4','Av4','B1','B4']}
        if not all(os.path.exists(v) for v in fp.values()):
            n_skip += 3; continue

        biases = {}
        for role in ['q', 'k', 'v']:
            full = f"{pfx}.{role}_proj"
            qkv_names.add(full)
            par, a = get_parent_attr(model, full)
            orig = getattr(par, a)
            biases[role] = orig.bias.data.cpu() if (isinstance(orig, nn.Linear) and orig.bias is not None) else None
            del orig
            setattr(par, a, nn.Identity())
        torch.cuda.empty_cache()

        Bm_cpu = torch.load(fp['B1'], map_location='cpu').to(torch.bfloat16)
        Bt_cpu = torch.load(fp['B4'], map_location='cpu').to(torch.bfloat16)

        for role, amk, atk in [('q','Aq1','Aq4'), ('k','Ak1','Ak4'), ('v','Av1','Av4')]:
            full = f"{pfx}.{role}_proj"
            Am = torch.load(fp[amk], map_location='cpu').to(torch.bfloat16)
            At = torch.load(fp[atk], map_location='cpu').to(torch.bfloat16)
            par, a = get_parent_attr(model, full)
            setattr(par, a, IDSLinearCPU(Am, Bm_cpu, At, Bt_cpu, biases[role]))
            n_joint += 1
            del Am, At

        del Bm_cpu, Bt_cpu
        torch.cuda.empty_cache()

    for name in _linear_names(model):
        if name in qkv_names:
            continue
        par, attr = get_parent_attr(model, name)
        mod = getattr(par, attr)
        if not isinstance(mod, nn.Linear):
            continue
        prefix = name.replace('.', '_')
        paths = {k: os.path.join(weights_path, f"{prefix}_{k}.pt")
                 for k in ['A1','B1','A4','B4']}
        if not all(os.path.exists(v) for v in paths.values()):
            n_skip += 1; continue

        bias = mod.bias.data.cpu() if mod.bias is not None else None
        del mod
        setattr(par, attr, nn.Identity())
        torch.cuda.empty_cache()

        Am = torch.load(paths['A1'], map_location='cpu').to(torch.bfloat16)
        Bm = torch.load(paths['B1'], map_location='cpu').to(torch.bfloat16)
        At = torch.load(paths['A4'], map_location='cpu').to(torch.bfloat16)
        Bt = torch.load(paths['B4'], map_location='cpu').to(torch.bfloat16)
        setattr(par, attr, IDSLinearCPU(Am, Bm, At, Bt, bias))
        n_ind += 1
        del Am, Bm, At, Bt
        torch.cuda.empty_cache()

    print(f"  joint QKV replaced: {n_joint}  independent: {n_ind}  skipped: {n_skip}")
    return model

# ── 1. load model ─────────────────────────────────────────────
t0 = time.time()
print(f"\n[1/3] Loading model (mode={args.mode})...")
model = AutoModel.from_pretrained(
    args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
if tokenizer.padding_side != 'left':
    tokenizer.padding_side = 'left'
print(f"  loaded  ({time.time()-t0:.0f}s)")

# ── 2. optionally replace layers ──────────────────────────────
_mode = args.mode
if _mode in ('whitening', 'whitening_mixed', 'baseline_a'):
    print(f"\n[2/3] Whitening / SVD-LLM (whitening-only, no closed-form update; static text SVD, t=0)...")
    model = replace_with_baseline_a(model, args.weights_path)
elif _mode == 'baseline_b':
    print(f"\n[2/3] Baseline B (static mixed-mask SVD)...")
    model = replace_with_baseline_b(model, args.weights_path)
elif _mode == 'ids':
    print(f"\n[2/3] IDS 2-anchor modules...")
    model = replace_with_ids(model, args.weights_path)
    set_alpha(1.0)  # use masked-regime weights for generation
elif _mode == 'ids_multi':
    print(f"\n[2/3] IDS 4-anchor modules...")
    model = replace_with_ids_multi(model, args.weights_path)
    set_mask_ratio(1.0)
elif _mode in ('tsvd', 'traject_svd'):
    print(f"\n[2/3] TSVD (TrajectSVD) static low-rank...")
    model = replace_with_tsvd(model, args.weights_path)
elif _mode == 'traject_svd_zh':
    print(f"\n[2/3] TrajectSVD-zh static low-rank...")
    model = replace_with_tsvd(model, args.weights_path)
elif _mode == 'traject_svd_mixed':
    print(f"\n[2/3] TrajectSVD-mixed static low-rank...")
    model = replace_with_tsvd(model, args.weights_path)
elif _mode in ('obs', 'obs_diff'):
    print(f"\n[2/3] OBS-Diff sparse weights...")
    model = replace_with_obs(model, args.weights_path)
elif _mode == 'obs_whitening':
    print(f"\n[2/3] OBS-Diff style whitening (log-decay weighted SVD)...")
    model = replace_with_static_ab(model, args.weights_path, "OBS-Whitening")
elif _mode == 'streaming_multistep':
    print(f"\n[2/3] Streaming multi-step whitening...")
    model = replace_with_streaming_multistep(model, args.weights_path)
elif _mode == 'joint_qkv':
    print(f"\n[2/3] Joint-QKV IDS modules...")
    model = replace_with_joint_qkv(model, args.weights_path)
else:
    print("\n[2/3] Uncompressed model — no layer replacement.")

torch.cuda.empty_cache()
if torch.cuda.is_available():
    used  = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU: {used:.1f}/{total:.1f} GB")

# ─────────────────────────────────────────────────────────────
# Generation helpers
# ─────────────────────────────────────────────────────────────

MAX_PROMPT_TOKENS = 1900  # keep under 2048 - gen_length headroom

def _to_chat_text(prompt_text):
    messages = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False)

@torch.no_grad()
def gen_responses(prompt_texts, gen_length, block_length, steps,
                  logits_eos_inf, confidence_eos_eot_inf):
    """Batched generation: list of prompt strings → list of decoded responses."""
    chat_texts = [_to_chat_text(p) for p in prompt_texts]
    enc = tokenizer(chat_texts, return_tensors='pt', add_special_tokens=False, padding=True)
    input_ids  = enc['input_ids'].to(device)
    attn_mask  = enc['attention_mask'].to(device)

    if input_ids.shape[1] > MAX_PROMPT_TOKENS:
        input_ids = input_ids[:, -MAX_PROMPT_TOKENS:]
        attn_mask = attn_mask[:, -MAX_PROMPT_TOKENS:]

    out = llada_generate(
        model, input_ids, attn_mask,
        steps=steps, gen_length=gen_length, block_length=block_length,
        temperature=0., cfg_scale=0., remasking='low_confidence',
        logits_eos_inf=logits_eos_inf, confidence_eos_eot_inf=confidence_eos_eot_inf,
    )
    decoded = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
    return [d.strip() for d in decoded]

def gen_response(prompt_text, gen_length, block_length, steps,
                 logits_eos_inf, confidence_eos_eot_inf):
    return gen_responses([prompt_text], gen_length, block_length, steps,
                         logits_eos_inf, confidence_eos_eot_inf)[0]


# ─────────────────────────────────────────────────────────────
# Answer extractors
# ─────────────────────────────────────────────────────────────

def extract_choice(text):
    """Extract first A/B/C/D from generated text."""
    m = re.search(r'\b([A-D])\b', text)
    return m.group(1) if m else None

def extract_choice_abcdj(text):
    """Extract first A-J from generated text (for MMLU-pro with up to 10 options)."""
    # Look for "The answer is X" pattern first
    m = re.search(r'[Tt]he answer is[:\s]+([A-J])\b', text)
    if m:
        return m.group(1)
    m = re.search(r'[Aa]nswer[:\s]+([A-J])\b', text)
    if m:
        return m.group(1)
    # Last capitalized option letter in text
    matches = re.findall(r'\b([A-J])\b', text)
    return matches[-1] if matches else None

def extract_number(text):
    """Extract final answer from GSM8K generated text.

    Priority:
    1. #### <number>  (GSM8K training format)
    2. The answer is <number>  (chain-of-thought format from our fewshot)
    3. Last number in text  (fallback)
    """
    m = re.search(r'####\s*(-?[\d,]+)', text)
    if m:
        return m.group(1).replace(',', '')
    m = re.search(r'[Tt]he answer is[^\d-]*(-?[\d,]+)', text)
    if m:
        return m.group(1).replace(',', '')
    nums = re.findall(r'-?[\d,]+\.?\d*', text)
    return nums[-1].replace(',', '') if nums else None

def extract_boxed(text):
    """Extract \\boxed{...} from generated text, fall back to last number."""
    m = re.search(r'\\boxed\{([^}]+)\}', text)
    if m:
        return m.group(1).strip()
    nums = re.findall(r'-?[\d,]+\.?\d*', text)
    return nums[-1].replace(',', '') if nums else None

def numbers_equal(a, b):
    """Compare two numeric strings for equality."""
    try:
        return abs(float(a.replace(',', '')) - float(b.replace(',', ''))) < 1e-6
    except Exception:
        return a.strip() == b.strip()

def math_gold(solution):
    """Extract gold answer from MATH solution (\\boxed{})."""
    m = re.search(r'\\boxed\{([^}]+)\}', solution)
    return m.group(1).strip() if m else solution.strip()

def math_match(pred, gold):
    if pred is None or gold is None:
        return False
    pred = pred.strip().replace(' ', '')
    gold = gold.strip().replace(' ', '')
    if pred == gold:
        return True
    return numbers_equal(pred, gold)


# ─────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────

LETTERS = 'ABCDEFGHIJ'

def _mmlu_example(row, include_answer=True):
    choices = row['choices']
    ans_idx = int(row['answer'])  # 0-3
    s = f"Question: {row['question'].strip()}\n"
    for i, c in enumerate(choices):
        s += f"{LETTERS[i]}. {c}\n"
    if include_answer:
        s += f"Answer: {LETTERS[ans_idx]}\n"
    else:
        s += "Answer:"
    return s

def build_mmlu_prompt(row, fewshot_rows):
    subject = row['subject'].replace('_', ' ')
    parts = [f"The following are multiple choice questions (with answers) about {subject}.\n"]
    for ex in fewshot_rows:
        parts.append(_mmlu_example(ex, include_answer=True))
    parts.append(_mmlu_example(row, include_answer=False))
    return '\n'.join(parts)

def _mmlu_pro_example(row, include_answer=True):
    opts = row['options']
    s = f"Question: {row['question'].strip()}\n"
    for i, o in enumerate(opts):
        s += f"{LETTERS[i]}. {o}\n"
    if include_answer:
        cot = row.get('cot_content', '').strip()
        s += f"Answer: Let's think step by step. {cot}\n"
    else:
        s += "Answer: Let's think step by step."
    return s

def build_mmlu_pro_prompt(row, fewshot_rows):
    parts = []
    for ex in fewshot_rows:
        parts.append(_mmlu_pro_example(ex, include_answer=True))
    parts.append(_mmlu_pro_example(row, include_answer=False))
    return '\n'.join(parts)

def build_hellaswag_prompt(row):
    ctx = row['ctx'].strip()
    endings = row['endings']
    s = f"Context: {ctx}\nWhat happens next?\n"
    for i, e in enumerate(endings):
        s += f"{LETTERS[i]}. {e.strip()}\n"
    s += "Answer:"
    return s

def _arc_choices(row):
    """Return list of (letter, text) and the gold letter."""
    labels = row['choices']['label']
    texts  = row['choices']['text']
    letter_map = {}
    for i, lbl in enumerate(labels):
        if lbl in 'ABCDE':
            letter_map[lbl] = texts[i]
        else:
            letter_map[LETTERS[i]] = texts[i]

    ans_key = row['answerKey']
    if ans_key not in LETTERS:
        # numeric key like "1","2","3","4"
        idx = int(ans_key) - 1
        gold = LETTERS[idx]
        letter_map = {LETTERS[i]: texts[i] for i in range(len(texts))}
    else:
        gold = ans_key
        # remap if labels were numeric
        if ans_key not in letter_map:
            letter_map = {LETTERS[i]: texts[i] for i in range(len(texts))}
            gold = LETTERS[labels.index(ans_key)] if ans_key in labels else LETTERS[int(ans_key)-1]

    return letter_map, gold

def build_arc_c_prompt(row):
    letter_map, gold = _arc_choices(row)
    s = f"Question: {row['question'].strip()}\n"
    for ltr in sorted(letter_map):
        s += f"{ltr}. {letter_map[ltr]}\n"
    s += "Answer:"
    return s, gold

# Official 4-shot examples from OpenCompass (gsm8k_gen_1d7fe4.py), fixed across all runs
_GSM8K_FEWSHOT = [
    ("Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?",
     "Angelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.\nFor the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.\nAngelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.\nHowever, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.\nThey also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.\nAnd they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.\nSo Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.\nThey want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75\nThey will need to plan to study 4 days to allow for all the time they need.\nThe answer is 4"),
    ("Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws.  Their opponents score double the 2 pointers but half the 3 pointers and free throws.  What's the total number of points scored by both teams added together?",
     "Mark's team scores 25 2 pointers, meaning they scored 25*2= 50 points in 2 pointers.\nHis team also scores 6 3 pointers, meaning they scored 8*3= 24 points in 3 pointers\nThey scored 10 free throws, and free throws count as one point so they scored 10*1=10 points in free throws.\nAll together his team scored 50+24+10= 84 points\nMark's opponents scored double his team's number of 2 pointers, meaning they scored 50*2=100 points in 2 pointers.\nHis opponents scored half his team's number of 3 pointers, meaning they scored 24/2= 12 points in 3 pointers.\nThey also scored half Mark's team's points in free throws, meaning they scored 10/2=5 points in free throws.\nAll together Mark's opponents scored 100+12+5=117 points\nThe total score for the game is both team's scores added together, so it is 84+117=201 points\nThe answer is 201"),
    ("Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles?",
     "When Bella buys 2/5 times more marbles, she'll have increased the number of marbles by 2/5*60 = 24\nThe total number of marbles she'll have is 60+24 = 84\nIf Bella currently has 60 marbles, and she has two times as many marbles as frisbees, she has 60/2 = 30 frisbees.\nIf Bella buys 2/5 times more frisbees, she'll have 2/5*30 = 12 more frisbees.\nThe total number of frisbees she'll have will increase to 30+12 = 42\nBella also has 20 more frisbees than deck cards, meaning she has 30-20 = 10 deck cards\nIf she buys 2/5 times more deck cards, she'll have 2/5*10 = 4 more deck cards.\nThe total number of deck cards she'll have is 10+4 = 14\nTogether, Bella will have a total of 14+42+84 = 140 items\nThe answer is 140"),
    ("A group of 4 fruit baskets contains 9 apples, 15 oranges, and 14 bananas in the first three baskets and 2 less of each fruit in the fourth basket. How many fruits are there?",
     "For the first three baskets, the number of apples and oranges in one basket is 9+15=24\nIn total, together with bananas, the number of fruits in one basket is 24+14=38 for the first three baskets.\nSince there are three baskets each having 38 fruits, there are 3*38=114 fruits in the first three baskets.\nThe number of apples in the fourth basket is 9-2=7\nThere are also 15-2=13 oranges in the fourth basket\nThe combined number of oranges and apples in the fourth basket is 13+7=20\nThe fourth basket also contains 14-2=12 bananas.\nIn total, the fourth basket has 20+12=32 fruits.\nThe four baskets together have 32+114=146 fruits.\nThe answer is 146"),
]

def build_gsm8k_prompt(row, fewshot_rows=None):
    parts = []
    for q, a in _GSM8K_FEWSHOT:
        parts.append(f"Question: {q}\nLet's think step by step\nAnswer: {a}")
    parts.append(f"Question: {row['question'].strip()}\nLet's think step by step\nAnswer:")
    return '\n\n'.join(parts)

def build_math_prompt(row, fewshot_rows):
    parts = []
    for ex in fewshot_rows:
        parts.append(f"Problem: {ex['problem'].strip()}\nSolution: {ex['solution'].strip()}")
    parts.append(f"Problem: {row['problem'].strip()}\nSolution:")
    return '\n\n'.join(parts)

def _gpqa_shuffle(row, idx):
    """Return shuffled list of options and the gold letter."""
    options = [
        row['Correct Answer'],
        row['Incorrect Answer 1'],
        row['Incorrect Answer 2'],
        row['Incorrect Answer 3'],
    ]
    rng = random.Random(idx)
    rng.shuffle(options)
    gold_letter = LETTERS[options.index(row['Correct Answer'])]
    return options, gold_letter

def build_gpqa_prompt(row, row_idx, fewshot_rows, fewshot_indices):
    parts = []
    for ex_row, ex_idx in zip(fewshot_rows, fewshot_indices):
        opts, gold_ltr = _gpqa_shuffle(ex_row, ex_idx)
        s = f"Question: {ex_row['Question'].strip()}\n"
        for i, o in enumerate(opts):
            s += f"{LETTERS[i]}. {o}\n"
        s += f"Answer: {gold_ltr}\n"
        parts.append(s)
    opts, gold_ltr = _gpqa_shuffle(row, row_idx)
    s = f"Question: {row['Question'].strip()}\n"
    for i, o in enumerate(opts):
        s += f"{LETTERS[i]}. {o}\n"
    s += "Answer:"
    parts.append(s)
    return '\n'.join(parts), gold_ltr


# ─────────────────────────────────────────────────────────────
# Per-benchmark evaluators
# ─────────────────────────────────────────────────────────────

# Per-item correctness records, keyed by benchmark: [{idx, gold, pred, correct}, ...].
# REQUIRED for paired significance testing (McNemar) between two arms: an aggregate acc alone
# cannot distinguish a real paired gap from noise. `idx` is the position in the (deterministically
# ordered, unshuffled) limited test set, so it aligns across runs; `gold` is stored so alignment
# can be verified rather than assumed.
PER_ITEM = {}


def _run_batched(items, build_fn, extract_fn, cfg, desc, gold_fn):
    """Generic batched evaluation loop. Also records per-item correctness into PER_ITEM under the
    benchmark key ('MMLU'->'mmlu', 'MMLU-Pro'->'mmlu_pro', 'ARC-C'->'arc_c', ...), so every
    benchmark routed through here supports paired significance testing between two arms."""
    bs = args.batch_size
    n_correct = n_total = 0
    recs = []
    bench_key = desc.lower().replace('-', '_')
    for start in tqdm(range(0, len(items), bs), desc=desc, leave=False):
        batch = items[start:start + bs]
        prompts, golds = [], []
        for item in batch:
            result = build_fn(item)
            if isinstance(result, tuple):
                prompt, gold = result
            else:
                prompt, gold = result, gold_fn(item)
            prompts.append(prompt)
            golds.append(gold)
        try:
            gens = gen_responses(prompts, cfg['gen_length'], cfg['block_length'], cfg['steps'],
                                 cfg['logits_eos_inf'], cfg['confidence_eos_eot_inf'])
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() and bs > 1:
                # OOM: fall back to single-sample for this batch
                torch.cuda.empty_cache()
                gens = [gen_response(p, cfg['gen_length'], cfg['block_length'], cfg['steps'],
                                     cfg['logits_eos_inf'], cfg['confidence_eos_eot_inf'])
                        for p in prompts]
            else:
                raise
        for j, (gen, gold) in enumerate(zip(gens, golds)):
            pred = extract_fn(gen)
            ok = bool(pred == gold)
            if ok:
                n_correct += 1
            n_total += 1
            recs.append({'idx': start + j, 'gold': None if gold is None else str(gold),
                         'pred': None if pred is None else str(pred), 'correct': ok})
    PER_ITEM[bench_key] = recs
    return n_correct / n_total if n_total > 0 else 0.0, n_total


def eval_mmlu(limit=None):
    from datasets import load_dataset
    cfg = BENCH_CONFIG['mmlu']
    print("\n  Loading MMLU dev split for few-shot...")
    dev_ds = load_dataset('cais/mmlu', 'all', split='dev')
    fewshot_by_subject = defaultdict(list)
    for row in dev_ds:
        fewshot_by_subject[row['subject']].append(row)

    test_ds = load_dataset('cais/mmlu', 'all', split='test')
    items = list(test_ds)
    if args.shuffle_eval:
        rng_s = random.Random(args.sample_seed)
        rng_s.shuffle(items)
    if limit:
        items = items[:min(limit, len(items))]

    def build(row):
        fewshot = fewshot_by_subject.get(row['subject'], [])[:cfg['num_fewshot']]
        return build_mmlu_prompt(row, fewshot), LETTERS[int(row['answer'])]

    return _run_batched(items, build, extract_choice, cfg, 'MMLU', None)


def eval_mmlu_pro(limit=None):
    from datasets import load_dataset
    cfg = BENCH_CONFIG['mmlu_pro']
    print("\n  Loading MMLU-Pro validation for few-shot...")
    val_ds = load_dataset('TIGER-Lab/MMLU-Pro', split='validation')
    fewshot_by_cat = defaultdict(list)
    for row in val_ds:
        fewshot_by_cat[row['category']].append(row)

    test_ds = load_dataset('TIGER-Lab/MMLU-Pro', split='test')
    if limit:
        test_ds = test_ds.select(range(min(limit, len(test_ds))))

    items = list(test_ds)

    def build(row):
        fewshot = fewshot_by_cat.get(row['category'], [])[:cfg['num_fewshot']]
        return build_mmlu_pro_prompt(row, fewshot), row['answer']

    return _run_batched(items, build, extract_choice_abcdj, cfg, 'MMLU-Pro', None)


def eval_hellaswag(limit=None):
    from datasets import load_dataset
    cfg = BENCH_CONFIG['hellaswag']
    ds = load_dataset('Rowan/hellaswag', split='validation')
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    items = list(ds)

    def build(row):
        return build_hellaswag_prompt(row), LETTERS[int(row['label'])]

    return _run_batched(items, build, extract_choice, cfg, 'Hellaswag', None)


def eval_arc_c(limit=None):
    from datasets import load_dataset
    cfg = BENCH_CONFIG['arc_c']
    ds = load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='test')
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    items = list(ds)

    def build(row):
        return build_arc_c_prompt(row)  # already returns (prompt, gold)

    return _run_batched(items, build, extract_choice, cfg, 'ARC-C', None)


def eval_gsm8k(limit=None):
    from datasets import load_dataset
    cfg = BENCH_CONFIG['gsm8k']
    # fewshot is now fixed via _GSM8K_FEWSHOT, no training set needed

    test_ds = load_dataset('gsm8k', 'main', split='test')
    if limit:
        test_ds = test_ds.select(range(min(limit, len(test_ds))))

    items = list(test_ds)

    def build(row):
        return build_gsm8k_prompt(row), row['answer'].split('####')[-1].strip().replace(',', '')

    def extract(gen):
        num = extract_number(gen)
        return num  # compared via numbers_equal, but _run_batched uses ==

    # GSM8K uses numerical comparison, can't use generic _run_batched for scoring
    bs = args.batch_size
    n_correct = n_total = 0
    recs = []
    for start in tqdm(range(0, len(items), bs), desc='GSM8K', leave=False):
        batch = items[start:start + bs]
        prompts = [build_gsm8k_prompt(row) for row in batch]
        golds   = [row['answer'].split('####')[-1].strip().replace(',', '') for row in batch]
        try:
            gens = gen_responses(prompts, cfg['gen_length'], cfg['block_length'], cfg['steps'],
                                 cfg['logits_eos_inf'], cfg['confidence_eos_eot_inf'])
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() and bs > 1:
                torch.cuda.empty_cache()
                gens = [gen_response(p, cfg['gen_length'], cfg['block_length'], cfg['steps'],
                                     cfg['logits_eos_inf'], cfg['confidence_eos_eot_inf'])
                        for p in prompts]
            else:
                raise
        for j, (gen, gold) in enumerate(zip(gens, golds)):
            pred = extract_number(gen)
            ok = bool(pred is not None and numbers_equal(pred, gold))
            if ok:
                n_correct += 1
            n_total += 1
            recs.append({'idx': start + j, 'gold': gold,
                         'pred': None if pred is None else str(pred), 'correct': ok})
    PER_ITEM['gsm8k'] = recs
    return n_correct / n_total if n_total > 0 else 0.0, n_total


def eval_math(limit=None):
    from datasets import load_dataset, concatenate_datasets
    cfg = BENCH_CONFIG['math']
    MATH_CATS = ['algebra', 'counting_and_probability', 'geometry',
                 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
    train_ds = concatenate_datasets(
        [load_dataset('EleutherAI/hendrycks_math', c, split='train') for c in MATH_CATS])
    fewshot = list(train_ds.select(range(cfg['num_fewshot'])))

    test_ds = concatenate_datasets(
        [load_dataset('EleutherAI/hendrycks_math', c, split='test') for c in MATH_CATS])
    if limit:
        test_ds = test_ds.select(range(min(limit, len(test_ds))))

    items = list(test_ds)
    bs = args.batch_size
    n_correct = n_total = 0
    recs = []
    for start in tqdm(range(0, len(items), bs), desc='MATH', leave=False):
        batch = items[start:start + bs]
        prompts = [build_math_prompt(row, fewshot) for row in batch]
        golds   = [math_gold(row['solution']) for row in batch]
        try:
            gens = gen_responses(prompts, cfg['gen_length'], cfg['block_length'], cfg['steps'],
                                 cfg['logits_eos_inf'], cfg['confidence_eos_eot_inf'])
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() and bs > 1:
                torch.cuda.empty_cache()
                gens = [gen_response(p, cfg['gen_length'], cfg['block_length'], cfg['steps'],
                                     cfg['logits_eos_inf'], cfg['confidence_eos_eot_inf'])
                        for p in prompts]
            else:
                raise
        for j, (gen, gold) in enumerate(zip(gens, golds)):
            pred = extract_boxed(gen)
            ok = bool(math_match(pred, gold))
            if ok:
                n_correct += 1
            n_total += 1
            recs.append({'idx': start + j, 'gold': None if gold is None else str(gold),
                         'pred': None if pred is None else str(pred), 'correct': ok})
    PER_ITEM['math'] = recs
    return n_correct / n_total if n_total > 0 else 0.0, n_total


def eval_gpqa(limit=None):
    from datasets import load_dataset
    import os as _os
    cfg = BENCH_CONFIG['gpqa']
    hf_token = None
    token_path = _os.path.expanduser('~/.cache/huggingface/token')
    if _os.path.exists(token_path):
        hf_token = open(token_path).read().strip()
    # GPQA is a gated dataset. On compute nodes OFFLINE mode blocks downloads.
    # Fix: on login node, run `python eval_benchmarks.py --download_only`
    # after requesting access at https://huggingface.co/datasets/Idavidrein/gpqa
    _was_offline = _os.environ.pop('HF_DATASETS_OFFLINE', None)
    try:
        ds = load_dataset('Idavidrein/gpqa', 'gpqa_main', split='train', token=hf_token)
    except Exception as e:
        if _was_offline:
            _os.environ['HF_DATASETS_OFFLINE'] = _was_offline
        msg = str(e)
        if 'gated' in msg.lower() or 'not found' in msg.lower():
            raise RuntimeError(
                "GPQA requires HuggingFace access. "
                "Visit https://huggingface.co/datasets/Idavidrein/gpqa "
                "→ agree to terms, then run: python eval_benchmarks.py --download_only"
            ) from None
        raise
    if _was_offline:
        _os.environ['HF_DATASETS_OFFLINE'] = _was_offline
    rows = list(ds)
    N = len(rows)
    if limit:
        rows = rows[:min(limit, N)]

    bs = args.batch_size
    n_correct = n_total = 0
    recs = []
    for start in tqdm(range(0, len(rows), bs), desc='GPQA', leave=False):
        batch_rows = rows[start:start + bs]
        prompts, golds = [], []
        for i, row in enumerate(batch_rows):
            abs_i = start + i
            fs_indices = [(abs_i + 1 + k) % N for k in range(cfg['num_fewshot'])]
            fewshot_rows = [ds[j] for j in fs_indices]
            prompt, gold = build_gpqa_prompt(row, abs_i, fewshot_rows, fs_indices)
            prompts.append(prompt)
            golds.append(gold)
        try:
            gens = gen_responses(prompts, cfg['gen_length'], cfg['block_length'], cfg['steps'],
                                 cfg['logits_eos_inf'], cfg['confidence_eos_eot_inf'])
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() and bs > 1:
                torch.cuda.empty_cache()
                gens = [gen_response(p, cfg['gen_length'], cfg['block_length'], cfg['steps'],
                                     cfg['logits_eos_inf'], cfg['confidence_eos_eot_inf'])
                        for p in prompts]
            else:
                raise
        for j, (gen, gold) in enumerate(zip(gens, golds)):
            pred = extract_choice(gen)
            ok = bool(pred == gold)
            if ok:
                n_correct += 1
            n_total += 1
            recs.append({'idx': start + j, 'gold': None if gold is None else str(gold),
                         'pred': None if pred is None else str(pred), 'correct': ok})
    PER_ITEM['gpqa'] = recs
    return n_correct / n_total if n_total > 0 else 0.0, n_total


# ─────────────────────────────────────────────────────────────
# [3/3] Main evaluation loop
# ─────────────────────────────────────────────────────────────

EVALUATORS = {
    'mmlu':      eval_mmlu,
    'mmlu_pro':  eval_mmlu_pro,
    'hellaswag': eval_hellaswag,
    'arc_c':     eval_arc_c,
    'gsm8k':     eval_gsm8k,
    'math':      eval_math,
    'gpqa':      eval_gpqa,
}

print(f"\n[3/3] Running benchmarks: {BENCHMARKS}")
print(f"  mode={args.mode}  limit={args.limit}  eval_type=generation  resume={args.resume}")

# Load prior results from latest checkpoint if --resume
results = {}
if args.resume:
    import glob
    ckpt_glob = os.path.join(RESULTS_ROOT, 'legacy_eval',
                             f'bench_gen_{args.mode}_*_ckpt.json')
    # Filter to checkpoints whose mode field exactly matches args.mode,
    # to avoid traject_svd accidentally picking up traject_svd_zh/mixed files.
    prior_ckpts = []
    for _p in sorted(glob.glob(ckpt_glob)):
        try:
            _d = json.load(open(_p))
            if _d.get('mode') == args.mode:
                prior_ckpts.append(_p)
        except Exception:
            pass
    if prior_ckpts:
        with open(prior_ckpts[-1]) as _f:
            _prior = json.load(_f)
        results = {k: v for k, v in _prior.get('results', {}).items()
                   if v.get('acc') is not None}
        print(f"  Resuming from {prior_ckpts[-1]}")
        print(f"  Already done: {sorted(results.keys())}")

t_start = time.time()

for bench in BENCHMARKS:
    if bench not in EVALUATORS:
        print(f"  WARNING: unknown benchmark '{bench}', skipping")
        continue
    if bench in results:
        print(f"\n  [{bench}] already done (resume), acc={results[bench]['acc']*100:.2f}%  skipping")
        continue
    t_bench = time.time()
    print(f"\n  [{bench}] starting...")
    try:
        acc, n = EVALUATORS[bench](limit=args.limit)
    except Exception as e:
        print(f"  [{bench}] FAILED: {e}")
        import traceback; traceback.print_exc()
        results[bench] = {'acc': None, 'n': 0, 'error': str(e)}
        continue

    target = TABLE2_TARGET.get(bench)
    delta  = f"  (target={target:.1f}%  Δ={acc*100 - target:+.1f}%)" if target else ""
    elapsed = (time.time() - t_bench) / 60
    print(f"  [{bench}] acc={acc*100:.2f}%  n={n}  time={elapsed:.1f}min{delta}")
    results[bench] = {'acc': round(acc, 6), 'n': n}

    # Checkpoint after each benchmark
    _ckpt = (args.output or os.path.join(
        os.path.join(RESULTS_ROOT, 'legacy_eval'),
        f"bench_gen_{args.mode}_{time.strftime('%Y%m%d_%H%M')}.json"
    )).replace('.json', '_ckpt.json')
    os.makedirs(os.path.dirname(_ckpt), exist_ok=True)
    with open(_ckpt, 'w') as f:
        json.dump({'mode': args.mode, 'eval_type': 'generation',
                   'results': results, 'per_item': PER_ITEM, 'note': 'checkpoint'}, f, indent=2)

total_min = (time.time() - t_start) / 60

# ── print summary ─────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  LLaDA-8B  mode={args.mode}  eval_type=generation  limit={args.limit}")
print(f"{'='*65}")
print(f"  {'benchmark':<12} {'acc':>8}  {'target':>8}  {'delta':>8}")
print(f"  {'-'*44}")
for bench in BENCHMARKS:
    if bench not in results or results[bench]['acc'] is None:
        print(f"  {bench:<12}  {'FAILED':>8}")
        continue
    acc = results[bench]['acc']
    n   = results[bench]['n']
    target = TABLE2_TARGET.get(bench, float('nan'))
    delta  = acc * 100 - target
    print(f"  {bench:<12}  {acc*100:7.2f}%  {target:7.1f}%  {delta:+7.2f}%  (n={n})")
print(f"{'='*65}")
print(f"  Total time: {total_min:.1f} min")

# ── save final JSON ───────────────────────────────────────────
if args.output is None:
    ts = time.strftime('%Y%m%d_%H%M')
    args.output = os.path.join(
        os.path.join(RESULTS_ROOT, 'legacy_eval'),
        f"bench_gen_{args.mode}_{ts}.json"
    )

os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, 'w') as f:
    json.dump({
        'mode':         args.mode,
        'eval_type':    'generation',
        'results':      results,
        'time_min':     round(total_min, 1),
        'limit':        args.limit,
        # provenance: which weights actually produced these numbers (no-cross-source guard)
        'weights_path': getattr(args, 'weights_path', None),
        'model_path':   getattr(args, 'model_path', None),
        # per-item correctness -> enables paired McNemar between arms
        'per_item':     PER_ITEM,
    }, f, indent=2)
print(f"  Results saved: {args.output}")
print("Done.")
