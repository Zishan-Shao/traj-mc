"""
lora_finetune.py -- LoRA fine-tuning on top of SVD-compressed LLaDA

Wraps each StaticLowRankCPU (compressed) layer with a LoRA adapter:
  y = A(B(x))  +  scale * lora_B(lora_A(x))

Only lora_A / lora_B are trained. Compressed A/B weights are frozen.
Training loss: masked diffusion (same as LLaDA pretraining), no teacher needed.

Usage:
  python lora_finetune.py --mode tsvd_zh --rank 16 --epochs 5
  python lora_finetune.py --mode tsvd_mixed --rank 16 --epochs 5
"""

import torch, argparse, os, sys, json, random, math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
from transformers import AutoTokenizer, AutoModel

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.path.join(REPO_ROOT, 'results')

parser = argparse.ArgumentParser()
parser.add_argument('--mode',         type=str,   default='tsvd_zh',
                    choices=['tsvd_zh', 'tsvd_mixed'])
parser.add_argument('--rank',         type=int,   default=16)
parser.add_argument('--lora_alpha',   type=float, default=16.0)
parser.add_argument('--model_path',   type=str,   default='GSAI-ML/LLaDA-8B-Instruct')
parser.add_argument('--llada_path',   type=str,   default=REPO_ROOT)
parser.add_argument('--weights_path', type=str,   default=None)
parser.add_argument('--data_path',    type=str,
                    default=os.path.join(REPO_ROOT, 'data', 'poem_data.json'))
parser.add_argument('--out_dir',      type=str,
                    default=os.path.join(RESULTS_ROOT, 'lora_weights'))
parser.add_argument('--epochs',       type=int,   default=5)
parser.add_argument('--lr',           type=float, default=5e-4)
parser.add_argument('--train_skip',   type=int,   default=100)
parser.add_argument('--gen_limit',    type=int,   default=50)
parser.add_argument('--steps',        type=int,   default=28)
parser.add_argument('--seed',         type=int,   default=42)
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
MASK_ID = 126336
sys.path.insert(0, args.llada_path)

WEIGHTS = {
    'tsvd_zh':    os.path.join(RESULTS_ROOT, 'weights', 'llada', 'tsvd_zh'),
    'tsvd_mixed': os.path.join(RESULTS_ROOT, 'weights', 'llada', 'tsvd_mixed'),
}
weights_path = args.weights_path or WEIGHTS[args.mode]
out_dir = os.path.join(args.out_dir, f"{args.mode}_r{args.rank}")
os.makedirs(out_dir, exist_ok=True)

print(f"Mode: {args.mode}  rank={args.rank}  alpha={args.lora_alpha}")
print(f"Weights: {weights_path}")
print(f"Out: {out_dir}")

# ── compressed model loader ───────────────────────────────────
class StaticLowRankCPU(nn.Module):
    def __init__(self, A, B, bias=None):
        super().__init__()
        self.A_cpu = A.to(torch.bfloat16).cpu()
        self.B_cpu = B.to(torch.bfloat16).cpu()
        self.has_bias = bias is not None
        if self.has_bias:
            self.bias_cpu = bias.to(torch.bfloat16).cpu()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        dev = x.device
        out = F.linear(F.linear(x, self.B_cpu.to(dev)), self.A_cpu.to(dev))
        if self.has_bias:
            out = out + self.bias_cpu.to(dev)
        return out

    @property
    def in_features(self):  return self.B_cpu.shape[1]
    @property
    def out_features(self): return self.A_cpu.shape[0]

def load_tsvd(model, weights_path):
    n_ok = n_skip = 0
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        prefix = name.replace('.', '_')
        pA = os.path.join(weights_path, f"{prefix}_A.pt")
        pB = os.path.join(weights_path, f"{prefix}_B.pt")
        if not (os.path.exists(pA) and os.path.exists(pB)):
            n_skip += 1
            continue
        parts = name.split('.')
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        bias = mod.bias.data.cpu() if mod.bias is not None else None
        setattr(parent, parts[-1], nn.Identity())
        torch.cuda.empty_cache()
        A = torch.load(pA, map_location='cpu')
        B = torch.load(pB, map_location='cpu')
        setattr(parent, parts[-1], StaticLowRankCPU(A, B, bias))
        n_ok += 1
        del A, B
        torch.cuda.empty_cache()
    print(f"  TSVD: replaced {n_ok}, skipped {n_skip}")
    return model

# ── LoRA wrapper ──────────────────────────────────────────────
class LoRAOverCompressed(nn.Module):
    """Adds a rank-r LoRA branch on top of a frozen StaticLowRankCPU layer."""
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        in_f, out_f = base.in_features, base.out_features
        self.lora_A = nn.Linear(in_f, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_f, bias=False)
        # init: lora_A random, lora_B zeros → ΔW = 0 at start
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        for p in self.base.parameters():
            p.requires_grad_(False)

    @property
    def in_features(self):  return self.base.in_features
    @property
    def out_features(self): return self.base.out_features

    def forward(self, x):
        base_out = self.base(x)
        # LoRA params stay float32 for gradient quality; cast x accordingly
        x_f = x.to(self.lora_A.weight.dtype)
        lora_out = self.scale * self.lora_B(self.lora_A(x_f))
        return base_out + lora_out.to(base_out.dtype)

def add_lora(model, rank, alpha):
    n_added = 0
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, StaticLowRankCPU):
            continue
        parts = name.split('.')
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], LoRAOverCompressed(mod, rank, alpha))
        n_added += 1
    print(f"  LoRA: added {n_added} adapters (rank={rank}, alpha={alpha})")
    return model, n_added

# ── load model ────────────────────────────────────────────────
print("\nLoading compressed model...")
model = AutoModel.from_pretrained(
    args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
model = load_tsvd(model, weights_path)

for p in model.parameters():
    p.requires_grad_(False)

print("Adding LoRA adapters...")
model, n_lora = add_lora(model, args.rank, args.lora_alpha)

# move LoRA params to device (StaticLowRankCPU buffers stay on CPU)
for mod in model.modules():
    if isinstance(mod, LoRAOverCompressed):
        mod.lora_A.to(device)
        mod.lora_B.to(device)

lora_params = [p for p in model.parameters() if p.requires_grad]
total_lora = sum(p.numel() for p in lora_params)
print(f"  Trainable LoRA params: {total_lora:,}  ({n_lora} layers × rank={args.rank})")

# ── data ──────────────────────────────────────────────────────
with open(args.data_path) as f:
    all_poems = json.load(f)

train_poems = all_poems[args.train_skip:]
eval_poems  = all_poems[:args.train_skip]
print(f"\nTrain: {len(train_poems)} | Eval: {len(eval_poems)} poems")

extra = ' 直接输出句子即可。'

def make_example(poem, task):
    if task == 'ftb':
        prompt_text = poem['first']  + '的下一句是什么？' + extra
        answer_text = poem['second']
    else:
        prompt_text = poem['second'] + '的上一句是什么？' + extra
        answer_text = poem['first']
    m = [{'role': 'user', 'content': prompt_text}]
    ctx = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    full_str = ctx + answer_text
    ids = tokenizer(full_str, return_tensors='pt')['input_ids'].to(device)
    prompt_len = tokenizer(ctx, return_tensors='pt')['input_ids'].shape[1]
    return ids, prompt_len

def random_mask(ids, prompt_len):
    ratio = random.uniform(0.15, 0.95)
    noisy = ids.clone()
    for pos in range(prompt_len, ids.shape[1]):
        if random.random() < ratio:
            noisy[0, pos] = MASK_ID
    return noisy

# ── train ─────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.01)
tasks = ['ftb', 'btf']

print(f"\nTraining {args.epochs} epochs (lr={args.lr})...")
for epoch in range(args.epochs):
    model.train()
    examples = [(p, t) for p in train_poems for t in tasks]
    random.shuffle(examples)
    total_loss = n_steps = 0

    for poem, task in tqdm.tqdm(examples, desc=f"Epoch {epoch+1}/{args.epochs}"):
        ids, prompt_len = make_example(poem, task)
        if ids.shape[1] - prompt_len == 0:
            continue
        noisy = random_mask(ids, prompt_len)
        mask_index = (noisy == MASK_ID)[0]
        if not mask_index.any():
            continue

        logits = model(noisy).logits
        loss = F.cross_entropy(logits[0][mask_index], ids[0][mask_index])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_steps += 1

    print(f"  Epoch {epoch+1}: avg_loss={total_loss/max(n_steps,1):.4f}  steps={n_steps}")

# ── save LoRA weights ─────────────────────────────────────────
lora_state = {}
for name, mod in model.named_modules():
    if isinstance(mod, LoRAOverCompressed):
        lora_state[name + '.lora_A'] = mod.lora_A.weight.data.cpu()
        lora_state[name + '.lora_B'] = mod.lora_B.weight.data.cpu()
torch.save(lora_state, os.path.join(out_dir, 'lora_weights.pt'))
print(f"\nLoRA weights saved → {out_dir}/lora_weights.pt")
print(f"  Keys: {len(lora_state)} tensors")

# ── generation eval ───────────────────────────────────────────
def _add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    return logits.exp() / ((-torch.log(noise)) ** temperature)

def _get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    ntt = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        ntt[i, :remainder[i]] += 1
    return ntt

@torch.no_grad()
def generate_seq(model, prompt, steps=28, gen_length=28):
    x = torch.full((1, prompt.shape[1] + gen_length), MASK_ID,
                   dtype=torch.long, device=device)
    x[:, :prompt.shape[1]] = prompt.clone()
    block_end = prompt.shape[1] + gen_length
    block_mask_index = (x[:, prompt.shape[1]:block_end] == MASK_ID)
    num_transfer_tokens = _get_num_transfer_tokens(block_mask_index, steps)

    for i in range(steps):
        mask_index = (x == MASK_ID)
        logits = model(x).logits
        x0 = torch.argmax(_add_gumbel_noise(logits, 0.), dim=-1)
        p    = F.softmax(logits, dim=-1)
        x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
        x0_p[:, block_end:] = -np.inf
        x0         = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -np.inf))
        transfer_index = torch.zeros_like(x0, dtype=torch.bool)
        _, sel = torch.topk(confidence[0], k=num_transfer_tokens[0, i])
        transfer_index[0, sel] = True
        x[transfer_index] = x0[transfer_index]
    return x

model.eval()
from difflib import SequenceMatcher

eval_poems_sub = eval_poems[:args.gen_limit]
print(f"\nGeneration eval ({args.steps} steps, {len(eval_poems_sub)} poems):")
results = {}
for task in tasks:
    if task == 'ftb':
        prompts = [p['first']  + '的下一句是什么？' + extra for p in eval_poems_sub]
        answers = [p['second'] for p in eval_poems_sub]
    else:
        prompts = [p['second'] + '的上一句是什么？' + extra for p in eval_poems_sub]
        answers = [p['first']  for p in eval_poems_sub]

    exact_hits = char_sims = 0
    for prompt_text, answer in tqdm.tqdm(zip(prompts, answers), total=len(prompts),
                                         desc=f'  [{task}]'):
        m = [{'role': 'user', 'content': prompt_text}]
        ps = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        iids = tokenizer(ps, return_tensors='pt')['input_ids'].to(device)
        out = generate_seq(model, iids, steps=args.steps)
        decoded = tokenizer.batch_decode(out[:, iids.shape[1]:],
                                         skip_special_tokens=True)[0].strip()
        exact_hits += int(answer in decoded)
        char_sims  += SequenceMatcher(None, answer, decoded).ratio()

    n = len(prompts)
    exact = exact_hits / n
    csim  = char_sims / n
    results[task] = {'exact_acc': exact, 'char_sim': csim}
    print(f"  [{args.mode}+lora_r{args.rank}] {task}: exact={exact:.3f}  char_sim={csim:.3f}")

out_json = {
    'mode': args.mode, 'rank': args.rank, 'alpha': args.lora_alpha,
    'epochs': args.epochs, 'lr': args.lr, 'steps': args.steps,
    'n_lora_layers': n_lora, 'total_lora_params': total_lora,
    'results': results,
}
with open(os.path.join(out_dir, 'eval_results.json'), 'w') as f:
    json.dump(out_json, f, indent=2)
print(f"\nEval results saved → {out_dir}/eval_results.json")
print("Done.")
