import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


class _SelectedHeadCaptured(RuntimeError):
    pass


class SelectiveHeadProjector:
    """Project only active answer-mask positions to the vocabulary."""

    def __init__(self, model):
        self.model = model
        self.head = model.get_output_embeddings()
        if not isinstance(self.head, nn.Linear):
            raise TypeError("selective generation requires an nn.Linear lm_head")
        self.selection = None
        self.hidden = None

        def capture(_module, inputs):
            self.hidden = inputs[0][self.selection].detach()
            raise _SelectedHeadCaptured()

        self.handle = self.head.register_forward_pre_hook(capture)

    @torch.no_grad()
    def __call__(self, batch, selection):
        self.selection = selection
        self.hidden = None
        try:
            self.model(batch)
        except _SelectedHeadCaptured:
            pass
        if self.hidden is None:
            raise RuntimeError("failed to capture hidden states before lm_head")
        logits = F.linear(self.hidden, self.head.weight, self.head.bias)
        if getattr(self.model.config, "scale_logits", False):
            logits.mul_(1.0 / math.sqrt(self.model.config.d_model))
        return logits

    def close(self):
        self.handle.remove()


@torch.no_grad()
def selected_cfg_logits(projector, x, selection, prompt_index, cfg_scale, mask_id):
    if cfg_scale > 0.:
        un_x = x.clone()
        un_x[prompt_index] = mask_id
        pair = torch.cat([x, un_x], dim=0)
        pair_selection = torch.cat([selection, selection], dim=0)
        selected = projector(pair, pair_selection)
        n = int(selection.sum())
        logits, un_logits = selected[:n], selected[n:]
        return un_logits + (cfg_scale + 1) * (logits - un_logits)
    return projector(x, selection)


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def get_num_transfer_tokens_host(mask_num, steps):
    """Host-side equivalent for the batch-1 fixed-size generation blocks.

    The number of initially masked tokens in a block is known from the sampler
    configuration.  Keeping this schedule as Python integers avoids a forced
    device-to-host synchronization at every denoising step.
    """
    base, remainder = divmod(int(mask_num), int(steps))
    return [base + int(index < remainder) for index in range(steps)]


def max_token_and_confidence(logits):
    """Return argmax token and its softmax probability without materializing softmax.

    softmax(z)[argmax(z)] = exp(max(z) - logsumexp(z)).  This avoids writing a
    second [active_tokens, vocabulary] tensor on every reverse step.
    """
    max_logits, token_ids = torch.max(logits, dim=-1)
    confidence = torch.exp(max_logits - torch.logsumexp(logits, dim=-1))
    return token_ids, confidence


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336,
             fast_path=False, memory_efficient_confidence=False):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    # The official benchmark uses temperature=0. Selective projection is exact
    # in that regime; nonzero-temperature Gumbel draws would require reproducing
    # the random tensor for every skipped prompt position.
    if temperature != 0:
        raise ValueError("selective generation currently requires temperature=0")
    projector = SelectiveHeadProjector(model)
    try:
        for num_block in range(num_blocks):
            if fast_path:
                # Every new block starts fully masked by construction.  This is
                # exactly the same integer schedule as get_num_transfer_tokens.
                num_transfer_tokens_host = get_num_transfer_tokens_host(
                    block_length, steps
                )
                num_transfer_tokens = None
            else:
                block_mask_index = (
                    x[:, prompt.shape[1] + num_block * block_length:
                      prompt.shape[1] + (num_block + 1) * block_length:] == mask_id
                )
                num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
                num_transfer_tokens_host = None

            for i in range(steps):
                mask_index = (x == mask_id)
                selection = mask_index.clone()
                selection[:, prompt.shape[1] + (num_block + 1) * block_length:] = False
                positions = torch.nonzero(selection[0], as_tuple=False).squeeze(1)
                logits = selected_cfg_logits(
                    projector, x, selection, prompt_index, cfg_scale, mask_id
                )

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                if (memory_efficient_confidence and remasking == 'low_confidence'
                        and temperature == 0):
                    x0, x0_p = max_token_and_confidence(logits_with_noise)
                else:
                    x0 = torch.argmax(logits_with_noise, dim=-1)
                    if remasking == 'low_confidence':
                        p = F.softmax(logits, dim=-1)
                        x0_p = torch.squeeze(
                            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)

                if remasking == 'random':
                    x0_p = torch.rand((x.shape[0], x.shape[1]), device=x.device)[selection]
                elif remasking != 'low_confidence':
                    raise NotImplementedError(remasking)

                take = (num_transfer_tokens_host[i] if fast_path
                        else int(num_transfer_tokens[0, i]))
                chosen = torch.topk(x0_p, k=take).indices
                x[0, positions[chosen]] = x0[chosen]
    finally:
        projector.close()
    return x


def main():
    device = 'cuda'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()
