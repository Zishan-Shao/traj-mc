import os

import time

import torch
import torch.nn as nn

from bigptq_arb import BRAGPTQ
from binary_arb import Binarization
from modelutils import find_layers, cleanup_memory
import model_utils
import logging

from paths import LOG_DIR, OUTPUT_DIR
from utils.mcs import apply_mcs


def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    for noisy_logger in ("urllib3", "filelock", "fsspec"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    return logger


def get_model(model):
    from transformers import AutoModelForCausalLM

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model.seqlen = getattr(model.config, "max_position_embeddings", 4096)
    if hasattr(model.config, "n_positions"):
        model.seqlen = model.config.n_positions
    return model


'''
The function is employed to calibrate and quantize models layer by layer.
'''
@torch.no_grad()
def quant_sequential(model, dataloader, dev, loaded_orders=None):
    print("Starting ...")

    for name, module in model.named_modules():
        module.global_name = args.model + name

    use_cache = model.config.use_cache
    model.config.use_cache = False

    mask_id = getattr(model.config, "mask_token_id", 126336)
    if not (
        hasattr(model, "model")
        and hasattr(model.model, "transformer")
        and hasattr(model.model.transformer, "blocks")
    ):
        raise ValueError(f"Unsupported LLaDA model structure: {type(model)}")

    transformer = model.model.transformer
    layers = transformer.blocks
    transformer.wte = transformer.wte.to(dev)
    for component in ("emb_drop", "ln_f", "ff_out"):
        if hasattr(transformer, component):
            setattr(transformer, component, getattr(transformer, component).to(dev))
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            raise ValueError

    layers[0] = Catcher(layers[0])
    for sample_index, batch in enumerate(dataloader):
        try:
            batch_input = batch[0].to(dev)
            noisy_batch, _ = apply_mcs(
                batch_input,
                mask_id,
                sample_index=sample_index,
                num_samples=args.nsamples,
                prefix_ratio=args.mcs_prefix_ratio,
                seed=args.seed_x,
            )
            model(noisy_batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    transformer.wte = transformer.wte.cpu()
    for component in ("emb_drop", "ln_f", "ff_out"):
        if hasattr(transformer, component):
            setattr(transformer, component, getattr(transformer, component).cpu())
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)

    sequential = [
        ["attn_out"],
        ["ff_out"],
        ["q_proj"],
        ["k_proj"],
        ["v_proj"],
        ["ff_proj"],
        ["up_proj"],
    ]

    fp_inputs_cache = model_utils.FPInputsCache(sequential)
    fp_inps = inps.clone()

    # SliM 分支：添加保存block order配置的字典
    if args.slim:
        mixed_block_orders = {}

    print("Ready.")

    for i in range(len(layers)):
        layer = layers[i].to(dev)

        full = find_layers(layer)

        fp_inputs_cache.add_hook(full)
        for j in range(args.nsamples):
            diffusion_input = fp_inps[j].unsqueeze(0)
            fp_inps[j] = layer(diffusion_input)[0]
        fp_inputs_cache.clear_hook()

        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                if (
                    not (args.minlayer <= i < args.maxlayer and args.quant_only in name)
                ) == (not args.invert):
                    continue
                braq_quantizer = Binarization(
                    subset[name].weight,
                    method=args.low_quant_method,
                    groupsize=groupsize,
                )
                gptq[name] = BRAGPTQ(
                    subset[name],
                    braq_quantizer,
                    salient_metric=args.salient_metric,
                    disable_gptq=args.disable_gptq,
                    method=args.low_quant_method,
                    order2_group=args.order2_group,
                    gptaq=args.gptaq,
                )
                gptq[name].fp_inp = fp_inputs_cache.fp_cache[name]

            if not gptq:
                continue

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            first_module_name = list(gptq.keys())[0]

            # # gptaq中的实现，不太行
            # handle = subset[first_module_name].register_forward_hook(add_batch(first_module_name))
            # for j in range(args.nsamples):
            #     outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            # handle.remove()

            # arb中的实现
            handles = []
            for name in gptq:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                diffusion_input = inps[j].unsqueeze(0)
                outs[j] = layer(diffusion_input)[0]
            for h in handles:
                h.remove()

            # ABMP uses Hessian salience to rank blocks. The legacy block-error
            # hooks were unused by the allocator and repeated expensive trial
            # quantization for every calibration sample.
            if args.slim and loaded_orders is None:
                for name in gptq:
                    gptq[name].get_salience(blocksize=args.blocksize)

            for name in gptq:
                if name != first_module_name:
                    gptq[name].H = gptq[first_module_name].H
                    if args.gptaq:
                        gptq[name].dXXT = gptq[first_module_name].dXXT

            # SliM 分支：准备保存block orders配置
            if args.slim:
                mixed_block_orders[i] = {}

            for name in gptq:
                # print(i, name)
                # print("Quantizing ...")
                # --- 核心修改：从 loaded_orders 中获取配置 ---
                orders_to_use = None
                if args.slim and loaded_orders is not None:
                    # 从加载的字典中安全地获取当前层、当前模块的 orders
                    if i in loaded_orders and name in loaded_orders[i]:
                        orders_to_use = loaded_orders[i][name]
                logging.info(f'{i} {name}')
                logging.info("Quantizing ...")
                info = gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    blocksize=args.blocksize,
                    num_p=args.num_p,
                    disable_mask=args.disable_mask,
                    no_mask_order=args.no_mask_order,
                    slim=args.slim,
                    saved_block_orders=orders_to_use,
                    abmp_ratio=args.abmp_ratio,
                )
                # SliM 分支：保存block orders配置
                if args.slim and 'block_orders' in info:
                    mixed_block_orders[i][name] = info['block_orders']
                gptq[name].free()

        for j in range(args.nsamples):
            diffusion_input = inps[j].unsqueeze(0)
            outs[j] = layer(diffusion_input)[0]
        fp_inputs_cache.clear_cache()

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    if args.slim:
        net = args.model.split("/")[-1]
        save_path = (
            OUTPUT_DIR
            / f"block_orders_{args.blocksize}_{args.low_quant_method}"
            / f"{net}_seed_{args.seed_x}.pt"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(mixed_block_orders, save_path)
        logging.info(f'Saved block orders to {save_path}')
    model.config.use_cache = use_cache
    cleanup_memory(verbose=True)


if __name__ == "__main__":
    import argparse
    from datautils import get_loaders

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model", type=str, help="Local path or Hugging Face ID for a LLaDA model."
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "low_quant_method",
        type=str,
        choices=["arb", "arb-x", 'arb-rc', 'braq'],
        help="alternating refined binarization method",
    )
    parser.add_argument(
        "--order2_group",
        action='store_true',
        help="division for salient weights",
    )
    parser.set_defaults(order2_group=False)
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--seed_x", type=int, default=0, help="Seed for MCS mask sampling."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--mcs_prefix_ratio",
        type=float,
        default=0.25,
        help="Fraction of leading tokens kept visible during MCS.",
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=128,
        help="Blocksize to use for adaptive mask selection.",
    )
    parser.add_argument(
        "--num_p",
        type=int,
        default=1,
        help="Number of division for non-salient weights",
    )
    parser.add_argument(
        "--salient_metric",
        type=str,
        default="magnitude",
        choices=["magnitude", "hessian"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="set the device to use for quantization.",
    )
    parser.add_argument(
        "--disable_gptq",
        action="store_true",
        help="disable GPTQ for quantization.",
    )
    parser.add_argument(
        "--minlayer", type=int, default=-1, help="Quant all layers with id >= this."
    )
    parser.add_argument(
        "--maxlayer", type=int, default=1000, help="Quant all layers with id < this."
    )
    parser.add_argument(
        "--quant_only",
        type=str,
        default="",
        help="Quant only layers that contain this text.",
    )
    parser.add_argument("--invert", action="store_true", help="Invert subset.")
    parser.add_argument(
        "--save",
        action="store_true",
    )
    parser.add_argument(
        "--gptaq",
        action="store_true",
    )
    parser.add_argument(
        "--disable_mask",
        action="store_true",
    )
    parser.add_argument(
        "--no_mask_order", type=int, default=1, help="no mask order for quantization."
    )
    parser.add_argument(
        "--slim",
        action="store_true",
        help="Enable SliM-LLM style mixed-order quantization for different blocks",
    )
    parser.add_argument(
        "--abmp_ratio",
        type=float,
        default=0.05,
        help="Fraction of blocks moved from 2-bit to each of 1-bit and 3-bit.",
    )
    parser.add_argument(
        "--load_block_orders",
        type=str,
        default=None,
        help="Path to the saved block orders file (.pt) to skip salience/error calculation.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="",
    )

    args = parser.parse_args()
    groupsize = args.blocksize

    device = args.device
    save_title = f"{args.model.split('/')[-1]}_{args.dataset}_{args.low_quant_method}_{groupsize}_{args.salient_metric}_nump_{args.num_p}_order2group_{args.order2_group}_gptaq_{args.gptaq}_disable_mask_{args.disable_mask}_no_mask_order_{args.no_mask_order}_slim_{args.slim}_abmp_ratio_{args.abmp_ratio}_mcs_prefix_{args.mcs_prefix_ratio}_seed_{args.seed_x}_maskx+salience+slim"
    save_file = str(OUTPUT_DIR / (save_title.replace("/", "_") + ".pt"))
    log_file = LOG_DIR / (
        save_title.replace("/", "_") + f"_{args.experiment}" + ".log"
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_file)

    model = get_model(args.model)
    model.eval()

    loaded_orders = None
    if args.load_block_orders:
        if os.path.exists(args.load_block_orders):
            logger.info("Loading block orders from %s", args.load_block_orders)
            loaded_orders = torch.load(
                args.load_block_orders,
                map_location="cpu",
                weights_only=False,
            )
        else:
            logger.warning(
                "Block orders not found at %s; computing them from scratch.",
                args.load_block_orders,
            )

    tick = time.time()
    dataloader, _ = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
    )
    quant_sequential(model, dataloader, device, loaded_orders=loaded_orders)
    logger.info("Quantization time: %.2f seconds", time.time() - tick)
    logger.info("Experiment: %s", args.experiment)

    if args.save:
        save_path = os.path.dirname(save_file)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model.save_pretrained(save_file)
