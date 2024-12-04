import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from tqdm import tqdm
from einops import rearrange, repeat
from omegaconf import OmegaConf

from diffusers import DDIMScheduler, DiffusionPipeline

from masactrl.diffuser_utils import MasaCtrlPipeline
from masactrl.masactrl_utils import AttentionBase
from masactrl.masactrl_utils import regiter_attention_editor_diffusers
from masactrl.masactrl import MutualSelfAttentionControl

from torchvision.utils import save_image
from torchvision.io import read_image
from pytorch_lightning import seed_everything

import argparse
import re

torch.cuda.set_device(0)  # set the GPU device

# Note that you may add your Hugging Face token to get access to the models
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def pathify(s):
    # Convert to lowercase and replace non-alphanumeric characters with underscores
    return re.sub(r'[^a-zA-Z0-9]', '_', s.lower())

def consistent_synthesis(args, seed, iteration, model, scheduler):
    seed_everything(seed)

    # Create the output directory based on prompt2
    out_dir_ori = os.path.join("masactrl_exp", pathify(args.prompt2), f"iteration_{iteration}")
    os.makedirs(out_dir_ori, exist_ok=True)

    prompts = [
        args.prompt1,
        args.prompt2,
    ]

    # inference the synthesized image with MasaCtrl
    # TODO: note that the hyper paramerter of MasaCtrl for SDXL may be not optimal
    STEP = 4
    #LAYER_LIST = [64, 74]  # run the synthesis with MasaCtrl at three different layer configs
    LAYER_LIST = [74]

    # initialize the noise map
    start_code = torch.randn([1, 4, 128, 128], device=device)
    start_code = start_code.expand(len(prompts), -1, -1, -1)

    # inference the synthesized image without MasaCtrl
    editor = AttentionBase()
    regiter_attention_editor_diffusers(model, editor)
    image_ori = model(prompts, latents=start_code, guidance_scale=args.guidance_scale).images

    for LAYER in LAYER_LIST:
        # hijack the attention module
        editor = MutualSelfAttentionControl(STEP, LAYER, model_type="SDXL")
        regiter_attention_editor_diffusers(model, editor)

        # inference the synthesized image
        image_masactrl = model(prompts, latents=start_code, guidance_scale=args.guidance_scale).images

        sample_count = len(os.listdir(out_dir_ori))
        out_dir = os.path.join(out_dir_ori, f"sample_{sample_count}")
        os.makedirs(out_dir, exist_ok=True)
        image_ori[0].save(os.path.join(out_dir, f"source_step{STEP}_layer{LAYER}.png"))
        image_ori[1].save(os.path.join(out_dir, f"without_step{STEP}_layer{LAYER}.png"))
        image_masactrl[-1].save(os.path.join(out_dir, f"masactrl_step{STEP}_layer{LAYER}.png"))
        with open(os.path.join(out_dir, f"prompts.txt"), "w") as f:
            for p in prompts:
                f.write(p + "\n")
            f.write(f"seed: {seed}\n")
        print("Syntheiszed images are saved in", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consistent Synthesis with MasaCtrl")
    parser.add_argument("--model_path", type=str, default="svjack/GenshinImpact_XL_Base", help="Path to the model")
    parser.add_argument("--prompt1", type=str, default="A portrait of an old man, facing camera, best quality", help="First prompt")
    parser.add_argument("--prompt2", type=str, default="A portrait of an old man, facing camera, smiling, best quality", help="Second prompt")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--num_iterations", type=int, default=1, help="Number of times to run the synthesis")
    parser.add_argument("--start_seed", type=int, default=42, help="Initial seed value")

    args = parser.parse_args()

    # If out_dir is not provided, use the default path based on prompt2
    if args.out_dir is None:
        args.out_dir = os.path.join("masactrl_exp", pathify(args.prompt2))

    # Initialize the model and scheduler once
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    model = DiffusionPipeline.from_pretrained(args.model_path, scheduler=scheduler)
    model.enable_model_cpu_offload()
    model.enable_sequential_cpu_offload()
    model.vae.enable_slicing()
    model.vae.enable_tiling()

    seed = args.start_seed
    for i in range(args.num_iterations):
        consistent_synthesis(args, seed, i, model, scheduler)
        seed += 1  # Increment seed for each iteration