import gradio as gr
import torch
from diffusers import DDIMScheduler, DiffusionPipeline
from masactrl.diffuser_utils import MasaCtrlPipeline
from masactrl.masactrl_utils import AttentionBase, regiter_attention_editor_diffusers
from masactrl.masactrl import MutualSelfAttentionControl
from pytorch_lightning import seed_everything
import os
import re

# 初始化设备和模型
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
weight_dtype = torch.float16

model = DiffusionPipeline.from_pretrained("svjack/GenshinImpact_XL_Base", scheduler=scheduler,
                                        torch_dtype=weight_dtype,
                                         ).to(device)
model.unet.set_default_attn_processor()

def pathify(s):
    return re.sub(r'[^a-zA-Z0-9]', '_', s.lower())

def consistent_synthesis(prompt1, prompt2, guidance_scale, seed, starting_step, starting_layer):
    seed_everything(seed)

    # 创建输出目录
    out_dir_ori = os.path.join("masactrl_exp", pathify(prompt2))
    os.makedirs(out_dir_ori, exist_ok=True)

    prompts = [prompt1, prompt2]

    # 初始化噪声图
    start_code = torch.randn([1, 4, 128, 128], dtype=weight_dtype, device=device)
    start_code = start_code.expand(len(prompts), -1, -1, -1)

    # 推理没有 MasaCtrl 的图像
    editor = AttentionBase()
    regiter_attention_editor_diffusers(model, editor)
    image_ori = model(prompts, latents=start_code, guidance_scale=guidance_scale).images

    images = []
    # 劫持注意力模块
    editor = MutualSelfAttentionControl(starting_step, starting_layer, model_type="SDXL")
    regiter_attention_editor_diffusers(model, editor)

    # 推理带 MasaCtrl 的图像
    image_masactrl = model(prompts, latents=start_code, guidance_scale=guidance_scale).images

    sample_count = len(os.listdir(out_dir_ori))
    out_dir = os.path.join(out_dir_ori, f"sample_{sample_count}")
    os.makedirs(out_dir, exist_ok=True)
    image_ori[0].save(os.path.join(out_dir, f"source_step{starting_step}_layer{starting_layer}.png"))
    image_ori[1].save(os.path.join(out_dir, f"without_step{starting_step}_layer{starting_layer}.png"))
    image_masactrl[-1].save(os.path.join(out_dir, f"masactrl_step{starting_step}_layer{starting_layer}.png"))
    with open(os.path.join(out_dir, f"prompts.txt"), "w") as f:
        for p in prompts:
            f.write(p + "\n")
        f.write(f"seed: {seed}\n")
        f.write(f"starting_step: {starting_step}\n")
        f.write(f"starting_layer: {starting_layer}\n")
    print("Synthesized images are saved in", out_dir)

    return [image_ori[0], image_ori[1], image_masactrl[-1]]

def create_demo_synthesis():
    with gr.Blocks() as demo:
        gr.Markdown("# **Genshin Impact XL MasaCtrl Image Synthesis**")  # 添加标题
        gr.Markdown("## **Input Settings**")
        with gr.Row():
            with gr.Column():
                prompt1 = gr.Textbox(label="Prompt 1", value="solo,ZHONGLI(genshin impact),1boy,highres,")
                prompt2 = gr.Textbox(label="Prompt 2", value="solo,ZHONGLI drink tea use chinese cup (genshin impact),1boy,highres,")
                with gr.Row():
                    starting_step = gr.Slider(label="Starting Step", minimum=0, maximum=999, value=4, step=1)
                    starting_layer = gr.Slider(label="Starting Layer", minimum=0, maximum=999, value=64, step=1)
                run_btn = gr.Button("Run")
            with gr.Column():
                guidance_scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=7.5, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, value=42, step=1)

        gr.Markdown("## **Output**")
        with gr.Row():
            image_source = gr.Image(label="Source Image")
            image_without_masactrl = gr.Image(label="Image without MasaCtrl")
            image_with_masactrl = gr.Image(label="Image with MasaCtrl")

        inputs = [prompt1, prompt2, guidance_scale, seed, starting_step, starting_layer]
        run_btn.click(consistent_synthesis, inputs, [image_source, image_without_masactrl, image_with_masactrl])

        gr.Examples(
            [
                ["solo,ZHONGLI(genshin impact),1boy,highres,", "solo,ZHONGLI drink tea use chinese cup (genshin impact),1boy,highres,", 42, 4, 64],
                ["solo,KAMISATO AYATO(genshin impact),1boy,highres,", "solo,KAMISATO AYATO smiling (genshin impact),1boy,highres,", 42, 4, 55]
            ],
            [prompt1, prompt2, seed, starting_step, starting_layer],
        )
    return demo

if __name__ == "__main__":
    demo_synthesis = create_demo_synthesis()
    demo_synthesis.launch(share = True)
