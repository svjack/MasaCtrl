# MotionCtrl and MasaCtrl: Genshin Impact Character Synthesis

This repository provides a guide to setting up and running the MotionCtrl and MasaCtrl projects for synthesizing Genshin Impact characters using machine learning models. The process involves cloning repositories, installing dependencies, and executing scripts to generate and visualize character images.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Synthesis](#running-the-synthesis)
- [Using Gradio Interface](#using-gradio-interface)
- [Example Prompts](#example-prompts)

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.10
- Conda (for environment management)
- Git
- Git LFS (Large File Storage)
- FFmpeg

## Installation

### Step 1: Clone the MotionCtrl Repository
Clone the MotionCtrl repository and install its dependencies:

```bash
git clone https://huggingface.co/spaces/svjack/MotionCtrl
cd MotionCtrl
pip install -r requirements.txt
```

### Step 2: Install System Dependencies
Update your package list and install necessary system packages:

```bash
sudo apt-get update
sudo apt-get install cbm git-lfs ffmpeg
```

### Step 3: Set Up Python Environment
Create a Conda environment with Python 3.10, activate it, and install the IPython kernel:

```bash
conda create -n py310 python=3.10
conda activate py310
pip install ipykernel
python -m ipykernel install --user --name py310 --display-name "py310"
```

### Step 4: Clone the MasaCtrl Repository
Clone the MasaCtrl repository and install its dependencies:

```bash
git clone https://github.com/svjack/MasaCtrl
cd MasaCtrl
pip install -r requirements.txt
pip install -U accelerate
pip install -U gradio
```

## Running the Synthesis

### Command Line Interface
Run the synthesis script to generate images of Genshin Impact characters:

```bash
python run_synthesis_genshin_impact_xl.py --model_path "svjack/GenshinImpact_XL_Base" \
 --prompt1 "solo,ZHONGLI\(genshin impact\),1boy,highres," \
 --prompt2 "solo,ZHONGLI drink tea use chinese cup \(genshin impact\),1boy,highres," --guidance_scale 5
```
- OR
```bash
python run_synthesis_genshin_impact_xl_offload.py --model_path "svjack/GenshinImpact_XL_Base" \
 --prompt1 "solo,ZHONGLI\(genshin impact\),1boy,highres," \
 --prompt2 "solo,ZHONGLI drink tea use chinese cup \(genshin impact\),1boy,highres," --guidance_scale 5
```
- OR
```bash
python run_synthesis_genshin_impact_xl_multi_offload.py --model_path "svjack/GenshinImpact_XL_Base" \
 --prompt1 "solo,ZHONGLI\(genshin impact\),1boy,highres," \
 --prompt2 "solo,ZHONGLI drink tea use chinese cup \(genshin impact\),1boy,highres," \
 --guidance_scale 5 \
 --num_iterations 1000 \
 --start_seed 42
```
- OR
```bash
  python run_synthesis_genshin_impact_xl_layers.py --model_path "svjack/GenshinImpact_XL_Base" \
   --prompt1 "solo, Ancient Chinese XINGQIU in Chinese traditional long gown \(genshin impact\),1boy,highres," \
   --prompt2 "solo, Ancient Chinese XINGQIU in Chinese traditional long gown pack clothes in a bag\(genshin impact\),1boy,highres," \
   --guidance_scale 5 \
   --start 14 --end 94 --step 10 --seed -1
```

### Gradio Interface (Locate in https://huggingface.co/spaces/svjack/Genshin-Impact-XL-MasaCtrl)
Alternatively, you can use the Gradio interface for a more interactive experience:

```bash
python run_synthesis_genshin_impact_xl_app.py
```

## Example Prompts

Here are some example prompts you can use to generate different character images: (Image with MasaCtrl more like Source Image: In terms of background and other aspects)

- **Zhongli Drinking Tea:**
  ```
  "solo,ZHONGLI(genshin impact),1boy,highres," -> "solo,ZHONGLI drink tea use chinese cup (genshin impact),1boy,highres,"
  ```
![Screenshot 2024-11-17 132742](https://github.com/user-attachments/assets/00451728-f2d5-4009-afa8-23baaabdc223)

- **Kamisato Ayato Smiling:**
  ```
  "solo,KAMISATO AYATO(genshin impact),1boy,highres," -> "solo,KAMISATO AYATO smiling (genshin impact),1boy,highres,"
  ```

![Screenshot 2024-11-17 133421](https://github.com/user-attachments/assets/7a920f4c-8a3a-4387-98d6-381a798566ef)

## MasaCtrl as Video demo 
### Use ToonCrafter to create Video (Source image as start, MasaCtrl image as end)
```shell
git clone https://huggingface.co/spaces/svjack/ToonCrafter-fp16 && cd ToonCrafter-fp16 && pip install -r requirements.txt
pip install --upgrade einops
python app.py
```
### Use Practical-RIFE to interplotation 
```shell
git clone https://github.com/svjack/Practical-RIFE && cd Practical-RIFE && pip install -r requirements.txt
python inference_video.py --multi=128 --video=../zhongli_sitting_down.mp4
python inference_video.py --multi=128 --video=../ayato_smiling.mp4
```
### Use Real-ESRGAN-Video to upscale 
```shell
pip install py-real-esrgan moviepy
```
```python
from video_upscaler_with_skip import *

input_video_path = "zhongli_sitting_down_128X_1024fps.mp4"
output_video_path = "zhongli_sitting_down_128X_1024fps_x4_sk2.mp4"
upscale_factor = 4
threshold = 2  # 设置阈值，0表示不跳过任何帧
upscale_video(input_video_path, output_video_path, upscale_factor, max_frames=None, threshold=threshold)

input_video_path = "ayato_smiling_128X_1024fps.mp4"
output_video_path = "ayato_smiling_128X_1024fps_x4_sk2.mp4"
upscale_factor = 4
threshold = 2  # 设置阈值，0表示不跳过任何帧
upscale_video(input_video_path, output_video_path, upscale_factor, max_frames=None, threshold=threshold)
```

<div style="display: flex; flex-direction: column; align-items: center;">
    <div style="margin-bottom: 10px;">
        <h3>Zhongli Drinking Tea:</h3>
    </div>
    <div style="margin-bottom: 10px;">
        <video controls autoplay src="https://github.com/user-attachments/assets/600efb3d-20bc-4791-86b8-2c3210dd65f3" style="width: 512px; height: 256px;"></video>
    </div>
    <div style="margin-bottom: 10px;">
        <video controls autoplay src="" style="width: 512px; height: 256px;"></video>
    </div>
    <div style="margin-bottom: 10px;">
        <h3>Kamisato Ayato Smiling:</h3>
    </div>
    <div style="margin-bottom: 10px;">
        <video controls autoplay src="https://github.com/user-attachments/assets/03740d07-a113-4874-ab21-2326477eb675" style="width: 1024px; height: 768px;"></video>
    </div>
    <div style="margin-bottom: 10px;">
        <video controls autoplay src="" style="width: 1024px; height: 768px;"></video>
    </div>
</div>

### Real-ESRGAN-Video Upscale conclusion

https://github.com/user-attachments/assets/607e7eb7-d41c-4740-9c8a-8369c31487da

https://github.com/user-attachments/assets/aaa9849e-0c53-4012-b6c3-9ceb9910f2f8

### APISR(https://github.com/svjack/APISR) Upscale conclusion

![zhongli_up (1)](https://github.com/user-attachments/assets/728ad8b1-53ab-40f7-ba50-3e6ec4049d54)


![lingren_up (1)](https://github.com/user-attachments/assets/ab9168d3-00bd-451f-b14a-0bde7ba5e79b)

<!--
- **Zhongli Drinking Tea:**
#### ToonCrafter output 
https://github.com/user-attachments/assets/70ea5cd8-1fd1-40cf-b645-6577d3347f4a
#### Output processed
https://github.com/user-attachments/assets/53b06bd0-1868-4fb0-8fb6-2201a2e092ea
- **Kamisato Ayato Smiling:**
#### ToonCrafter output 
https://github.com/user-attachments/assets/6132ebb4-3fa2-4fcc-9c0e-3ca93027935d
#### Output processed
https://github.com/user-attachments/assets/5938eda0-0109-450b-aa4f-953485bd6182
-->

## More models can be tried by (https://github.com/the-database/VideoJaNai) Use model from (https://openmodeldb.info/)

## MasaCtrl: Tuning-free <span style="text-decoration: underline"><font color="Tomato">M</font></span>utu<span style="text-decoration: underline"><font color="Tomato">a</font></span>l <span style="text-decoration: underline"><font color="Tomato">S</font></span>elf-<span style="text-decoration: underline"><font color="Tomato">A</font></span>ttention <span style="text-decoration: underline"><font color="Tomato">Control</font></span> for Consistent Image Synthesis and Editing

Pytorch implementation of [MasaCtrl: Tuning-free Mutual Self-Attention Control for **Consistent Image Synthesis and Editing**](https://arxiv.org/abs/2304.08465)

[Mingdeng Cao](https://github.com/ljzycmd),
[Xintao Wang](https://xinntao.github.io/),
[Zhongang Qi](https://scholar.google.com/citations?user=zJvrrusAAAAJ),
[Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ),
[Xiaohu Qie](https://scholar.google.com/citations?user=mk-F69UAAAAJ),
[Yinqiang Zheng](https://scholar.google.com/citations?user=JD-5DKcAAAAJ)

[![arXiv](https://img.shields.io/badge/ArXiv-2304.08465-brightgreen)](https://arxiv.org/abs/2304.08465)
[![Project page](https://img.shields.io/badge/Project-Page-brightgreen)](https://ljzycmd.github.io/projects/MasaCtrl/)
[![demo](https://img.shields.io/badge/Demo-Hugging%20Face-brightgreen)](https://huggingface.co/spaces/TencentARC/MasaCtrl)
[![demo](https://img.shields.io/badge/Demo-Colab-brightgreen)](https://colab.research.google.com/drive/1DZeQn2WvRBsNg4feS1bJrwWnIzw1zLJq?usp=sharing)
[![Open in OpenXLab](https://cdn-static.openxlab.org.cn/app-center/openxlab_app.svg)](https://openxlab.org.cn/apps/detail/MingDengCao/MasaCtrl)

---

<div align="center">
<img src="https://huggingface.co/TencentARC/MasaCtrl/resolve/main/assets/overview.gif">
<i> MasaCtrl enables performing various consistent non-rigid image synthesis and editing without fine-tuning and optimization. </i>
</div>


## Updates
- [2024/8/17] We add AttnProcessor based MasaCtrlProcessor, please check `masactrl/masactrl_processor.py` and `run_synthesis_sdxl_processor.py`. You can integrate MasaCtrl into official Diffuser pipeline by register the attention processor. 
- [2023/8/20] MasaCtrl supports SDXL (and other variants) now. ![sdxl_example](https://huggingface.co/TencentARC/MasaCtrl/resolve/main/assets/sdxl_example.jpg)
- [2023/5/13] The inference code of MasaCtrl with T2I-Adapter is available.
- [2023/4/28] [Hugging Face demo](https://huggingface.co/spaces/TencentARC/MasaCtrl) released.
- [2023/4/25] Code released.
- [2023/4/17] Paper is available [here](https://arxiv.org/abs/2304.08465).

---

## Introduction

We propose MasaCtrl, a tuning-free method for non-rigid consistent image synthesis and editing. The key idea is to combine the `contents` from the *source image* and the `layout` synthesized from *text prompt and additional controls* into the desired synthesized or edited image, by querying semantically correlated features with **Mutual Self-Attention Control**.


## Main Features

### 1 Consistent Image Synthesis and Editing

MasaCtrl can perform prompt-based image synthesis and editing that changes the layout while maintaining contents of source image.

>*The target layout is synthesized directly from the target prompt.*

<details><summary>View visual results</summary>
<div align="center">
<img src="https://huggingface.co/TencentARC/MasaCtrl/resolve/main/assets/results_synthetic.png">
<i>Consistent synthesis results</i>

<img src="https://huggingface.co/TencentARC/MasaCtrl/resolve/main/assets/results_real.png">
<i>Real image editing results</i>
</div>
</details>



### 2 Integration to Controllable Diffusion Models

Directly modifying the text prompts often cannot generate target layout of desired image, thus we further integrate our method into existing proposed controllable diffusion pipelines (like T2I-Adapter and ControlNet) to obtain stable synthesis and editing results.

>*The target layout controlled by additional guidance.*

<details><summary>View visual results</summary>
<div align="center">
<img src="https://huggingface.co/TencentARC/MasaCtrl/resolve/main/assets/results_w_adapter.png">
<i>Synthesis (left part) and editing (right part) results with T2I-Adapter</i>
</div>
</details>

### 3 Generalization to Other Models: Anything-V4

Our method also generalize well to other Stable-Diffusion-based models.

<details><summary>View visual results</summary>
<div align="center">
<img src="https://huggingface.co/TencentARC/MasaCtrl/resolve/main/assets/anythingv4_synthetic.png">
<i>Results on Anything-V4</i>
</div>
</details>


### 4 Extension to Video Synthesis

With dense consistent guidance, MasaCtrl enables video synthesis

<details><summary>View visual results</summary>
<div align="center">
<img src="https://huggingface.co/TencentARC/MasaCtrl/resolve/main/assets/results_w_adapter_consistent.png">
<i>Video Synthesis Results (with keypose and canny guidance)</i>
</div>
</details>


## Usage

### Requirements
We implement our method with [diffusers](https://github.com/huggingface/diffusers) code base with similar code structure to [Prompt-to-Prompt](https://github.com/google/prompt-to-prompt). The code runs on Python 3.8.5 with Pytorch 1.11. Conda environment is highly recommended.

```base
pip install -r requirements.txt
```

### Checkpoints

**Stable Diffusion:**
We mainly conduct expriemnts on Stable Diffusion v1-4, while our method can generalize to other versions (like v1-5). You can download these checkpoints on their official repository and [Hugging Face](https://huggingface.co/).

**Personalized Models:**
You can download personlized models from [CIVITAI](https://civitai.com/) or train your own customized models.


### Demos

**Notebook demos**

To run the synthesis with MasaCtrl, single GPU with at least 16 GB VRAM is required. 

The notebook `playground.ipynb` and `playground_real.ipynb` provide the synthesis and real editing samples, respectively.

**Online demos**

We provide [![demo](https://img.shields.io/badge/Demo-Hugging%20Face-brightgreen)](https://huggingface.co/spaces/TencentARC/MasaCtrl) with Gradio app. Note that you may copy the demo into your own space to use the GPU. Online Colab demo [![demo](https://img.shields.io/badge/Demo-Colab-brightgreen)](https://colab.research.google.com/drive/1DZeQn2WvRBsNg4feS1bJrwWnIzw1zLJq?usp=sharing) is also available. 

**Local Gradio demo**

You can launch the provided Gradio demo locally with

```bash
CUDA_VISIBLE_DEVICES=0 python app.py
```


### MasaCtrl with T2I-Adapter

Install [T2I-Adapter](https://github.com/TencentARC/T2I-Adapter) and prepare the checkpoints following their provided tutorial. Assuming it has been successfully installed and the root directory is `T2I-Adapter`. 

Thereafter copy the core `masactrl` package and the inference code `masactrl_w_adapter.py` to the root directory of T2I-Adapter

```bash
cp -r MasaCtrl/masactrl T2I-Adapter/
cp MasaCtrl/masactrl_w_adapter/masactrl_w_adapter.py T2I-Adapter/
```

**[Updates]** Or you can clone the repo [MasaCtrl-w-T2I-Adapter](https://github.com/ljzycmd/T2I-Adapter-w-MasaCtrl) directly to your local space.

Last, you can inference the images with following command (with sketch adapter)

```bash
python masactrl_w_adapter.py \
--which_cond sketch \
--cond_path_src SOURCE_CONDITION_PATH \
--cond_path CONDITION_PATH \
--cond_inp_type sketch \
--prompt_src "A bear walking in the forest" \
--prompt "A bear standing in the forest" \
--sd_ckpt models/sd-v1-4.ckpt \
--resize_short_edge 512 \
--cond_tau 1.0 \
--cond_weight 1.0 \
--n_samples 1 \
--adapter_ckpt models/t2iadapter_sketch_sd14v1.pth
```

NOTE: You can download the sketch examples [here](https://huggingface.co/TencentARC/MasaCtrl/tree/main/sketch_example).

For real image, the DDIM inversion is performed to invert the image into the noise map, thus we add the inversion process into the original DDIM sampler. **You should replace the original file `T2I-Adapter/ldm/models/diffusion/ddim.py` with the exteneded version `MasaCtrl/masactrl_w_adapter/ddim.py` to enable the inversion function**. Then you can edit the real image with following command (with sketch adapter)

```bash
python masactrl_w_adapter.py \
--src_img_path SOURCE_IMAGE_PATH \
--cond_path CONDITION_PATH \
--cond_inp_type image \
--prompt_src "" \
--prompt "a photo of a man wearing black t-shirt, giving a thumbs up" \
--sd_ckpt models/sd-v1-4.ckpt \
--resize_short_edge 512 \
--cond_tau 1.0 \
--cond_weight 1.0 \
--n_samples 1 \
--which_cond sketch \
--adapter_ckpt models/t2iadapter_sketch_sd14v1.pth \
--outdir ./workdir/masactrl_w_adapter_inversion/black-shirt
```

NOTE: You can download the real image editing example [here](https://huggingface.co/TencentARC/MasaCtrl/tree/main/black_shirt_example).

## Acknowledgements

We thank the awesome research works [Prompt-to-Prompt](https://github.com/google/prompt-to-prompt), [T2I-Adapter](https://github.com/TencentARC/T2I-Adapter).


## Citation

```bibtex
@InProceedings{cao_2023_masactrl,
    author    = {Cao, Mingdeng and Wang, Xintao and Qi, Zhongang and Shan, Ying and Qie, Xiaohu and Zheng, Yinqiang},
    title     = {MasaCtrl: Tuning-Free Mutual Self-Attention Control for Consistent Image Synthesis and Editing},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {22560-22570}
}
```


## Contact

If you have any comments or questions, please [open a new issue](https://github.com/TencentARC/MasaCtrl/issues/new/choose) or feel free to contact [Mingdeng Cao](https://github.com/ljzycmd) and [Xintao Wang](https://xinntao.github.io/).
