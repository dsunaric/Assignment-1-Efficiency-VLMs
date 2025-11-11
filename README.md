# Assignment 1 - Evaluating efficient Vision Language Models (VLMs)
## Student Project @ TU Wien 191.021 Introduction to Computational Sustainability, 2025 Winter Semester
### Students: Patrick Ennemoser, Dragana Sunaric, Daniel Martin Pühringer

## 1. Project Introduction
This repository contains our experiments on LLava, a Vision-Language Model (VLM) used for image caption generation.
We evaluate model quality, efficiency, and energy consumption under different quantization configurations using Google Colab (T4 GPU).

Our goal is to measure the trade-offs between model performance (CIDEr) and computational efficiency (e.g., VRAM usage, latency, throughput) across varying precision levels.

## 2. Experiment Config
### 2.1. Specs
Runtime: Google Colab Pro (T4 GPU, 15 GB VRAM) with Python 3.12.12

### 2.2. Experiment setup
Our project consistes of three different experiences:
1. Baseline FP16 model:
   - [Source Code in Google Colab](https://colab.research.google.com/drive/1B_Fx7Wp7Eza_O8QZClDg8-R1Z033BlnE?usp=sharing)
   - [Source Code Notebook File](Exp1_FP16_Comp_Sust_assignment1.ipynb)
   - Normal configuration without any further adjustments
3. Weight-only 8-bit:
   - [Source Code in Google Colab](https://colab.research.google.com/drive/1WtK9vDrSsb6iOyisC0QKNbG8-_8KvsBd?usp=sharing)
   - [Source Code Notebook File](Exp2_8Bit_Only_Comp_Sust_assignment1.ipynb)
   - This configuration quantizes only the attention and MLP weights to INT8 (using BitsAndBytes) while keeping activations and the vision tower in FP16
   - This adjustment reduces model size and therefore memory usage by 50%
5. Quantized vision tower to INT8:
   - [Source Code in Google Colab](https://colab.research.google.com/drive/1zjTEqyJk8DRfc8iG29zNU0QBCbUsMQvB?usp=sharing)
   - [Source Code Notebook File](Exp3_Quant_INT8_Comp_Sust_assignment1.ipynb)
   - This configuration quantizes only the vision tower to INT8 (using BitsAndBytes)
   - This adjustment does not reduce the size of the model.

For each of these three experiments, we calculated the following metrics
- Quality: CIDEr score
    - The CIDEr Score is widely used for benchmarking image descriptions.
    - [Paper: "CIDEr: Consensus-based Image Description Evaluation"](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vedantam_CIDEr_Consensus-Based_Image_2015_CVPR_paper.pdf)
- Efficiency: VRAM, latency per image, throughput, model size
    - VRAM refers to Video RAM, which is dedicated to the GPU, and is controlled by the GPU. It is measured by _[torch.cuda.memory.max_mem](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_allocated.html)_
    - Latency per image and throughput indicate how long the inference of one images takes. The more tokens are being used, the longer the inference takes. The throughput is around 1-1.5 images (rough estimate) per second in the default setting with around 15+ tokens per second.
    - Model size refers to the size of the model, which is around 14 GB in the default setting. The model is described in the chapter below.
- Energy
    - This is a bonus metric and measures the CO2eq emissions in kg for the inference of the model
    - It is estimated using codecarbon, which uses nvidia-smi under the hood

The results are collected and visualized in [this Google Colab Notebook](https://colab.research.google.com/drive/1AbVfa65_UrQtIeBSyoSuNizjdDYq5ZYa?usp=sharing)

### 2.3. Model
Used model: LLaVA-1.5 [Github](https://llava-vl.github.io/)
LLaVA is a multimodal large language model that integrates a vision encoder (ViT) and a language decoder based on LLaMA. 

### 2.4. Quantizer
For quantization, we used the [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) library.

- **Baseline (FP16):** No quantization applied, model loaded with float16 precision.
- **Weight-only 8-bit:** Used `bitsandbytes`'s `LLM.int8()` quantization method.
  - Quantizes only the linear layers (attention and MLP weights) of the language model to INT8.
  - Vision encoder (ViT) remains in FP16.
- **Quantized Vision Tower:** Used `bitsandbytes` `linear8` quantization method to quantize the vision encoder (ViT) to INT8.
  - Language model remains in FP16.

### 2.5. Precision per module

| Experiment | Vision Encoder (ViT) | Language Model (LLaMA) | Attention / MLP Layers | Activations |
|-------------|----------------------|--------------------------|------------------------|--------------|
| Baseline FP16 | FP16 | FP16 | FP16 | FP16 |
| Weight-only 8-bit | FP16 | INT8 (weights only) | INT8 | FP16 |
| Quantized Vision Tower | INT8 | FP16 | FP16 | FP16 |


### 2.6. Dataset split & seed
Used Dataset: Coco 2017 val, [source](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset) 
Subset indices: first 100 images (subset 1-100)
Image preprocessing: no preprocessing needed, all images sizes in the Coco val 2017 dataset are smaller than 1024px (for comparability)

### 2.7. Hyper-Parameters
No hyper-parameters were used other than the parameters for quantization, which are explained in the experiments section.

