# LVLM-DFD: Unlocking the capabilities of large vision-language models for generalizable and explainable deepfake detection

This repository contains the official implementation of the ICML paper **"Unlocking the capabilities of large vision-language models for generalizable and explainable deepfake detection"**.

## 🔥 Overview

This work introduces a novel approach for deepfake detection that leverages the power of Large Vision-Language Models (LVLMs). Our method combines visual understanding with natural language reasoning to provide both accurate detection and explainable results for deepfake detection.


## 📋 Requirements

### Dependencies

```bash
pip install -r requirements.txt
```

### Additional Requirements

- CUDA-compatible GPU with at least 16GB VRAM
- Python 3.8+
- PyTorch 1.13+

## 🚀 Quick Start

### 1. Model Setup

Download the pre-trained checkpoints:

```bash

Download our fine-tuned checkpoint from https://pan.baidu.com/s/1jPgpi-zluxeXGUAPrqJv0Q?pwd=iuy7
Place in: checkpoint/ckpt.pth
```

### 2. Prepare Input Images

Place your test images in the `input/` directory:

```
input/
├── 1_img.png
├── 2_img.png
├── 3_img.png
└── 4_img.png
```

### 3. Configuration

Update the model paths in `inference.py`:

```python
args = {
    'model': 'openllama_peft',
    'ckpt_path': 'checkpoint/ckpt.pth',  # Update this path
    'max_tgt_len': 128,
    'lora_r': 32,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
}
```

## 📁 Project Structure

```
code/
├── inference.py              # Main inference script
├── model/
│   ├── openllama.py         # Main LVLM model implementation
│   ├── AnomalyGPT_models.py # Anomaly detection modules
│   ├── clip/                # CLIP model components
│   └── ImageBind/           # ImageBind multimodal encoder
├── input/                   # Input images directory
├── output/                  # Output results directory
└── checkpoint/              # Model checkpoints directory
```

## 📊 Usage Examples

python inference.py

## 📝 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{yu2025unlocking,
  title={Unlocking the Capabilities of Large Vision-Language Models for Generalizable and Explainable Deepfake Detection},
  author={Yu, Peipeng and Fei, Jianwei and Gao, Hui and Feng, Xuan and Xia, Zhihua and Chang, Chip Hong},
  journal={arXiv preprint arXiv:2503.14853},
  year={2025}
}
```

## 🙏 Acknowledgments

This work builds upon several excellent open-source projects:
- [AnomalyGPT](https://github.com/CASIA-IVA-Lab/AnomalyGPT) - Detecting Anomalies using Large Vision-Language Models
- [ImageBind](https://github.com/facebookresearch/ImageBind) - Multi-modal encoder


