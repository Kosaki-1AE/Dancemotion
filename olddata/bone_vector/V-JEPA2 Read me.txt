Thanks for the detailed info. The issue you're facing —
`AssertionError: Torch not compiled with CUDA enabled` — means **your PyTorch installation doesn't support CUDA**, despite having GPUs available on the system.

---

### ✅ Let's fix it step-by-step **from the beginning** to successfully run the `.ipynb` notebook using GPU:

---

## 🔧 **Step 1: Create and Activate Conda Environment**

Create a new Conda environment using Python 3.12:

```bash
conda create -n ai python=3.12 -y
conda activate ai
```

---

## ⚙️ **Step 2: Install CUDA-Compatible PyTorch**

Check your system supports **CUDA 12.1** (you already confirmed this).

Install GPU-compatible PyTorch using this command:


> 🔍 You must install it via the official CUDA-specific index URL, otherwise you’ll get CPU-only PyTorch by default — which caused your `AssertionError`.
'''bash
!pip uninstall -y torch torchvision torchaudio
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Restart Kernal
Test installation

import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

cmd--
nvidia-smi
---

## 📦 **Step 3: Install Required Packages**

Now install the rest of the packages:

```bash
pip install einops timm pillow
pip install git+https://github.com/huggingface/transformers
pip install git+https://github.com/huggingface/accelerate
pip install git+https://github.com/huggingface/diffusers
pip install huggingface_hub
pip install sentencepiece bitsandbytes protobuf decord numpy
```

---

## 📁 **Step 4: Clone the Repository**

```bash
git clone https://github.com/facebookresearch/vjepa2.git
cd vjepa2
pip install -e .
```

---

## 🔐 **Step 5: HuggingFace Login**

```bash
huggingface-cli login
```

Paste your token:

```
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
```

---

## 🎥 **Step 6: Download a Sample Video**

```bash
wget https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/bowling/-WH-lxmGJVY_000005_000015.mp4 -O "sample_video.mp4"
```

---

## 📓 **Step 7: Jupyter and IPywidgets Setup**

```bash
conda install -c conda-forge notebook ipywidgets -y
```

Then start your notebook:

```bash
jupyter notebook
```

---

## 🧠 **Step 8: Model Weights Configuration**

Update the following line in your notebook:

```python
pt_model_path = "YOUR_MODEL_PATH"
```

Make sure it points to the **local pretrained weights file** for the PyTorch model (a `.pt` or `.pth` file).
If you don’t have one, you can try to download from an official/pretrained source or convert from HuggingFace if applicable.

wget https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt -P YOUR_DIR  (Large file error wget:Read follow python code to download)
wget https://dl.fbaipublicfiles.com/vjepa2/evals/ssv2-vitg-384-64x2x3.pt -P YOUR_DIR
---

## ✅ You Should Now Be Able to Run This Cell Without Error:

```python
model_hf = AutoModel.from_pretrained(hf_model_name)
model_hf.cuda().eval()
```



vitg-384.pt

import requests

url = "https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt"
filename = "vitg-384.pt"

with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open(filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

print("Download completed.")


ssv2_classes.json

import json
import requests

url = "https://huggingface.co/datasets/huggingface/label-files/resolve/d79675f2d50a7b1ecf98923d42c30526a51818e2/something-something-v2-id2label.json"
response = requests.get(url)

if response.status_code == 200:
    with open("ssv2_classes.json", "w", encoding="utf-8") as f:
        json.dump(response.json(), f, ensure_ascii=False, indent=2)
    print("✅ SSV2 classes JSON saved as ssv2_classes.json")
else:
    print(f"❌ Failed to download. Status code: {response.status_code}")

