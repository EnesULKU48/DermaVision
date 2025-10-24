# 🌿 DermaVision - No-Code Skin Type Analysis Plugin

> *"Your skin tells a story — and technology is finally listening."*

---

## 📖 The Story

Every woman sees a story in the mirror. Sometimes it looks tired, sometimes glowing, sometimes hiding. But it should never remain silent.

DermaVision was born to **amplify that voice**. We didn't see dryness, oiliness, wrinkles, or dark spots as flaws — we saw them as **data**. Because technology only becomes meaningful when it touches not just numbers, but the **human essence**.

More than a camera, beyond an analysis... We built a technology that doesn't just look at your skin — it **understands** it.

### 🎯 What DermaVision Does

**Current Version:**
- **Skin Type Classification:** Detects dry, normal, or oily skin type
- **Confidence Score:** Provides prediction confidence (0.00 - 1.00)

**Future Vision:**
- Advanced feature analysis (wrinkles, dark circles, pigmentation, texture)
- Multi-dimensional skin health scoring
- Personalized recommendations based on analysis

> **This is an early-stage project with significant room for improvement. We need the community's help to make it better!**

---

## 🚀 Why No-Code?

- ⚡ Go from idea → prototype in minutes
- 🌟 Focus on insights, not code
- 🤝 Built for everyone: designers, dermatologists, researchers
- 🔧 Visual debugging of every block
- 📱 Instant webcam-based testing

---

## ✨ Key Features

- ⏺ Real-time webcam-based **skin type classification**
- 🧬 Transformer-powered DINOv2 backbone
- 📂 No-code environment: **AugeLab Studio**
- 🔹 Custom Plugin Blocks:
  - `SingleFrameCapture` — capture when triggered
  - `SkinTypeClassifier` — classify skin type
- 🔦 Live visualization and block-based connections
- 💡 Runs lightweight on local machines

---

## 🤖 How It Works

### Architecture Overview

**SingleFrameCapture** saves the current webcam frame only on trigger (e.g., key press)

**SkinTypeClassifier** loads the `best_dino_sweaty.pth` model and processes the frame

**Outputs:**
- `Skin Type`: "dry", "normal", or "oily"
- `Confidence`: float (0.00 – 1.00)

---

## 📦 Installation & Setup

### Prerequisites

- Windows 10/11
- [AugeLab Studio](https://augelab.ai) installed
- Webcam or camera device

### Clone This Repository

```bash
git lfs install
git clone https://github.com/futureactionai/DermaVision.git
```

### Project Structure

```
DermaVision/
├── augelab/
│   ├── SingleFrameCapture.py
│   └── SkinTypeClassifier.py
├── model/
│   ├── best_dino_sweaty.pth
│   ├── requirements.txt
│   └── train_dinov2_finetune.py
├── .gitattributes
├── .gitignore
├── DermaVision.pmod
├── LICENSE
└── README.md
```

### Add Blocks to AugeLab

1. Navigate to:
   ```
   C:\Users\<YOUR_USERNAME>\AppData\Roaming\AugeLab Studio\marketplace\custom_blocks
   ```
2. Copy the `.py` files from `augelab/` folder
3. Restart **AugeLab Studio**
4. Open `DermaVision.pmod` file to run the plugin

---

## 🔬 Model Training (Optional)

### Dataset Folder Format

```
dataset/
├── dry/
├── normal/
└── oily/
```

### Install Requirements

```bash
pip install -r model/requirements.txt
```

### Train Your Own Model

```bash
python model/train_dinov2_finetune.py
```

**Output:** `best_dino_sweaty.pth`

#### Training Datasets

The model was trained on the following Kaggle datasets:
- 🔗 [Oily & Dry Skin Dataset](https://www.kaggle.com/datasets/manithj/oily-and-dry-skin-dataset)
- 🔗 [Normal/Dry/Oily Skin Type](https://www.kaggle.com/datasets/ritikasinghkatoch/normaldryoily-skin-type)

---

## ⚠️ Current Limitations & Development Status

**This is an experimental project with known issues:**

- ❌ Model accuracy is **not reliable** in real-world conditions
- ❌ Only basic skin type classification (dry/normal/oily)
- ❌ Lighting, camera quality, and angles heavily affect results
- ❌ No advanced feature analysis yet (wrinkles, texture, etc.)
- ❌ Limited training data diversity

### 🛠️ What We're Working On

This project needs significant improvements. You can:
- **Use the pre-trained model** (`best_dino_sweaty.pth`) to test and experiment
- **Fine-tune your own model** using `train_dinov2_finetune.py` with better datasets
- **Contribute improvements** to make this project production-ready

**We openly acknowledge this is a work in progress and welcome all contributions!**

---

## 💚 Open Source Contribution

DermaVision is community-driven and welcomes all contributions!

### How You Can Help

- 🎯 **Improve Models:** Train with more diverse datasets
- 🚀 **Add Features:** Create new detection blocks (wrinkles, acne, pores, etc.)
- 📖 **Documentation:** Help us write better guides
- 🔹 **Share Use Cases:** Show us what you've built
- 🤝 **Submit PRs:** Evolve this project together

> *Fork the repo, create your block, and share your vision.*

---

## 📸 Demo

![DermaVision Demo](https://github-production-user-asset-6210df.s3.amazonaws.com/234184906/505447875-1f37f857-1caf-4f92-accc-417c716b21cf.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20251024%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20251024T183450Z&X-Amz-Expires=300&X-Amz-Signature=e76dc0f0ce10c33e708d9d84210e15c36363af1719f9e65b6fb37e828d8e2f0a&X-Amz-SignedHeaders=host)


---


## 🤝 Join the Community

- 💡  Report bugs or suggest improvements
- 💬 Share results and ideas
- ⭐ **Star this repo** to stay updated
- 🔔 **Watch** for new releases

---

### 📞 Contact

For questions or collaboration opportunities, feel free to open an issue or reach out through GitHub Discussions.

---

> *"If you want to understand skin, let it speak through data."*

**Let's build a smarter, kinder skincare future — together.** 🚀
