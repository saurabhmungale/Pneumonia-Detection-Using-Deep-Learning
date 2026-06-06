# 🫁 Pneumonia Detection Using Deep Learning (VGG19)

A deep learning project to detect **Pneumonia from Chest X-ray images** using Transfer Learning with VGG19.

---

## 📌 Project Overview

This project uses a pre-trained **VGG19 CNN model** (trained on ImageNet) combined with custom dense layers to classify chest X-ray images as either:
- ✅ **NORMAL**
- 🔴 **PNEUMONIA**

---

## 📂 Dataset

- **Source:** [Chest X-Ray Images (Pneumonia) - Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Split:** Train / Validation / Test
- **Classes:** NORMAL, PNEUMONIA

---

## 🏗️ Model Architecture

```
Input (128 × 128 × 3)
        ↓
VGG19 Base (Pre-trained, Frozen)
        ↓
Flatten
        ↓
Dense(4608, ReLU)
        ↓
Dropout(0.2)
        ↓
Dense(1152, ReLU)
        ↓
Dense(2, Softmax)  →  NORMAL / PNEUMONIA
```

- **Transfer Learning** — VGG19 base layers are frozen
- **Custom head** — trained specifically on chest X-ray data
- **Dropout** — added to reduce overfitting

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| TensorFlow / Keras | Model building & training |
| OpenCV | Image reading & preprocessing |
| VGG19 (ImageNet) | Transfer learning base |
| Matplotlib / Seaborn | Visualization |
| Kaggle API | Dataset download |

---

## 📊 Data Augmentation (Training Only)

| Technique | Value |
|---|---|
| Rescale | 1/255 |
| Horizontal Flip | Yes |
| Vertical Flip | Yes |
| Rotation Range | 40° |
| Width / Height Shift | 0.4 |
| Shear Range | 0.2 |
| Fill Mode | Nearest |

---

## 🚀 How to Run

1. Clone this repo
2. Open the notebook in **Google Colab**
3. Upload your `kaggle.json` API key
4. Run all cells in order

```bash
# Install dependencies
pip install kaggle tensorflow opencv-python
```

---

## 📁 Project Structure

```
pneumonia-detection-deep-learning/
│
├── pneumonia_detection.ipynb   # Main notebook
└── README.md                   # Project documentation
```

---

## 👤 Author

**Saurabh Mungale**  
B.E. Final Year | Data Science & Generative AI  
📎 [GitHub](https://github.com/saurabhmungale) | [LinkedIn](https://linkedin.com/in/saurabhmungale)
