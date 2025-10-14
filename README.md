# 🦴 Bone Fracture Detection (X‑ray)

A carefully engineered **computer vision** project for detecting **bone fractures** from X‑ray images.  
The pipeline emphasizes **patient‑wise splitting**, robust preprocessing/augmentation, **CNN backbones** (e.g., DenseNet/EfficientNet via `timm`) for **classification**, and an optional **object detection** path (YOLOv8) for **localizing** fracture regions.  
Evaluation follows medical‑AI best practices: **AUROC, mAP, sensitivity@specificity**, per‑class thresholds, and **Grad‑CAM** explainability.

> ⚠️ **Medical disclaimer**: This code is for **research & education** only and must **not** be used for clinical decision‑making.

---

## 🧠 Problem
Given a musculoskeletal X‑ray, determine whether a **fracture** is present (and optionally **where**).  
Two complementary tracks are supported:
- **Image classification**: fracture vs. no‑fracture (or multi‑class by bone/region).
- **Object detection** *(optional)*: localize suspected fracture regions with bounding boxes.

This is a **class‑imbalanced** task; metrics must be reported **per‑class** and with proper threshold tuning.

---

## 🗂️ Dataset
- **Input**: AP/LAT X‑ray images (PNG/JPEG or DICOM).  
- **Labels**: binary (`fracture` / `normal`) or multi‑class (`humerus`, `radius/ulna`, `femur`, `tibia/fibula`, `hand`, etc.).  
- **Splits**: **patient‑wise** train/val/test to avoid leakage (no patient appearing in multiple splits).  
- **Recommended image size**: 320–512 px (trade‑off accuracy vs. speed).

> Replace links/paths with the exact dataset you used and cite the source/license.

---

## 🧰 Pipeline Overview
1. **Load & split** by *patient ID*.  
2. **Preprocess**: convert to RGB, histogram equalization / CLAHE (optional), normalize to ImageNet mean/std.  
3. **Augment** (train): RandomResizedCrop/Resize, HorizontalFlip (careful with laterality), Rotation (≤10°), Brightness/Contrast (mild).  
4. **Backbone**: `timm` models (e.g., **EfficientNet‑B0/B3**, **DenseNet121**) with global pooling and **sigmoid/softmax** head.  
5. **Loss**: `BCEWithLogitsLoss` (multi‑label) or `CrossEntropyLoss` (multi‑class); **Focal loss** optional for imbalance.  
6. **Optimization**: AdamW + cosine schedule / ReduceLROnPlateau; AMP enabled.  
7. **Thresholds**: tune per‑class thresholds on **validation** to maximize F1 or Youden‑J.  
8. **Explainability**: **Grad‑CAM** for positive predictions.  
9. **(Optional) Detection**: YOLOv8 for bounding‑box localization; evaluate with **mAP@0.5:0.95**.

---

## 📈 Metrics & Reporting
For **classification**:
- **AUROC (per‑class)**, **macro AUROC**, **micro AUROC**  
- **Average Precision (mAP)** for multi‑label setups  
- **Sensitivity / Specificity** at tuned thresholds

For **detection** (optional):
- **mAP@0.5**, **mAP@0.5:0.95**, per‑class AP

**Example Results (replace with your numbers):**
| Task | Model | AUROC (macro) | mAP (cls) | Sen@Spec=0.80 |
|---|---|---:|---:|---:|
| Classification | EfficientNet‑B3 | 0.93 | 0.58 | 0.77 |
| Detection | YOLOv8‑s | — | **0.41** (bbox) | — |

---

## 🧩 Repository Structure (suggested)
```
Bone-Fracture-Detection/
├─ notebooks/
│  ├─ 01_explore.ipynb
│  ├─ 02_train_classification.ipynb
│  ├─ 03_eval_cam.ipynb
│  └─ 04_yolov8_detection.ipynb         # optional
├─ src/
│  ├─ data.py          # Dataset/Dataloader (patient-wise split)
│  ├─ transforms.py    # Albumentations/torchvision augmentations
│  ├─ models.py        # timm backbones → classification heads
│  ├─ losses.py        # BCE/Focal; class-balanced weights
│  ├─ train.py         # train loop (AMP, early stop, checkpoint)
│  ├─ eval.py          # AUROC/AP, threshold sweep, Sen@Spec
│  ├─ cam.py           # Grad-CAM visualizations
│  ├─ detect_yolo.py   # optional: YOLOv8 inference/eval
│  └─ utils.py
├─ configs/            # YAMLs (model/img_size/augmentation)
├─ reports/figures/    # Curves, CAMs, detections
├─ data/               # (gitignored) images, metadata, splits
├─ models/             # (gitignored) checkpoints
├─ requirements.txt
├─ .gitignore
└─ README.md
```

---

## ⚙️ Installation
```bash
git clone https://github.com/ziaee-mohammad/Bone-Fracture-Detection.git
cd Bone-Fracture-Detection
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**requirements.txt (example)**
```
torch
torchvision
timm
albumentations
opencv-python
pandas
numpy
scikit-learn
matplotlib
seaborn
pytorch-grad-cam
ultralytics    # optional: YOLOv8 for detection
pyyaml
```

---

## 🚀 Usage

### 1) Train (classification)
```bash
python -m src.train   --data_dir data/fracture_xray   --train_csv splits/train.csv   --val_csv   splits/val.csv   --model     efficientnet_b3   --img_size  384   --batch     16   --loss      bce   --epochs    25   --amp
```

### 2) Evaluate & Threshold Tuning
```bash
python -m src.eval   --ckpt models/effb3_best.pt   --val_csv splits/val.csv   --metrics auroc ap   --tune_threshold f1
```

### 3) Grad‑CAM
```bash
python -m src.cam   --ckpt models/effb3_best.pt   --image path/to/xray.png   --target fracture   --save  reports/figures/cam_fracture.png
```

### 4) Detection (optional)
```bash
python -m src.detect_yolo   --model yolov8s.pt   --source data/fracture_xray/images/val   --save  reports/figures/dets/
```

---

## 🔬 Implementation Notes
- Always ensure **patient‑wise** split; avoid view‑level leakage.  
- Prefer mild augmentations suitable for radiographs; avoid heavy geometric transforms.  
- Track **class prevalence** and consider **loss weighting** / **Focal loss** for rare classes.  
- **Per‑class thresholds** improve downstream utility compared to a single global threshold.  
- Consider **calibration** (temperature scaling / isotonic) for better probability estimates.

---

## 🔐 Ethics & Privacy
- Remove PHI and anonymize images/metadata.  
- Obey dataset licenses and institutional review requirements.  
- This project is **not** a medical device.

---

## 👤 Author
**Mohammad Ziaee** — Computer Engineer | AI & Data Science  
📧 moha2012zia@gmail.com  
🔗 https://github.com/ziaee-mohammad

---

## 🏷 Tags
```
data-science
machine-learning
deep-learning
computer-vision
medical-imaging
xray
fracture-detection
multi-label-classification
pytorch
grad-cam
yolov8
timm
```
