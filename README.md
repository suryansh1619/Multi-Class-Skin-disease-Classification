# 🩺 MultiClass Skin Disease Classification

A deep learning-based web application for classifying skin diseases from lesion images using multiple state-of-the-art CNN models with preprocessing and visual performance analysis.   
Video link [https://drive.google.com/file/d/1ljeRBnuP5HUiupC1hOVyxMMJayxR8J5g/view?usp=drive_link](https://drive.google.com/file/d/1q-4zpYVMbxuX5QPBRzSSjd4raeVlAThq/view?usp=drive_link)
---

## 🔍 Project Overview

This project leverages six powerful CNN architectures to predict skin disease types from lesion images. It incorporates advanced preprocessing including hair removal and contrast enhancement to improve classification accuracy.

The web application is built using **Streamlit** and provides a user-friendly interface to upload an image and view top-2 class predictions along with their probabilities for each model.

---

## 🚀 Demo

Upload an image and get real-time predictions from:

- ✅ ConvNeXtSmall
- ✅ DenseNet201
- ✅ EfficientNetV2S
- ✅ InceptionV3
- ✅ ResNet50V2
- ✅ Xception

Each model displays:

- 🔹 Top-1 prediction class and confidence
- 🔹 Top-2 prediction class and confidence

---

## 🧠 Models Used

These models were trained and fine-tuned on the **HAM10000** dataset. They classify images into one of the following **7 skin disease classes**:

| Label  | Description                          |
|--------|--------------------------------------|
| akiec  | Actinic keratoses                    |
| bcc    | Basal cell carcinoma                 |
| bkl    | Benign keratosis-like lesions        |
| df     | Dermatofibroma                       |
| mel    | Melanoma                             |
| nv     | Melanocytic nevi                     |
| vasc   | Vascular lesions                     |

---

## 📈 Model Evaluation & Visualizations

Each model has been evaluated using the following metrics:

- **Accuracy**
- **Confusion Matrix**
- **Classification Report**
- **ROC-AUC Curve (One-vs-Rest, Macro Averaged)**
- **Precision-Recall Curve**

---

### ✅ Model Performance Summary (Top-1 Accuracy)

| Model             | Accuracy (%) |
|-------------------|--------------|
| ConvNeXtSmall     | 89.1         |
| DenseNet201       | 88.9         |
| EfficientNetV2S   | 89.5         |
| InceptionV3       | 85.6         |
| ResNet50V2        | 83.9         |
| Xception          | 88.7         |

---
