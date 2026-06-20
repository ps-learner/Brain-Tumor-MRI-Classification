# 🧠 NeuroScan AI — Brain Tumor MRI Classification

An end-to-end deep learning project for **brain tumor MRI image classification** using a **Custom CNN**, **EfficientNetB0**, **MobileNetV2**, and **InceptionV3**, followed by deployment in a medically themed **Streamlit** application with **Grad-CAM explainability**.

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Real-World Use Cases](#real-world-use-cases)
- [Project Highlights](#project-highlights)
- [Architecture Flow](#architecture-flow)
- [Repository Structure](#repository-structure)
- [Dataset Description](#dataset-description)
- [Model Results](#model-results)
- [Grad-CAM Explainability](#grad-cam-explainability)
- [App Screenshots](#app-screenshots)
- [How to Run](#how-to-run)
- [Deployment](#deployment)
- [Tech Stack](#tech-stack)
- [Clinical Disclaimer](#clinical-disclaimer)
- [Author](#author)
- [License](#license)

---

## Problem Statement

Brain tumors can present with subtle visual patterns in MRI scans, and early identification is clinically important. This project builds a deep learning-based MRI classifier that predicts whether a brain image belongs to one of four classes:

- **glioma**
- **meningioma**
- **pituitary**
- **notumor**

The goal is not to replace radiologists, but to build a strong **AI-assisted screening and educational decision-support system** that demonstrates technical rigor, explainability, and deployment readiness.

---

## Real-World Use Cases

| Use Case | Stakeholder | Value |
|---|---|---|
| MRI screening support | Radiologists | Faster first-pass categorization |
| Academic medical AI demo | Evaluators / recruiters | End-to-end project depth |
| Resource-constrained clinical settings | Hospitals | Lightweight inference options with MobileNetV2 |
| Explainable AI showcase | Researchers | Grad-CAM visualization of model attention |

---

## Project Highlights

- Google Colab GPU training workflow
- Drive-mounted dataset pipeline
- Custom CNN from scratch
- Transfer learning with EfficientNetB0, MobileNetV2, InceptionV3
- Class imbalance handling using class weights
- Data augmentation for robust training
- Evaluation with confusion matrix, ROC-AUC, precision, recall, and F1-score
- Grad-CAM visual explanations
- Multi-page Streamlit app with medical-themed UI
- Recruiter-grade GitHub repository structure

---

## Architecture Flow

```text
                  ┌──────────────────────────┐
                  │   Brain MRI Image Input  │
                  └────────────┬─────────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │  Resize + Normalize      │
                  │  224x224 RGB             │
                  └────────────┬─────────────┘
                               │
                               ▼
            ┌────────────────────────────────────────┐
            │          Model Candidates              │
            │ Custom CNN | EfficientNetB0 |          │
            │ MobileNetV2 | InceptionV3              │
            └────────────┬───────────────────────────┘
                         │
                         ▼
              ┌──────────────────────────┐
              │  Softmax Classification  │
              │  4 Classes               │
              └────────────┬─────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        ▼                                     ▼
┌─────────────────────┐             ┌─────────────────────┐
│ Confidence Scores   │             │ Grad-CAM Heatmap    │
└─────────────────────┘             └─────────────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │ Streamlit Clinical UI    │
              └──────────────────────────┘
```

---

## Repository Structure

```text
brain-tumor-mri-classification/
│
├── Brain_Tumor_MRI_Classification.ipynb
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── .streamlit/
│   └── config.toml
│
├── models/
│   ├── custom_cnn_best.h5
│   ├── efficientnetb0_best.h5
│   ├── mobilenetv2_best.h5
│   ├── inceptionv3_best.h5
│   ├── model_comparison.csv
│   └── meta.json
│
├── plots/
│   ├── 01_class_distribution.png
│   ├── 02_sample_images_grid.png
│   ├── 03_augmented_samples.png
│   ├── 04_custom_cnn_training_history.png
│   ├── 05_efficientnetb0_training_history.png
│   ├── 06_mobilenetv2_training_history.png
│   ├── 07_confusion_matrix_custom_cnn.png
│   ├── 08_confusion_matrix_efficientnet.png
│   ├── 09_confusion_matrix_mobilenet.png
│   ├── 10_roc_curves.png
│   ├── 11_model_comparison_bar.png
│   └── 12–15 gradcam plots
│
├── screenshots/
│   ├── home_dashboard.png
│   ├── dataset_explorer.png
│   ├── model_results.png
│   ├── mri_classifier_upload.png
│   ├── mri_classifier_result.png
│   └── gradcam_view.png
│
└── data/
    └── .gitkeep
```

---

## Dataset Description

| Class | Clinical Meaning |
|---|---|
| glioma | Often aggressive malignant tumor |
| meningioma | Usually benign tumor on protective lining |
| notumor | Healthy MRI with no visible tumor |
| pituitary | Tumor at the base of the brain |

> Dataset should be stored externally and **not committed** to GitHub due to size.

---

## Model Results

| Model | Test Accuracy | Macro F1 | Parameters | Best For |
|---|---:|---:|---:|---|
| Custom CNN | Add after training | Add after training | Lower | Baseline |
| EfficientNetB0 | Add after training | Add after training | Moderate | Accuracy |
| MobileNetV2 | Add after training | Add after training | Low | Speed |
| InceptionV3 | Add after training | Add after training | High | Robustness |

Update this table after running the notebook and exporting `models/model_comparison.csv`.

---

## Grad-CAM Explainability

Grad-CAM helps visualize **where the model focuses** in the MRI when making a prediction. This improves interpretability and strengthens the project’s value as a medical AI demo.

Use Grad-CAM outputs to answer:
- Is the model focusing on anatomically plausible tumor regions?
- Is it relying on irrelevant borders or imaging artifacts?
- Can we present AI predictions more transparently to evaluators?

---

## App Screenshots

Add screenshots after running the Streamlit app:

- `screenshots/home_dashboard.png`
- `screenshots/dataset_explorer.png`
- `screenshots/model_results.png`
- `screenshots/mri_classifier_upload.png`
- `screenshots/mri_classifier_result.png`
- `screenshots/gradcam_view.png`

---

## How to Run

### 1) Train in Google Colab
- Open `Brain_Tumor_MRI_Classification.ipynb`
- Mount Google Drive
- Set dataset path
- Enable GPU: `Runtime > Change runtime type > GPU`
- Run all cells

### 2) Local Streamlit Setup

#### Windows
```bash
python -m venv brain_tumor_env
brain_tumor_env\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

#### macOS / Linux
```bash
python3 -m venv brain_tumor_env
source brain_tumor_env/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## Deployment

### Streamlit Cloud
1. Push repository to GitHub
2. Ensure `app.py` is in root
3. Add `.streamlit/config.toml`
4. Upload or link model artifacts
5. Deploy through [Streamlit Community Cloud](https://streamlit.io/cloud)

> If `.h5` files exceed GitHub limits, use **Git LFS** or place model download links in the README.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Programming | Python |
| Deep Learning | TensorFlow, Keras |
| CV / Image Ops | OpenCV, Pillow |
| Analysis | NumPy, Pandas, Scikit-learn |
| Visualization | Matplotlib, Seaborn, Plotly |
| Explainability | tf-keras-vis |
| Deployment | Streamlit |
| Training Platform | Google Colab |

---

## Clinical Disclaimer

This project is intended for **educational, research, and demonstration purposes only**. It is **not a diagnostic medical device**, is **not FDA-approved**, and must **not** be used as a substitute for clinical judgment, radiologist interpretation, or medical decision-making.

---

## Author

**Pratyusha Sharma**   
GitHub: [Add your GitHub link]  
LinkedIn: [Add your LinkedIn link]

---

## License

This project is licensed under the **MIT License**.