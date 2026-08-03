# 🧬 VEP: Hybrid Multimodal Genomic Variant Pathogenicity Predictor

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end deep learning framework for classifying human genetic variant pathogenicity (ClinVar GRCh38). **VEP** combines sequence-level 1D Convolutional Neural Networks (CNN) with dense Multi-Layer Perceptrons (MLP) fed by clinical and population continuous biomarkers to break performance bottlenecks of single-modality predictors.

---

## 🌟 Key Highlights & Performance

* **🎯 84% Test Classification Performance:** Resolved class imbalance and baseline plateau issues using class-weighted binary cross-entropy and multimodal late-fusion.
* **⚡ Dual-Arm Multimodal Core (Module 1):**
  * **Arm 1 (1D CNN):** Extracts structural allele switch patterns across 8 binary sequence encoding channels.
  * **Arm 2 (Dense MLP):** Processes 5 engineered non-redundant clinical biomarkers (Submitter Consensus, Origin Inheritance Mode, ClinVar Review Stars, Somatic Impact, and Gene Pathogenicity Density).
* **🌐 Production REST API:** Includes a lightweight, decoupled API gateway (`Flask` / `FastAPI`) for real-time inference and integration with frontend dashboards.

---

## 🏗️ System Architecture

```text
 ┌─────────────────────────────────────────────────────────────┐
 │                    Input Variant Sample                     │
 └──────────────────────────────┬──────────────────────────────┘
                                │
         ┌──────────────────────┴──────────────────────┐
         ▼                                             ▼
  [Sequence Arm]                                [Tabular Arm]
  8 Binary Allele Switches                      5 Clinical Biomarkers
  (Ref/Alt A, C, G, T)                          (Submitters, Origin, Stars, etc.)
         │                                             │
         ▼                                             ▼
  ┌──────────────┐                              ┌──────────────┐
  │  1D CNN Core │                              │  Dense MLP   │
  └──────┬───────┘                              └──────┬───────┘
         │ (32 Vector)                                 │ (16 Vector)
         └──────────────────────┬──────────────────────┘
                                │ Concatenation (48 Vector)
                                ▼
                        ┌──────────────┐
                        │ Late Fusion  │
                        └──────┬───────┘
                                │
                                ▼
                   [ Pathogenicity Output ]
                    (Benign vs. Pathogenic)
```
# VEP

## 📂 Repository Structure

```text
VEP/
├── data/
│   ├── raw/                      # NCBI ClinVar variant summary files
│   └── processed/                # Compressed, chunked Parquet feature matrices
├── notebooks/                    # Model design, data engineering & training experiments
├── app.py                        # Production REST API script for model inference
├── multi_armed_model_84acc.pth   # Saved PyTorch model weights (84% accuracy)
├── model_config.json             # Preprocessing feature mapping schema
├── requirements.txt              # Dependency declarations
└── README.md                     # Project documentation
```

---

## 🧪 Tabular Feature Matrix (Arm 2)

| Feature Name | Biological / Clinical Significance | Engineering Applied |
|--------------|------------------------------------|---------------------|
| **Tab_Scaled_Submitters** | Measures consensus depth across laboratories | Log1p transformation (`NumberSubmitters`) |
| **Tab_Origin_Numeric** | Differentiates inherited vs. spontaneous variants | Numeric encoding (`OriginSimple`) |
| **Tab_Review_Score** | Standardized star rating of curation quality | Ordinal star mapping (`ReviewStatus`) |
| **Tab_Somatic_Impact** | Identifies documented tumor driver mutations | Binary flag (`SomaticClinicalImpact`) |
| **Tab_Gene_Vulnerability** | Measures gene intolerance to mutations | Dataset pathogenicity density ratio |

---

# 🚀 Quickstart Guide

## 1. Clone & Install Dependencies

```bash
git clone https://github.com/jayatigupta05/VEP.git
cd VEP
pip install -r requirements.txt
```

## 2. Run the Live Inference API

```bash
python app.py
```

The API server will launch locally at:

```
http://127.0.0.1:5000
```

(or port **8000**, depending on your configuration).

---

## 3. Send a Sample Prediction Request

Send a **POST** request to `/predict`.

### Request

```json
{
  "ref_allele": "A",
  "alt_allele": "G",
  "number_submitters": 8,
  "origin_simple": "germline",
  "review_stars": 2.0,
  "has_somatic_impact": false,
  "gene_vulnerability": 0.85
}
```

### Response

```json
{
  "confidence_score": "84.12%",
  "pathogenicity_probability": 0.8412,
  "prediction": "Pathogenic",
  "status": "success"
}
```

---

## 🔮 Future Roadmap (Module 2)

- **Relational Graph Neural Network (GNN):** Integrate PyTorch Geometric to construct chromosome neighborhood graphs and model inter-gene relationships using message passing.
- **Interactive Frontend Visualizer:** Develop a web-based interface for variant exploration and graph visualization.

---

## 📜 License

Distributed under the **MIT License**. See the `LICENSE` file for more information.