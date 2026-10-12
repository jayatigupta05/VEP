# GenoHybrid-VEP
### A Multimodal Deep Learning Framework for Genomic Variant Classification

GenoHybrid-VEP is a research project exploring the classification of genomic variants using complementary sequence, tabular, and graph-based representations. The project combines a convolutional neural network (CNN), a multilayer perceptron (MLP), and a graph convolutional network (GCN), with fusion experiments designed to investigate whether combining these modalities improves classification.

The planned application layer will allow users to upload a VCF file through a web interface, process variants through a FastAPI backend, and view downloadable predictions and model explanations.

> **Project status:** Research pipeline and model experiments are under development. The FastAPI service and VCF-upload frontend described below are planned components and should not be considered implemented until they are available in the repository.

> **Research disclaimer:** This is a research prototype, not a clinical diagnostic system. Predictions must not be used as the sole basis for medical decisions.

---

## Contents

- [Overview](#overview)
- [Key goals](#key-goals)
- [Architecture](#architecture)
- [Model components](#model-components)
- [Data pipeline](#data-pipeline)
- [Validation and evaluation](#validation-and-evaluation)
- [Planned VCF upload application](#planned-vcf-upload-application)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Running the research pipeline](#running-the-research-pipeline)
- [Planned API design](#planned-api-design)
- [Input and output expectations](#input-and-output-expectations)
- [Reproducibility and data management](#reproducibility-and-data-management)
- [Limitations](#limitations)

## Overview

Interpreting genomic variants requires combining evidence from different sources. A local DNA sequence window can represent nucleotide context, structured annotations can encode known variant characteristics, and a graph can represent relationships defined between variants.

GenoHybrid-VEP investigates these sources through three model branches:

1. **Sequence branch:** a CNN learns local patterns from an encoded DNA sequence surrounding a variant.
2. **Tabular branch:** an MLP processes structured genomic annotations and derived features.
3. **Graph branch:** a GCN learns node representations using node features and a defined graph structure.

The project experiments with individual branches, two-branch combinations, prediction-level late fusion, and a tri-modal gated fusion model.

The long-term goal is to expose a trained and validated model through an application where a user uploads a compatible VCF file and receives a structured prediction report.

## Key goals

- Build a reproducible pipeline for preparing genomic variant data.
- Extract reference-sequence and evolutionary-conservation features.
- Train and compare sequence, tabular, and graph-based models.
- Investigate whether multimodal fusion improves predictive performance.
- Evaluate generalization using gene-wise validation where configured.
- Develop a FastAPI backend for VCF upload, validation, inference, and result retrieval.
- Build a frontend for submitting VCF files and inspecting prediction results.

## Architecture

### Research pipeline

```text
ClinVar variant data + GRCh38 reference resources
                    |
                    v
        Data cleaning and filtering
                    |
                    v
   Sequence and annotation feature extraction
                    |
                    v
        Processed variant feature table
                    |
          +---------+---------+
          |         |         |
          v         v         v
       Sequence  Tabular    Graph
         CNN       MLP       GCN
          |         |         |
          +---------+---------+
                    |
                    v
        Fusion experiments / gate
                    |
                    v
       Variant-class predictions
                    |
                    v
        Validation and evaluation
```

### Planned application

```text
User
 |
 v
Frontend: VCF upload and results interface
 |
 v
FastAPI backend
 |
 +--> File validation and parsing
 |
 +--> Variant normalization and feature extraction
 |
 +--> Load trained model artifacts
 |
 +--> Run inference and format predictions
 |
 v
Structured results and downloadable report
```

The application diagram describes the intended design. Its components will become available as implementation progresses.

## Model components

### 1. Sequence CNN

The sequence branch processes a nucleotide window surrounding each variant. The current research pipeline uses a 201-base reference window and includes a 101-base encoded input in the sequence-model workflow.

The sequence is converted into a numerical representation compatible with the CNN. Convolutional layers learn local sequence patterns that may be predictive of the target class.

### 2. Tabular MLP

The tabular branch processes structured features derived from variant metadata and genomic annotations. Features used by the model depend on the preprocessing configuration and availability in the dataset.

Potential feature sources in the current pipeline include:

- Evolutionary conservation scores from phyloP and phastCons.
- Molecular consequence annotations.
- Other derived variant-level features supported by the preprocessing code.

Feature transformations must be consistent between training and inference. Features that are unavailable must not be presented as observed measurements.

### 3. Graph convolutional network

The graph branch represents variants as nodes and uses the graph edges defined by the project to aggregate information from neighboring nodes.

The meaning of the graph depends on its actual construction procedure. Node identity, edge-generation rules, feature alignment, and partitioning must be documented and checked to avoid accidental label leakage.

### 4. Fusion experiments

The project investigates several approaches:

- Individual unimodal baselines.
- Sequence–tabular fusion.
- Tabular–graph fusion.
- Fixed-weight and validation-selected late fusion.
- Tri-modal gated fusion.

Fixed-weight late fusion combines predictions using specified coefficients. A learned gate can instead assign input-dependent weights according to its implemented architecture.

These methods are evaluated as separate experiments. Learned gate weights should not be confused with manually specified fusion coefficients.

## Data pipeline

The preprocessing workflow is organized around three research notebooks.

### Notebook 1 — Data pipeline

The data pipeline prepares the variant table and derives the features required by the model branches. Its configured workflow includes:

- Loading and filtering ClinVar variant data.
- Mapping the target labels used by the three-class task.
- Working with GRCh38 reference resources.
- Extracting sequence windows.
- Annotating evolutionary conservation using phyloP and phastCons tracks.
- Looking up molecular consequence annotations.
- Running integrity checks and saving the processed dataset.

The exact label mapping, filtering criteria, feature schema, and sample count are defined by the executed notebook and its configuration.

### Notebook 2 — Model development

The model-development notebook trains the individual branches and explores fusion strategies. It includes model definitions, training procedures, checkpoint handling, evaluation, and gate analysis.

### Notebook 3 — Validation

The validation notebook evaluates generalization using its configured validation protocol, including gene-wise grouping where implemented. Results should be reported with the exact split strategy, model configuration, and checkpoint provenance.

### Reference resources

The data pipeline may require large reference files, including:

- `hg38.2bit`
- `hg38.phyloP100way.bw`
- `hg38.phastCons100way.bw`
- ClinVar variant summary and VCF files

These files are intentionally kept outside the normal Git repository because of their size and licensing, distribution, and versioning considerations. Follow the data-source terms and document the exact versions used.

## Validation and evaluation

Model performance should be assessed using more than accuracy, particularly when class frequencies are imbalanced.

Recommended metrics include:

- Accuracy
- Macro-F1
- Balanced accuracy
- Per-class precision and recall
- Confusion matrix
- Multiclass ROC-AUC, where appropriate
- Per-class precision–recall analysis, where appropriate

### Avoiding information leakage

- Fit data-dependent preprocessing only on the training partition.
- Keep validation and test data out of training and model-selection operations as required by the evaluation protocol.
- Prevent related gene groups from crossing partitions when using gene-wise validation.
- Select fusion weights using the designated validation data, not the held-out test data.
- For stacking-style fusion, consider out-of-fold predictions for training the downstream fusion component.
- Report fold-level and aggregated out-of-fold results separately from development-set results.

No performance metric should be treated as a verified final result unless it is generated from the corresponding current run and documented evaluation procedure.

## Planned VCF upload application

The intended user workflow is:

1. Open the web application.
2. Upload a supported VCF file.
3. Review file-validation feedback.
4. Submit the file for processing.
5. Track the processing status.
6. View predictions and available supporting information.
7. Download the resulting report.

### Planned frontend

The frontend is intended to provide:

- VCF file selection and upload.
- File format and validation feedback.
- Processing status and error messages.
- A results table with variant coordinates and predicted classes.
- Available confidence scores or class probabilities.
- Downloadable prediction results.
- Clear limitations and a research-use disclaimer.

### Planned FastAPI backend

The backend is intended to provide:

- Upload validation and size limits.
- VCF parsing and variant-level input checks.
- Reference-build and allele compatibility checks.
- Feature extraction compatible with the training pipeline.
- Loading of versioned model and preprocessing artifacts.
- Batch inference.
- Structured JSON responses and downloadable results.
- Logging, error handling, and health checks.

**Important implementation requirement:** inference must use the same feature definitions, ordering, encodings, reference build, and preprocessing transformations used during training. A VCF upload endpoint alone does not make the research model production-ready.

## Repository structure

The structure below is a suggested organization for the research code and planned application. Adjust it to match the files that actually exist in the repository.

```text
GenoHybrid-VEP/
├── notebooks/
│   ├── GenoHybrid_VEP_Notebook_1_Data_Pipeline.ipynb
│   ├── GenoHybrid_VEP_Notebook_2_Foundation_and_Sequence_Encoding.ipynb
│   └── GenoHybrid_VEP_Notebook_3_Validation.ipynb
├── src/
│   └── # Reusable preprocessing, model, and inference modules
├── results/
│   └── # Evaluation tables and figures
├── checkpoints/
│   └── # Model checkpoints, stored separately if large
├── data/
│   └── # Small permitted sample data or schema examples only
├── frontend/
│   └── # Planned VCF upload and results UI
├── api/
│   └── # Planned FastAPI application
├── model_config.json
├── requirements.txt
├── .gitignore
└── README.md
```

Do not commit large raw datasets, genomic reference tracks, private genomic files, or oversized model checkpoints directly to GitHub. The actual directory names and notebook filenames may differ; update this tree to reflect the repository's real contents.

## Requirements

The research environment may include:

- Python 3.10 or later, subject to compatibility with the selected PyTorch and PyTorch Geometric versions.
- PyTorch
- PyTorch Geometric and compatible extension wheels, where required by the GCN implementation.
- pandas and NumPy
- scikit-learn
- Biopython or other VCF/reference-processing libraries used by the notebooks
- Libraries for reading BigWig and 2bit reference resources
- Jupyter or Google Colab

The planned API layer will require FastAPI, an ASGI server such as Uvicorn, and a supported VCF parsing library. The frontend framework and package requirements will be documented when selected.

The exact dependencies should be taken from the tested environment and pinned in the project dependency files. PyTorch, CUDA, and PyTorch Geometric versions must be compatible with one another.

## Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd GenoHybrid-VEP
```

Replace the placeholder with the repository's actual URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

If the repository contains a tested dependency file:

```bash
pip install -r requirements.txt
```

Install the appropriate PyTorch and PyTorch Geometric builds for your environment by following their compatibility instructions. Do not assume that one CUDA wheel configuration works on every Colab runtime or local machine.

### 4. Configure the data paths

Set the paths expected by the notebooks or configuration files. Download the required reference resources from their authoritative providers and place them in the configured directories.

Large reference resources and processed datasets are not included in this repository by default.

## Running the research pipeline

The following is the intended order of execution. Use the notebook names and paths present in the actual repository.

1. Run **Notebook 1** to prepare and validate the processed feature dataset.
2. Run **Notebook 2** to train the individual branches and execute fusion experiments.
3. Run **Notebook 3** to perform the configured validation procedure and generate evaluation artifacts.

Some stages can be computationally expensive, especially reference annotation, graph construction, and neural-network training. The required data files, checkpoints, and configuration must be available before running dependent stages.

The exact commands for standalone training and inference will be documented once reusable command-line entry points are implemented.

## Planned API design

The following endpoints are a proposed interface, not a claim that the endpoints currently exist.

| Method | Endpoint | Intended purpose |
|---|---|---|
| `GET` | `/health` | Check whether the API is running. |
| `POST` | `/api/v1/predictions` | Upload a VCF file and submit it for prediction. |
| `GET` | `/api/v1/jobs/{job_id}` | Retrieve processing status if asynchronous jobs are implemented. |
| `GET` | `/api/v1/results/{job_id}` | Retrieve completed results if job-based processing is implemented. |
| `GET` | `/api/v1/results/{job_id}/download` | Download a prediction report. |

For small inputs, synchronous inference may be sufficient. Larger VCF files may require background processing, persistent job state, and explicit upload and execution limits.

The API contract should specify supported file types, maximum upload size, accepted genome assemblies, error response formats, and how results are associated with a particular model version.

## Input and output expectations

### Input

The planned application accepts a VCF file compatible with the implemented preprocessing pipeline.

Before inference, the system must establish that it can interpret the file correctly, including the genome assembly, chromosome naming, genomic positions, reference alleles, and alternate alleles.

The first implementation should clearly document supported VCF versions and variant types. Variants that cannot be processed reliably should receive explicit validation errors or be reported as unsupported, rather than silently receiving predictions.

### Output

The intended result for each supported variant may include:

- Variant identifier or normalized variant key.
- Chromosome and position.
- Reference and alternate alleles.
- Predicted class.
- Per-class probabilities, if calibrated and exposed by the model.
- Processing status and any relevant warnings.
- Model version and reference-build metadata.

The output schema must reflect what the trained model actually produces. Probabilities should not be described as calibrated clinical risk estimates unless calibration has been evaluated.

## Reproducibility and data management

For reproducible experiments, record:

- Dataset source and retrieval date or version.
- Reference genome assembly and reference-resource versions.
- Filtering criteria and target-label mapping.
- Sample size, class distribution, and random seeds.
- Train/validation/test split strategy and gene-grouping rules.
- Feature schema and preprocessing configuration.
- Model architecture, optimizer, loss, epoch settings, and checkpoint-selection rule.
- Dependency versions and hardware/runtime configuration.
- Evaluation metrics and the source prediction files used to calculate them.

Keep code and lightweight documentation in Git. Keep large datasets, reference files, and large checkpoints in suitable external storage, and document how to obtain the exact artifacts needed to reproduce a run.

Do not upload personal or sensitive genomic data to a public repository. Genomic files can contain highly sensitive information even when obvious personal identifiers have been removed.

## Limitations

- The project is a research prototype and is not a clinically validated diagnostic tool.
- Results depend on dataset selection, class balance, feature availability, reference build, and evaluation design.
- Synthetic or development-set performance, where applicable, does not establish real-world clinical performance.
- Graph results depend on the validity of the graph construction and the prevention of information leakage.
- Fusion can increase complexity without necessarily improving generalization.
- Missing annotations must be handled explicitly and consistently.
- VCF support requires careful handling of genome builds, multiallelic records, normalization, and unsupported variant types.
- The planned FastAPI and frontend components require testing for security, scalability, reproducibility, and correctness before deployment.
