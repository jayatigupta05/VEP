# GenoHybrid-VEP

GenoHybrid-VEP is a research prototype for classifying genomic variants as
`Benign`, `Pathogenic`, or `VUS` using a tri-modal deep learning workflow:

- a sequence CNN over local GRCh38 sequence context
- a biomarker/tabular MLP over conservation and annotation features
- a graph convolutional network over related variant nodes

The repository currently contains the research notebooks, trained checkpoint
artifacts, a VCF prediction script, and a small Flask interface for local
testing.

> Research use only: this project is not a clinical diagnostic system. Do not
> use predictions as the sole basis for medical or health decisions.

## Repository Contents
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
.
|-- app.py
|-- predict.py
|-- requirements.txt
|-- checkpoints/
|-- data/
|   |-- processed/
|   |-- raw/
|   `-- references/
|-- final_validation/
|-- results/
|-- GenoHybrid_VEP_Notebook_1_Data_Pipeline.ipynb
|-- GenoHybrid_VEP_Notebook_2_Foundation_and_Sequence_Encoding.ipynb
`-- GenoHybrid_VEP_Notebook_3_Validation.ipynb
```

Important files:

- `predict.py`: command-line VCF prediction pipeline.
- `app.py`: local Flask upload interface for testing VCF prediction.
- `checkpoints/`: model checkpoint artifacts used by inference.
- `data/processed/`: processed feature table expected by inference.
- `data/references/`: local reference-resource storage.
- `final_validation/` and `results/`: validation and experiment outputs.

## Model Overview

The prediction script uses three model branches:

- `sequence_cnn_best.pt`: sequence-only CNN branch.
- `biomarker_mlp_best.pt`: tabular biomarker branch.
- `gcn_v2_best.pt`: graph branch.

The final prediction uses locked late-fusion weights:

```text
0.40 biomarker + 0.30 graph + 0.30 sequence
```

The output classes are:

- `0`: Benign
- `1`: Pathogenic
- `2`: VUS

## Requirements

Python 3.10 or newer is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

The main packages are:

- Flask
- NumPy
- pandas
- scikit-learn
- PyTorch
- PyTorch Geometric
- pyarrow
- pysam
- py2bit
- pyBigWig
- matplotlib
- tqdm

PyTorch Geometric installation can depend on your Python, PyTorch, CUDA, and
operating-system versions. If a plain `pip install -r requirements.txt` fails,
install the compatible PyTorch and PyTorch Geometric wheels for your machine.

## Required Data And Artifacts

For full VCF inference, the project directory must contain:

```text
checkpoints/
|-- sequence_cnn_best.pt
|-- biomarker_mlp_best.pt
`-- gcn_v2_best.pt

data/
|-- processed/
|   `-- genohybrid_features.parquet
`-- reference/
    |-- hg38.2bit
    |-- hg38.phyloP100way.bw
    `-- hg38.phastCons100way.bw
```

The current repository uses `data/references/` in some places, while
`predict.py` expects `data/reference/`. Before running inference, either rename
the folder to `data/reference/` or update the `reference_dir` path in
`predict.py`.

Large genomic reference files are not suitable for normal Git tracking. Keep
them local or in external storage and document their source/version.

## Input VCF Expectations

The predictor accepts GRCh38 VCF or VCF.GZ files.

Current behavior:

- multi-allelic rows are split into separate ALT alleles
- only SNVs are predicted
- non-SNV or unsupported rows are kept in the output with status information
- Ensembl VEP `CSQ` or SnpEff `ANN` annotations are used when available
- missing gene/consequence annotation falls back to less informative defaults

Best results require the input VCF to include gene and consequence annotations.

## Run Prediction From The CLI

```bash
python predict.py \
  --vcf path/to/input.vcf \
  --output genohybrid_predictions.csv \
  --project-dir .
```

On Windows PowerShell:

```powershell
python predict.py `
  --vcf path\to\input.vcf `
  --output genohybrid_predictions.csv `
  --project-dir .
```

The output CSV includes variant metadata, branch probabilities, final fused
probabilities, predicted class, and confidence.

## Run The Local Web Interface

`app.py` provides a simple Flask interface for uploading a VCF and downloading
the prediction CSV.

First check the predictor import. The current `app.py` imports:

```python
from genohybrid_vcf_predict import predict_vcf
```

If the predictor file is named `predict.py`, either rename it to
`genohybrid_vcf_predict.py` or change that import to:

```python
from predict import predict_vcf
```

Also update `PROJECT_DIR` in `app.py` if your project is not located at the
default Colab/Drive path.

Then run:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

The web interface accepts `.vcf` and `.vcf.gz` files up to the configured upload
limit and writes output files under `prediction_outputs/`.

## Research Notebooks

The notebooks capture the development workflow:

1. `GenoHybrid_VEP_Notebook_1_Data_Pipeline.ipynb`
   - prepares ClinVar-derived data
   - extracts sequence/conservation features
   - creates processed training data

2. `GenoHybrid_VEP_Notebook_2_Foundation_and_Sequence_Encoding.ipynb`
   - trains sequence, biomarker, graph, and fusion models
   - writes checkpoint artifacts

3. `GenoHybrid_VEP_Notebook_3_Validation.ipynb`
   - runs validation experiments
   - saves final metrics and plots

## Validation Outputs

Validation artifacts are stored in `final_validation/`, including:

- fold-level prediction CSVs
- summary metrics
- classification reports
- confusion matrices
- ROC and precision-recall plots

Treat these as experiment artifacts from the recorded run. Re-run validation
when data, preprocessing, checkpoints, or evaluation logic changes.

## Data Management

Do not commit:

- raw patient or private genomic files
- large reference genomes or BigWig tracks
- large generated datasets
- oversized checkpoints
- local uploads or prediction outputs

The `.gitignore` currently excludes:

```text
data/references
data/raw
__pycache__
```

Consider also ignoring local upload/output folders if you use the Flask app:

```text
uploads/
prediction_outputs/
```

## Limitations

- This is a research prototype, not a validated clinical tool.
- Performance depends on dataset choice, class balance, feature availability,
  annotation quality, and validation design.
- Missing gene or consequence annotations reduce graph and biomarker context.
- The inference path must use the same reference build, feature definitions,
  preprocessing logic, and checkpoint architecture used during training.
- Probability outputs should not be interpreted as calibrated clinical risk
  unless calibration has been separately evaluated.

## License

Add a `LICENSE` file before distributing or reusing this repository publicly.
