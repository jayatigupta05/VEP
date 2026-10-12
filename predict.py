#!/usr/bin/env python3
"""
GenoHybrid-VEP: VCF -> tri-modal prediction

Expected project layout
-----------------------
/content/drive/MyDrive/variant_effect_prediction/
├── checkpoints/
│   ├── sequence_cnn_best.pt
│   ├── biomarker_mlp_best.pt
│   └── gcn_v2_best.pt
└── data/
    ├── processed/
    │   └── genohybrid_features.parquet
    └── references/
        ├── hg38.2bit
        ├── hg38.phyloP100way.bw
        └── hg38.phastCons100way.bw

Input
-----
A GRCh38 VCF. Multi-allelic rows are split into separate ALT alleles.
Only SNVs are predicted.

Important annotation note
-------------------------
The final GenoHybrid-VEP biomarker and graph branches use GeneSymbol and
consequence_category.

This script can recover them from:
  1. Ensembl VEP CSQ annotations, or
  2. SnpEff ANN annotations.

If neither is present, consequence_category becomes "unknown" and GeneSymbol
is blank. The model can still run, but the graph and biomarker branches become
less informative.

The ordinary input VCF also does not normally contain the ClinVar-specific
metadata used during development:
  Tab_Review_Score
  Tab_Origin_Numeric
  Tab_Somatic_Impact

For deployment, these are intentionally left missing and replaced using the
training-set medians by the same preprocessing logic used in training.

Output
------
CSV with:
  CHROM, POS, REF, ALT, GeneSymbol, consequence_category,
  sequence probability, biomarker probability, graph probability,
  final fused probabilities, predicted class and confidence.

Final locked fusion:
  0.40 biomarker + 0.30 graph + 0.30 sequence

Example
-------
python genohybrid_vcf_predict.py \
    --vcf /content/input.vcf \
    --output /content/genohybrid_predictions.csv
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

try:
    import pysam
except ImportError:
    raise SystemExit(
        "Missing pysam. Install with: pip install pysam"
    )

try:
    import py2bit
except ImportError:
    raise SystemExit(
        "Missing py2bit. Install with: pip install py2bit"
    )

try:
    import pyBigWig
except ImportError:
    raise SystemExit(
        "Missing pyBigWig. Install with: pip install pyBigWig"
    )

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv
except ImportError:
    raise SystemExit(
        "Missing torch-geometric. Install with: pip install torch-geometric"
    )


# ============================================================
# LOCKED PROJECT SETTINGS
# ============================================================

SEED = 42
NUM_CLASSES = 3
SEQ_WINDOW = 101

W_BIO = 0.40
W_GRAPH = 0.30
W_SEQ = 0.30

K_SAME_CONSEQUENCE = 4
K_CONSERVATION = 3
K_PROX = 2
MAX_BP = 5000

CLASS_NAMES = {
    0: "Benign",
    1: "Pathogenic",
    2: "VUS"
}

BIO_NUMERIC_FEATURES = [
    "phylop",
    "phastcons",
    "Tab_Review_Score",
    "Tab_Origin_Numeric",
    "Tab_Somatic_Impact"
]

GRAPH_NUMERIC_FEATURES = [
    "phylop",
    "phastcons",
    "Tab_Origin_Numeric",
    "Tab_Somatic_Impact"
]

CONSEQUENCE_COLUMN = "consequence_category"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL DEFINITIONS
# ============================================================

class SequenceCNN(nn.Module):

    def __init__(self, embedding_dim=64):
        super().__init__()

        self.conv1 = nn.Conv1d(
            4, 32, kernel_size=7, padding=3
        )
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(
            32, 64, kernel_size=5, padding=2
        )
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(
            64, 64, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm1d(64)

        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.30)

        self.fc = nn.Sequential(
            nn.Linear(128, embedding_dim),
            nn.ReLU()
        )

    def forward(self, x):

        x = self.pool(
            F.relu(
                self.bn1(
                    self.conv1(x)
                )
            )
        )

        x = self.pool(
            F.relu(
                self.bn2(
                    self.conv2(x)
                )
            )
        )

        x = F.relu(
            self.bn3(
                self.conv3(x)
            )
        )

        avg = torch.mean(
            x,
            dim=2
        )

        mx = torch.max(
            x,
            dim=2
        ).values

        x = torch.cat(
            [avg, mx],
            dim=1
        )

        x = self.dropout(x)

        return self.fc(x)


class SequenceOnlyModel(nn.Module):

    def __init__(self, num_classes=3):
        super().__init__()

        self.encoder = SequenceCNN(
            embedding_dim=64
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.classifier(
            self.encoder(x)
        )


class BiomarkerMLP(nn.Module):

    def __init__(
        self,
        input_dim,
        embedding_dim=32
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(32, embedding_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.network(x)


class BiomarkerOnlyModel(nn.Module):

    def __init__(
        self,
        input_dim,
        num_classes=3
    ):
        super().__init__()

        self.encoder = BiomarkerMLP(
            input_dim=input_dim,
            embedding_dim=32
        )

        self.classifier = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.classifier(
            self.encoder(x)
        )


class VariantGCN(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=32,
        embedding_dim=32
    ):
        super().__init__()

        self.conv1 = GCNConv(
            input_dim,
            hidden_dim
        )

        self.norm1 = nn.BatchNorm1d(
            hidden_dim
        )

        self.conv2 = GCNConv(
            hidden_dim,
            embedding_dim
        )

        self.norm2 = nn.BatchNorm1d(
            embedding_dim
        )

        self.dropout = nn.Dropout(
            0.30
        )

    def forward(
        self,
        x,
        edge_index
    ):

        x = self.conv1(
            x,
            edge_index
        )

        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(
            x,
            edge_index
        )

        x = self.norm2(x)
        x = F.relu(x)

        return x


class GCNOnlyModel(nn.Module):

    def __init__(
        self,
        input_dim,
        num_classes=3
    ):
        super().__init__()

        self.encoder = VariantGCN(
            input_dim=input_dim,
            hidden_dim=32,
            embedding_dim=32
        )

        self.classifier = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(32, num_classes)
        )

    def forward(
        self,
        x,
        edge_index
    ):

        return self.classifier(
            self.encoder(
                x,
                edge_index
            )
        )


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def extract_state_dict(checkpoint):

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]

        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]

    return checkpoint


def load_checkpoint_state(path):

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False
    )

    return extract_state_dict(
        checkpoint
    )


# ============================================================
# REPRODUCE ORIGINAL DEVELOPMENT TRAIN SPLIT
# ============================================================

def add_group_id(df):

    df = df.copy()

    df["group_id"] = (
        df["GeneSymbol"]
        .astype("string")
    )

    missing_gene = (
        df["group_id"].isna()
        |
        (
            df["group_id"]
            .fillna("")
            .str.len()
            == 0
        )
    )

    df.loc[
        missing_gene,
        "group_id"
    ] = (
        "NO_GENE_"
        +
        df.loc[
            missing_gene,
            "Chromosome"
        ].astype(str)
        +
        "_"
        +
        (
            df.loc[
                missing_gene,
                "Start"
            ]
            // 1_000_000
        ).astype(str)
    )

    return df


def reproduce_train_split(df):

    df = add_group_id(df)

    outer_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.10,
        random_state=SEED
    )

    trainval_idx, _ = next(
        outer_split.split(
            df,
            y=df["Label"],
            groups=df["group_id"]
        )
    )

    trainval_df = (
        df.iloc[
            trainval_idx
        ]
        .copy()
    )

    inner_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.1111,
        random_state=SEED
    )

    train_idx, _ = next(
        inner_split.split(
            trainval_df,
            y=trainval_df["Label"],
            groups=trainval_df["group_id"]
        )
    )

    train_df = (
        trainval_df.iloc[
            train_idx
        ]
        .copy()
        .reset_index(drop=True)
    )

    return train_df


# ============================================================
# FEATURE PREPROCESSING
# ============================================================

def fit_feature_preprocessor(
    train_part,
    numeric_features
):

    medians = {}

    for feature in numeric_features:

        median = train_part[
            feature
        ].median()

        if pd.isna(median):
            median = 0.0

        medians[feature] = float(
            median
        )

    categories = sorted(
        train_part[
            CONSEQUENCE_COLUMN
        ]
        .fillna("unknown")
        .astype(str)
        .unique()
    )

    numeric = train_part[
        numeric_features
    ].copy()

    for feature in [
        "phylop",
        "phastcons"
    ]:

        numeric[
            f"{feature}_missing"
        ] = (
            train_part[
                feature
            ]
            .isna()
            .astype(np.float32)
        )

    for feature in numeric_features:

        numeric[feature] = (
            numeric[feature]
            .fillna(
                medians[feature]
            )
        )

    scaler = StandardScaler()

    scaler.fit(
        numeric.astype(
            np.float32
        )
    )

    return (
        medians,
        categories,
        scaler
    )


def transform_features(
    dataframe,
    numeric_features,
    medians,
    categories,
    scaler
):

    temp = (
        dataframe
        .reset_index(drop=True)
        .copy()
    )

    numeric = temp[
        numeric_features
    ].copy()

    for feature in [
        "phylop",
        "phastcons"
    ]:

        numeric[
            f"{feature}_missing"
        ] = (
            temp[feature]
            .isna()
            .astype(np.float32)
        )

    for feature in numeric_features:

        numeric[feature] = (
            numeric[feature]
            .fillna(
                medians[feature]
            )
        )

    numeric_scaled = (
        scaler.transform(
            numeric.astype(
                np.float32
            )
        )
    )

    category_to_index = {
        value: i
        for i, value
        in enumerate(categories)
    }

    categorical = np.zeros(
        (
            len(temp),
            len(categories)
        ),
        dtype=np.float32
    )

    values = (
        temp[
            CONSEQUENCE_COLUMN
        ]
        .fillna("unknown")
        .astype(str)
    )

    for i, value in enumerate(values):

        if value in category_to_index:

            categorical[
                i,
                category_to_index[value]
            ] = 1.0

    return np.concatenate(
        [
            numeric_scaled.astype(
                np.float32
            ),
            categorical
        ],
        axis=1
    )


# ============================================================
# SEQUENCE ENCODING
# ============================================================

BASE_TO_INDEX = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3
}


def encode_variant_sequence(
    seq201,
    ref,
    alt,
    window_size=101
):

    seq201 = str(
        seq201
    ).upper()

    ref = str(
        ref
    ).upper()

    alt = str(
        alt
    ).upper()

    center = 100
    half = window_size // 2

    cropped = seq201[
        center-half:
        center+half+1
    ]

    encoded = np.zeros(
        (
            4,
            window_size
        ),
        dtype=np.float32
    )

    for pos, base in enumerate(
        cropped
    ):

        if base in BASE_TO_INDEX:

            encoded[
                BASE_TO_INDEX[base],
                pos
            ] = 1.0

    middle = window_size // 2

    encoded[
        :,
        middle
    ] = 0.0

    if ref in BASE_TO_INDEX:

        encoded[
            BASE_TO_INDEX[ref],
            middle
        ] = 1.0

    if alt in BASE_TO_INDEX:

        encoded[
            BASE_TO_INDEX[alt],
            middle
        ] = -1.0

    return encoded


class QuerySequenceDataset(Dataset):

    def __init__(self, dataframe):
        self.df = (
            dataframe
            .reset_index(drop=True)
            .copy()
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[
            idx
        ]

        seq = encode_variant_sequence(
            row["seq201"],
            row["ReferenceAlleleVCF"],
            row["AlternateAlleleVCF"],
            window_size=SEQ_WINDOW
        )

        return torch.tensor(
            seq,
            dtype=torch.float32
        )


@torch.no_grad()
def get_sequence_probabilities(
    model,
    dataframe
):

    dataset = QuerySequenceDataset(
        dataframe
    )

    loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0
    )

    model.eval()

    probabilities = []

    for seq in loader:

        seq = seq.to(
            device
        )

        logits = model(
            seq
        )

        prob = torch.softmax(
            logits,
            dim=1
        )

        probabilities.append(
            prob.cpu().numpy()
        )

    return np.concatenate(
        probabilities,
        axis=0
    )


@torch.no_grad()
def get_biomarker_probabilities(
    model,
    X
):

    model.eval()

    tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=device
    )

    logits = model(
        tensor
    )

    return (
        torch.softmax(
            logits,
            dim=1
        )
        .cpu()
        .numpy()
    )


# ============================================================
# REFERENCE SEQUENCE / CONSERVATION
# ============================================================

def ucsc_chrom(chrom):

    chrom = str(
        chrom
    ).replace(
        "chr",
        ""
    )

    return f"chr{chrom}"


def extract_seq201(
    genome,
    chrom,
    pos_1based
):

    chrom = ucsc_chrom(
        chrom
    )

    center0 = int(
        pos_1based
    ) - 1

    start0 = center0 - 100
    end0 = center0 + 101

    if start0 < 0:
        return None

    try:

        seq = genome.sequence(
            chrom,
            start0,
            end0
        )

    except Exception:

        return None

    if seq is None:
        return None

    seq = seq.upper()

    if len(seq) != 201:
        return None

    return seq


def bigwig_value(
    bw,
    chrom,
    pos_1based
):

    chrom = ucsc_chrom(
        chrom
    )

    p0 = int(
        pos_1based
    ) - 1

    try:

        values = bw.values(
            chrom,
            p0,
            p0 + 1
        )

        if not values:
            return np.nan

        value = values[0]

        if value is None:
            return np.nan

        if isinstance(
            value,
            float
        ) and np.isnan(value):
            return np.nan

        return float(
            value
        )

    except Exception:

        return np.nan


# ============================================================
# CONSEQUENCE MAPPING
# ============================================================

def consequence_category(
    raw_consequence
):

    text = str(
        raw_consequence
    ).lower()

    if "missense_variant" in text:
        return "missense"

    if (
        "stop_gained" in text
        or "stop_lost" in text
        or "start_lost" in text
    ):
        return "nonsense_or_stop"

    if "synonymous_variant" in text:
        return "synonymous"

    if "splice" in text:
        return "splice"

    if "intron_variant" in text:
        return "intron"

    if (
        "5_prime_utr_variant" in text
        or "3_prime_utr_variant" in text
    ):
        return "utr"

    if (
        "upstream_gene_variant" in text
        or "downstream_gene_variant" in text
    ):
        return "near_gene"

    if text.strip() == "":
        return "unknown"

    return "other"


# ============================================================
# PARSE VEP CSQ / SNPEFF ANN
# ============================================================

def get_vep_csq_fields(header):

    if "CSQ" not in header.info:
        return None

    description = str(
        header.info["CSQ"].description
    )

    marker = "Format:"

    if marker not in description:
        return None

    field_string = (
        description
        .split(
            marker,
            1
        )[1]
        .strip()
        .strip('"')
    )

    return [
        x.strip()
        for x in field_string.split("|")
    ]


def parse_annotation(
    record,
    alt,
    csq_fields
):

    gene = ""
    consequences = []

    # Ensembl VEP CSQ
    if (
        csq_fields is not None
        and "CSQ" in record.info
    ):

        raw_csq = record.info.get(
            "CSQ"
        )

        if raw_csq is not None:

            if isinstance(
                raw_csq,
                str
            ):
                entries = raw_csq.split(",")
            else:
                entries = list(
                    raw_csq
                )

            for entry in entries:

                values = str(
                    entry
                ).split("|")

                row = {
                    field:
                    values[i]
                    if i < len(values)
                    else ""
                    for i, field
                    in enumerate(csq_fields)
                }

                allele = row.get(
                    "Allele",
                    ""
                )

                if (
                    allele
                    and allele != alt
                ):
                    continue

                if not gene:

                    gene = (
                        row.get(
                            "SYMBOL",
                            ""
                        )
                        or row.get(
                            "Gene",
                            ""
                        )
                    )

                consequence = row.get(
                    "Consequence",
                    ""
                )

                if consequence:
                    consequences.append(
                        consequence
                    )

    # SnpEff ANN fallback
    if (
        not consequences
        and "ANN" in record.info
    ):

        raw_ann = record.info.get(
            "ANN"
        )

        if raw_ann is not None:

            if isinstance(
                raw_ann,
                str
            ):
                entries = raw_ann.split(",")
            else:
                entries = list(
                    raw_ann
                )

            for entry in entries:

                values = str(
                    entry
                ).split("|")

                if len(values) < 4:
                    continue

                ann_alt = values[0]

                if (
                    ann_alt
                    and ann_alt != alt
                ):
                    continue

                annotation = values[1]
                gene_name = values[3]

                if (
                    not gene
                    and gene_name
                ):
                    gene = gene_name

                if annotation:
                    consequences.append(
                        annotation
                    )

    raw_consequence = "&".join(
        sorted(
            set(
                consequences
            )
        )
    )

    return (
        gene,
        raw_consequence,
        consequence_category(
            raw_consequence
        )
    )


# ============================================================
# VCF -> QUERY TABLE
# ============================================================

def parse_vcf_to_dataframe(
    vcf_path,
    genome,
    bw_phylop,
    bw_phastcons
):

    vcf = pysam.VariantFile(
        str(
            vcf_path
        )
    )

    csq_fields = get_vep_csq_fields(
        vcf.header
    )

    rows = []

    for record in vcf.fetch():

        chrom = str(
            record.chrom
        ).replace(
            "chr",
            ""
        )

        pos = int(
            record.pos
        )

        ref = str(
            record.ref
        ).upper()

        for alt_raw in (
            record.alts
            or []
        ):

            alt = str(
                alt_raw
            ).upper()

            base_row = {
                "Chromosome":
                    chrom,

                "Start":
                    pos,

                "ReferenceAlleleVCF":
                    ref,

                "AlternateAlleleVCF":
                    alt,

                "variant_key":
                    f"{chrom}:{pos}:{ref}>{alt}",

                "status":
                    "OK"
            }

            # SNVs only
            if (
                len(ref) != 1
                or len(alt) != 1
                or ref not in BASE_TO_INDEX
                or alt not in BASE_TO_INDEX
            ):

                base_row["status"] = (
                    "SKIPPED_NON_SNV"
                )

                rows.append(
                    base_row
                )

                continue

            seq201 = extract_seq201(
                genome,
                chrom,
                pos
            )

            if seq201 is None:

                base_row["status"] = (
                    "SKIPPED_SEQUENCE_EXTRACTION_FAILED"
                )

                rows.append(
                    base_row
                )

                continue

            reference_base = seq201[
                100
            ]

            if reference_base != ref:

                base_row["status"] = (
                    f"SKIPPED_REF_MISMATCH_"
                    f"GENOME_{reference_base}"
                )

                rows.append(
                    base_row
                )

                continue

            (
                gene_symbol,
                raw_consequence,
                category
            ) = parse_annotation(
                record,
                alt,
                csq_fields
            )

            base_row.update({

                "GeneSymbol":
                    gene_symbol,

                "molecular_consequence_raw":
                    raw_consequence,

                "consequence_category":
                    category,

                "seq201":
                    seq201,

                "phylop":
                    bigwig_value(
                        bw_phylop,
                        chrom,
                        pos
                    ),

                "phastcons":
                    bigwig_value(
                        bw_phastcons,
                        chrom,
                        pos
                    ),

                # ClinVar-specific metadata is not normally
                # available in arbitrary user VCFs.
                # Leave missing -> training median during transform.
                "Tab_Review_Score":
                    np.nan,

                "Tab_Origin_Numeric":
                    np.nan,

                "Tab_Somatic_Impact":
                    np.nan
            })

            rows.append(
                base_row
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# GRAPH V2 BUILDER
# ============================================================

def build_variant_graph_edges_v2_fast(
    dataframe,
    graph_train_medians,
    k_same_consequence=K_SAME_CONSEQUENCE,
    k_conservation=K_CONSERVATION,
    k_prox=K_PROX,
    max_bp=MAX_BP
):

    temp = (
        dataframe
        .reset_index(drop=True)
        .copy()
    )

    edges = set()

    temp["_gene"] = (
        temp["GeneSymbol"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    temp["_consequence"] = (
        temp[
            "consequence_category"
        ]
        .fillna(
            "unknown"
        )
        .astype(str)
    )

    temp["_phylop"] = (
        temp["phylop"]
        .fillna(
            graph_train_medians[
                "phylop"
            ]
        )
        .astype(float)
    )

    temp["_phastcons"] = (
        temp["phastcons"]
        .fillna(
            graph_train_medians[
                "phastcons"
            ]
        )
        .astype(float)
    )

    valid_gene = (
        (
            temp["_gene"]
            != ""
        )
        &
        (
            temp["_gene"]
            != "-"
        )
        &
        (
            temp["_gene"]
            .str.lower()
            != "nan"
        )
    )

    def add_edge(i, j):

        i = int(i)
        j = int(j)

        if i == j:
            return

        edges.add(
            (i, j)
        )

        edges.add(
            (j, i)
        )

    for _, gene_group in (
        temp.loc[
            valid_gene
        ]
        .groupby(
            "_gene",
            sort=False
        )
    ):

        if len(
            gene_group
        ) < 2:
            continue

        # ----------------------------------------------------
        # A. same gene + same consequence
        # ----------------------------------------------------

        for _, subgroup in (
            gene_group
            .groupby(
                "_consequence",
                sort=False
            )
        ):

            if len(
                subgroup
            ) < 2:
                continue

            subgroup = (
                subgroup
                .sort_values(
                    "Start"
                )
            )

            idxs = (
                subgroup.index
                .to_numpy()
            )

            pos = (
                subgroup[
                    "Start"
                ]
                .to_numpy()
            )

            m = len(
                subgroup
            )

            for i in range(
                m
            ):

                candidates = []

                for offset in range(
                    1,
                    k_same_consequence + 1
                ):

                    li = i - offset
                    ri = i + offset

                    if li >= 0:

                        candidates.append(
                            (
                                abs(
                                    pos[i]
                                    - pos[li]
                                ),
                                idxs[li]
                            )
                        )

                    if ri < m:

                        candidates.append(
                            (
                                abs(
                                    pos[ri]
                                    - pos[i]
                                ),
                                idxs[ri]
                            )
                        )

                candidates.sort(
                    key=lambda x:
                    x[0]
                )

                for _, neighbor in (
                    candidates[
                        :k_same_consequence
                    ]
                ):

                    add_edge(
                        idxs[i],
                        neighbor
                    )

        # ----------------------------------------------------
        # B. same gene + conservation similarity
        # ----------------------------------------------------

        gene_indices = (
            gene_group.index
            .to_numpy()
        )

        conservation = (
            gene_group[
                [
                    "_phylop",
                    "_phastcons"
                ]
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        means = conservation.mean(
            axis=0,
            keepdims=True
        )

        stds = conservation.std(
            axis=0,
            keepdims=True
        )

        stds[
            stds < 1e-6
        ] = 1.0

        conservation_norm = (
            conservation - means
        ) / stds

        m = len(
            gene_group
        )

        n_neighbors = min(
            k_conservation + 1,
            m
        )

        if n_neighbors > 1:

            nbrs = NearestNeighbors(
                n_neighbors=n_neighbors,
                metric="euclidean"
            )

            nbrs.fit(
                conservation_norm
            )

            _, neighbor_positions = (
                nbrs.kneighbors(
                    conservation_norm
                )
            )

            for i in range(
                m
            ):

                added = 0

                for j_pos in (
                    neighbor_positions[
                        i
                    ]
                ):

                    if j_pos == i:
                        continue

                    add_edge(
                        gene_indices[
                            i
                        ],
                        gene_indices[
                            j_pos
                        ]
                    )

                    added += 1

                    if (
                        added
                        >= k_conservation
                    ):
                        break

        # ----------------------------------------------------
        # C. same-gene genomic proximity
        # ----------------------------------------------------

        positional = (
            gene_group
            .sort_values(
                "Start"
            )
        )

        idxs = (
            positional.index
            .to_numpy()
        )

        pos = (
            positional[
                "Start"
            ]
            .to_numpy()
        )

        m = len(
            positional
        )

        for i in range(
            m
        ):

            candidates = []

            for offset in range(
                1,
                k_prox + 3
            ):

                li = i - offset
                ri = i + offset

                if li >= 0:

                    distance = (
                        pos[i]
                        - pos[li]
                    )

                    if (
                        distance
                        <= max_bp
                    ):

                        candidates.append(
                            (
                                distance,
                                idxs[li]
                            )
                        )

                if ri < m:

                    distance = (
                        pos[ri]
                        - pos[i]
                    )

                    if (
                        distance
                        <= max_bp
                    ):

                        candidates.append(
                            (
                                distance,
                                idxs[ri]
                            )
                        )

            candidates.sort(
                key=lambda x:
                x[0]
            )

            for _, neighbor in (
                candidates[
                    :k_prox
                ]
            ):

                add_edge(
                    idxs[i],
                    neighbor
                )

    if not edges:

        return torch.empty(
            (
                2,
                0
            ),
            dtype=torch.long
        )

    edge_array = np.asarray(
        list(
            edges
        ),
        dtype=np.int64
    )

    return torch.tensor(
        edge_array.T,
        dtype=torch.long
    )


# ============================================================
# GRAPH INFERENCE
# ============================================================

@torch.no_grad()
def graph_probabilities_with_reference_context(
    model,
    query_df,
    train_reference_df,
    graph_medians,
    graph_categories,
    graph_scaler
):

    query_genes = (
        query_df[
            "GeneSymbol"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    valid_genes = set(
        query_genes[
            (
                query_genes != ""
            )
            &
            (
                query_genes != "-"
            )
        ]
        .tolist()
    )

    # Pull same-gene training nodes as graph context.
    if valid_genes:

        reference_context = (
            train_reference_df[
                train_reference_df[
                    "GeneSymbol"
                ]
                .fillna("")
                .astype(str)
                .isin(
                    valid_genes
                )
            ]
            .copy()
            .reset_index(drop=True)
        )

    else:

        reference_context = (
            train_reference_df
            .iloc[
                0:0
            ]
            .copy()
        )

    # Mark which nodes are queries.
    reference_context[
        "_is_query"
    ] = False

    query_copy = (
        query_df
        .copy()
        .reset_index(drop=True)
    )

    query_copy[
        "_is_query"
    ] = True

    combined = pd.concat(
        [
            reference_context,
            query_copy
        ],
        ignore_index=True,
        sort=False
    )

    # Guarantee metadata columns exist.
    for col in [
        "Tab_Origin_Numeric",
        "Tab_Somatic_Impact"
    ]:

        if col not in combined.columns:
            combined[col] = np.nan

    X_graph = transform_features(
        combined,
        GRAPH_NUMERIC_FEATURES,
        graph_medians,
        graph_categories,
        graph_scaler
    )

    edge_index = (
        build_variant_graph_edges_v2_fast(
            combined,
            graph_medians
        )
    )

    graph = Data(
        x=torch.tensor(
            X_graph,
            dtype=torch.float32
        ),
        edge_index=edge_index
    ).to(
        device
    )

    model.eval()

    logits = model(
        graph.x,
        graph.edge_index
    )

    probabilities = (
        torch.softmax(
            logits,
            dim=1
        )
        .cpu()
        .numpy()
    )

    query_mask = (
        combined[
            "_is_query"
        ]
        .to_numpy(
            dtype=bool
        )
    )

    return probabilities[
        query_mask
    ]


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def predict_vcf(
    vcf_path,
    output_path,
    project_dir
):

    project_dir = Path(
        project_dir
    )

    processed_path = (
        project_dir
        / "data"
        / "processed"
        / "genohybrid_features.parquet"
    )

    reference_dir = (
        project_dir
        / "data"
        / "references"
    )

    checkpoint_dir = (
        project_dir
        / "checkpoints"
    )

    hg38_2bit = (
        reference_dir
        / "hg38.2bit"
    )

    phylop_bw = (
        reference_dir
        / "hg38.phyloP100way.bw"
    )

    phastcons_bw = (
        reference_dir
        / "hg38.phastCons100way.bw"
    )

    seq_checkpoint = (
        checkpoint_dir
        / "sequence_cnn_best.pt"
    )

    bio_checkpoint = (
        checkpoint_dir
        / "biomarker_mlp_best.pt"
    )

    gcn_checkpoint = (
        checkpoint_dir
        / "gcn_v2_best.pt"
    )

    required_files = [
        processed_path,
        hg38_2bit,
        phylop_bw,
        phastcons_bw,
        seq_checkpoint,
        bio_checkpoint,
        gcn_checkpoint
    ]

    missing = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing:

        raise FileNotFoundError(
            "Missing required project files:\n"
            +
            "\n".join(
                missing
            )
        )

    print(
        "Device:",
        device
    )

    print(
        "Loading processed training features..."
    )

    full_df = pd.read_parquet(
        processed_path
    )

    train_df = reproduce_train_split(
        full_df
    )

    print(
        "Reproduced development training rows:",
        len(train_df)
    )

    # --------------------------------------------------------
    # Reconstruct training preprocessing
    # --------------------------------------------------------

    (
        bio_medians,
        bio_categories,
        bio_scaler
    ) = fit_feature_preprocessor(
        train_df,
        BIO_NUMERIC_FEATURES
    )

    (
        graph_medians,
        graph_categories,
        graph_scaler
    ) = fit_feature_preprocessor(
        train_df,
        GRAPH_NUMERIC_FEATURES
    )

    X_bio_train_shape = transform_features(
        train_df.iloc[
            :2
        ],
        BIO_NUMERIC_FEATURES,
        bio_medians,
        bio_categories,
        bio_scaler
    ).shape[1]

    X_graph_train_shape = transform_features(
        train_df.iloc[
            :2
        ],
        GRAPH_NUMERIC_FEATURES,
        graph_medians,
        graph_categories,
        graph_scaler
    ).shape[1]

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    sequence_model = SequenceOnlyModel(
        num_classes=NUM_CLASSES
    ).to(
        device
    )

    biomarker_model = BiomarkerOnlyModel(
        input_dim=X_bio_train_shape,
        num_classes=NUM_CLASSES
    ).to(
        device
    )

    gcn_model = GCNOnlyModel(
        input_dim=X_graph_train_shape,
        num_classes=NUM_CLASSES
    ).to(
        device
    )

    sequence_model.load_state_dict(
        load_checkpoint_state(
            seq_checkpoint
        )
    )

    biomarker_model.load_state_dict(
        load_checkpoint_state(
            bio_checkpoint
        )
    )

    gcn_model.load_state_dict(
        load_checkpoint_state(
            gcn_checkpoint
        )
    )

    sequence_model.eval()
    biomarker_model.eval()
    gcn_model.eval()

    print(
        "Models loaded."
    )

    # --------------------------------------------------------
    # Load reference tracks
    # --------------------------------------------------------

    genome = py2bit.open(
        str(
            hg38_2bit
        )
    )

    bw_phylop = pyBigWig.open(
        str(
            phylop_bw
        )
    )

    bw_phastcons = pyBigWig.open(
        str(
            phastcons_bw
        )
    )

    print(
        "Parsing VCF and generating features..."
    )

    all_rows = parse_vcf_to_dataframe(
        vcf_path,
        genome,
        bw_phylop,
        bw_phastcons
    )

    if all_rows.empty:

        raise ValueError(
            "No variants were found in the VCF."
        )

    valid_mask = (
        all_rows[
            "status"
        ]
        == "OK"
    )

    valid_df = (
        all_rows.loc[
            valid_mask
        ]
        .copy()
        .reset_index(drop=True)
    )

    print(
        "Total ALT alleles:",
        len(all_rows)
    )

    print(
        "Valid SNVs:",
        len(valid_df)
    )

    if len(
        valid_df
    ) == 0:

        all_rows.to_csv(
            output_path,
            index=False
        )

        print(
            "No valid SNVs to predict."
        )

        return all_rows

    missing_annotation_fraction = (
        (
            valid_df[
                "consequence_category"
            ]
            == "unknown"
        )
        .mean()
    )

    if (
        missing_annotation_fraction
        > 0
    ):

        print(
            f"WARNING: "
            f"{missing_annotation_fraction:.1%} "
            f"of valid variants lack VEP/SnpEff consequence annotation."
        )

    missing_gene_fraction = (
        valid_df[
            "GeneSymbol"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .mean()
    )

    if (
        missing_gene_fraction
        > 0
    ):

        print(
            f"WARNING: "
            f"{missing_gene_fraction:.1%} "
            f"of valid variants lack GeneSymbol annotation."
        )

    # --------------------------------------------------------
    # Sequence arm
    # --------------------------------------------------------

    seq_prob = get_sequence_probabilities(
        sequence_model,
        valid_df
    )

    # --------------------------------------------------------
    # Biomarker arm
    # --------------------------------------------------------

    X_bio = transform_features(
        valid_df,
        BIO_NUMERIC_FEATURES,
        bio_medians,
        bio_categories,
        bio_scaler
    )

    bio_prob = get_biomarker_probabilities(
        biomarker_model,
        X_bio
    )

    # --------------------------------------------------------
    # Graph arm
    # --------------------------------------------------------

    graph_prob = (
        graph_probabilities_with_reference_context(
            gcn_model,
            valid_df,
            train_df,
            graph_medians,
            graph_categories,
            graph_scaler
        )
    )

    # --------------------------------------------------------
    # Locked tri-modal fusion
    # --------------------------------------------------------

    final_prob = (
        W_BIO * bio_prob
        +
        W_GRAPH * graph_prob
        +
        W_SEQ * seq_prob
    )

    predicted_class = np.argmax(
        final_prob,
        axis=1
    )

    confidence = np.max(
        final_prob,
        axis=1
    )

    prediction_columns = pd.DataFrame({

        "seq_prob_benign":
            seq_prob[:, 0],

        "seq_prob_pathogenic":
            seq_prob[:, 1],

        "seq_prob_vus":
            seq_prob[:, 2],

        "bio_prob_benign":
            bio_prob[:, 0],

        "bio_prob_pathogenic":
            bio_prob[:, 1],

        "bio_prob_vus":
            bio_prob[:, 2],

        "graph_prob_benign":
            graph_prob[:, 0],

        "graph_prob_pathogenic":
            graph_prob[:, 1],

        "graph_prob_vus":
            graph_prob[:, 2],

        "final_prob_benign":
            final_prob[:, 0],

        "final_prob_pathogenic":
            final_prob[:, 1],

        "final_prob_vus":
            final_prob[:, 2],

        "predicted_label":
            predicted_class,

        "predicted_class":
            [
                CLASS_NAMES[
                    int(x)
                ]
                for x in predicted_class
            ],

        "confidence":
            confidence
    })

    valid_result = pd.concat(
        [
            valid_df.reset_index(
                drop=True
            ),
            prediction_columns
        ],
        axis=1
    )

    # --------------------------------------------------------
    # Reinsert skipped variants
    # --------------------------------------------------------

    final_output = (
        all_rows.copy()
    )

    prediction_cols = [
        col
        for col
        in prediction_columns.columns
    ]

    for col in prediction_cols:

        final_output[
            col
        ] = np.nan

    valid_indices = (
        np.where(
            valid_mask.to_numpy()
        )[0]
    )

    for col in prediction_cols:

        final_output.loc[
            valid_indices,
            col
        ] = (
            valid_result[
                col
            ]
            .to_numpy()
        )

    # --------------------------------------------------------
    # Save compact output
    # --------------------------------------------------------

    preferred_columns = [
        "Chromosome",
        "Start",
        "ReferenceAlleleVCF",
        "AlternateAlleleVCF",
        "variant_key",
        "GeneSymbol",
        "molecular_consequence_raw",
        "consequence_category",
        "phylop",
        "phastcons",
        "status",

        "seq_prob_benign",
        "seq_prob_pathogenic",
        "seq_prob_vus",

        "bio_prob_benign",
        "bio_prob_pathogenic",
        "bio_prob_vus",

        "graph_prob_benign",
        "graph_prob_pathogenic",
        "graph_prob_vus",

        "final_prob_benign",
        "final_prob_pathogenic",
        "final_prob_vus",

        "predicted_label",
        "predicted_class",
        "confidence"
    ]

    preferred_columns = [
        col
        for col in preferred_columns
        if col in final_output.columns
    ]

    final_output = final_output[
        preferred_columns
    ]

    final_output.to_csv(
        output_path,
        index=False
    )

    print(
        "\nPrediction complete."
    )

    print(
        "Saved:",
        output_path
    )

    print(
        "\nPredicted class counts:"
    )

    print(
        final_output[
            "predicted_class"
        ]
        .value_counts(
            dropna=False
        )
    )

    return final_output


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "GenoHybrid-VEP "
            "GRCh38 VCF -> "
            "Benign/Pathogenic/VUS predictions"
        )
    )

    parser.add_argument(
        "--vcf",
        required=True,
        help=(
            "Input GRCh38 VCF/VCF.GZ"
        )
    )

    parser.add_argument(
        "--output",
        default=(
            "genohybrid_predictions.csv"
        ),
        help=(
            "Output CSV path"
        )
    )

    parser.add_argument(
        "--project-dir",
        default=(
            "/content/drive/MyDrive/"
            "variant_effect_prediction"
        ),
        help=(
            "GenoHybrid-VEP project directory"
        )
    )

    args = parser.parse_args()

    predict_vcf(
        vcf_path=Path(
            args.vcf
        ),
        output_path=Path(
            args.output
        ),
        project_dir=Path(
            args.project_dir
        )
    )


if __name__ == "__main__":
    main()
