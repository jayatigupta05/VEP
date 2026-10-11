#!/usr/bin/env python3
"""
Simple local testing UI for GenoHybrid-VEP.

Run:
    pip install flask
    python app.py

Then open:
    http://127.0.0.1:5000

Expected files in the same folder:
    app.py
    genohybrid_vcf_predict.py

The predictor will use:
    /content/drive/MyDrive/variant_effect_prediction

by default. Change PROJECT_DIR below if running outside Colab/Drive.
"""

from pathlib import Path
import uuid

from flask import (
    Flask,
    request,
    render_template_string,
    send_file,
    flash,
    redirect,
    url_for
)

import pandas as pd

from genohybrid_vcf_predict import predict_vcf


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(
    "/content/drive/MyDrive/variant_effect_prediction"
)

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "prediction_outputs"

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ALLOWED_SUFFIXES = (
    ".vcf",
    ".vcf.gz"
)

MAX_UPLOAD_MB = 50


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

app.secret_key = (
    "genohybrid-local-testing-key"
)

app.config[
    "MAX_CONTENT_LENGTH"
] = (
    MAX_UPLOAD_MB
    * 1024
    * 1024
)


# ============================================================
# HTML
# ============================================================

PAGE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <title>GenoHybrid-VEP</title>

    <style>

        body {
            font-family:
                Arial,
                Helvetica,
                sans-serif;

            background:
                #f6f7f9;

            color:
                #222;

            margin:
                0;

            padding:
                0;
        }

        .container {
            width:
                min(
                    1100px,
                    92%
                );

            margin:
                40px auto;
        }

        .card {
            background:
                white;

            border:
                1px solid #e1e4e8;

            border-radius:
                12px;

            padding:
                24px;

            margin-bottom:
                24px;

            box-shadow:
                0 2px 8px
                rgba(
                    0,
                    0,
                    0,
                    0.04
                );
        }

        h1 {
            margin-top:
                0;

            font-size:
                28px;
        }

        h2 {
            font-size:
                20px;

            margin-top:
                0;
        }

        .subtitle {
            color:
                #666;

            margin-bottom:
                24px;
        }

        .upload-box {
            border:
                2px dashed #c8ccd0;

            border-radius:
                10px;

            padding:
                28px;

            text-align:
                center;

            background:
                #fafbfc;
        }

        input[type=file] {
            margin:
                16px 0;
        }

        button,
        .button-link {
            background:
                #111;

            color:
                white;

            border:
                none;

            border-radius:
                8px;

            padding:
                11px 18px;

            cursor:
                pointer;

            font-size:
                14px;

            text-decoration:
                none;

            display:
                inline-block;
        }

        button:hover,
        .button-link:hover {
            opacity:
                0.88;
        }

        .flash {
            background:
                #fff2f2;

            border:
                1px solid #f2b9b9;

            color:
                #8b1e1e;

            padding:
                12px;

            border-radius:
                8px;

            margin-bottom:
                18px;
        }

        .summary-grid {
            display:
                grid;

            grid-template-columns:
                repeat(
                    auto-fit,
                    minmax(
                        160px,
                        1fr
                    )
                );

            gap:
                12px;

            margin-bottom:
                20px;
        }

        .metric {
            background:
                #f8f9fa;

            border:
                1px solid #eceff1;

            border-radius:
                9px;

            padding:
                14px;
        }

        .metric-label {
            color:
                #666;

            font-size:
                12px;

            margin-bottom:
                5px;
        }

        .metric-value {
            font-size:
                21px;

            font-weight:
                600;
        }

        .table-wrap {
            overflow-x:
                auto;
        }

        table {
            width:
                100%;

            border-collapse:
                collapse;

            font-size:
                13px;
        }

        th,
        td {
            border-bottom:
                1px solid #e9ecef;

            padding:
                9px;

            text-align:
                left;

            white-space:
                nowrap;
        }

        th {
            background:
                #f8f9fa;

            position:
                sticky;

            top:
                0;
        }

        .small {
            color:
                #777;

            font-size:
                12px;
        }

        .warning {
            background:
                #fff8e5;

            border:
                1px solid #f3d27a;

            padding:
                12px;

            border-radius:
                8px;

            margin:
                12px 0;
        }

    </style>
</head>


<body>

<div class="container">

    <div class="card">

        <h1>
            GenoHybrid-VEP
        </h1>

        <div class="subtitle">
            Local GRCh38 VCF pathogenicity prediction test interface
        </div>

        {% with messages = get_flashed_messages() %}
            {% if messages %}

                {% for message in messages %}

                    <div class="flash">
                        {{ message }}
                    </div>

                {% endfor %}

            {% endif %}
        {% endwith %}


        <form
            method="POST"
            action="/predict"
            enctype="multipart/form-data"
        >

            <div class="upload-box">

                <strong>
                    Upload a GRCh38 VCF
                </strong>

                <br>

                <span class="small">
                    Supported:
                    .vcf and .vcf.gz
                    · Max {{ max_upload_mb }} MB
                </span>

                <br>

                <input
                    type="file"
                    name="vcf_file"
                    accept=".vcf,.vcf.gz"
                    required
                >

                <br>

                <button type="submit">
                    Run prediction
                </button>

            </div>

        </form>

        <div class="warning">

            Best results require the VCF to contain
            Ensembl VEP <code>CSQ</code>
            or SnpEff <code>ANN</code>
            annotations for gene and consequence information.

        </div>

    </div>


    {% if result %}

        <div class="card">

            <h2>
                Prediction summary
            </h2>

            <div class="summary-grid">

                <div class="metric">

                    <div class="metric-label">
                        Total records
                    </div>

                    <div class="metric-value">
                        {{ result.total }}
                    </div>

                </div>


                <div class="metric">

                    <div class="metric-label">
                        Valid predictions
                    </div>

                    <div class="metric-value">
                        {{ result.valid }}
                    </div>

                </div>


                <div class="metric">

                    <div class="metric-label">
                        Benign
                    </div>

                    <div class="metric-value">
                        {{ result.benign }}
                    </div>

                </div>


                <div class="metric">

                    <div class="metric-label">
                        Pathogenic
                    </div>

                    <div class="metric-value">
                        {{ result.pathogenic }}
                    </div>

                </div>


                <div class="metric">

                    <div class="metric-label">
                        VUS
                    </div>

                    <div class="metric-value">
                        {{ result.vus }}
                    </div>

                </div>

            </div>


            <a
                class="button-link"
                href="{{ result.download_url }}"
            >
                Download full CSV
            </a>

        </div>


        <div class="card">

            <h2>
                Preview
            </h2>

            <div class="small">
                Showing up to the first
                {{ result.preview_count }}
                rows.
            </div>

            <br>

            <div class="table-wrap">

                {{ result.table_html | safe }}

            </div>

        </div>

    {% endif %}

</div>

</body>

</html>
"""


# ============================================================
# HELPERS
# ============================================================

def valid_filename(
    filename
):
    """
    Accept .vcf or .vcf.gz only.
    """

    filename = (
        filename
        .lower()
        .strip()
    )

    return (
        filename.endswith(
            ".vcf"
        )
        or
        filename.endswith(
            ".vcf.gz"
        )
    )


def make_summary(
    df,
    output_filename
):

    predicted = (
        df[
            "predicted_class"
        ]
        if "predicted_class"
        in df.columns
        else pd.Series(
            dtype=str
        )
    )

    valid_count = int(
        predicted
        .notna()
        .sum()
    )

    benign = int(
        (
            predicted
            == "Benign"
        )
        .sum()
    )

    pathogenic = int(
        (
            predicted
            == "Pathogenic"
        )
        .sum()
    )

    vus = int(
        (
            predicted
            == "VUS"
        )
        .sum()
    )

    preview_columns = [
        "Chromosome",
        "Start",
        "ReferenceAlleleVCF",
        "AlternateAlleleVCF",
        "GeneSymbol",
        "consequence_category",
        "predicted_class",
        "confidence",
        "final_prob_benign",
        "final_prob_pathogenic",
        "final_prob_vus",
        "status"
    ]

    preview_columns = [
        column
        for column
        in preview_columns
        if column
        in df.columns
    ]

    preview_df = (
        df[
            preview_columns
        ]
        .head(
            50
        )
        .copy()
    )

    probability_columns = [
        "confidence",
        "final_prob_benign",
        "final_prob_pathogenic",
        "final_prob_vus"
    ]

    for column in probability_columns:

        if column in preview_df.columns:

            preview_df[
                column
            ] = (
                preview_df[
                    column
                ]
                .round(
                    4
                )
            )

    table_html = (
        preview_df
        .to_html(
            index=False,
            classes="result-table",
            border=0,
            escape=True
        )
    )

    return {
        "total":
            len(
                df
            ),

        "valid":
            valid_count,

        "benign":
            benign,

        "pathogenic":
            pathogenic,

        "vus":
            vus,

        "preview_count":
            min(
                len(df),
                50
            ),

        "table_html":
            table_html,

        "download_url":
            url_for(
                "download_result",
                filename=output_filename
            )
    }


# ============================================================
# ROUTES
# ============================================================

@app.route(
    "/",
    methods=[
        "GET"
    ]
)
def index():

    return render_template_string(
        PAGE_TEMPLATE,
        result=None,
        max_upload_mb=MAX_UPLOAD_MB
    )


@app.route(
    "/predict",
    methods=[
        "POST"
    ]
)
def predict():

    if "vcf_file" not in request.files:

        flash(
            "No file was uploaded."
        )

        return redirect(
            url_for(
                "index"
            )
        )

    uploaded_file = request.files[
        "vcf_file"
    ]

    if (
        uploaded_file.filename
        is None
        or uploaded_file.filename
        == ""
    ):

        flash(
            "Please select a VCF file."
        )

        return redirect(
            url_for(
                "index"
            )
        )

    if not valid_filename(
        uploaded_file.filename
    ):

        flash(
            "Only .vcf and .vcf.gz files are supported."
        )

        return redirect(
            url_for(
                "index"
            )
        )


    run_id = (
        uuid.uuid4()
        .hex[
            :10
        ]
    )

    if (
        uploaded_file.filename
        .lower()
        .endswith(
            ".vcf.gz"
        )
    ):

        upload_path = (
            UPLOAD_DIR
            /
            f"{run_id}.vcf.gz"
        )

    else:

        upload_path = (
            UPLOAD_DIR
            /
            f"{run_id}.vcf"
        )


    output_filename = (
        f"genohybrid_predictions_{run_id}.csv"
    )

    output_path = (
        OUTPUT_DIR
        /
        output_filename
    )


    uploaded_file.save(
        upload_path
    )


    try:

        prediction_df = predict_vcf(
            vcf_path=upload_path,
            output_path=output_path,
            project_dir=PROJECT_DIR
        )


    except Exception as error:

        flash(
            f"Prediction failed: {error}"
        )

        return redirect(
            url_for(
                "index"
            )
        )


    result = make_summary(
        prediction_df,
        output_filename
    )


    return render_template_string(
        PAGE_TEMPLATE,
        result=result,
        max_upload_mb=MAX_UPLOAD_MB
    )


@app.route(
    "/download/<filename>",
    methods=[
        "GET"
    ]
)
def download_result(
    filename
):

    file_path = (
        OUTPUT_DIR
        /
        filename
    )

    if not file_path.exists():

        flash(
            "Requested result file was not found."
        )

        return redirect(
            url_for(
                "index"
            )
        )

    return send_file(
        file_path,
        as_attachment=True,
        download_name=filename
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print(
        "\nGenoHybrid-VEP local interface"
    )

    print(
        "Open:"
    )

    print(
        "http://127.0.0.1:5000\n"
    )

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )
