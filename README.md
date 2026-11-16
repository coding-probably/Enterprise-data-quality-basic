<<<<<<< HEAD
=======
# Enterprise Data Quality & Observability Platform

A basic Python + pandas implementation that combines dataset profiling, automated validation,
issue reporting, historical distribution monitoring, and a weighted data quality score.

## 1. Setup

```bash
pip install pandas numpy matplotlib
```

Matplotlib is optional. If it is not installed, profiling and validation still run; only charts are skipped.

## 2. Folder structure

```text
enterprise_data_quality_platform/
├── data/
│   └── customer_data.csv       # Put your dataset here
├── history/
├── reports/
├── visualizations/
├── config.json
├── data_quality_platform.py
└── README.md
```

## 3. Run

```bash
python data_quality_platform.py
```

Change `dataset_path` in `config.json` if your CSV has a different name or location.

## 4. Configure your own rules

Edit `config.json`. Examples:

- `required_columns`: columns that must exist and must not contain nulls.
- `unique_columns`: primary/candidate keys that must be unique.
- `unique_combinations`: composite unique keys, e.g. `[["Account_ID", "Trade_Date"]]`.
- `expected_types`: `integer`, `number`, `date`, or `text`.
- `ranges`: numeric min/max rules.
- `regex`: regular-expression format checks.
- `allowed_values`: accepted category values.
- `date_rules`: date parsing, not-in-future, and min/max checks.
- `foreign_keys`: compare a column to a reference CSV.
- `accuracy_checks`: compare values to a trusted reference CSV using a key.
- `outlier_columns`: columns to check using IQR or Z-score.
- `freshness`: maximum age in hours; can use file modified time or a timestamp column.
- `distribution_monitoring`: compare category proportions with the first saved baseline.

### Foreign-key example

Put `customers_reference.csv` in the project folder or adjust the path:

```json
"foreign_keys": [
  {
    "column": "Customer_ID",
    "reference_file": "data/customers_reference.csv",
    "reference_column": "Customer_ID"
  }
]
```

### Accuracy/reference comparison example

```json
"accuracy_checks": [
  {
    "column": "Country",
    "key": "Customer_ID",
    "reference_file": "data/trusted_customer_reference.csv"
  }
]
```

The reference file must contain both the configured `key` and `column`.

### Distribution monitoring

Set `enabled` to `true`. The first run saves `history/distribution_baseline.json`.
Later runs compare category shares against that baseline. Set `update_baseline` to `true` only
when you intentionally want to replace the baseline. This simple implementation monitors
categorical distributions; it is not a full statistical drift detector.

## 5. Output reports

- `reports/column_profile.csv`: type, missing %, unique count, duplicate occurrences, numeric/date summaries.
- `reports/categorical_frequency.csv`: category counts and percentages.
- `reports/quality_issues.csv`: issue ID, rule, dimension, affected row indexes, severity, status and message.
- `reports/quality_score.csv`: dimension scores and overall weighted score.
- `reports/run_summary.json`: run-level summary.
- `visualizations/`: numeric histograms and correlation heatmap (when matplotlib is installed).

## 6. Important design notes

- Accuracy is only checked when a trusted reference file is configured. Statistical outliers are
  warnings, not proof of inaccurate data.
- Referential integrity and accuracy checks require reference CSVs.
- Freshness uses file modification time unless `timestamp_column` is configured.
- The first distribution-monitoring run creates a baseline. Run it again on a later dataset to compare.
- The quality score is a practical starter metric: it penalizes the proportion of distinct affected
  rows per dimension. Tune the weights and scoring policy for your actual business rules.
- The program does not delete, overwrite, or automatically correct source data.
- This is a single-process CSV prototype. For production use, add run history, persistent storage,
  authentication, audit logs, stronger drift metrics, and tests.
>>>>>>> a7c72b9 (Initial commit)
