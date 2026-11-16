
"""
ENTERPRISE DATA QUALITY & OBSERVABILITY PLATFORM

Features:
1. CSV data loading
2. Data profiling
3. Categorical frequency analysis
4. Data visualizations
5. Completeness validation
6. Uniqueness validation
7. Data type validation
8. Range validation
9. Regex validation
10. Allowed category validation
11. Date validation
12. Referential integrity validation
13. Standardization validation
14. Accuracy checks against reference datasets
15. Outlier detection
16. Data freshness monitoring
17. Schema validation
18. Historical distribution drift monitoring
19. Data quality scoring
20. Cleaning-compatible issue report generation

Run:
    py -3.15 data_quality_platform.py

Dependencies:
    py -3.15 -m pip install pandas numpy matplotlib
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import re
import warnings

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ============================================================
# 1. PROJECT PATHS AND DEFAULT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"

DEFAULT_CONFIG = {
    "dataset_path": "data/customer_data.csv",
    "reports_dir": "reports",
    "visualizations_dir": "visualizations",
    "history_dir": "history",
    "categorical_columns": [],

    "schema": {
        "expected_columns": [],
        "allow_extra_columns": True
    },

    "quality_weights": {
        "completeness": 0.20,
        "accuracy": 0.20,
        "uniqueness": 0.15,
        "consistency": 0.15,
        "validity": 0.15,
        "timeliness": 0.15
    },

    "rules": {
        "required_columns": [],
        "blank_check_columns": [],
        "unique_columns": [],
        "unique_combinations": [],
        "check_duplicate_rows": True,
        "expected_types": {},
        "ranges": {},
        "regex": {},
        "allowed_values": {},
        "date_rules": {},
        "foreign_keys": [],
        "standard_values": {},
        "accuracy_checks": [],
        "outlier_columns": [],
        "outlier_method": "iqr",
        "zscore_threshold": 3.0,
        "freshness": {},
        "distribution_monitoring": {
            "enabled": False,
            "columns": [],
            "max_category_share_change": 0.20,
            "update_baseline": False
        }
    }
}


def load_config():
    """Load configuration or create a default configuration."""

    if not CONFIG_PATH.exists():
        print("config.json not found. Creating default configuration.")

        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)

        return DEFAULT_CONFIG.copy()

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config


# ============================================================
# 2. DATASET LOADING
# ============================================================

def load_dataset(path):
    """Load a CSV dataset and normalize whitespace-only values."""

    path = Path(path)

    if not path.is_absolute():
        path = ROOT / path

    if not path.exists():
        raise FileNotFoundError(
            f"\nDataset not found: {path}\n"
            "Place your CSV inside the data folder or "
            "change dataset_path in config.json."
        )

    df = pd.read_csv(path)

    # Reset index so row numbers remain consistent with the
    # original CSV and the interactive cleaning script.
    df = df.reset_index(drop=True)

    # Convert whitespace-only strings into missing values.
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].map(
            lambda value: (
                np.nan
                if isinstance(value, str) and value.strip() == ""
                else value
            )
        )

    if df.empty:
        raise ValueError("The CSV dataset contains no data rows.")

    if len(df.columns) == 0:
        raise ValueError("The CSV dataset contains no columns.")

    return df, path


# ============================================================
# 3. DATA TYPE DETECTION
# ============================================================

def detect_type(series):
    """Automatically detect a column's general data type."""

    non_null = series.dropna()

    if non_null.empty:
        return "empty"

    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if pd.api.types.is_integer_dtype(series):
        return "integer"

    if pd.api.types.is_float_dtype(series):
        return "number"

    numeric = pd.to_numeric(non_null, errors="coerce")

    if numeric.notna().mean() >= 0.90:
        return "number"

    dates = pd.to_datetime(
        non_null,
        format="mixed",
        errors="coerce"
    )

    if dates.notna().mean() >= 0.80:
        return "date"

    return "text"


# ============================================================
# 4. REPORT WRITING
# ============================================================

def write_csv(rows, output_path, columns=None):
    """Write a list of dictionaries to a CSV file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        rows,
        columns=columns
    ).to_csv(
        output_path,
        index=False
    )


# ============================================================
# 5. DATA PROFILING
# ============================================================

def profile_dataset(df, reports_dir):

    rows = []
    total = len(df)

    for col in df.columns:

        series = df[col]
        dtype = detect_type(series)
        non_null = series.dropna()

        row = {
            "column": col,
            "detected_type": dtype,
            "total_rows": total,
            "missing_count": int(series.isna().sum()),
            "missing_percentage": round(
                series.isna().mean() * 100, 2
            ),
            "unique_values": int(series.nunique(dropna=True)),
            "duplicate_occurrences": int(
                non_null.duplicated().sum()
            ),
            "completeness_percentage": round(
                series.notna().mean() * 100, 2
            ),
            "min": None,
            "max": None,
            "mean": None,
            "median": None
        }

        if dtype in ("integer", "number"):

            numeric = pd.to_numeric(
                series,
                errors="coerce"
            ).dropna()

            if not numeric.empty:
                row.update({
                    "min": numeric.min(),
                    "max": numeric.max(),
                    "mean": round(numeric.mean(), 4),
                    "median": round(numeric.median(), 4)
                })

        elif dtype == "date":

            dates = pd.to_datetime(
                series,
                format="mixed",
                errors="coerce"
            ).dropna()

            if not dates.empty:
                row["min"] = str(dates.min())
                row["max"] = str(dates.max())

        rows.append(row)

    write_csv(
        rows,
        reports_dir / "column_profile.csv"
    )

    print("Data profiling completed.")

    return rows


# ============================================================
# 6. CATEGORICAL FREQUENCY ANALYSIS
# ============================================================

def categorical_analysis(df, reports_dir, configured_columns=None):

    configured_columns = configured_columns or []

    rows = []

    for col in df.columns:

        dtype = detect_type(df[col])

        if dtype in ("text", "boolean") or col in configured_columns:

            counts = (
                df[col]
                .fillna("<MISSING>")
                .astype(str)
                .value_counts(dropna=False)
            )

            denominator = len(df)

            for value, count in counts.items():

                rows.append({
                    "column": col,
                    "value": value,
                    "frequency": int(count),
                    "percentage": round(
                        count / denominator * 100, 2
                    ) if denominator else 0
                })

    write_csv(
        rows,
        reports_dir / "categorical_frequency.csv",
        ["column", "value", "frequency", "percentage"]
    )

    print("Categorical analysis completed.")


# ============================================================
# 7. DATA VISUALIZATION
# ============================================================

def create_visualizations(df, viz_dir):

    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not installed. Skipping visualizations.")
        return

    viz_dir.mkdir(parents=True, exist_ok=True)

    hist_dir = viz_dir / "histograms"
    hist_dir.mkdir(parents=True, exist_ok=True)

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    # Generate histograms.
    for col in numeric_cols:

        values = df[col].dropna()

        if values.empty:
            continue

        plt.figure()

        plt.hist(
            values,
            bins=20,
            edgecolor="black"
        )

        plt.title(f"Distribution: {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()

        safe_name = re.sub(r'[\\/*?:"<>|]', "_", str(col))

        plt.savefig(hist_dir / f"{safe_name}.png")
        plt.close()

    # Generate correlation matrix and heatmap.
    if len(numeric_cols) >= 2:

        corr = df[numeric_cols].corr(numeric_only=True)

        corr.to_csv(
            viz_dir / "correlation_matrix.csv"
        )

        plt.figure(figsize=(9, 7))

        plt.imshow(
            corr,
            cmap="coolwarm",
            vmin=-1,
            vmax=1
        )

        plt.colorbar(label="Correlation")

        plt.xticks(
            range(len(corr.columns)),
            corr.columns,
            rotation=45,
            ha="right"
        )

        plt.yticks(
            range(len(corr.index)),
            corr.index
        )

        plt.title("Correlation Matrix")
        plt.tight_layout()

        plt.savefig(
            viz_dir / "correlation_heatmap.png"
        )

        plt.close()

    print("Visualizations completed.")


# ============================================================
# 8. ISSUE GENERATION HELPER
# ============================================================

def add_issue(
    issues,
    rule_id,
    dimension,
    column,
    row_indexes,
    message,
    severity="medium",
    sample_values=None
):

    indexes = list(row_indexes)

    issues.append({

        "issue_id": f"ISS-{len(issues) + 1:05d}",

        "rule_id": rule_id,

        "dimension": dimension,

        "column": column or "",

        "affected_rows": len(indexes),

        "row_indexes": ";".join(
            str(x) for x in indexes[:100]
        ),

        "sample_values": ";".join(
            map(str, (sample_values or [])[:5])
        ),

        "severity": severity,

        "message": message,

        "detected_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "status": "OPEN"

    })


# ============================================================
# 9. CONFIGURED DATA VALIDATION RULES
# ============================================================

def run_configured_rules(
    df,
    config,
    dataset_path,
    reports_dir,
    history_dir
):

    issues = []

    rules = config.get("rules", {})

    # --------------------------------------------------------
    # A. COMPLETENESS: MISSING VALUES
    # --------------------------------------------------------

    for col in rules.get("required_columns", []):

        if col not in df.columns:

            add_issue(
                issues,
                "REQUIRED_COLUMN",
                "completeness",
                col,
                [],
                f"Required column '{col}' is missing.",
                "critical"
            )

            continue

        bad = df.index[
            df[col].isna()
        ].tolist()

        if bad:

            add_issue(
                issues,
                "NOT_NULL",
                "completeness",
                col,
                bad,
                f"Required column '{col}' contains missing values.",
                "high"
            )

    # --------------------------------------------------------
    # B. COMPLETENESS: BLANK VALUES
    # --------------------------------------------------------

    for col in rules.get("blank_check_columns", []):

        if col in df.columns:

            mask = (
                df[col]
                .astype("string")
                .str.strip()
                .eq("")
                .fillna(False)
            )

            bad = df.index[mask].tolist()

            if bad:

                add_issue(
                    issues,
                    "NO_BLANKS",
                    "completeness",
                    col,
                    bad,
                    f"Column '{col}' contains blank values.",
                    "medium"
                )

    # --------------------------------------------------------
    # C. UNIQUENESS: UNIQUE COLUMNS
    # --------------------------------------------------------

    for col in rules.get("unique_columns", []):

        if col in df.columns:

            # Only flag subsequent duplicates.
            mask = (
                df[col].notna()
                & df[col].duplicated(keep="first")
            )

            bad = df.index[mask].tolist()

            if bad:

                add_issue(
                    issues,
                    "UNIQUE_COLUMN",
                    "uniqueness",
                    col,
                    bad,
                    f"Column '{col}' contains duplicate values.",
                    "high",
                    df.loc[mask, col].astype(str).tolist()
                )

    # --------------------------------------------------------
    # D. UNIQUENESS: COMPOSITE KEYS
    # --------------------------------------------------------

    for cols in rules.get("unique_combinations", []):

        missing_cols = [
            col for col in cols
            if col not in df.columns
        ]

        if missing_cols:

            add_issue(
                issues,
                "COMPOSITE_COLUMNS_MISSING",
                "schema",
                ",".join(cols),
                [],
                f"Missing columns: {missing_cols}",
                "high"
            )

            continue

        mask = df.duplicated(
            subset=cols,
            keep="first"
        )

        bad = df.index[mask].tolist()

        if bad:

            add_issue(
                issues,
                "UNIQUE_COMBINATION",
                "uniqueness",
                ",".join(cols),
                bad,
                f"Combination {cols} contains duplicate records.",
                "high"
            )

    # --------------------------------------------------------
    # E. UNIQUENESS: DUPLICATE ROWS
    # --------------------------------------------------------

    if rules.get("check_duplicate_rows", True):

        mask = df.duplicated(keep="first")

        bad = df.index[mask].tolist()

        if bad:

            add_issue(
                issues,
                "DUPLICATE_ROW",
                "uniqueness",
                "*",
                bad,
                "Entire duplicate rows were detected.",
                "medium"
            )

    # --------------------------------------------------------
    # F. EXPECTED DATA TYPES
    # --------------------------------------------------------

    expected_types = rules.get("expected_types", {})

    for col, expected in expected_types.items():

        if col not in df.columns:

            add_issue(
                issues,
                "TYPE_COLUMN_MISSING",
                "schema",
                col,
                [],
                f"Expected type configured for missing column '{col}'.",
                "high"
            )

            continue

        expected = expected.lower()

        series = df[col]

        if expected in ("integer", "int"):

            converted = pd.to_numeric(
                series,
                errors="coerce"
            )

            bad_mask = series.notna() & (
                converted.isna()
                | ((converted % 1) != 0)
            )

        elif expected in ("number", "float", "numeric"):

            converted = pd.to_numeric(
                series,
                errors="coerce"
            )

            bad_mask = series.notna() & converted.isna()

        elif expected in ("date", "datetime"):

            converted = pd.to_datetime(
                series,
                format="mixed",
                errors="coerce"
            )

            bad_mask = series.notna() & converted.isna()

        elif expected in ("text", "string"):

            bad_mask = series.notna() & ~series.map(
                lambda value: isinstance(value, str)
            )

        else:
            continue

        bad = df.index[bad_mask].tolist()

        if bad:

            add_issue(
                issues,
                "EXPECTED_TYPE",
                "validity",
                col,
                bad,
                f"Values in '{col}' do not match expected type '{expected}'.",
                "medium",
                series.loc[bad_mask].astype(str).tolist()
            )

    # --------------------------------------------------------
    # G. NUMERIC RANGE VALIDATION
    # --------------------------------------------------------

    for col, limits in rules.get("ranges", {}).items():

        if col not in df.columns:

            add_issue(
                issues,
                "RANGE_COLUMN_MISSING",
                "schema",
                col,
                [],
                f"Range rule refers to missing column '{col}'.",
                "high"
            )

            continue

        values = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        mask = df[col].notna() & values.isna()

        if "min" in limits:
            mask |= values.notna() & (
                values < limits["min"]
            )

        if "max" in limits:
            mask |= values.notna() & (
                values > limits["max"]
            )

        bad = df.index[mask].tolist()

        if bad:

            add_issue(
                issues,
                "VALUE_RANGE",
                "validity",
                col,
                bad,
                f"Values in '{col}' are outside range {limits}.",
                "medium",
                df.loc[mask, col].astype(str).tolist()
            )

    # --------------------------------------------------------
    # H. REGEX FORMAT VALIDATION
    # --------------------------------------------------------

    for col, pattern in rules.get("regex", {}).items():

        if col not in df.columns:

            add_issue(
                issues,
                "REGEX_COLUMN_MISSING",
                "schema",
                col,
                [],
                f"Regex rule refers to missing column '{col}'.",
                "high"
            )

            continue

        mask = (
            df[col].notna()
            & ~df[col].astype(str).str.match(
                pattern,
                na=False
            )
        )

        bad = df.index[mask].tolist()

        if bad:

            add_issue(
                issues,
                "REGEX_FORMAT",
                "validity",
                col,
                bad,
                f"Values in '{col}' do not match the configured pattern.",
                "medium",
                df.loc[mask, col].astype(str).tolist()
            )

    # --------------------------------------------------------
    # I. ALLOWED CATEGORY VALIDATION
    # --------------------------------------------------------

    for col, allowed in rules.get("allowed_values", {}).items():

        if col not in df.columns:

            add_issue(
                issues,
                "CATEGORY_COLUMN_MISSING",
                "schema",
                col,
                [],
                f"Allowed-values rule refers to missing column '{col}'.",
                "high"
            )

            continue

        mask = (
            df[col].notna()
            & ~df[col].isin(allowed)
        )

        bad = df.index[mask].tolist()

        if bad:

            add_issue(
                issues,
                "ALLOWED_VALUES",
                "validity",
                col,
                bad,
                f"Column '{col}' contains values outside allowed categories.",
                "medium",
                df.loc[mask, col].astype(str).tolist()
            )

    # --------------------------------------------------------
    # J. DATE VALIDATION
    # --------------------------------------------------------

    for col, date_rule in rules.get("date_rules", {}).items():

        if col not in df.columns:

            add_issue(
                issues,
                "DATE_COLUMN_MISSING",
                "schema",
                col,
                [],
                f"Date rule refers to missing column '{col}'.",
                "high"
            )

            continue

        dates = pd.to_datetime(
            df[col],
            format="mixed",
            errors="coerce"
        )

        mask = df[col].notna() & dates.isna()

        if date_rule.get("not_in_future", False):

            today = pd.Timestamp.now().normalize()

            mask |= dates.notna() & (dates > today)

        if "min" in date_rule:

            mask |= dates.notna() & (
                dates < pd.Timestamp(date_rule["min"])
            )

        if "max" in date_rule:

            mask |= dates.notna() & (
                dates > pd.Timestamp(date_rule["max"])
            )

        bad = df.index[mask].tolist()

        if bad:

            add_issue(
                issues,
                "DATE_RULE",
                "validity",
                col,
                bad,
                f"Column '{col}' violates configured date rules.",
                "medium",
                df.loc[mask, col].astype(str).tolist()
            )

    # --------------------------------------------------------
    # K. FOREIGN KEY VALIDATION
    # --------------------------------------------------------

    for fk in rules.get("foreign_keys", []):

        child_col = fk["column"]
        ref_col = fk["reference_column"]
        ref_path = ROOT / fk["reference_file"]

        if child_col not in df.columns:

            add_issue(
                issues,
                "FK_CHILD_MISSING",
                "consistency",
                child_col,
                [],
                f"Foreign-key child column '{child_col}' is missing.",
                "high"
            )

            continue

        if not ref_path.exists():

            add_issue(
                issues,
                "FK_REFERENCE_MISSING",
                "consistency",
                child_col,
                [],
                f"Reference file '{fk['reference_file']}' was not found.",
                "high"
            )

            continue

        ref_df = pd.read_csv(ref_path)

        if ref_col not in ref_df.columns:

            add_issue(
                issues,
                "FK_REFERENCE_COLUMN_MISSING",
                "consistency",
                ref_col,
                [],
                f"Reference column '{ref_col}' was not found.",
                "high"
            )

            continue

        valid_values = set(
            ref_df[ref_col].dropna().astype(str)
        )

        mask = (
            df[child_col].notna()
            & ~df[child_col].astype(str).isin(valid_values)
        )

        bad = df.index[mask].tolist()

        if bad:

            add_issue(
                issues,
                "FOREIGN_KEY",
                "consistency",
                child_col,
                bad,
                f"Values in '{child_col}' do not exist in reference column '{ref_col}'.",
                "high",
                df.loc[mask, child_col].astype(str).tolist()
            )

    # --------------------------------------------------------
    # L. STANDARD VALUE VALIDATION
    # --------------------------------------------------------

    for col, mapping in rules.get("standard_values", {}).items():

        if col not in df.columns:
            continue

        allowed = set(mapping.keys())

        mask = (
            df[col].notna()
            & ~df[col].astype(str).isin(allowed)
        )

        bad = df.index[mask].tolist()

        if bad:

            add_issue(
                issues,
                "STANDARD_VALUE",
                "consistency",
                col,
                bad,
                f"Column '{col}' contains non-standard labels.",
                "low",
                df.loc[mask, col].astype(str).tolist()
            )

    # --------------------------------------------------------
    # M. ACCURACY VALIDATION AGAINST REFERENCE DATA
    # --------------------------------------------------------

    for check in rules.get("accuracy_checks", []):

        col = check["column"]
        key = check["key"]
        ref_file = ROOT / check["reference_file"]

        if col not in df.columns:
            continue

        if not ref_file.exists():

            add_issue(
                issues,
                "ACCURACY_REFERENCE_MISSING",
                "accuracy",
                col,
                [],
                f"Accuracy reference file '{check['reference_file']}' was not found.",
                "high"
            )

            continue

        ref_df = pd.read_csv(ref_file)

        if (
            key not in df.columns
            or key not in ref_df.columns
            or col not in ref_df.columns
        ):

            add_issue(
                issues,
                "ACCURACY_COLUMNS_MISSING",
                "accuracy",
                col,
                [],
                "Accuracy comparison requires matching key and value columns.",
                "high"
            )

            continue

        lookup = (
            ref_df.drop_duplicates(key)
            .set_index(key)[col]
            .astype(str)
        )

        expected = df[key].map(lookup)

        mask = (
            df[key].notna()
            & expected.notna()
            & df[col].notna()
            & (df[col].astype(str) != expected)
        )

        bad = df.index[mask].tolist()

        if bad:

            add_issue(
                issues,
                "REFERENCE_MISMATCH",
                "accuracy",
                col,
                bad,
                f"Column '{col}' differs from the trusted reference dataset.",
                "high",
                df.loc[mask, col].astype(str).tolist()
            )

    # --------------------------------------------------------
    # N. OUTLIER DETECTION
    # --------------------------------------------------------

    for col in rules.get("outlier_columns", []):

        if col not in df.columns:
            continue

        values = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        valid = values.dropna()

        if len(valid) < 4:
            continue

        method = rules.get(
            "outlier_method",
            "iqr"
        ).lower()

        if method == "zscore":

            std = valid.std()

            if std == 0 or pd.isna(std):
                continue

            scores = (
                values - valid.mean()
            ).abs() / std

            mask = scores > rules.get(
                "zscore_threshold",
                3.0
            )

        else:

            q1, q3 = valid.quantile([0.25, 0.75])

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            mask = (
                values.notna()
                & ((values < lower) | (values > upper))
            )

        bad = df.index[
            mask.fillna(False)
        ].tolist()

        if bad:

            add_issue(
                issues,
                "OUTLIER",
                "accuracy",
                col,
                bad,
                f"Potential statistical outliers detected in '{col}' using {method.upper()}.",
                "low",
                df.loc[bad, col].astype(str).tolist()
            )

    # --------------------------------------------------------
    # O. DATA FRESHNESS VALIDATION
    # --------------------------------------------------------

    freshness = rules.get("freshness", {})

    max_age_hours = freshness.get("max_age_hours")

    if max_age_hours is not None:

        timestamp_column = freshness.get("timestamp_column")

        if timestamp_column:

            if timestamp_column in df.columns:

                timestamps = pd.to_datetime(
                    df[timestamp_column],
                    format="mixed",
                    errors="coerce",
                    utc=True
                )

                latest = timestamps.max()

                if pd.notna(latest):

                    age_hours = (
                        pd.Timestamp.now(tz="UTC") - latest
                    ).total_seconds() / 3600

                else:
                    age_hours = float("inf")

            else:
                age_hours = float("inf")

        else:

            modified = datetime.fromtimestamp(
                dataset_path.stat().st_mtime,
                tz=timezone.utc
            )

            age_hours = (
                datetime.now(timezone.utc) - modified
            ).total_seconds() / 3600

        if age_hours > max_age_hours:

            add_issue(
                issues,
                "STALE_DATASET",
                "timeliness",
                "",
                [],
                f"Dataset is {age_hours:.1f} hours old; "
                f"maximum allowed is {max_age_hours} hours.",
                "high"
            )

    # --------------------------------------------------------
    # P. SCHEMA VALIDATION
    # --------------------------------------------------------

    schema = config.get("schema", {})

    expected_columns = schema.get(
        "expected_columns",
        []
    )

    missing = [
        col for col in expected_columns
        if col not in df.columns
    ]

    for col in missing:

        add_issue(
            issues,
            "SCHEMA_MISSING_COLUMN",
            "validity",
            col,
            [],
            f"Expected schema column '{col}' is missing.",
            "critical"
        )

    if schema.get("allow_extra_columns", True) is False:

        extra = [
            col for col in df.columns
            if col not in expected_columns
        ]

        for col in extra:

            add_issue(
                issues,
                "SCHEMA_EXTRA_COLUMN",
                "validity",
                col,
                [],
                f"Unexpected column '{col}' is present.",
                "low"
            )

    # --------------------------------------------------------
    # Q. HISTORICAL DISTRIBUTION DRIFT MONITORING
    # --------------------------------------------------------

    drift_cfg = rules.get(
        "distribution_monitoring",
        {}
    )

    if drift_cfg.get("enabled", False):

        baseline_file = (
            history_dir / "distribution_baseline.json"
        )

        monitored = drift_cfg.get(
            "columns",
            []
        )

        current = {}

        for col in monitored:

            if col in df.columns:

                counts = (
                    df[col]
                    .fillna("<MISSING>")
                    .astype(str)
                    .value_counts(normalize=True)
                )

                current[col] = counts.to_dict()

        if baseline_file.exists():

            with open(
                baseline_file,
                "r",
                encoding="utf-8"
            ) as f:

                baseline = json.load(f)

            threshold = drift_cfg.get(
                "max_category_share_change",
                0.20
            )

            for col, distribution in current.items():

                old = baseline.get(col, {})

                categories = set(old) | set(distribution)

                changed = []

                for category in categories:

                    delta = abs(
                        distribution.get(category, 0)
                        - old.get(category, 0)
                    )

                    if delta > threshold:
                        changed.append(category)

                if changed:

                    add_issue(
                        issues,
                        "DISTRIBUTION_DRIFT",
                        "consistency",
                        col,
                        [],
                        f"Distribution changed beyond {threshold:.0%} "
                        f"for categories: {changed[:10]}.",
                        "medium"
                    )

        if (
            not baseline_file.exists()
            or drift_cfg.get("update_baseline", False)
        ):

            history_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            with open(
                baseline_file,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    current,
                    f,
                    indent=2
                )

    return issues


# ============================================================
# 10. GENERATE CLEANING-COMPATIBLE ISSUE REPORT
# ============================================================

def create_cleaning_issue_report(
    df,
    issues,
    reports_dir
):
    """
    Convert detailed validation issues into individual
    row-level records for data_cleaning.py.

    Row_Number is 1-based.
    """

    issue_type_mapping = {

        "NOT_NULL": "Missing_Value",
        "NO_BLANKS": "Blank_Value",

        "UNIQUE_COLUMN": "Duplicate_Value",
        "UNIQUE_COMBINATION": "Duplicate_Value",
        "DUPLICATE_ROW": "Duplicate_Value",

        "EXPECTED_TYPE": "Invalid_Format",
        "VALUE_RANGE": "Out_Of_Range",
        "REGEX_FORMAT": "Invalid_Format",

        "ALLOWED_VALUES": "Invalid_Category",
        "DATE_RULE": "Invalid_Date",

        "FOREIGN_KEY": "Invalid_Reference",
        "STANDARD_VALUE": "Inconsistent_Value",

        "REFERENCE_MISMATCH": "Accuracy_Mismatch",

        "OUTLIER": "Outlier"
    }

    cleaning_issues = []

    issue_counter = 1

    for issue in issues:

        column = issue["column"]

        # Skip dataset-level and schema-level issues.
        # They do not refer to a specific cell.
        if column not in df.columns:
            continue

        row_indexes = issue["row_indexes"]

        if not row_indexes:
            continue

        for index in row_indexes.split(";"):

            try:
                row_index = int(index)
            except ValueError:
                continue

            if row_index not in df.index:
                continue

            current_value = df.at[
                row_index,
                column
            ]

            if pd.isna(current_value):
                current_value = "NULL"
            else:
                current_value = str(current_value)

            cleaning_issues.append({

                "Issue_ID": issue_counter,

                "Row_Number": row_index + 1,

                "Column_Name": column,

                "Issue_Type": issue_type_mapping.get(
                    issue["rule_id"],
                    "Other_Issue"
                ),

                "Current_Value": current_value,

                "Rule": issue["message"],

                "Severity": issue["severity"].title()

            })

            issue_counter += 1

    columns = [
        "Issue_ID",
        "Row_Number",
        "Column_Name",
        "Issue_Type",
        "Current_Value",
        "Rule",
        "Severity"
    ]

    write_csv(
        cleaning_issues,
        reports_dir / "quality_issues.csv",
        columns
    )

    print(
        f"Cleaning issue report generated: "
        f"{len(cleaning_issues)} actionable issues."
    )


# ============================================================
# 11. DATA QUALITY SCORE CALCULATION
# ============================================================

def calculate_quality_score(
    df,
    issues,
    config,
    reports_dir
):

    dimensions = [
        "completeness",
        "accuracy",
        "uniqueness",
        "consistency",
        "validity",
        "timeliness"
    ]

    weights = config.get(
        "quality_weights",
        DEFAULT_CONFIG["quality_weights"]
    )

    total_rows = max(len(df), 1)

    dimension_scores = {}

    for dimension in dimensions:

        affected = set()

        for issue in issues:

            if issue["dimension"] != dimension:
                continue

            if issue["row_indexes"]:

                affected.update(
                    issue["row_indexes"].split(";")
                )

            elif issue["affected_rows"] == 0:

                # Dataset-level issues count as one failed check.
                affected.add(
                    f"dataset-level-{issue['issue_id']}"
                )

        penalty_rate = min(
            len(affected) / total_rows,
            1.0
        )

        dimension_scores[dimension] = round(
            (1 - penalty_rate) * 100,
            2
        )

    total_weight = sum(
        weights.get(dimension, 0)
        for dimension in dimensions
    )

    overall = sum(
        dimension_scores[dimension]
        * weights.get(dimension, 0)
        for dimension in dimensions
    )

    overall = (
        round(overall / total_weight, 2)
        if total_weight
        else 0
    )

    rows = []

    for dimension in dimensions:

        rows.append({

            "dimension": dimension,

            "score": dimension_scores[dimension],

            "weight": weights.get(dimension, 0)

        })

    rows.append({

        "dimension": "OVERALL",

        "score": overall,

        "weight": total_weight

    })

    write_csv(
        rows,
        reports_dir / "quality_score.csv"
    )

    return overall, dimension_scores


# ============================================================
# 12. MAIN EXECUTION
# ============================================================

def main():

    print("\n==========================================")
    print(" ENTERPRISE DATA QUALITY PLATFORM")
    print("==========================================\n")

    # Load configuration.
    config = load_config()

    # Load dataset.
    df, dataset_path = load_dataset(
        config["dataset_path"]
    )

    # Create output directories.
    reports_dir = ROOT / config.get(
        "reports_dir",
        "reports"
    )

    viz_dir = ROOT / config.get(
        "visualizations_dir",
        "visualizations"
    )

    history_dir = ROOT / config.get(
        "history_dir",
        "history"
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    history_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"Loaded {len(df)} rows and "
        f"{len(df.columns)} columns."
    )

    print(f"Dataset: {dataset_path.name}\n")

    # STEP 1: Data profiling.
    profile_dataset(
        df,
        reports_dir
    )

    # STEP 2: Categorical analysis.
    categorical_analysis(
        df,
        reports_dir,
        config.get("categorical_columns", [])
    )

    # STEP 3: Visualizations.
    create_visualizations(
        df,
        viz_dir
    )

    # STEP 4: Run validation rules.
    issues = run_configured_rules(
        df,
        config,
        dataset_path,
        reports_dir,
        history_dir
    )

    # STEP 5: Save detailed diagnostic report.
    detailed_columns = [
        "issue_id",
        "rule_id",
        "dimension",
        "column",
        "affected_rows",
        "row_indexes",
        "sample_values",
        "severity",
        "message",
        "detected_at_utc",
        "status"
    ]

    write_csv(
        issues,
        reports_dir / "quality_issues_detailed.csv",
        detailed_columns
    )

    # STEP 6: Generate report for data_cleaning.py.
    create_cleaning_issue_report(
        df,
        issues,
        reports_dir
    )

    # STEP 7: Calculate quality score.
    overall, scores = calculate_quality_score(
        df,
        issues,
        config,
        reports_dir
    )

    # STEP 8: Generate run summary.
    summary = {

        "dataset": dataset_path.name,

        "rows": len(df),

        "columns": len(df.columns),

        "issues_found": len(issues),

        "overall_quality_score": overall,

        "dimension_scores": scores,

        "run_at_utc": datetime.now(
            timezone.utc
        ).isoformat()

    }

    with open(
        reports_dir / "run_summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    # STEP 9: Display final results.
    print("\n==========================================")
    print("         DATA QUALITY SUMMARY")
    print("==========================================")

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Issues found: {len(issues)}")

    print(
        f"\nOverall Data Quality Score: "
        f"{overall}/100"
    )

    print("\nDimension Scores:")

    for dimension, score in scores.items():

        print(
            f"{dimension.title():15s}: "
            f"{score:6.2f}/100"
        )

    print("\nReports generated successfully.")

    print(f"Reports folder: {reports_dir}")
    print(f"Visualizations folder: {viz_dir}")

    print("\nGenerated files:")
    print("1. column_profile.csv")
    print("2. categorical_frequency.csv")
    print("3. quality_issues.csv")
    print("4. quality_issues_detailed.csv")
    print("5. quality_score.csv")
    print("6. run_summary.json")

    print("\nOriginal dataset was not modified.")

    print("==========================================\n")


# ============================================================
# 13. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:
        main()

    except FileNotFoundError as error:

        print("\nFILE ERROR:")
        print(error)

    except json.JSONDecodeError as error:

        print("\nCONFIGURATION ERROR:")
        print("Check whether config.json contains valid JSON.")
        print(error)

    except Exception as error:

        print("\nUNEXPECTED ERROR:")
        print(error)

        raise
