
import pandas as pd
from pathlib import Path
from datetime import datetime
import re


# =====================================================
# CONFIGURATION
# =====================================================

INPUT_FILE = Path("data/customer_data.csv")
ISSUES_FILE = Path("reports/quality_issues.csv")

OUTPUT_FILE = Path("data/cleaned_dataset.csv")
LOG_FILE = Path("reports/cleaning_log.csv")

# Row_Number in quality_issues.csv represents the
# data record number, starting from 1 (excluding header).


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def get_value(value):

    if pd.isna(value):
        return "NULL"

    return str(value)


def convert_value(value, original_value):

    """Try to preserve the original column's data type."""

    if pd.isna(original_value):
        return value

    if isinstance(original_value, bool):
        return value.lower() in ["true", "1", "yes"]

    if isinstance(original_value, int):
        return int(value)

    if isinstance(original_value, float):
        return float(value)

    return value


def get_issue_details(issue):

    issue_type = str(issue.get("Issue_Type", "Unknown"))
    column = str(issue.get("Column_Name", "Unknown"))
    row = issue.get("Row_Number", None)

    print("\n" + "=" * 55)
    print("DATA QUALITY ISSUE DETECTED")
    print("=" * 55)

    print("Issue ID:", issue.get("Issue_ID", "N/A"))
    print("Issue Type:", issue_type)
    print("Column:", column)
    print("Affected row:", row)
    print("Current value:", issue.get("Current_Value", "N/A"))
    print("Severity:", issue.get("Severity", "N/A"))

    if pd.notna(issue.get("Rule", None)):
        print("Failed rule:", issue["Rule"])


def get_actions(issue_type):

    """
    Determine available actions from the detected
    issue type, not from a specific dataset.
    """

    issue = issue_type.lower().replace(" ", "_")

    if any(x in issue for x in ["duplicate"]):

        return [
            "Enter a corrected value",
            "Set value as NULL",
            "Delete affected row",
            "Skip"
        ]

    if any(x in issue for x in ["outlier"]):

        return [
            "Enter corrected value",
            "Set value as NULL",
            "Keep existing value",
            "Delete affected row",
            "Skip"
        ]

    if any(x in issue for x in [
        "missing", "null", "blank", "empty"
    ]):

        return [
            "Enter a value manually",
            "Delete affected row",
            "Skip"
        ]

    return [
        "Enter corrected value",
        "Set value as NULL",
        "Delete affected row",
        "Skip"
    ]


# =====================================================
# USER INPUT
# =====================================================

def select_action(issue_type):

    actions = get_actions(issue_type)

    print("\nChoose how to resolve this issue:")

    for i, action in enumerate(actions, start=1):
        print(f"{i}. {action}")

    while True:

        choice = input("\nEnter choice: ").strip()

        if choice.isdigit():

            choice = int(choice)

            if 1 <= choice <= len(actions):
                return choice, actions[choice - 1]

        print("Invalid choice. Try again.")


def enter_corrected_value(column, original_value):

    while True:

        new_value = input(
            f"Enter corrected value for {column}: "
        )

        try:
            return convert_value(new_value, original_value)

        except ValueError:
            print("Invalid value for this column's data type.")


# =====================================================
# CLEANING LOG
# =====================================================

cleaning_logs = []


def add_log(issue, old_value, new_value, action):

    cleaning_logs.append({

        "Issue_ID": issue.get("Issue_ID", "N/A"),

        "Row_Number": issue.get("Row_Number", "N/A"),

        "Column_Name": issue.get("Column_Name", "N/A"),

        "Issue_Type": issue.get("Issue_Type", "Unknown"),

        "Original_Value": get_value(old_value),

        "New_Value": get_value(new_value),

        "Action": action,

        "Timestamp": datetime.now().isoformat(timespec="seconds")

    })


# =====================================================
# REMEDIATION ENGINE
# =====================================================

def clean_data(df, issues):

    df = df.copy()

    total_issues = len(issues)

    resolved = 0
    skipped = 0

    print("\n==========================================")
    print("       DATA REMEDIATION ENGINE")
    print("==========================================")

    print("Total detected issues:", total_issues)

    for _, issue in issues.iterrows():

        get_issue_details(issue)

        # -------------------------------------------------
        # Identify affected column
        # -------------------------------------------------

        column = issue.get("Column_Name")

        if pd.isna(column) or column not in df.columns:

            print("Column not found. Skipping issue.")
            skipped += 1
            continue

        # -------------------------------------------------
        # Identify affected row
        # -------------------------------------------------

        row_number = issue.get("Row_Number")

        if pd.isna(row_number):

            print("Row number missing. Skipping issue.")
            skipped += 1
            continue

        try:
            row_number = int(row_number)

        except (ValueError, TypeError):

            print("Invalid row number. Skipping issue.")
            skipped += 1
            continue

        if row_number < 1:

            print("Invalid row number. Skipping issue.")
            skipped += 1
            continue

        # Convert original CSV data record number
        # to its original pandas index.

        row_index = row_number - 1

        if row_index not in df.index:

            print("Record no longer exists. Skipping issue.")
            skipped += 1
            continue

        old_value = df.at[row_index, column]

        issue_type = str(
            issue.get("Issue_Type", "Unknown")
        )

        # -------------------------------------------------
        # Ask user
        # -------------------------------------------------

        choice, action = select_action(issue_type)

        new_value = old_value

        try:

            if action == "Enter a value manually" or \
               action == "Enter corrected value" or \
               action == "Enter a corrected value":

                new_value = enter_corrected_value(
                    column, old_value
                )

                df.at[row_index, column] = new_value

            elif action == "Set value as NULL" or \
                 action == "Set as NULL":

                df.at[row_index, column] = pd.NA
                new_value = pd.NA

            elif action == "Delete affected row":

                df = df.drop(index=row_index)
                new_value = "ROW DELETED"

            elif action == "Keep existing value":

                new_value = old_value

            elif action == "Skip":

                skipped += 1
                continue

            # Record every accepted action.
            add_log(
                issue,
                old_value,
                new_value,
                action
            )

            resolved += 1

            print("\nAction applied successfully.")

        except (ValueError, TypeError) as error:

            print("Could not apply change:", error)
            print("Issue skipped.")
            skipped += 1

    print("\n==========================================")
    print("       REMEDIATION SUMMARY")
    print("==========================================")

    print("Total issues:", total_issues)
    print("Actions completed:", resolved)
    print("Issues skipped:", skipped)
    print("Remaining records:", len(df))

    return df


# =====================================================
# SAVE CLEANED DATASET
# =====================================================

def save_results(df):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    LOG_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nSave the cleaned dataset?")
    print("1. Save")
    print("2. Exit without saving")

    choice = input("Choose: ").strip()

    if choice == "1":

        df.to_csv(OUTPUT_FILE, index=False)

        pd.DataFrame(cleaning_logs).to_csv(
            LOG_FILE,
            index=False
        )

        print("\nCleaned dataset saved:", OUTPUT_FILE)
        print("Cleaning log saved:", LOG_FILE)

    else:

        print("\nChanges were not saved.")


# =====================================================
# MAIN
# =====================================================

def main():

    print("\n==========================================")
    print("     ENTERPRISE DATA CLEANING SYSTEM")
    print("==========================================")

    if not INPUT_FILE.exists():

        print("Input dataset not found:", INPUT_FILE)
        return

    if not ISSUES_FILE.exists():

        print("Quality issues report not found:", ISSUES_FILE)
        print("Run data_quality_platform.py first.")
        return

    # Load original dataset and issue report.

    df = pd.read_csv(INPUT_FILE)

    issues = pd.read_csv(ISSUES_FILE)

    # Validate report structure.

    required_columns = [
        "Row_Number",
        "Column_Name",
        "Issue_Type"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in issues.columns
    ]

    if missing_columns:

        print("Invalid issue report.")
        print("Missing columns:", missing_columns)
        return

    if issues.empty:

        print("No data quality issues found.")
        return

    # Process issues.

    cleaned_df = clean_data(df, issues)

    # Save results.

    save_results(cleaned_df)


if __name__ == "__main__":

    main()
