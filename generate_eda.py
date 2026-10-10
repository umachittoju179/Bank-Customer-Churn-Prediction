
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "European_Bank.csv"
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"

OUT.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=160, bbox_inches="tight")
    plt.close()


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")

    df = pd.read_csv(DATA)
    df.columns = df.columns.str.strip()

    required = {"Exited"}
    if not required.issubset(df.columns):
        raise ValueError("Dataset must contain the Exited column.")

    df = df.drop_duplicates().dropna(subset=["Exited"]).copy()
    df["Exited"] = pd.to_numeric(df["Exited"], errors="raise").astype(int)

    if not set(df["Exited"].unique()).issubset({0, 1}):
        raise ValueError("Exited must contain only 0 and 1.")

    df["ChurnStatus"] = df["Exited"].map(
        {0: "Retained", 1: "Churned"}
    )

    churn_rate = float(df["Exited"].mean())
    summary = [
        "BANK CUSTOMER CHURN — EDA REPORT",
        "=" * 40,
        f"Rows: {len(df)}",
        f"Columns: {len(df.columns) - 1}",
        f"Duplicate rows removed: {int(pd.read_csv(DATA).duplicated().sum())}",
        f"Missing values remaining in source columns: {int(df.drop(columns=['ChurnStatus']).isna().sum().sum())}",
        f"Churned customers: {int(df['Exited'].sum())}",
        f"Retained customers: {int((df['Exited'] == 0).sum())}",
        f"Overall churn rate: {churn_rate:.2%}",
        "",
        "CHURN BY CATEGORICAL FEATURES",
    ]

    # 1. Churn class distribution
    counts = df["ChurnStatus"].value_counts().reindex(
        ["Retained", "Churned"], fill_value=0
    )
    plt.figure(figsize=(7, 5))
    sns.barplot(x=counts.index, y=counts.values)
    plt.title("Bank Customer Churn Distribution")
    plt.xlabel("Customer status")
    plt.ylabel("Number of customers")
    save_plot("01_churn_distribution.png")

    # 2. Churn rates for categorical features
    categorical = [
        c for c in ["Geography", "Gender", "IsActiveMember",
                    "HasCrCard", "NumOfProducts"]
        if c in df.columns
    ]

    for index, col in enumerate(categorical, start=2):
        rates = (
            df.groupby(col, dropna=False)["Exited"]
            .agg(["mean", "count"])
            .reset_index()
        )
        rates["mean"] *= 100

        plt.figure(figsize=(8, 5))
        sns.barplot(data=rates, x=col, y="mean")
        plt.title(f"Churn Rate by {col}")
        plt.xlabel(col)
        plt.ylabel("Churn rate (%)")
        plt.xticks(rotation=25)
        save_plot(f"{index:02d}_churn_by_{col}.png")

        summary.append(f"\n{col}:")
        summary.append(rates.to_string(index=False))

    # 3. Age and tenure distributions by churn status
    for col, filename, title in [
        ("Age", "07_age_distribution.png", "Age Distribution by Churn Status"),
        ("Tenure", "08_tenure_distribution.png", "Tenure by Churn Status"),
        ("CreditScore", "09_credit_score_distribution.png",
         "Credit Score by Churn Status"),
        ("Balance", "10_balance_distribution.png",
         "Account Balance by Churn Status"),
        ("EstimatedSalary", "11_salary_distribution.png",
         "Estimated Salary by Churn Status"),
    ]:
        if col in df.columns:
            plt.figure(figsize=(8, 5))
            sns.histplot(
                data=df,
                x=col,
                hue="ChurnStatus",
                stat="density",
                common_norm=False,
                element="step",
                fill=False,
            )
            plt.title(title)
            save_plot(filename)

    # 4. Numerical correlation heatmap
    numeric = df.select_dtypes(include="number").drop(
        columns=["CustomerId", "Exited"], errors="ignore"
    )
    if not numeric.empty:
        corr = numeric.corr()
        plt.figure(figsize=(11, 8))
        sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
        plt.title("Numerical Feature Correlation Heatmap")
        save_plot("12_correlation_heatmap.png")

    # 5. Numerical summary by churn status
    numerical_summary = (
        df.groupby("ChurnStatus")
        .mean(numeric_only=True)
        .T
    )
    numerical_summary.to_csv(OUT / "numerical_summary_by_churn.csv")

    summary.extend([
        "",
        "NUMERICAL MEANS BY CUSTOMER STATUS",
        numerical_summary.to_string(),
        "",
        "Interpretation note:",
        "Differences are descriptive associations, not proof of causation.",
        "Use held-out model results and business validation before decisions.",
    ])

    (REPORTS / "eda_summary.txt").write_text(
        "\n".join(summary), encoding="utf-8"
    )

    print("\nEDA completed successfully.")
    print(f"Charts and numerical results: {OUT}")
    print(f"Written summary: {REPORTS / 'eda_summary.txt'}")
    print("\nOverall churn rate:", f"{churn_rate:.2%}")
    print("\nCharts generated:")
    for path in sorted(OUT.glob("*.png")):
        print("-", path.name)


if __name__ == "__main__":
    main()
