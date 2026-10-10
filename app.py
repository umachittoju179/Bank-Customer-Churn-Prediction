
from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    brier_score_loss,
)

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Bank Customer Churn Analytics",
    page_icon="🏦",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "European_Bank.csv"
MODEL_PATH = BASE_DIR / "outputs" / "churn_model.pkl"

sns.set_theme(style="whitegrid")

# =========================================================
# CSS - READABLE TEXT COLORS
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #F4F7FB;
        color: #1E293B;
    }

    .block-container {
        padding-top: 1.5rem;
        max-width: 1500px;
    }

    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: #1E293B;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #10233F !important;
    }

    .hero {
        background: linear-gradient(120deg, #10233F, #145B75);
        padding: 30px;
        border-radius: 18px;
        color: #FFFFFF;
        margin-bottom: 24px;
    }

    .hero h1, .hero b {
        color: #FFFFFF !important;
    }

    .hero p {
        color: #E0EDF5 !important;
        font-size: 1.05rem;
    }

    .section-heading {
        color: #10233F !important;
        font-size: 1.45rem;
        font-weight: 700;
        margin: 12px 0 15px;
    }

    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #DCE5EF;
        border-radius: 15px;
        padding: 20px;
        min-height: 130px;
        box-shadow: 0 3px 12px rgba(20, 45, 70, 0.05);
    }

    .kpi-label {
        color: #475569 !important;
        font-size: 0.92rem;
        font-weight: 600;
    }

    .kpi-value {
        color: #10233F !important;
        font-size: 2rem;
        font-weight: 750;
        margin-top: 8px;
    }

    .kpi-note {
        color: #64748B !important;
        font-size: 0.82rem;
        margin-top: 4px;
    }

    .info-card {
        background: #FFFFFF;
        color: #1E293B;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #DCE5EF;
        margin-bottom: 12px;
    }

    .info-card h2, .info-card h4, .info-card p {
        color: #1E293B !important;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        padding: 15px;
        border: 1px solid #DCE5EF;
        border-radius: 12px;
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #10233F !important;
    }

    [data-testid="stSidebar"] {
        background: #EAF0F7;
    }

    [data-testid="stSidebar"] * {
        color: #1E293B;
    }

    .stTabs [data-baseweb="tab"] {
        color: #334155 !important;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        color: #0F766E !important;
    }

    input, textarea {
        color: #1E293B !important;
        background: #FFFFFF !important;
    }

    div[data-baseweb="select"] * {
        color: #1E293B !important;
    }

    .stButton button, .stFormSubmitButton button {
        background: #0F766E !important;
        border: none;
        border-radius: 9px;
    }

    .stButton button p, .stFormSubmitButton button p {
        color: #FFFFFF !important;
        font-weight: 600;
    }

    .result-card {
        background: #FFFFFF;
        border: 1px solid #DCE5EF;
        border-radius: 15px;
        padding: 22px;
        margin: 10px 0;
    }

    .result-card h3, .result-card p {
        color: #10233F !important;
    }

    .footer {
        text-align: center;
        color: #64748B !important;
        padding: 24px 0;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# LOAD DATA AND MODEL
# =========================================================

@st.cache_data
def load_data(path, modified):
    return pd.read_csv(path)


@st.cache_resource
def load_model(path, modified):
    return joblib.load(path)


if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

if not MODEL_PATH.exists():
    st.error(f"Model not found: {MODEL_PATH}")
    st.stop()

try:
    data = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
    model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
except Exception as error:
    st.error(f"Unable to load project files: {error}")
    st.stop()


# =========================================================
# TARGET COLUMN AND DATA PREPARATION
# =========================================================

def find_target_column(df):
    for col in df.columns:
        if col.lower() in ("exited", "churn", "attrition", "is_churn"):
            return col
    return None


target_col = find_target_column(data)

if target_col is None:
    st.error("Target column not found. Expected Exited, Churn or Attrition.")
    st.write("Available columns:", list(data.columns))
    st.stop()


def normalize_target(series):
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")
        return values.where(values.isin([0, 1]))

    mapping = {
        "0": 0, "1": 1,
        "no": 0, "yes": 1,
        "false": 0, "true": 1,
        "stayed": 0, "left": 1,
        "retained": 0, "churned": 1,
    }
    return series.astype(str).str.strip().str.lower().map(mapping)


data[target_col] = normalize_target(data[target_col])
valid_data = data.dropna(subset=[target_col]).copy()
valid_data[target_col] = valid_data[target_col].astype(int)
valid_data = valid_data[valid_data[target_col].isin([0, 1])]

if valid_data.empty:
    st.error("No valid 0/1 target values found.")
    st.stop()

total_customers = len(valid_data)
customers_left = int((valid_data[target_col] == 1).sum())
customers_stayed = int((valid_data[target_col] == 0).sum())
churn_rate = customers_left / total_customers
retention_rate = customers_stayed / total_customers


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def show_kpi(label, value, note, emoji):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{emoji} {label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def expected_features(estimator):
    if hasattr(estimator, "feature_names_in_"):
        return list(estimator.feature_names_in_)

    if hasattr(estimator, "named_steps"):
        for step in estimator.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    return None


def churn_probability(estimator, features):
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(features)[0]
        classes = list(estimator.classes_)
        if 1 in classes:
            return float(probabilities[classes.index(1)])
        return float(probabilities[-1])

    return None


def plot_distribution():
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["Stayed", "Left"]
    values = [customers_stayed, customers_left]
    bars = ax.bar(
        labels,
        values,
        color=["#0D9488", "#F97316"],
        width=0.55,
    )
    ax.set_title("Customer Churn Distribution", fontsize=14, pad=12)
    ax.set_ylabel("Number of Customers")
    ax.set_ylim(0, max(values) * 1.18)
    ax.bar_label(bars, labels=[f"{v:,}" for v in values], padding=4)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def plot_group_churn(df, column):
    grouped = df.groupby(column, dropna=False)[target_col].agg(
        ["mean", "count"]
    ).reset_index()
    grouped["Churn Rate (%)"] = grouped["mean"] * 100
    grouped[column] = grouped[column].fillna("Missing").astype(str)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(
        grouped[column],
        grouped["Churn Rate (%)"],
        color="#168AAD",
        width=0.6,
    )
    ax.set_title(f"Churn Rate by {column}")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_ylim(0, max(100, grouped["Churn Rate (%)"].max() * 1.15))
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig, grouped


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <h1>🏦 Bank Customer Churn Prediction</h1>
        <p>
            Explore customer behavior, evaluate model performance, and
            estimate the likelihood that a bank customer may leave.
        </p>
        <br>
        <b>CUSTOMER ANALYTICS • CHURN INSIGHTS • MACHINE LEARNING</b>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-heading">Bank Analytics Overview</div>',
    unsafe_allow_html=True,
)

k1, k2, k3 = st.columns(3)

with k1:
    show_kpi("Total Customers", f"{total_customers:,}",
             "Valid customer records", "👥")
with k2:
    show_kpi("Customers Who Left", f"{customers_left:,}",
             f"{churn_rate:.2%} of customers", "📉")
with k3:
    show_kpi("Overall Churn Rate", f"{churn_rate:.2%}",
             f"Retention rate: {retention_rate:.2%}", "🎯")

with st.sidebar:
    st.title("🏦 Bank Analytics")
    st.caption("CUSTOMER INTELLIGENCE")
    st.divider()
    st.markdown("### Dashboard")
    st.write("Explore churn trends, evaluate a model, and predict churn.")
    st.divider()
    st.markdown("### Technology")
    st.write("Python · Pandas · scikit-learn · Streamlit")
    st.caption("Evaluation is attempted on a held-out split when supported.")

tab_overview, tab_customers, tab_features, tab_evaluation, tab_predict = st.tabs(
    [
        "📊 Overview",
        "👥 Customer Analysis",
        "🧬 Feature Analysis",
        "🧪 Model Evaluation",
        "🎯 Predict Churn",
    ]
)

# =========================================================
# TAB 1 - OVERVIEW
# =========================================================

with tab_overview:
    st.markdown(
        '<div class="section-heading">Customer Churn at a Glance</div>',
        unsafe_allow_html=True,
    )
    st.write("Understand retention patterns and check the loaded dataset.")

    chart_col, summary_col = st.columns([1.4, 1])

    with chart_col:
        st.markdown("#### 📊 Churn Distribution")
        fig = plot_distribution()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with summary_col:
        st.markdown("#### Retention Summary")
        st.markdown(
            f"""
            <div class="info-card">
                <h4>🟢 Customers Retained</h4>
                <h2>{customers_stayed:,}</h2>
                <p>{retention_rate:.2%} of valid customers.</p>
            </div>
            <div class="info-card">
                <h4>🟠 Customers Who Left</h4>
                <h2>{customers_left:,}</h2>
                <p>{churn_rate:.2%} of valid customers.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### 🧹 Dataset Quality Checks")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Rows", f"{len(data):,}")
    q2.metric("Columns", f"{data.shape[1]:,}")
    q3.metric("Missing Cells", f"{data.isna().sum().sum():,}")
    q4.metric("Duplicate Rows", f"{data.duplicated().sum():,}")

    st.caption(
        "Review missing values, duplicates, target definitions and potential "
        "data leakage before using this model in production."
    )

    st.markdown("#### Dataset Preview")
    st.dataframe(valid_data.head(10), use_container_width=True)

# =========================================================
# TAB 2 - CUSTOMER ANALYSIS
# =========================================================

with tab_customers:
    st.markdown(
        '<div class="section-heading">👥 Customer Analysis</div>',
        unsafe_allow_html=True,
    )
    st.write("Compare churn patterns across customer groups.")

    groups = [
        c for c in [
            "Geography", "Gender", "IsActiveMember",
            "HasCrCard", "NumOfProducts", "Tenure"
        ] if c in valid_data.columns
    ]

    if groups:
        group_col = st.selectbox("Choose a customer attribute", groups)
        fig, group_table = plot_group_churn(valid_data, group_col)
        left, right = st.columns([1.2, 1])

        with left:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with right:
            table = group_table[
                [group_col, "count", "Churn Rate (%)"]
            ].copy()
            table.columns = ["Segment", "Customers", "Churn Rate (%)"]
            table["Churn Rate (%)"] = table["Churn Rate (%)"].round(2)
            st.dataframe(table, use_container_width=True)
    else:
        st.info("No standard customer grouping columns were found.")

    if "Age" in valid_data.columns:
        st.markdown("#### Churn by Age Group")
        age_df = valid_data.copy()
        age_df["Age Group"] = pd.cut(
            pd.to_numeric(age_df["Age"], errors="coerce"),
            bins=[0, 25, 35, 45, 55, 65, 120],
            labels=["18–25", "26–35", "36–45", "46–55", "56–65", "66+"],
            include_lowest=True,
        )
        age_rates = age_df.groupby(
            "Age Group", observed=False
        )[target_col].mean() * 100

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(age_rates.index.astype(str), age_rates.values,
                      color="#0D9488")
        ax.set_ylabel("Churn Rate (%)")
        ax.set_ylim(0, max(100, float(age_rates.max()) * 1.15))
        ax.bar_label(bars, fmt="%.1f%%", padding=3)
        ax.set_title("Churn Rate by Age Group")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    if "Balance" in valid_data.columns:
        st.markdown("#### Account Balance by Churn Status")
        balance_df = valid_data.copy()
        balance_df["Churn Status"] = balance_df[target_col].map(
            {0: "Stayed", 1: "Left"}
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(
            data=balance_df,
            x="Churn Status",
            y="Balance",
            hue="Churn Status",
            legend=False,
            ax=ax,
            palette={"Stayed": "#0D9488", "Left": "#F97316"},
        )
        ax.set_title("Account Balance by Churn Status")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# =========================================================
# TAB 3 - FEATURE ANALYSIS
# =========================================================

with tab_features:
    st.markdown(
        '<div class="section-heading">🧬 Feature Analysis</div>',
        unsafe_allow_html=True,
    )
    st.write("Explore numerical relationships and feature distributions.")

    numeric = valid_data.select_dtypes(include=np.number)

    if numeric.shape[1] >= 2:
        st.markdown("#### Feature Correlation Heatmap")
        corr = numeric.corr()
        fig, ax = plt.subplots(
            figsize=(max(8, len(corr.columns) * 0.7),
                     max(5, len(corr.columns) * 0.6))
        )
        sns.heatmap(
            corr,
            annot=len(corr.columns) <= 12,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            linewidths=0.4,
            ax=ax,
        )
        ax.set_title("Feature Correlation Heatmap")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.caption(
            "Correlation does not establish causation and may not capture "
            "nonlinear relationships."
        )

    numeric_features = [c for c in numeric.columns if c != target_col]

    if numeric_features:
        st.markdown("#### Compare Features by Churn Status")
        feature = st.selectbox(
            "Select a numerical feature",
            numeric_features,
            key="feature_comparison",
        )

        summary = valid_data.groupby(target_col)[feature].agg(
            ["mean", "median", "std", "count"]
        ).rename(index={0: "Stayed", 1: "Left"})
        st.dataframe(summary.round(3), use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        for class_value, label, color in [
            (0, "Stayed", "#0D9488"),
            (1, "Left", "#F97316"),
        ]:
            values = pd.to_numeric(
                valid_data.loc[valid_data[target_col] == class_value, feature],
                errors="coerce",
            ).dropna()
            if not values.empty:
                ax.hist(values, bins=30, alpha=0.55, label=label, color=color)

        ax.set_title(f"{feature} Distribution")
        ax.set_xlabel(feature)
        ax.set_ylabel("Customer Count")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# =========================================================
# TAB 4 - MODEL EVALUATION
# =========================================================

with tab_evaluation:
    st.markdown(
        '<div class="section-heading">🧪 Model Evaluation</div>',
        unsafe_allow_html=True,
    )
    st.write(
        "A fresh copy of the saved estimator is trained on the training "
        "portion of a split and evaluated on the test portion."
    )

    features = expected_features(model)
    if not features or not all(c in valid_data.columns for c in features):
        standard = [
            "CreditScore", "Geography", "Gender", "Age", "Tenure",
            "Balance", "NumOfProducts", "HasCrCard",
            "IsActiveMember", "EstimatedSalary",
        ]
        features = standard if all(c in valid_data.columns for c in standard) else [
            c for c in valid_data.columns if c != target_col
        ]

    try:
        eval_df = valid_data[features + [target_col]].dropna().copy()
        X = eval_df[features]
        y = eval_df[target_col].astype(int)

        if y.nunique() < 2 or y.value_counts().min() < 2:
            raise ValueError("Both classes need at least two records.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        eval_model = clone(model)
        eval_model.fit(X_train, y_train)
        predictions = eval_model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)

        probabilities = None
        if hasattr(eval_model, "predict_proba"):
            probs = eval_model.predict_proba(X_test)
            classes = list(eval_model.classes_)
            probabilities = probs[:, classes.index(1)] if 1 in classes else probs[:, -1]

        auc = roc_auc_score(y_test, probabilities) if probabilities is not None else None
        brier = brier_score_loss(y_test, probabilities) if probabilities is not None else None

        a, b, c, d = st.columns(4)
        a.metric("Accuracy", f"{accuracy:.1%}")
        b.metric("Precision", f"{precision:.1%}")
        c.metric("Recall", f"{recall:.1%}")
        d.metric("F1-Score", f"{f1:.1%}")

        e, f = st.columns(2)
        e.metric("ROC-AUC", f"{auc:.3f}" if auc is not None else "N/A")
        f.metric("Brier Score", f"{brier:.3f}" if brier is not None else "N/A")

        st.caption(
            "Precision measures the reliability of predicted churn cases; "
            "recall measures the share of actual churn cases detected. "
            "A lower Brier score is better. Metrics depend on the split and "
            "training procedure."
        )

        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y_test, predictions, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Predicted Stayed", "Predicted Left"],
            yticklabels=["Actual Stayed", "Actual Left"],
            ax=ax,
        )
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Actual")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        if probabilities is not None and y_test.nunique() == 2:
            st.markdown("#### ROC Curve")
            fig, ax = plt.subplots(figsize=(7, 4))
            RocCurveDisplay.from_predictions(
                y_test, probabilities, ax=ax, name="Churn Model"
            )
            ax.plot([0, 1], [0, 1], "--", color="gray", label="Random baseline")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.markdown("#### Classification Report")
        report = classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Stayed", "Left"],
            output_dict=True,
            zero_division=0,
        )
        st.dataframe(pd.DataFrame(report).transpose().round(3),
                     use_container_width=True)

        st.caption(
            "This evaluates a cloned model retrained on the split; it does "
            "not necessarily represent the original saved model's performance. "
            "Model selection on the full dataset can make scores optimistic."
        )

    except Exception as error:
        st.error("Model evaluation could not be completed.")
        st.code(str(error))
        st.info(
            "Check that the saved model accepts the selected columns and "
            "includes the preprocessing used during training."
        )

# =========================================================
# TAB 5 - PREDICT CHURN
# =========================================================

with tab_predict:
    st.markdown(
        '<div class="section-heading">🎯 Predict Churn for a Customer</div>',
        unsafe_allow_html=True,
    )

    st.write("Enter customer details below.")

    st.markdown(
        """
        <div class="info-card">
            <b>Customer information</b>
            <p>Complete the fields and click Predict Churn to view the result.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("customer_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            credit_score = st.number_input(
                "Credit Score", 300, 900, 650
            )
            geography = st.selectbox(
                "Geography", ["France", "Germany", "Spain"]
            )
            gender = st.selectbox("Gender", ["Female", "Male"])
            age = st.number_input("Age", 18, 100, 40)

        with col2:
            tenure = st.number_input("Tenure (years)", 0, 10, 5)
            balance = st.number_input(
                "Account Balance", min_value=0.0,
                value=50000.0, step=1000.0, format="%.2f"
            )
            products = st.selectbox(
                "Number of Products", [1, 2, 3, 4]
            )

        with col3:
            has_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
            active_member = st.selectbox(
                "Is Active Member?", ["Yes", "No"]
            )
            salary = st.number_input(
                "Estimated Salary", min_value=0.0,
                value=60000.0, step=1000.0, format="%.2f"
            )

        submitted = st.form_submit_button(
            "🔍 Predict Churn",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        customer = pd.DataFrame([{
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": products,
            "HasCrCard": 1 if has_card == "Yes" else 0,
            "IsActiveMember": 1 if active_member == "Yes" else 0,
            "EstimatedSalary": salary,
        }])

        # Match binary column representation when the source uses strings.
        for col, label in [
            ("HasCrCard", has_card),
            ("IsActiveMember", active_member),
        ]:
            if col in valid_data.columns and valid_data[col].dtype == object:
                customer[col] = label

        try:
            required = expected_features(model)

            if required:
                missing = [c for c in required if c not in customer.columns]
                if missing:
                    st.error(
                        "The model expects fields not provided by this form: "
                        + ", ".join(missing)
                    )
                    st.info(
                        "Update the form to match the features used when "
                        "the model was trained."
                    )
                    st.stop()

                customer_input = customer[required]
            else:
                customer_input = customer

            prediction = int(model.predict(customer_input)[0])

            try:
                probability = churn_probability(model, customer_input)
            except Exception:
                probability = None

            # IMPORTANT: Prediction Result remains visible after submission.
            st.divider()
            st.markdown(
                '<div class="section-heading">📋 Prediction Result</div>',
                unsafe_allow_html=True,
            )

            if probability is not None:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <p><b>Estimated Churn Probability</b></p>
                        <h1 style="color:#0F766E !important;">
                            {probability:.1%}
                        </h1>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(float(np.clip(probability, 0, 1)))

                if prediction == 1:
                    st.warning(
                        "The model predicts that this customer may leave."
                    )
                else:
                    st.success(
                        "The model predicts that this customer is likely to stay."
                    )

                if probability < 0.20:
                    st.info("Illustrative risk category: Lower")
                elif probability < 0.50:
                    st.info("Illustrative risk category: Moderate")
                else:
                    st.warning("Illustrative risk category: Higher")

            else:
                st.info(
                    "This model does not provide a churn probability. "
                    "The predicted class is displayed below."
                )
                if prediction == 1:
                    st.warning("The model predicts that this customer may leave.")
                else:
                    st.success(
                        "The model predicts that this customer is likely to stay."
                    )

            st.markdown("#### Customer Details")
            st.dataframe(customer, use_container_width=True)

            st.caption(
                "This is a statistical estimate, not a guarantee. "
                "Predictions may be wrong and should not be the sole basis "
                "for customer decisions. Risk categories are illustrative "
                "and have not been validated as business thresholds."
            )

        except Exception as error:
            st.error("Prediction could not be completed.")
            st.code(str(error))
            st.info(
                "Check the model's expected feature names, column types, "
                "and original preprocessing pipeline."
            )

# =========================================================
# FOOTER
# =========================================================

st.divider()
st.markdown(
    """
    <div class="footer">
        BANK CUSTOMER CHURN ANALYTICS<br>
        Powered by Python • Pandas • scikit-learn • Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
