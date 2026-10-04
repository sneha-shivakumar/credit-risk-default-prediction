"""
Credit Risk — Default Prediction on Credit Card Clients
--------------------------------------------------------
Predicts whether a credit card customer will default on their payment
next month, using the UCI "Default of Credit Card Clients" dataset
(30,000 customers, Taiwan, 2005).

Covers:
  - Data cleaning (fixing undocumented category codes)
  - Logistic Regression (interpretable baseline)
  - Random Forest (stronger model + feature importance)
  - A risk-segmentation view: scoring every customer and bucketing
    them into Low / Medium / High risk, the way a risk team would
    actually use a model's output day to day.

Usage:
    python credit_risk_model.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, classification_report,
    confusion_matrix,
)

TARGET = "default.payment.next.month"


# ---------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------

def load_and_clean_data(path="data/UCI_Credit_Card.csv"):
    """Load the dataset and fix undocumented category codes."""
    df = pd.read_csv(path)

    # EDUCATION: 1=grad school, 2=university, 3=high school, 4=others;
    # 0, 5, 6 aren't documented in the dataset's codebook -> fold into "others" (4)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})

    # MARRIAGE: 1=married, 2=single, 3=others; 0 is undocumented -> fold into "others" (3)
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    return df


def engineer_features(df):
    """Build the feature matrix and target vector."""
    y = df[TARGET]
    X = df.drop(columns=["ID", TARGET])
    return X, y


# ---------------------------------------------------------------
# Models
# ---------------------------------------------------------------

def run_logistic_regression(X_train, X_test, y_train, y_test, scaler):
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    logreg.fit(X_train_s, y_train)

    y_pred = logreg.predict(X_test_s)
    y_prob = logreg.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print(f"\n[Logistic Regression] Accuracy: {acc*100:.2f}%  |  ROC-AUC: {auc:.3f}")
    print(classification_report(y_test, y_pred, target_names=["No default", "Default"]))

    return logreg, y_pred, y_prob, acc, auc


def run_random_forest(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print(f"\n[Random Forest] Accuracy: {acc*100:.2f}%  |  ROC-AUC: {auc:.3f}")
    print(classification_report(y_test, y_pred, target_names=["No default", "Default"]))

    return rf, y_pred, y_prob, acc, auc


# ---------------------------------------------------------------
# Plots
# ---------------------------------------------------------------

def plot_confusion_matrix(y_test, y_pred, title, save_path):
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=["No default", "Default"],
                yticklabels=["No default", "Default"])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curves(results, save_path):
    plt.figure(figsize=(6, 5))
    for name, (y_test, y_prob, auc) in results.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_feature_importance(rf, feature_names, save_path, top_n=10):
    importances = pd.Series(rf.feature_importances_, index=feature_names)
    top = importances.sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(7, 5))
    sns.barplot(x=top.values, y=top.index, hue=top.index, palette="viridis", legend=False)
    plt.title(f"Top {top_n} Drivers of Default Risk (Random Forest)")
    plt.xlabel("Relative importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_risk_segments(risk_df, save_path):
    seg_rates = risk_df.groupby("risk_band")["default.payment.next.month"].mean().reindex(
        ["Low", "Medium", "High"]
    )
    seg_counts = risk_df["risk_band"].value_counts().reindex(["Low", "Medium", "High"])

    fig, ax1 = plt.subplots(figsize=(7, 5))
    bars = ax1.bar(seg_counts.index, seg_counts.values, color=["#4C9A6B", "#E8A33D", "#C0392B"])
    ax1.set_ylabel("Number of customers")
    ax1.set_title("Customer Base by Risk Band — with Actual Default Rate")
    for bar, rate in zip(bars, seg_rates.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                  f"{rate*100:.1f}% actually defaulted", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------
# Risk segmentation — the business-facing output
# ---------------------------------------------------------------

def build_risk_segments(rf, X, y, df):
    """Score every customer and bucket into Low/Medium/High risk."""
    probs = rf.predict_proba(X)[:, 1]
    risk_df = df.copy()
    risk_df["default_probability"] = probs
    risk_df["risk_band"] = pd.cut(
        probs, bins=[-0.01, 0.2, 0.5, 1.0], labels=["Low", "Medium", "High"]
    )
    return risk_df


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------

if __name__ == "__main__":
    df = load_and_clean_data()
    X, y = engineer_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    logreg, lr_pred, lr_prob, lr_acc, lr_auc = run_logistic_regression(
        X_train, X_test, y_train, y_test, scaler
    )
    rf, rf_pred, rf_prob, rf_acc, rf_auc = run_random_forest(X_train, X_test, y_train, y_test)

    plot_confusion_matrix(y_test, lr_pred, "Logistic Regression — Confusion Matrix",
                           "images/logreg_confusion_matrix.png")
    plot_confusion_matrix(y_test, rf_pred, "Random Forest — Confusion Matrix",
                           "images/rf_confusion_matrix.png")
    plot_roc_curves(
        {"Logistic Regression": (y_test, lr_prob, lr_auc),
         "Random Forest": (y_test, rf_prob, rf_auc)},
        "images/roc_comparison.png",
    )
    plot_feature_importance(rf, X.columns, "images/feature_importance.png")

    # Risk segmentation across the FULL dataset (what a risk team would see)
    risk_df = build_risk_segments(rf, X, y, df)
    plot_risk_segments(risk_df, "images/risk_segments.png")
    risk_df[["ID", "LIMIT_BAL", "AGE", "default_probability", "risk_band",
             "default.payment.next.month"]].to_csv("data/customer_risk_scores.csv", index=False)

    print("\n=== Summary ===")
    print(f"Logistic Regression: {lr_acc*100:.2f}% accuracy, {lr_auc:.3f} AUC")
    print(f"Random Forest:       {rf_acc*100:.2f}% accuracy, {rf_auc:.3f} AUC")
    print("\nRisk band breakdown (full customer base):")
    print(risk_df.groupby("risk_band").agg(
        customers=("ID", "count"),
        actual_default_rate=("default.payment.next.month", "mean"),
    ))
