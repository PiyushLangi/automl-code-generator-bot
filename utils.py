import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


def plot_target_distribution(df, target, problem_type):
    fig, ax = plt.subplots(figsize=(7, 3))
    if problem_type == 'classification':
        counts = df[target].value_counts()
        sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, palette='Blues_d')
        ax.set_title(f'Class Distribution — {target}')
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        for i, v in enumerate(counts.values):
            ax.text(i, v + 0.5, str(v), ha='center', fontsize=10)
    else:
        sns.histplot(df[target].dropna(), bins=30, kde=True, ax=ax, color='steelblue')
        ax.set_title(f'Target Distribution — {target}')
        ax.set_xlabel(target)
    plt.tight_layout()
    return fig


def plot_missing_values(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, max(2, len(missing) * 0.4)))
    pct = (missing / len(df) * 100).round(1)
    sns.barplot(x=pct.values, y=pct.index.tolist(), ax=ax, palette='Reds_d')
    ax.set_title('Missing Values (%)')
    ax.set_xlabel('% Missing')
    for i, v in enumerate(pct.values):
        ax.text(v + 0.3, i, f'{v}%', va='center', fontsize=9)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df, target):
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) < 2:
        return None
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(min(10, len(corr.columns)), min(8, len(corr.columns))))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax, square=True, linewidths=0.5, annot_kws={"size": 8})
    ax.set_title('Feature Correlation Heatmap')
    plt.tight_layout()
    return fig


def render_metric_cards(info):
    cols = st.columns(4)
    cols[0].metric("Rows", f"{info['n_rows']:,}")
    cols[1].metric("Columns", info['n_cols'])
    cols[2].metric("Missing %", f"{info['missing_pct']}%")
    cols[3].metric("Duplicate Rows", info['duplicate_rows'])

    cols2 = st.columns(4)
    cols2[0].metric("Numeric Features", info['n_numeric'])
    cols2[1].metric("Categorical Features", info['n_categorical'])
    cols2[2].metric("Problem Type", info['problem_type'].capitalize())
    if info['problem_type'] == 'classification':
        imb = "Yes ⚠️" if info.get('is_imbalanced') else "No ✅"
        cols2[3].metric("Imbalanced?", imb)
    else:
        cols2[3].metric("Target Skewness", info.get('target_skewness', 'N/A'))
