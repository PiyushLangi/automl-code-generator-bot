import pandas as pd
import numpy as np

def remove_irrelevant_columns(df, target):
    df = df.copy()
    drop_cols = []

    for col in df.columns:
        col_lower = col.lower()

        # 🚨 NEVER DROP TARGET
        if col == target:
            continue

        # 1. Remove ID-like columns
        if 'id' in col_lower:
            drop_cols.append(col)

        # 2. Remove fully unique columns
        elif df[col].nunique() == len(df) and df[col].dtype == 'object':
            drop_cols.append(col)

        # 3. Remove leakage columns
        elif target.lower() in col_lower:
            drop_cols.append(col)

        # 4. Remove data leakage columns (post-target information like score/reason/category)
        elif any(keyword in col_lower for keyword in ['score', 'reason', 'category']):
            drop_cols.append(col)

    df = df.drop(columns=drop_cols, errors='ignore')

    return df, drop_cols

def analyze_data(df, target):

    # 🔥 AUTO REMOVE ID + LEAKAGE COLUMNS
    df, dropped_cols = remove_irrelevant_columns(df, target)
    info = {}

    # ✅ store dropped columns
    info['dropped_columns'] = dropped_cols

    info['feature_columns'] = [col for col in df.columns if col != target]

    # Basic shape
    info['n_rows'], info['n_cols'] = df.shape

    # Target info
    info['target'] = target
    info['target_dtype'] = str(df[target].dtype)

    # Problem type
    info['problem_type'] = detect_problem_type(df, target)

    # Class imbalance (classification only)
    if info['problem_type'] == 'classification':
        counts = df[target].value_counts()
        info['n_classes'] = len(counts)
        info['class_distribution'] = counts.to_dict()
        minority = counts.min()
        majority = counts.max()
        info['imbalance_ratio'] = round(majority / minority, 2) if minority > 0 else 999
        info['is_imbalanced'] = info['imbalance_ratio'] > 3

    # Missing values
    missing = df.isnull().sum()
    info['missing_counts'] = missing[missing > 0].to_dict()
    info['missing_pct'] = round(missing.sum() / (df.shape[0] * df.shape[1]) * 100, 2)
    info['has_missing'] = info['missing_pct'] > 0

    # Feature types
    features = df.drop(columns=[target])
    info['numeric_features'] = features.select_dtypes(include=[np.number]).columns.tolist()
    info['categorical_features'] = features.select_dtypes(include=['object', 'category']).columns.tolist()
    info['n_numeric'] = len(info['numeric_features'])
    info['n_categorical'] = len(info['categorical_features'])

    # High cardinality categoricals
    high_card = [c for c in info['categorical_features'] if df[c].nunique() > 20]
    info['high_cardinality_cols'] = high_card

    # Skewness in numeric features
    if info['numeric_features']:
        skews = df[info['numeric_features']].skew().abs()
        info['highly_skewed_cols'] = skews[skews > 2].index.tolist()
        info['avg_skewness'] = round(skews.mean(), 2)
    else:
        info['highly_skewed_cols'] = []
        info['avg_skewness'] = 0

    # Duplicate rows
    info['duplicate_rows'] = int(df.duplicated().sum())

    # Outliers (IQR method) in numeric
    outlier_cols = []
    for col in info['numeric_features']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
        if outliers > 0.05 * len(df):
            outlier_cols.append(col)
    info['outlier_cols'] = outlier_cols
    info['has_outliers'] = len(outlier_cols) > 0

    # Regression target stats
    if info['problem_type'] == 'regression':
        info['target_skewness'] = round(float(df[target].skew()), 2)
        info['target_range'] = [float(df[target].min()), float(df[target].max())]

    return info, df


def detect_problem_type(df, target):
    col = df[target]
    n_unique = col.nunique()

    # Strong classification detection
    if col.dtype == 'object' or col.dtype.name == 'category':
        return 'classification'
    
    # Binary or small discrete values → classification
    if n_unique <= 10:
        return 'classification'
    
    return 'regression'


def get_data_summary_text(info):
    """Create a natural language summary for the LLM prompt."""
    lines = [
        f"Dataset: {info['n_rows']} rows, {info['n_cols']} columns",
        f"Problem type: {info['problem_type']}",
        f"Target column: {info['target']} (dtype: {info['target_dtype']})",
        f"Numeric features: {info['n_numeric']}, Categorical features: {info['n_categorical']}",
        f"Missing data: {info['missing_pct']}% overall, affected columns: {list(info['missing_counts'].keys())}",
        f"Duplicate rows: {info['duplicate_rows']}",
        f"Outlier-heavy columns: {info['outlier_cols']}",
        f"Highly skewed columns: {info['highly_skewed_cols']}",
        f"High cardinality categoricals: {info['high_cardinality_cols']}",
    ]
    if info['problem_type'] == 'classification':
        lines.append(f"Number of classes: {info['n_classes']}")
        lines.append(f"Class imbalance ratio: {info['imbalance_ratio']} ({'IMBALANCED' if info['is_imbalanced'] else 'balanced'})")
    else:
        lines.append(f"Target skewness: {info.get('target_skewness', 'N/A')}")
    return "\n".join(lines)
