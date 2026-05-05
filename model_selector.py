def suggest_models(info):
    """
    Intelligently suggest ML models based on 10+ data characteristics.
    Returns a ranked list of (model_name, reason) tuples.
    """
    problem_type = info['problem_type']
    n_rows = info['n_rows']
    n_cols = info['n_cols']
    has_missing = info['has_missing']
    has_outliers = info['has_outliers']
    n_categorical = info['n_categorical']
    n_numeric = info['n_numeric']
    high_card = len(info['high_cardinality_cols']) > 0
    skewed = info['avg_skewness'] > 1.5

    suggestions = []

    if problem_type == 'classification':
        is_imbalanced = info.get('is_imbalanced', False)
        n_classes = info.get('n_classes', 2)
        is_binary = n_classes == 2

        # Large dataset
        if n_rows >= 10000:
            suggestions.append(("XGBClassifier", "Excellent for large datasets; handles missing values, outliers, and mixed feature types natively."))
            suggestions.append(("RandomForestClassifier", "Robust ensemble; handles high dimensionality and noisy data well."))
            if is_imbalanced:
                suggestions.append(("BalancedRandomForestClassifier", "Specifically designed for imbalanced class distributions."))
            suggestions.append(("GradientBoostingClassifier", "High accuracy boosting; great when feature interactions matter."))

        # Medium dataset
        elif n_rows >= 1000:
            suggestions.append(("RandomForestClassifier", "Best default for medium datasets; robust and interpretable."))
            suggestions.append(("XGBClassifier", "Strong performer; handles mixed features and missing values."))
            if is_binary:
                suggestions.append(("LogisticRegression", "Fast, interpretable baseline for binary classification."))
            suggestions.append(("SVC", "Effective in high-dimensional spaces with good kernel choice."))

        # Small dataset
        else:
            suggestions.append(("LogisticRegression", "Works well with small data; less prone to overfitting."))
            suggestions.append(("KNeighborsClassifier", "Simple and effective for small datasets."))
            suggestions.append(("DecisionTreeClassifier", "Interpretable; good for small tabular data."))
            if not is_imbalanced:
                suggestions.append(("GaussianNB", "Extremely fast; good baseline for small clean datasets."))

        # Categorical-heavy override
        if n_categorical > n_numeric and not high_card:
            suggestions.insert(0, ("RandomForestClassifier", "Best choice when most features are categorical (handles encoding well)."))

    else:  # regression
        target_skew = abs(info.get('target_skewness', 0))

        if n_rows >= 10000:
            suggestions.append(("XGBRegressor", "Top performer for large regression datasets; handles missing data and outliers."))
            suggestions.append(("RandomForestRegressor", "Robust ensemble; low variance, handles nonlinear relationships."))
            suggestions.append(("GradientBoostingRegressor", "Great accuracy when precise predictions are needed."))
            if not skewed and not has_outliers:
                suggestions.append(("Ridge", "Fast and interpretable linear baseline."))

        elif n_rows >= 1000:
            suggestions.append(("RandomForestRegressor", "Best default for medium regression; robust to outliers."))
            suggestions.append(("XGBRegressor", "Strong performer with mixed feature types."))
            if not has_outliers:
                suggestions.append(("LinearRegression", "Good interpretable baseline if relationships are linear."))
            suggestions.append(("SVR", "Effective when feature space is complex."))

        else:
            if not has_outliers and not skewed:
                suggestions.append(("LinearRegression", "Best for small, clean datasets with linear relationships."))
            else:
                suggestions.append(("HuberRegressor", "Robust to outliers; better than LinearRegression on messy small data."))
            suggestions.append(("DecisionTreeRegressor", "Non-linear; captures complex patterns in small data."))
            suggestions.append(("KNeighborsRegressor", "Simple, non-parametric; works well on small datasets."))

        # Highly skewed target
        if target_skew > 1.5:
            suggestions.insert(0, ("XGBRegressor", "Handles skewed targets well; more robust than linear models on non-normal distributions."))

        # Outlier override
        if has_outliers:
            suggestions.insert(0, ("HuberRegressor", "Specifically robust to outliers in both features and target."))

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for model, reason in suggestions:
        if model not in seen:
            seen.add(model)
            unique.append((model, reason))

    return unique[:3]  # Top 3


def get_model_import(model_name):
    imports = {
        "XGBClassifier": "from xgboost import XGBClassifier",
        "XGBRegressor": "from xgboost import XGBRegressor",
        "RandomForestClassifier": "from sklearn.ensemble import RandomForestClassifier",
        "RandomForestRegressor": "from sklearn.ensemble import RandomForestRegressor",
        "GradientBoostingClassifier": "from sklearn.ensemble import GradientBoostingClassifier",
        "GradientBoostingRegressor": "from sklearn.ensemble import GradientBoostingRegressor",
        "LogisticRegression": "from sklearn.linear_model import LogisticRegression",
        "LinearRegression": "from sklearn.linear_model import LinearRegression",
        "Ridge": "from sklearn.linear_model import Ridge",
        "HuberRegressor": "from sklearn.linear_model import HuberRegressor",
        "SVC": "from sklearn.svm import SVC",
        "SVR": "from sklearn.svm import SVR",
        "KNeighborsClassifier": "from sklearn.neighbors import KNeighborsClassifier",
        "KNeighborsRegressor": "from sklearn.neighbors import KNeighborsRegressor",
        "DecisionTreeClassifier": "from sklearn.tree import DecisionTreeClassifier",
        "DecisionTreeRegressor": "from sklearn.tree import DecisionTreeRegressor",
        "GaussianNB": "from sklearn.naive_bayes import GaussianNB",
        "BalancedRandomForestClassifier": "from imblearn.ensemble import BalancedRandomForestClassifier",
    }
    return imports.get(model_name, f"# No import found for {model_name}")
