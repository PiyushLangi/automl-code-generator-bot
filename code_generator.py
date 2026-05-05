from groq import Groq
from data_analyzer import get_data_summary_text
from model_selector import get_model_import
import os

def get_client():
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return None
    return Groq(api_key=api_key)

SYSTEM_PROMPT = """You are an expert ML engineer. Generate complete, production-quality Python code.
Rules:
- Always include imports, preprocessing, train/test split, model training, evaluation, and a prediction example
- Handle missing values, categorical encoding, and feature scaling where needed
- Include clear inline comments
- Use sklearn pipelines where appropriate
- For classification: show accuracy, classification_report, confusion matrix
- For regression: show RMSE, MAE, R2 score
- Add a feature importance plot if the model supports it
- Output ONLY the Python code, no explanations outside the code
"""

def generate_code(model_name, info, df_columns):
    """Generate complete ML code using Groq LLM."""
    client = get_client()                                    
    if not client:                                           
        return "# Please enter Groq API key in sidebar."    
    
    data_summary = get_data_summary_text(info)
    model_import = get_model_import(model_name)
    target = info['target']
    problem_type = info['problem_type']
    numeric_features = info['numeric_features']
    categorical_features = info['categorical_features']
    has_missing = info['has_missing']
    has_outliers = info['has_outliers']
    is_imbalanced = info.get('is_imbalanced', False)

    prompt = f"""Generate complete Python ML code for the following task:

DATA SUMMARY:
{data_summary}

ALL COLUMNS: {list(df_columns)}
NUMERIC FEATURES: {numeric_features}
CATEGORICAL FEATURES: {categorical_features}
TARGET COLUMN: {target}
PROBLEM TYPE: {problem_type}
SELECTED MODEL: {model_name}
MODEL IMPORT: {model_import}

SPECIAL CONDITIONS TO HANDLE:
- Missing values: {has_missing}
- Outliers present: {has_outliers}
- Class imbalance: {is_imbalanced}

REQUIREMENTS:
1. Load data with: df = pd.read_csv('your_data.csv')
2. Full preprocessing pipeline (imputation, encoding, scaling as needed)
3. Train/test split (80/20)
4. Train {model_name}
5. Evaluate with appropriate metrics for {problem_type}
6. Show feature importances if supported
7. Show a sample prediction
8. Add warnings/recommendations in comments based on the data characteristics

Generate ONLY the complete Python code."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=3000,
        temperature=0.2,
    )

    code = response.choices[0].message.content
    # Strip markdown fences if present
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code:
        code = code.split("```")[1].split("```")[0].strip()
    return code


def generate_data_insights(info):
    """Use LLM to give smart natural language insights about the data."""
    client = get_client()                                              
    if not client:                                                     
        return "Please enter Groq API key in sidebar for insights."   
    data_summary = get_data_summary_text(info)

    prompt = f"""Given this dataset analysis, provide 4-5 concise, actionable insights and warnings a data scientist should know before modeling:

{data_summary}

Format as bullet points. Be specific and practical. No generic advice."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a senior data scientist. Give sharp, specific, actionable insights."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=600,
        temperature=0.3,
    )
    return response.choices[0].message.content
