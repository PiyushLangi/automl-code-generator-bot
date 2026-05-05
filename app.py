import os
import streamlit as st
import pandas as pd
from data_analyzer import analyze_data
from model_selector import suggest_models
from code_generator import generate_code, generate_data_insights
from utils import (
    plot_target_distribution,
    plot_missing_values,
    plot_correlation_heatmap,
    render_metric_cards,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoML Bot",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 AutoML Code Generator Bot")
st.caption("Upload your CSV → Get smart model recommendations → Generate production-ready Python code")

# ── Sidebar: API key ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_key = st.text_input("Groq API Key", type="password",
                              help="Get your free key at console.groq.com")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        st.success("API key set ✅")
    else:
        st.warning("Enter your free Groq API key to enable AI features.")

    st.markdown("---")
    st.markdown("**Model used:** Llama 3.3 70B via Groq")
    st.markdown("**Privacy:** Your data never leaves your machine. Only the data *summary* is sent to Groq for code generation.")
    st.markdown("---")
    st.markdown("**Get free Groq key:** [console.groq.com](https://console.groq.com)")

# ── File upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("📂 Upload your CSV dataset", type=["csv"])

if not uploaded_file:
    st.info("👆 Upload a CSV file to get started.")
    st.stop()

df = pd.read_csv(uploaded_file)

# ── Dataset preview ───────────────────────────────────────────────────────────
with st.expander("📋 Dataset Preview", expanded=True):
    st.dataframe(df.head(10), use_container_width=True)

# ── Target column selection ───────────────────────────────────────────────────
target = st.selectbox("🎯 Select Target Column (what you want to predict)", df.columns)

if not target:
    st.stop()

# ── Analyze data ──────────────────────────────────────────────────────────────
with st.spinner("Analyzing your dataset..."):
    info, df_cleaned = analyze_data(df, target)

st.subheader("📊 Dataset Analysis")
render_metric_cards(info)

st.info(f"📌 Problem Type: {info['problem_type'].upper()}")

st.subheader("🧹 Cleaned Dataset Preview")
st.dataframe(df_cleaned.head(), use_container_width=True)

st.subheader("🔍 Feature Engineering Info")

st.markdown("**Why columns removed?**")
st.caption("ID columns, unique columns, and leakage columns are automatically removed.")

st.markdown("**Dropped Columns:**")
st.write(info['dropped_columns'])

if len(info['dropped_columns']) > 5:
    st.warning("⚠️ Many columns were removed. Review your dataset.")

st.markdown("**Features Used for Modeling:**")
st.write(info['feature_columns'])

st.metric("Features Used", len(info['feature_columns']))

# ── Visualizations ─────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    fig_target = plot_target_distribution(df_cleaned, target, info['problem_type'])
    st.pyplot(fig_target)

with col2:
    fig_missing = plot_missing_values(df_cleaned)
    if fig_missing:
        st.pyplot(fig_missing)
    else:
        st.success("✅ No missing values found!")

fig_corr = plot_correlation_heatmap(df_cleaned, target)
if fig_corr:
    with st.expander("🔗 Correlation Heatmap"):
        st.pyplot(fig_corr)

# ── AI Data Insights ──────────────────────────────────────────────────────────
if groq_key:
    with st.expander("🧠 AI Data Insights", expanded=True):
        with st.spinner("Generating AI insights about your data..."):
            insights = generate_data_insights(info)
        st.markdown(insights)
else:
    st.info("Add your Groq API key in the sidebar to get AI-powered data insights.")

# ── Model recommendations ─────────────────────────────────────────────────────
st.subheader("🏆 Model Recommendations")
suggestions = suggest_models(info)

model_names = [m for m, _ in suggestions]
model_reasons = {m: r for m, r in suggestions}

for i, (model, reason) in enumerate(suggestions):
    badge = "🥇 Best Match" if i == 0 else f"#{i+1}"
    st.markdown(f"**{badge} — `{model}`**")
    st.caption(f"  {reason}")

default_model = model_names[0]
selected_model = st.selectbox("🔧 Choose Model", model_names, index=0)
st.success(f"✅ Recommended: {default_model}")
st.info(f"💡 **Why this model?** {model_reasons[selected_model]}")

# ── Generate Code ──────────────────────────────────────────────────────────────
st.subheader("💻 Generate Python Code")

if not groq_key:
    st.warning("Add your free Groq API key in the sidebar to generate code.")
else:
    if st.button("⚡ Generate Complete ML Code", type="primary", use_container_width=True):
        with st.spinner(f"Generating production-ready code for {selected_model}..."):
            code = generate_code(selected_model, info, df_cleaned.columns)

        st.success("✅ Code generated successfully!")
        st.code(code, language="python")

        # Download button
        st.download_button(
            label="⬇️ Download as .py file",
            data=code,
            file_name=f"ml_model_{selected_model.lower()}.py",
            mime="text/plain",
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown("### 🚀 How to run this code")
        st.code("""
# 1. Install dependencies
pip install pandas scikit-learn xgboost imbalanced-learn matplotlib seaborn

# 2. Place your CSV in the same folder

# 3. Run the script
python ml_model_yourmodel.py
        """, language="bash")
