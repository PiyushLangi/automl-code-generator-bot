# 🤖 AutoML Code Generator Bot

An intelligent AutoML system that analyzes datasets, detects problem types, recommends optimal machine learning models, and generates complete production-ready Python code using LLMs.

---

## 🔥 Description

An intelligent AutoML system designed to automate the machine learning workflow. The application analyzes datasets, performs feature engineering with data leakage prevention, detects problem types, and recommends optimal models based on dataset characteristics. It integrates LLM (Groq + Llama 3) to generate complete, production-ready Python ML pipelines including preprocessing, training, and evaluation. Built with an interactive Streamlit interface for seamless user experience and explainable insights.


---

## 🚀 Features

* 📊 Automatic dataset analysis (rows, columns, missing values, skewness)
* 🧠 Intelligent problem type detection (classification / regression)
* 🔒 Data leakage prevention (removes ID, score, reason, category columns)
* ⚙️ Smart model recommendation engine (based on dataset characteristics)
* 🧹 Automated feature engineering insights
* 📈 Interactive visualizations (correlation heatmap, missing values, distribution)
* 🤖 LLM-powered code generation (Groq + Llama 3)
* 💡 AI-generated data insights and warnings
* 🎯 Cleaned dataset preview for transparency

---

## 🛠️ Tech Stack

* **Programming:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **Visualization:** Matplotlib, Seaborn
* **Frontend/UI:** Streamlit
* **LLM Integration:** Groq API (Llama 3)
* **Version Control:** Git, GitHub

---

## 📁 Project Structure

```
automl-code-generator-bot/
│
├── app.py                  # Main Streamlit application
├── data_analyzer.py        # Dataset analysis and preprocessing logic
├── model_selector.py       # Intelligent model recommendation engine
├── code_generator.py       # LLM-based ML code generation (Groq + Llama 3)
├── utils.py                # Visualization and helper functions
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
└── images/                 # Screenshots for README
```

---

## 🧠 How It Works

```text
Upload CSV
   ↓
Automatic Data Cleaning (remove leakage + ID columns)
   ↓
Dataset Analysis (EDA)
   ↓
Problem Type Detection
   ↓
Model Recommendation
   ↓
Visualization & Insights
   ↓
LLM generates complete ML code
```

---

## 📸 Screenshots

![alt text](Main_UI.png)


### 📊 Dataset Analysis

![alt text](Dataset_Analysis.png)


### 🧹 Cleaned Dataset Preview

![alt text](Cleaned_Dataset_Preview.png)


### 🏆 Model Recommendations

![alt text](Model_Recommendation.png)


### 📈 Visualizations

![alt text](Visualization1.png)

![alt text](Visualization2.png)


### 💻 Generate Python Code

![alt text](Generate_Python_ML_code1.png)

![alt text](Generate_Python_ML_code2.png)

---

## ▶️ How to Run the Project

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/automl-code-generator-bot.git
cd automl-code-generator-bot
```

---

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run the app

```bash
streamlit run app.py
```

---

### 4️⃣ Open in browser

```
http://localhost:8501
```

---

## 🔑 API Setup (IMPORTANT)

This project uses **Groq API (Llama 3)** for code generation.

1. Go to: https://console.groq.com
2. Create a free account
3. Generate API key
4. Paste it in the sidebar inside the app

---

## 💡 Use Cases

* Beginners learning Machine Learning
* Data Analysts automating ML workflows
* Rapid prototyping of ML models
* Educational tool for understanding pipelines

---

## ⚠️ Important Note

* The system prevents **data leakage** by removing columns like:

  * IDs
  * Scores
  * Reasons
  * Categories
* This ensures realistic and unbiased model recommendations.

---

## 🚀 Future Improvements

* 🔥 Auto model training inside app
* 📊 Accuracy comparison dashboard
* 💾 Model saving (pickle)
* 🌐 Deployment (Streamlit Cloud)
* 🤖 Agent-based automation

---

## 👨‍💻 Author

**Piyush Langi**
📍 Mumbai, India

* GitHub: https://github.com/PiyushLangi
* LinkedIn: https://linkedin.com/in/piyush-langi-72647226b

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
