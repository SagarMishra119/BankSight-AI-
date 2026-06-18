# 🏦 BankSight AI – Deposit Subscription Predictor

Live Demo : https://banksight-ai.onrender.com/

A modern, interactive **Machine Learning-powered Streamlit web application** that predicts whether a bank customer is likely to subscribe to a **Term Deposit** based on demographic, financial, and marketing campaign data.

The application features a premium glassmorphism UI, real-time predictions, interactive analytics, and model insights.

---

# 🚀 Features

## 📊 Dashboard

* Modern glassmorphism interface
* Project overview and model summary
* Feature group visualization
* Quick navigation to prediction module

## 🔮 Customer Prediction

* Real-time deposit subscription prediction
* Probability/confidence scoring
* Personalized recommendations
* Downloadable prediction reports

## 📈 Analytics & Insights

* Interactive Plotly visualizations
* Subscription probability gauge
* Feature importance charts
* Customer profile radar chart
* Dataset-level insights and trends

## 🧠 Model Insights

* Random Forest model explanation
* Feature importance overview
* End-to-end prediction pipeline visualization
* Business intelligence insights

## ℹ️ About Section

* Dataset information
* Technology stack
* Model limitations and notes

---

# 🏗️ Project Structure

```text
BankSight-AI/
│
├── app.py                 # Main Streamlit application
├── model.pkl              # Trained Random Forest model
├── features.pkl           # Training feature schema
├── cat_cols.pkl           # Categorical columns list
├── bank.csv               # Dataset
├── requirements.txt       # Dependencies
│
└── README.md
```

---

# 🧠 Machine Learning Pipeline

## Algorithm

* Random Forest Classifier

## Input Features

### Customer Demographics

* Age
* Job
* Marital Status
* Education

### Financial Information

* Balance
* Credit Default
* Housing Loan
* Personal Loan

### Campaign Information

* Contact Type
* Day
* Month
* Call Duration
* Campaign Calls

### Previous Marketing History

* Previous Contacts
* Days Since Last Contact
* Previous Campaign Outcome

---

## Processing Pipeline

```text
Raw Input
    ↓
Feature Engineering
    ↓
One-Hot Encoding
    ↓
Feature Alignment
    ↓
Random Forest Prediction
    ↓
Probability Score
```

### Engineered Feature

```python
was_contacted_before = (previous > 0)
```

---

# 📦 Installation

## 1. Clone Repository

```bash
git clone https://github.com/SagarMishra119/BankSight-AI-
cd banksight-ai
```

## 2. Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Dependencies:

```text
streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.26.0
scikit-learn==1.6.1
joblib>=1.3.0
plotly>=5.20.0
```

---

# ▶️ Running the Application

```bash
streamlit run app.py
```

Application URL:

```text
http://localhost:8501
```

---

# 📊 Dataset

The project uses the **Bank Marketing Dataset** from the UCI Machine Learning Repository.

### Dataset Summary

| Metric         | Value  |
| -------------- | ------ |
| Records        | 11,162 |
| Features       | 16     |
| Target Classes | 2      |

### Target Variable

```text
deposit
```

Values:

* Yes → Customer subscribes to term deposit
* No → Customer does not subscribe

---

# 📈 Prediction Output

The application predicts:

### Likely to Subscribe

or

### Unlikely to Subscribe

Additionally, it provides:

* Subscription Probability
* Confidence Score
* Recommended Business Actions
* Downloadable Prediction Report

---

# 🎨 User Interface Highlights

* Glassmorphism Design
* Dark Premium Theme
* Interactive Plotly Charts
* Responsive Layout
* Analytics Dashboard
* Dynamic KPI Cards
* Report Download Support

---

# ⚠️ Important Notes

## Duration Leakage

The feature:

```text
duration
```

is only known after a phone call has occurred.

For production-grade pre-call prediction systems, this feature should be removed to prevent data leakage.

## Educational Purpose

This project is intended for:

* Academic Demonstrations
* Machine Learning Learning Projects
* Data Science Portfolios
* College Presentations

Model performance may vary depending on dataset splits and training configurations.

---

# 🛠️ Technology Stack

| Layer             | Technology    |
| ----------------- | ------------- |
| Frontend          | Streamlit     |
| Machine Learning  | Scikit-Learn  |
| Data Processing   | Pandas, NumPy |
| Visualization     | Plotly        |
| Model Persistence | Joblib        |
| Language          | Python        |

---

# 💡 Business Value

BankSight AI helps banks:

* Identify high-potential customers
* Improve campaign efficiency
* Reduce marketing costs
* Increase conversion rates
* Enable data-driven decision making

---

# 🔮 Future Enhancements

* SHAP Explainability
* XGBoost Integration
* Customer Segmentation
* Batch Predictions
* Model Comparison Dashboard
* Cloud Deployment
* User Authentication
* CRM Integration

---

# 👨‍💻 Author

**BankSight AI**

Machine Learning-powered Bank Marketing Campaign Intelligence System built using:

* Python
* Streamlit
* Scikit-Learn
* Plotly
* Random Forest Classifier

---

# 📜 License

This project is released for educational and research purposes.

Feel free to modify, extend, and use it for:

* Learning
* Academic Projects
* Portfolio Development
* Research Work
