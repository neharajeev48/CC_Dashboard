# 📡 Customer Churn Intelligence Dashboard

> **Objective:** Understand why telecom customers stay or leave — using 4 layers of analytics.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## 🎯 Overview

A production-grade Streamlit analytics dashboard that provides a **360° view of customer churn** in a telecom context. Built around one core objective: **understand the why behind customer exits and how to prevent them**.

The dashboard covers all **4 types of analytics**:

| Layer | Question | Tabs |
|-------|----------|------|
| 📊 **Descriptive** | What does our customer base look like? | Tab 1 |
| 🔍 **Diagnostic** | Why are customers churning? | Tab 2 |
| 🤖 **Predictive** | Who is likely to churn next? | Tab 3 |
| 💡 **Prescriptive** | What actions should we take? | Tab 4 |

---

## 📊 Dataset

**File:** `CustomerChurn.csv`  
**Rows:** 300 customers | **Columns:** 11

| Column | Description |
|--------|-------------|
| `CustomerID` | Unique customer identifier |
| `Age` | Customer age (18–79) |
| `TenureMonths` | Months as customer (1–60) |
| `MonthlySpend` | Monthly billing amount ($) |
| `SubscriptionType` | Basic / Standard / Premium |
| `ServiceUsage` | Monthly service usage (units) |
| `SupportCalls` | Number of support calls made |
| `AvgRating` | Customer satisfaction rating (1–5) |
| `HasContract` | 1 = has contract, 0 = no contract |
| `IsActive` | 1 = currently active, 0 = inactive |
| `Churn` | **Target variable** — 1 = churned, 0 = retained |

---

## 🖥️ Dashboard Features

### 📊 Tab 1 — Descriptive Analysis
- Overall churn donut chart
- **Drill-down sunburst charts** (Subscription → Churn, Contract → Churn)
- Numeric distribution histograms (Churned vs Retained overlay)
- Churn rate by tenure band (color-scaled bar chart)
- **Heatmap:** Age × Subscription churn rate matrix
- Churn rate by support call volume
- Customer count by rating band

### 🔍 Tab 2 — Diagnostic Analysis
- Multi-variable scatter plots with churn overlay
- **Statistical significance testing** (t-tests, Cohen's d effect sizes)
- Full **Pearson correlation matrix** (lower triangle)
- Churn rate trend lines across tenure & spend deciles
- Box plots comparing churned vs retained distributions
- Violin plots for subscription and contract comparisons

### 🤖 Tab 3 — Predictive Analysis
- **3 ML models:** Random Forest, Gradient Boosting, Logistic Regression
- Cross-validated **ROC curves** & AUC scores
- Feature importance rankings (RF + GB ensemble)
- Predicted churn probability distribution
- Customer **Risk Tier segmentation** (Low / Medium / High)
- Confusion matrix visualization
- **Live churn predictor** — input any customer profile, get instant probability

### 💡 Tab 4 — Prescriptive Analysis
- Prioritized retention intervention cards (HIGH / MEDIUM / LOW priority)
- **Intervention impact simulation** — projected churn rate under each scenario
- Monthly revenue recovery estimates per intervention
- **Customer Strategy Quadrant Map** (Engagement vs Risk score)
- Segment-level action recommendations

### 🗃️ Tab 5 — Data Explorer
- Full dataset with derived features + predicted churn probabilities
- Sortable, filterable dataframe
- Download full dataset or high-risk customers as CSV

---

## 🚀 Deployment on Streamlit Cloud

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: Customer Churn Dashboard"
git remote add origin https://github.com/YOUR_USERNAME/CustomerChurn_Dashboard.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy!**

> ⚠️ Make sure `CustomerChurn.csv` is committed to your repo alongside `app.py`.

---

## 💻 Run Locally

```bash
# Clone / navigate to the project
cd CustomerChurn_Dashboard

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📐 Key Findings

Based on the dataset analysis:

1. **29% overall churn rate** — significant revenue risk
2. **No-contract customers** churn at ~2.5× the rate of contracted customers
3. **First-year customers** (0-12 months) are the most vulnerable segment
4. **High support call volume (4+)** is the strongest behavioral churn signal
5. **Low service usage** customers are disengaged and at elevated risk
6. **Contract lock-in** is the single highest-ROI retention lever

---

## 🗂️ File Structure

```
CustomerChurn_Dashboard/
├── app.py                  # Main Streamlit application (~900 lines)
├── CustomerChurn.csv       # Dataset
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .gitignore
└── .streamlit/
    └── config.toml         # App theme (dark, telecom blue)
```

---

## 🛠️ Tech Stack

- **[Streamlit](https://streamlit.io)** — Web app framework
- **[Plotly](https://plotly.com)** — Interactive charts (donut, sunburst, heatmap, scatter, etc.)
- **[scikit-learn](https://scikit-learn.org)** — Random Forest, Gradient Boosting, Logistic Regression
- **[pandas](https://pandas.pydata.org)** + **[numpy](https://numpy.org)** — Data manipulation
- **[scipy](https://scipy.org)** — Statistical testing

---

*Built to answer one question: **Why do telecom customers stay or leave?***
