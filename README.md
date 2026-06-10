# Creditwise-Loan-Approval-Prediction-System
Automated loan approval system using Machine Learning (Logistic Regression, KNN, Naive Bayes) to evaluate credit risk, optimize lending decisions, and minimize NPAs. Features end-to-end binary classification pipeline with EDA, feature engineering, and robust evaluation (Precision, Recall, F1-score).

# Loan Approval Prediction System 🏦💰

An end-to-end Supervised Machine Learning pipeline built to predict loan approval using customer credit profiles, demographic data, and financial history. This project implements binary classification algorithms and comprehensive evaluation metrics to minimize credit risk for financial institutions.

## 📌 Project Overview
Manual verification of loan applications is slow and vulnerable to human errors. This project automates the decision-making process by training classification models to predict whether a loan application should be **Approved** or **Rejected** based on historical patterns of creditworthiness.

## 🛠️ Tech Stack & Libraries Used
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
- **Environment:** Jupyter Notebook

## 📋 Data Dictionary
The historical dataset contains key personal, financial, and credit attributes of the applicants:
- `Applicant_Income` / `Coapplicant_Income`: Monthly income profiles.
- `Age` / `Marital_Status` / `Dependents` / `Education_Level`: Demographic data.
- `Credit_Score`: Bureau historical credit rating.
- `Existing_Loans` / `DTI_Ratio`: Current financial liabilities and Debt-to-Income ratio.
- `Savings` / `Collateral_Value`: Assets held by the applicant.
- `Loan_Amount` / `Loan_Term` / `Loan_Purpose`: Requested loan specifications.
- `Loan_Approved`: **[Target Variable]** (1 = Approved, 0 = Rejected).

## 🔄 Machine Learning Pipeline

### 1. Data Cleaning & Exploratory Data Analysis (EDA)
- Handled missing values and eliminated redundant features.
- Visualized distributions of numerical variables and frequency of categorical factors.
- Outlier detection and correlation analysis to avoid multi-collinearity.

### 2. Feature Engineering
- Categorical encoding for non-numeric variables.
- Feature scaling to normalize data distributions.
- Handled class imbalance to ensure stable model training.

### 3. Model Training & Implementation
We implemented and compared three powerful supervised learning algorithms:
- **Logistic Regression** (Baseline statistical classification)
- **K-Nearest Neighbors (KNN)** (Distance-based instance learning)
- **Naive Bayes** (Probabilistic classification based on Bayes' Theorem)

### 4. Evaluation Strategy
In banking risk assessment, Accuracy alone is misleading. Therefore, models were evaluated using:
- **Precision:** To ensure we don't reject good customers.
- **Recall:** To minimize the risk of approving potential defaulters (Crucial for banking).
- **F1-Score:** To maintain a balance between Precision and Recall.

## 📂 Project Structure
```text
├── data/                  # Contains the dataset
├── notebooks/
│   └── credit_wise_loan_approval.ipynb  # Main Jupyter Notebook with core pipeline code
├── loan_approval_problem_statement.pdf  # Detailed corporate problem statement document
├── README.md              # Project documentation and summary
└── requirements.txt       # Python dependencies (pandas, numpy, scikit-learn)
