# Loan Approval Prediction System

## Project Overview

This project is a complete machine learning solution designed to automate the credit risk assessment process. Instead of relying on manual reviews, I built a system that predicts the likelihood of loan approval based on an applicant's financial and demographic profile. 

The goal was to move beyond simple "Yes/No" answers and provide a data-driven risk score that helps in understanding *why* a loan might be considered high-risk.

---

## Project Structure

* **`notebook.ipynb`**: This is where the core engineering happens. It covers everything from raw data ingestion and outlier handling to training the final classification model.
* **`app.py`**: A production-ready Streamlit interface. It allows users to input applicant details and get an instant, visual risk report.
* **`loan_model.pkl`**: The "brain" of the app. This isn't just the model; it’s a serialized dictionary containing the trained classifier, the feature scalers, and the encoders to ensure the UI processes data exactly like the training environment.

---

## Technical Workflow

### 1. Data Preparation & Cleaning
Real-world financial data is rarely clean. My first priority was ensuring the model wasn't skewed by "noise":
* **Imputation Strategy:** I used the median for numeric columns to stay robust against extreme outliers (like unrealistic income or age entries).
* **Feature Encoding:** Categorical data was converted using Label Encoders so the math-based models could process them.
* **Scaling:** I applied `StandardScaler` to features like income and loan amounts. Without this, the model might incorrectly prioritize larger numbers over smaller, more significant percentages.

### 2. Model Selection
I didn't jump straight to a complex model. I followed a logical progression:
* **Linear Regression:** Used as a baseline to see if the relationships were simple and linear.
* **Decision Tree:** To capture more complex, rule-based logic.
* **Random Forest (Final Choice):** I settled on this because it’s an ensemble method. It handles the non-linear interactions between things like "Credit Score" and "Loan-to-Income Ratio" much better than single-model approaches, leading to higher accuracy.

### 3. The Dashboard Logic
The Streamlit app is built for usability. I split the interface to make it feel like a professional banking tool:
* **Demographics (Sidebar):** Keeps static info like Age and Education separate.
* **Financials (Main Panel):** Focuses on the "moving parts" like Income and Loan Amount.
* **Real-time Calculation:** The app calculates the debt-to-income ratio on the fly before passing it to the model for a prediction.

---

## Key Insights & Features

* **Probability-Based Decisions:** The system doesn't just say "Rejected." It provides a risk probability. This allows a human to see if a rejection was a "near miss" or a high-risk red flag.
* **Handling Data Leakage:** By saving the `StandardScaler` inside the pickle file, the UI uses the exact same mean and variance from the training set. This is a critical step that prevents the model from giving wrong predictions in production.
* **Sanity Checks:** The training pipeline filters out unrealistic data points (like employment years exceeding age) to ensure the model learns from logical human behavior.

---

## How to Run the Project

1. **Environment Setup:** `pip install pandas numpy scikit-learn streamlit`
2. **Model Training:** Run all cells in `notebook.ipynb` to generate the updated `loan_model.pkl`.
3. **Launch UI:** Execute `streamlit run app.py` in your terminal.

---

**Author:** Hasya Patel  
**Tech Stack:** Python, Scikit-learn, Pandas, Streamlit, Pickle
