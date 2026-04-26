# Financial Fraud Detection with Random Forest 🕵️‍♂️💸

## Overview
This repository contains a complete, end-to-end machine learning pipeline designed to detect fraudulent financial transactions. Utilizing the **PaySim synthetic dataset** (comprising over 6.3 million records), this project tackles the critical challenge of extreme class imbalance to accurately identify malicious activity without generating excessive false alarms.

## The Challenge: Class Imbalance
In the raw dataset, fraudulent transactions account for less than **0.14%** of the total volume. Standard machine learning models struggle with this environment, often achieving artificially high accuracy simply by blindly predicting the majority class ("Not Fraud") and missing the actual crimes.

## Methodology & Preprocessing
To build a robust predictive model, the following data engineering pipeline was implemented:

* **One-Hot Encoding:** Categorical transaction types (e.g., `TRANSFER`, `CASH_OUT`) were numerically encoded to allow for independent mathematical comparison.
* **Feature Scaling (StandardScaler):** Transaction amounts and account balances were scaled to Z-scores. This prevented columns with massive dollar ranges from biasing the model against smaller numerical features like the transaction `step` (time).
* **Random Undersampling:** The extreme class imbalance was resolved by randomly sampling 8,213 legitimate transactions to perfectly match the 8,213 fraudulent transactions. This created a balanced, unbiased training subset of 16,426 rows.

## Model Performance & Results
A **Random Forest Classifier** (`n_estimators=100`) was trained on the balanced dataset and evaluated on an unseen 20% test split. The ensemble approach yielded exceptional results:

* **Overall Accuracy:** 99%
* **Fraud Recall:** 1.00 
* **Precision:** 0.99

> **Key Insight:** *Out of 1,664 actual fraud attempts in the final exam test set, the model successfully caught 1,660. Feature importance analysis revealed that the model relied most heavily on sender and receiver account balances (`oldbalanceOrg`, `newbalanceDest`) to flag suspicious behavior, rather than just the raw transaction amount.*

## Tech Stack
* **Language:** Python 3
* **Libraries:** Pandas, Scikit-Learn, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Environment:** Google Colab / Jupyter Notebook

## How to Run
1. Open the `.ipynb` notebook in Google Colab or your local Jupyter environment.
2. The notebook includes a script using `kagglehub` to automatically fetch the PaySim dataset.
3. Run the cells sequentially to observe the Exploratory Data Analysis (EDA), pipeline preprocessing, model training, and the final confusion matrix visual outputs.

---
**Author:** [Zhyrgalbek Kalykov](https://github.com/zkalykov)  
**Institution:** North American University  
**Course:** COMP 2319 - Introduction to Artificial Intelligence
