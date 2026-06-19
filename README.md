# Disease Prediction using Medical Data 🩺⚕️

This repository contains an advanced Machine Learning pipeline designed to predict medical diagnoses (Breast Cancer malignancy) using structured patient data. Developed as part of the CodeAlpha Machine Learning Internship.

## 🎯 Objective
To accurately predict the presence of disease based on complex medical features (tumor radius, texture, perimeter, etc.) while strictly optimizing for **Recall** to minimize dangerous False Negatives in a diagnostic setting.

## 🛠️ Architecture & Ensembling Strategy
* **Language:** Python
* **Libraries:** Scikit-Learn, XGBoost, Pandas, NumPy
* **Core Methodology:** This pipeline bypasses single-algorithm limitations by utilizing a **Voting Classifier** (`voting='soft'`). 
* **Base Learners:** The ensemble acts as a convex combination of three distinct predictive algorithms:
  1. Logistic Regression (Linear baseline)
  2. Support Vector Machine / SVC (Non-linear boundary detection)
  3. XGBoost (Gradient boosted decision trees)

## 📈 Dataset
The model is trained on the **Breast Cancer Wisconsin (Diagnostic) Dataset**, which contains computed features from digitized images of fine needle aspirates (FNA) of breast masses.

## 🚀 Execution Guide
1. Open `disease_prediction.ipynb` in Google Colab or Jupyter.
2. Run the cell to load the data, apply standard scaling, and train the ensemble model.
3. The final output provides a comprehensive classification report highlighting Precision, Recall, and F1-Scores.

---
*Developed by Muhammad Zeeshan | AI/ML Engineer*
