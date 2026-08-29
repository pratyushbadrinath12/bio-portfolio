# Diabetes Prediction with Machine Learning

## Overview

This project uses machine learning to predict the presence of diabetes in patients using clinical and demographic data. It leverages binary classification to model diagnostic risks based on health metrics.

**Award:** Developed as a winning entry for the **NN Scholarship**.

---

## Dataset & Features

The model analyzes standard medical diagnostic metrics to evaluate patient risk:

* **Features:** Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, and Age.


* **Target (`Outcome`):** Binary classification indicating diabetes diagnosis (`1` = positive, `0` = negative).



---

## Methodology

* **Algorithm:** Logistic Regression implemented via `scikit-learn`.


* **Data Split:** 90% training / 10% test split.


* **Classification Strategy:** Sigmoid S-curve mapping feature inputs to probability outputs using a $0.5$ decision threshold.



---
