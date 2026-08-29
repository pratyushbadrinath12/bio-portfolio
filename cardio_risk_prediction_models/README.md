# Cardiovascular Risk Prediction: Cox Regression vs. Decision Trees

## Overview

This repository contains a comparative analysis exploring how different computational models predict 10-year heart disease risk. It compares traditional survival analysis—**Cox Proportional Hazards Regression**—against a machine learning approach—**Decision Trees**—to evaluate their approaches to modeling clinical risk factors.

This project was created as a special assignment for **World Heart Day** for my high school biology class.

## Dataset

* **Cohort Size:** 4,000 patient records.


* **Structure:** Synthetic dataset modeled after the Framingham Heart Study.


* **Features:** 14 clinical, demographic, and lifestyle variables including age, blood pressure, total cholesterol, BMI, glucose levels, and smoking status.



## Model Implementations

* **Cox Regression:** Implemented using Python's `lifelines` library to evaluate long-term hazard ratios across patient variables.


* **Decision Tree:** Implemented using `scikit-learn` with depth constraints to extract straightforward decision rules.
