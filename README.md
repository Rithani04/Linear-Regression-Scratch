# Linear Regression from Scratch (Gradient Descent Implementation)

This repository demonstrates a **from-scratch implementation** of Simple Linear Regression in Python, without using `scikit-learn` or any other high-level ML libraries for the core algorithm.  
It serves as both a practical coding exercise and an illustration of the underlying mathematics of regression models.

---

## 📌 Overview
Linear Regression is one of the most fundamental algorithms in machine learning.  
It models the relationship between an **independent variable (X)** and a **dependent variable (y)** by fitting a straight line:
ŷ = wX + b


This project implements:
- Gradient Descent optimization
- Mean Squared Error (MSE) as the loss function
- Visualization of predictions vs. actual data

---

## 🧠 Mathematical Foundation

### 1. Hypothesis Function
ŷ = wX + b

Where:
- **w** = weight (slope)
- **b** = bias (intercept)
- **X** = feature values
- **ŷ** = predicted output

### 2. Loss Function (MSE)
We use **Mean Squared Error** to measure the model's prediction error: MSE = (1/m) * Σ (yᵢ - ŷᵢ)²


### 3. Gradient Descent Updates
To minimize the loss, we iteratively update:
w := w - α * (2/m) * Xᵀ(ŷ - y)
b := b - α * (2/m) * Σ(ŷ - y)

Where:
- **α** = learning rate  
- **m** = number of training examples

---



