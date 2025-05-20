# 🛡️ Phishing Domain Detection using Machine Learning

Detect and classify domain names as **legitimate** or **phishing** using supervised machine learning models. This project is built with Python and leverages feature engineering and classification techniques to identify suspicious domains that may be used in phishing attacks.

---

## 📌 Project Overview

Phishing websites often use deceptive or slightly altered domain names to trick users. This project trains a machine learning model to distinguish between real and fake domains by analyzing lexical features of domain names.

---

## 📂 Project Structure
phishing-domain-detector/
│
├── dataset_full.csv # Full dataset with labeled domains
├── Jayesh Badoge project on ML(Phishing Domain Detection)-checkpoint.ipynb # Main Jupyter notebook (auto-saved checkpoint)
├── Untitled-checkpoint.ipynb # Unnamed notebook (auto-saved checkpoint)
├── Logistic Regression-checkpoint.py # Logistic Regression Python script (checkpoint)
├── README.md # Project documentation

---

## 🔍 Features

- Binary classification: `Real` vs `Phishing`
- Feature extraction from domain names (e.g., length, digits, special characters, entropy)
- Comparison of multiple ML models (e.g., Logistic Regression, Random Forest)
- Evaluation using accuracy, precision, recall, F1-score, and confusion matrix
- Visualization of data distributions and model performance

---

## 🧰 Tech Stack

- Python 3.x
- Pandas
- scikit-learn
- matplotlib
- NumPy

---

## 🚀 How to Run

1. **Clone this repository:**
   ```bash
   git clone https://github.com/jaybadoge07/Phishing-Domain-Detection.gitector
   cd phishing-domain-det
