# PredictiLoan

> Loan Approval Prediction Using Logistic Regression and SVM

<p align='center'>
   <img width="500" alt="predictiloan_poster" src="https://github.com/user-attachments/assets/d7f73d2d-63b7-4559-8534-aaf2244d7fad" />
</p>

---

## Problem Statement

Loan approval prediction is crucial for financial institutions to automate decisions and minimize risk. This project compares the performance of two modeling approaches:

* **Logistic Regression** using engineered numerical and categorical features.
* **SVM** with optimized hyperparameters for high-dimensional decision boundaries.

---

## Dataset Overview

We used the [Loan Approval Prediction dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset), containing:

* Applicant demographic info (education, dependents, employment status)
* Financial details (income, loan amount, asset values)
* Credit history (CIBIL score)

After preprocessing, categorical features were encoded, numerical features scaled, and imbalance handled via SMOTE.

---

## Features

| Feature Group | Description                                                                                      |
| ------------- | ------------------------------------------------------------------------------------------------ |
| Demographics  | `no_of_dependents`, `education`, `self_employed`                                                 |
| Financial     | `income_annum`, `loan_amount`, `loan_term`                                                       |
| Credit        | `cibil_score`                                                                                    |
| Assets        | `residential_assets_value`, `commercial_assets_value`, `luxury_assets_value`, `bank_asset_value` |
| Target        | `loan_status` (Approved=1, Rejected=0)                                                           |

---

## Model Comparison: Logistic Regression vs. SVM

### Final Comparison Table

| Metric        | Logistic Regression       | SVM                       |
| ------------- | ------------------------- | ------------------------- |
| Test Accuracy | **91.57%**                | **93.68%**                |
| CV F1 Score   | 0.9402                    | 0.9558                    |
| Input Format  | Preprocessed tabular data | Preprocessed tabular data |
| Train Samples | size(train.csv)           | size(train.csv)           |
| Model Type    | Linear classifier         | Kernel-based classifier   |

> Confusion matrix visualizations are saved under `figures/`.

| Confusion Matrix (Logistic Regression)                                                       | Confusion Matrix (SVM)                                                                 |
| -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| <img width="640" height="480" alt="logreg_confusion_matrix" src="https://github.com/user-attachments/assets/80ad56f2-f7cf-4263-9b86-f92a26d9a798" /> | <img width="640" height="480" alt="svm_confusion_matrix" src="https://github.com/user-attachments/assets/af621263-4a74-49cf-8ecd-9081b6780a70" /> |

---

### Performance Insights

| ROC Curve (Logistic Regression)                                                        | ROC Curve (SVM)                                                                  |
| -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| <img width="640" height="480" alt="logreg_roc_curve" src="https://github.com/user-attachments/assets/6e11a756-47d9-4e61-a9c3-c28a838e087e" /> | <img width="640" height="480" alt="svm_roc_curve" src="https://github.com/user-attachments/assets/55bfe4f4-8864-4d75-a542-e799dd4293aa" /> |

> Both models performed well, but SVM achieved slightly better recall and F1, making it more effective for catching potential loan defaults.

---

## Project Structure

```
PredictiLoan/
├── data/
│   ├── raw/
│   ├── processed/
│   └── curated/
├── models/
├── figures/
├── notebooks/
│   ├── 01_data_setup.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_model_evaluation.ipynb
├── src/
├── main.py
└── README.md
```

---

## How to Run

1. Clone the repo and install requirements:

   ```bash
   pip install -r requirements.txt
   ```
2. Run `main.py` to preprocess the data, train both models, and save them to the `models/` directory.
3. Open `03_model_evaluation.ipynb` to view metrics, confusion matrices, and performance curves.

---

## Submission

This project was developed as part of the **PredictiLoan Series**, showcasing a comparison between linear and kernel-based classifiers in the context of financial risk assessment.

---

## Author

<div>
<table align="center">
  <tr>
    <td align="center">
      <a href="https://github.com/YassienTawfikk" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/126521373?v=4" width="150px;" alt="Yassien Tawfik"/>
        <br>
        <sub><b>Yassien Tawfik</b></sub>
      </a>
    </td>
  </tr>
</table>
</div>
