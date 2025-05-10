# Loan Applicant Credit Risk Prediction

## Project Overview
This study analyzes loan applicants' credit risk, focusing on predicting whether an applicant will be 90 days past due. The objective is to evaluate different predictive models and address class imbalance to improve the identification of high-risk applicants. The study applied SMOTE and PCA to balance the dataset and reduce dimensionality, finding that Random Forest performed best, particularly in improving recall for identifying high-risk applicants. Future work can further optimize Random Forest, explore alternative models, and test the model with real-world data to ensure better generalization and decision-making.

### Objectives of the Study
- Evaluate predictive modeling techniques to identify the most effective model
- Gain insights into applicant data and its relation to creditworthiness
- Communicate findings clearly for stakeholders with varied technical backgrounds

## Introduction
The project predicts whether a loan applicant is 90 days past due based on various credit-related features. The target variable, 'NA90', indicates whether the applicant is classified as high-risk. Accurate identification of high-risk applicants is critical for minimizing loan default losses. Python libraries such as pandas, NumPy, scikit-learn, and seaborn were used to explore and model the dataset.

## Exploratory Data Analysis
- Total records: 376,208  
- Total features: 249  
- Key issue: class imbalance in 'NA90' (14.3% positive class)  
- Tools used: pie chart, bar chart, correlation heatmap  
- Purpose: understand data distribution, guide feature engineering, and select meaningful predictors

## Data Cleaning & Preprocessing
- Verified no duplicate or missing entries
- Removed features with >50% invalid negative values
- Corrected remaining negative values to 0
- Dropped identifier columns (e.g., Loan ID) from feature set
- Applied StandardScaler for normalization

## 📊 Initial Model Evaluation
- Train/test split: 70%/30%
- Models: Decision Tree, Random Forest, KNN
- Evaluated using accuracy, precision, recall, and F1
- Random Forest performed best in accuracy (85.8%) but needed better recall

## 💸 Class Balancing with SMOTE + PCA
- Applied SMOTE to balance classes
- Used PCA to reduce dimensionality (13 components captured ~50% variance)

## 🔧 Hyperparameter Tuning
- Random Forest tuned with `RandomizedSearchCV`
- KNN tuned with `GridSearchCV`
- Best recall scores:
  - Random Forest: 72.86%
  - KNN: 70.26%

## 🔄 Model Comparison & Results
- Random Forest achieved better recall and F1 than KNN
- Recall prioritized to avoid false negatives (missing high-risk applicants)
- Final recall after SMOTE + PCA for Random Forest: 62.71%

## 📖 Conclusion & Future Work

- Random Forest is most effective for predicting credit risk.
- Future enhancements:
  - Optimize Random Forest hyperparameters.
  - Explore other balancing methods (e.g., ADASYN).
  - Try feature selection and alternate dimensionality reduction (e.g., t‑SNE).
  - Test model on real‑world or unseen datasets for validation.

## Features

- **Automated Class Balancing**: Apply SMOTE to synthetically up‑sample minority (high‑risk) applicants.
- **Dimensionality Reduction**: Reduce 249 features to 13 principal components while retaining ≈50% variance via PCA.
- **Hyperparameter Search**: Reproducible tuning with `RandomizedSearchCV` and `GridSearchCV`.
- **Pretrained Artifacts**: Ready‑to‑use `scaler.pkl` and `pca_model.pkl` streamline inference.
- **Transparent Workflow**: Notebooks document EDA, cleaning, modeling, and evaluation for stakeholders.

## Files Included
- `notebooks/`: Jupyter notebooks for analysis  
- `scripts/`: Python scripts for modeling  
- `model/`: Pretrained model components (`scaler.pkl`, `pca_model.pkl`)  
- `.gitignore`: Excludes unnecessary files from tracking

## 📌 Note on Data and Model Files
The original training data is proprietary and not included in this repository.

However:
- Trained `scaler.pkl` and `pca_model.pkl` files are provided
- Full code is included to reproduce preprocessing, modeling, and evaluation steps

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/Credit.ipynb
