# Climate Change Twitter Analysis (Academic Report)

This repository documents an empirical analysis of the Climate Change Twitter dataset using a reproducible Python notebook: `twitter_climate_change_analysis.ipynb`.

## 1. Introduction

The primary research question is whether structured metadata from climate-related tweets (excluding tweet text) enables accurate inference of user stance (support, against, neutral). This study explicitly focuses on feature-level indicators and bypasses raw text embedding due to the scope of this analysis.

## 2. Dataset and preprocessing

- Data file: `The Climate Change Twitter Dataset.csv`.
- Variables included: `created_at`, `lat`, `lng`, `topic`, `sentiment`, `stance`, `gender`, `temperature_avg`, `aggressiveness`.

### 2.1 Data cleaning
- Remove duplicate observations based on `id`.
- Drop rows with missing `stance` or `sentiment`.
- Impute `temperature_avg` missing values with column mean.
- Parse datetime to extract `year` and `month`.
- Enforce categorical data types for `topic`, `gender`, `stance`, and `aggressiveness`.

### 2.2 Feature selection
- Retain columns: `created_at`, `year`, `month`, `lat`, `lng`, `topic`, `sentiment`, `stance`, `gender`, `temperature_avg`, `aggressiveness`.
- Persist cleaned data as `clean_climate_data.csv`.

## 3. Descriptive analysis

The notebook includes exploratory data analysis with:
- summary statistics
- distributions for `sentiment`, `stance`, `gender`, `topic`, `aggressiveness`
- temperature vs sentiment relationship
- temporal trend of tweet volume by year

These diagnostics are presented with appropriate visualizations and academic narrative.

## 4. Predictive modeling

### 4.1 Encoding and modeling variables
- Label encoding applied to: `topic`, `gender`, `aggressiveness`, and target `stance`.
- Engineered variables: `topic_enc`, `gender_enc`, `aggr_enc`, `stance_encoded`.
- Predictor set: `sentiment`, `temperature_avg`, `topic_enc`, `gender_enc`, `aggr_enc`.

### 4.2 Train-test partition
- Dataset sampled up to 200,000 rows for computational efficiency.
- 70/30 train-test split using `random_state=42`.

### 4.3 Algorithms evaluated
1. Logistic Regression (`sklearn.linear_model.LogisticRegression`)
2. Random Forest (`sklearn.ensemble.RandomForestClassifier`, `n_estimators=100`, `max_depth=12`)
3. Gradient Boosting (`sklearn.ensemble.GradientBoostingClassifier`, `n_estimators=100`, `learning_rate=0.1`, `max_depth=4`)

**Note:** Support Vector Machine was initially considered, but the final implementation uses Gradient Boosting due to superior class discrimination in the structured feature space.

## 5. Evaluation framework

Model performance is evaluated using:
- accuracy
- macro and weighted precision, recall, F1
- class-level precision/recall/F1/support (classification report)
- confusion matrices
- ROC-AUC curves (one-vs-rest multi-class strategy)

A summary table is computed for direct model comparison.

## 6. Limitations and academic notes

- This analysis does not use tweet text tokens or NLP-based feature extraction (e.g., TF-IDF, transformers).
- The modeling exercise is therefore constrained to metadata and sentiment scores already provided in the dataset.
- Class imbalance is present and addressed via evaluation prioritizing macro metrics and optionally balanced class weights.

## 7. Reproducibility

1. Set up an environment (Python 3.8+).
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Execute:
   ```bash
   jupyter notebook twitter_climate_change_analysis.ipynb
   ```

## 8. Future work

- Introduce tweet text representation (e.g., TF-IDF, embeddings) for richer stance inference.
- Apply cross-validation and hyperparameter optimization (`GridSearchCV`, `RandomizedSearchCV`).
- Evaluate fairness and demographic bias across `gender`, `topic`, and `geolocation`.
- Explore model deployment for interactive dashboard or policy monitoring.

