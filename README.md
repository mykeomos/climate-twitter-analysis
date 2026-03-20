# Climate Change Twitter Analysis

This repository contains a data science analysis pipeline for the Climate Change Twitter dataset, implemented in `twitter_climate_change_analysis.ipynb`.

## Project overview

- Objective: analyze public climate-change sentiment and stance from tweets, and build predictive models to classify user stance.
- Data source: `The Climate Change Twitter Dataset.csv` (tweets with attributes such as sentiment, stance, topic, gender, aggressiveness, temperature, and timestamp).
- Workflow: data wrangling → exploratory data diagnostics and visualizations → modeling and evaluation.

## 1. Data preparation

1. Load dataset (`pandas.read_csv`).
2. Inspect structure and basic stats with `df.info()` and `df.describe()`.
3. Remove duplicate tweets by `id` using `drop_duplicates`.
4. Handle missing values:
   - Drop rows missing target features `stance` and `sentiment`.
   - Fill missing `temperature_avg` values with the column mean.
5. Convert `created_at` to datetime and extract `year`, `month`.
6. Cast categorical fields:
   - `gender`, `topic`, `stance`, `aggressiveness` as `category`.
7. Select main feature subset:
   - columns: `created_at`, `year`, `month`, `lat`, `lng`, `topic`, `sentiment`, `stance`, `gender`, `temperature_avg`, `aggressiveness`.
8. Save cleaned dataset to `clean_climate_data.csv`.

## 2. Descriptive analytics

The notebook visualizes and summarizes:
- Summary statistics (`describe`).
- Sentiment score distribution (histogram).
- Stance distribution (bar chart).
- Gender distribution (bar chart).
- Relationship of temperature vs sentiment (scatter plot).
- Trend of tweets per year (line plot).
- Topic distribution (bar chart).
- Aggressiveness distribution (bar chart).

## 3. Predictive analytics

### 3.1 Encoding
- Label-encode `topic`, `gender`, `aggressiveness`, and target `stance` with `LabelEncoder`.
- New numeric columns: `topic_enc`, `gender_enc`, `aggr_enc`, `stance_encoded`.

### 3.2 Sampling for memory efficiency
- For model training, sample up to 200k rows with `df.sample(sample_size, random_state=42)`.

### 3.3 Features and target
- features: `sentiment`, `temperature_avg`, `topic_enc`, `gender_enc`, `aggr_enc`.
- target: `stance_encoded`.

### 3.4 Train/test split
- `train_test_split(test_size=0.3, random_state=42)`.

### 3.5 Models built
1. Logistic Regression (`sklearn.linear_model.LogisticRegression`)
2. Random Forest (`sklearn.ensemble.RandomForestClassifier`, `n_estimators=100`, `max_depth=12`)
3. Support Vector Machine
   - `StandardScaler` for numeric features
   - `LinearSVC` wrapped in `CalibratedClassifierCV` for probability outputs

### 3.6 Evaluation (expected in notebook)
- Predicted values for each model: `lr_pred`, `rf_pred`, `svm_pred`.
- Add standard metrics (accuracy, precision, recall, F1, ROC AUC) if not already present.

## 4. How to run

1. Create/activate Python environment (recommended Python 3.8+).
2. Install requirements:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Run the notebook:
   ```bash
   jupyter notebook twitter_climate_change_analysis.ipynb
   ```
4. If needed, run script extraction or convert to Python using nbconvert.

## 5. Notes and next steps

- Add explicit model performance table (accuracy, recall, precision, F1, ROC AUC) to evaluate model selection.
- Explore text-level NLP features (e.g., tweet text, sentiment in text, topic modeling).
- Add cross-validation and hyperparameter tuning (`GridSearchCV`/`RandomizedSearchCV`).
- Deploy predictions and design a dashboard summarizing global climate tweet behavior.
