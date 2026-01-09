# Zomato Restaurant Reviews – EDA & ML Capstone

## Project Type
**EDA + Classification**  
**Contribution:** Individual

## Overview
This project analyzes Zomato restaurant data using two CSV files:

- `Zomato Restaurant names and Metadata.csv`
- `Zomato Restaurant reviews.csv`

The work has two main goals:

1. Perform structured **Exploratory Data Analysis (EDA)** to understand restaurant pricing, cuisines, ratings, and review behaviour.
2. Build **classification models** that predict whether a review is **high-rated** (rating ≥ 4) using numeric features engineered from the data.

## Dataset & Preprocessing

### Files
- **Metadata:** Restaurant name, link, cost, collections, cuisines, timings  
- **Reviews:** Restaurant, reviewer, review text, rating, metadata string, time, number of pictures

### Key Cleaning & Wrangling Steps
- Stripped leading/trailing spaces from all string columns.
- Standardised restaurant names to a lowercased `restaurant_key` and **merged** both datasets on this key.
- Parsed `Cost` into numeric `cost_for_two`.
- Converted rating text into numeric `Rating_num`.
- Filled missing values for:
  - `Collections` → `"No collection tagged"`
  - `Timings` → `"Not available"`
  - `Reviewer` → `"Anonymous"`
- Dropped rows where both `Review` and `Rating` were missing.
- Parsed review time to `Time_parsed` and created the final analysis dataframe `df_final`.

## Exploratory Data Analysis (EDA)

EDA follows the **UBM** structure:

- **Univariate:**  
  - Distribution of ratings  
  - Distribution of `cost_for_two`  
  - Top 10 most-reviewed restaurants  
  - Count of primary cuisine types  
  - Distribution of number of reviews per restaurant  

- **Bivariate:**  
  - Average rating by cost bucket (Low/Medium/High)  
  - Average rating by primary cuisine  
  - Rating vs number of reviews (scatter)  
  - Cost for two vs number of reviews (scatter)  
  - Pictures vs rating patterns  

- **Multivariate:**  
  - Correlation heatmap of numeric features  
  - Pairplot of selected numeric features  
  - Combined views of cost, rating and review volume  

Each chart is documented with:
1. Why the chart type was chosen  
2. Key insights  
3. Potential positive/negative business impact  

## Hypothesis Testing

Several business-focused hypotheses were tested, for example:

- Whether reviews **with pictures** have significantly different average ratings than those **without pictures**.
- Whether higher-cost restaurants tend to receive higher ratings.
- Whether longer reviews (word count) are associated with higher ratings.

Appropriate statistical tests (e.g., **two-sample t-test**) were used, and conclusions are based on p-values and effect directions.

## Machine Learning Model

### Target
- `high_rating` (binary):  
  - 1 → rating ≥ 4  
  - 0 → rating < 4  

### Features
Numeric features engineered from `df_final`, including:

- `cost_for_two_log` (log-transformed cost for two)
- `review_word_count_log` (log of word count in review text)
- `review_char_len` (number of characters in review)
- `Pictures` (number of pictures attached)
- `review_hour` (hour of the day the review was posted)

### Models Implemented

1. **Model 1 – Logistic Regression (Baseline)**  
   - Pipeline: `StandardScaler` + `LogisticRegression(class_weight='balanced')`  
   - Evaluated with Accuracy, Precision, Recall, F1-score, ROC–AUC.  
   - Provides a simple, interpretable baseline for the binary classification task.

2. **Model 1 (Tuned) – Logistic Regression + GridSearchCV**  
   - Hyperparameter tuning on `C` using **GridSearchCV** with 5-fold cross-validation and F1-score as the objective.  
   - Best parameters and cross-validated F1-score are reported in the notebook.  
   - Shows improvement over the baseline in terms of F1-score and more balanced Precision/Recall.

3. **Model 2 – Random Forest (Baseline)**  
   - `RandomForestClassifier` with `class_weight='balanced'`, `n_estimators=200`.  
   - Captures non-linear relationships between features and the target.  
   - Evaluated using the same metric set (Accuracy, Precision, Recall, F1, ROC–AUC).

4. **Model 2 (Tuned) – Random Forest + GridSearchCV**  
   - Hyperparameters tuned: `n_estimators`, `max_depth`, `min_samples_split`.  
   - Used **GridSearchCV** (3-fold, F1-score) to find a better bias–variance trade-off.  
   - Tuned model performance is compared with baseline Random Forest to quantify gains.

5. **Model 3 – Gradient Boosting (Baseline)**  
   - `GradientBoostingClassifier` with `n_estimators=100`, `learning_rate=0.1`, `max_depth=3`.  
   - Boosting-based model aimed at capturing more complex patterns than Logistic Regression.  

6. **Model 3 (Tuned) – Gradient Boosting + GridSearchCV**  
   - Hyperparameters tuned: `n_estimators`, `learning_rate`, `max_depth`.  
   - GridSearchCV used with F1-score to choose the best configuration.  
   - Tuned model metrics (Accuracy, F1, ROC–AUC, etc.) are reported and compared with the baseline Gradient Boosting and other models.

For all models, performance is compared using a **metric bar chart** and confusion matrices to understand trade-offs between Precision, Recall, and overall classification quality.

### Model Persistence

- The best-performing model is saved to disk (e.g., using `joblib.dump`) as a serialized file.  
- The saved model is then reloaded and used to score a small sample of unseen data for a sanity check, demonstrating readiness for deployment.

## Tech Stack

- **Language:** Python  
- **Libraries:**  
  - Data: `pandas`, `numpy`  
  - Visualisation: `matplotlib`, `seaborn`  
  - ML & Evaluation: `scikit-learn` (`LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`, `GridSearchCV`, metrics)  
  - Stats: `scipy` (for hypothesis testing)  
- **Environment:** Jupyter / Google Colab

## How to Run

1. Clone the repository or download the project folder.
2. Ensure the following files are in the **same directory** as the notebook:
   - `Zomato Restaurant names and Metadata.csv`
   - `Zomato Restaurant reviews.csv`
3. Install required libraries (if needed):

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn scipy
<img width="451" height="690" alt="image" src="https://github.com/user-attachments/assets/5826db44-fc70-4860-9ec7-1d0e60527a92" />
