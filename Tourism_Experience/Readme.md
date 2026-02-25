# Tourism Experience Analytics  
## Classification, Prediction & Recommendation System (Python • SQL/DuckDB • ML • Streamlit)

**Author:** Shubhra Das  
**Date:** 25 Feb 2026  
**Domain:** Tourism Analytics  

> An end-to-end analytics + ML project that transforms multi-table tourism data into three decision engines:  
> **(1) Rating Prediction (Regression)**, **(2) Visit Mode Prediction (Classification)**, and **(3) Personalized Attraction Recommendations (Hybrid Recommender)**.  
> Includes data integration, feature engineering, EDA (U-B-M rule), model evaluation, SQL analytics using DuckDB, and deployment using Streamlit.

---

## ✨ Skills Demonstrated
- Data Cleaning & Preprocessing  
- Data Integration (multi-table joins)  
- Exploratory Data Analysis & Visualization (U-B-M rule)  
- SQL Analytics using **DuckDB**  
- Machine Learning: **Regression**, **Classification**, **Recommendation System**  
- Model packaging & reproducible inference (**joblib**)  
- Streamlit app deployment

---

## 📌 Business Problem
Tourism platforms generate rich interaction signals (who visited which attraction, when, in what mode, and how they rated the experience).  
This project enhances user experience by:

- Predicting **satisfaction rating (1–5)**
- Classifying **Visit Mode** *(Business, Couples, Family, Friends, Solo)*
- Recommending **Top-N attractions** using a **hybrid recommender**

---

## ✅ Solution Overview
### 1) Rating Prediction (Regression)
Predicts user satisfaction rating (1–5) for a user-attraction context.

**Best model:** GradientBoostingRegressor  
**Performance (test set):**
- **MSE:** 0.236257 *(noted as RMSE label in notebook output)*
- **MAE:** 0.264200  
- **R²:** 0.749148  

**Saved artifact:** `artifacts/rating_regression_model.joblib`

---

### 2) Visit Mode Prediction (Classification)
Predicts VisitMode using contextual and user/attraction features (excluding Rating and VisitMode to reduce leakage).

**Best model:** RandomForestClassifier  
**Performance (test set):**
- **Accuracy:** 0.523238  
- **Macro F1:** 0.423752  

**Saved artifact:** `artifacts/visitmode_classifier.joblib`

---

### 3) Recommendation System (Hybrid)
A hybrid recommender combining:
- **Collaborative Filtering:** user-item rating matrix + cosine similarity (item-item)
- **Content-Based Similarity:** one-hot encoding of attraction attributes + cosine similarity
- **Hybrid Scoring:** weighted blending of normalized collaborative scores and content boosts

**Saved artifacts:**
- `artifacts/user_item_matrix.joblib`
- `artifacts/item_similarity_collab.joblib`
- `artifacts/items_features.joblib`
- `artifacts/items_content_encoder.joblib`
- `artifacts/item_similarity_content.joblib`

---

## 🧾 Dataset Description
The dataset is provided as multiple tables (transactions + dimension/lookup tables). Excel files were converted to CSV for processing.

### Tables and Size
| Table | Rows | Description |
|------|------|-------------|
| Transaction | 52,930 | User visits with ratings + context |
| User | 33,530 | User geo attributes (continent/region/country/city) |
| City | 9,143 | City names + country mapping |
| Item (Attractions) | 30 | Attraction details + addresses |
| Type | 17 | Attraction type lookup |
| Mode | 6 | VisitMode lookup |
| Country | 165 | Country lookup with region mapping |
| Region | 22 | Region lookup with continent mapping |
| Continent | 6 | Continent lookup |

### Core Columns Used
- **Transaction:** `TransactionId, UserId, VisitYear, VisitMonth, VisitMode, AttractionId, Rating`
- **User:** `UserId, ContinentId, RegionId, CountryId, CityId`
- **Item:** `AttractionId, AttractionCityId, AttractionTypeId, Attraction, AttractionAddress`

---

## 🔧 Data Pipeline
### Data Cleaning
- Converted `.xlsx` tables to `.csv`
- Standardized column names
- Type conversions for keys and rating
- Rating clipped to **[1, 5]**
- Dropped rows missing essential identifiers (`UserId`, `AttractionId`)

### Data Integration (JOINs)
- `transactions ⟕ users` on `UserId`
- `... ⟕ items` on `AttractionId`
- Cities joined twice for:
  - `UserCityName`
  - `AttractionCityName`
- Joined `types` for `AttractionType`
- Joined `visit_modes` to map VisitModeId → VisitMode label

**Consolidated dataset:** **52,930 rows × 18 columns**

### Feature Engineering
- Missing values: categorical → `"Unknown"`; numeric → median (excluding Rating)
- User-level aggregates:
  - `user_rating_count, user_rating_mean, user_rating_std`
- Attraction-level aggregates:
  - `attraction_rating_count, attraction_rating_mean`

**After feature engineering:** **52,930 rows × 23 columns**  
Saved as: `data/master_cleaned.csv`

---

## 📊 EDA Highlights (U-B-M Rule)
- **Univariate:** rating distribution, visits by month, top attractions, top countries  
- **Bivariate:** avg rating by VisitMode, rating distribution by AttractionType, yearly trends  
- **Multivariate:** VisitMode × AttractionType heatmap, VisitMode mix by month, bubble chart

*(Add screenshots in `/images` and link them here if you upload them.)*

---

## 🧠 SQL Analytics (DuckDB)
DuckDB enables running SQL directly on the consolidated dataframe for fast, reproducible analysis.

Examples included:
- Top attractions by average rating and count
- VisitMode distribution
- Seasonality patterns (visits by month, visit mode mix)
- Region/country market distribution

---

## 🖥️ Streamlit App
The Streamlit app demonstrates end-to-end usage:
- Predict **VisitMode**
- Predict **Rating**
- Generate **Top-N hybrid recommendations**
- Uses **saved joblib artifacts** for repeatable inference

### Run locally
1. Ensure these are present in your project folder:
   - `app.py`
   - `data/master_cleaned.csv`
   - `artifacts/` (joblib files)
2. Create/activate a Python environment  
3. Install dependencies:

```bash
pip install -r requirements.txt
