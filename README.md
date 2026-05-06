# 🛒 Machine Learning on Brazilian Olist E-Commerce Data

**DS Machine Learning Techniques | Project**  
**Student:** Sai Pramod Kumar Podila  
**Institution:** CICCC Vancouver — Diploma in Data Science
**Dataset:** [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle)

---

## 📋 Project Overview

This capstone applies four machine learning techniques to a single real-world dataset — 100,000+ orders from the Brazilian Olist e-commerce platform (2016–2018). Each part is independent and uses the Olist dataset as its foundation.

I selected this domain because of my professional background at **Amazon India's LMAQ (Last Mile Analytics & Quality)** team, where I worked on delivery verification data quality across 11 countries. This project extends that expertise into predictive modelling using techniques applicable to Vancouver-based companies in logistics, healthcare, and SaaS.

---

## 🗂️ Repository Structure

```
ds-ml-ecommerce/
├── part1_regression/
│   └── part1_notebook.ipynb      # Predicting delivery days (7 models)
├── part2_classification/
│   └── part2_notebook.ipynb      # Predicting on-time delivery (5 models)
├── part3_timeseries/
│   └── part3_notebook.ipynb      # Forecasting daily orders (ARIMA + RF)
├── part4_clustering/
│   └── part4_notebook.ipynb      # Customer segmentation (3 algorithms)
├── .gitignore                    # data/ folder excluded (too large for GitHub)
└── README.md
```

> **Note:** The raw data CSVs are excluded via `.gitignore`. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and place all 9 CSV files in a `data/` folder at the project root before running any notebook.

---

## 📦 Part 1 — Regression: Predicting Delivery Days

**Goal:** Predict how many days an order takes from purchase to delivery.  
**Target:** `delivery_days` (continuous)  
**Dataset:** 109,842 orders × 7 features

| Model | RMSE | MAE | R² |
|---|---|---|---|
| **Random Forest (Tuned)** ⭐ | **6.898** | **4.683** | **0.294** |
| Random Forest | 7.110 | 4.962 | 0.250 |
| Neural Network | 7.275 | 5.079 | 0.215 |
| Decision Tree | 7.310 | 5.127 | 0.207 |
| Linear Regression | 7.460 | 5.307 | 0.174 |
| Ridge Regression | 7.460 | 5.307 | 0.174 |
| Lasso Regression | 7.464 | 5.314 | 0.173 |
| SVR | 7.522 | 4.947 | 0.160 |

**Key findings:**
- `same_state` (customer vs seller in same Brazilian state) is the dominant feature at **53.4%** importance
- 89.4% of predictions land within ±10 days of actual delivery
- Tuning with GridSearchCV (300 trees, no depth limit, min_samples_split=5) improved RMSE by 0.212 days

---

## 📦 Part 2 — Classification: Predicting On-Time Delivery

**Goal:** Predict whether an order will arrive on or before the estimated date.  
**Target:** `on_time` (binary: 1 = on-time, 0 = late)  
**Dataset:** 110,171 orders × 7 features | Class imbalance: 92.1% on-time / 7.9% late

| Model | F1 | Accuracy | Precision | Recall |
|---|---|---|---|---|
| **Random Forest (Tuned)** ⭐ | **0.883** | **88.5%** | 0.880 | 0.885 |
| Random Forest | 0.707 | 61.9% | 0.874 | 0.619 |
| Neural Network | 0.697 | 60.7% | 0.870 | 0.607 |
| Decision Tree | 0.551 | 44.4% | 0.875 | 0.444 |
| SVC | 0.546 | 44.0% | 0.873 | 0.440 |
| Logistic Regression | 0.494 | 39.1% | 0.876 | 0.391 |

**Key findings:**
- F1-score used as primary metric due to class imbalance (not accuracy)
- `class_weight='balanced'` applied to all models
- Tuning improved F1 by +0.176 — the largest single gain across all four parts
- Best params: 200 trees, no depth limit, min_samples_split=2

---

## 📦 Part 3 — Time Series: Forecasting Daily Orders

**Goal:** Forecast daily order volume using one classical and one ML method.  
**Target:** `daily_orders` (count per day)  
**Dataset:** 655 days (Jan 2017–Oct 2018) | 524 train / 131 test (chronological split)

| Method | RMSE | MAE |
|---|---|---|
| ARIMA(1,1,1) | 130.64 | 109.68 |
| **RF + Lag Features** ⭐ | **55.77** | **38.02** |

**Key findings:**
- ADF test: original series p=0.0645 (non-stationary) → first differencing → p=0.0000 (stationary), d=1
- ARIMA(1,1,1) selected via manual grid search (AIC=5658.71) across 12 (p,d,q) combinations
- 8 lag features engineered: lag_1, lag_7, lag_14, rolling_mean_7, rolling_mean_14, rolling_std_7, day_of_week, month
- RF lag features: **57.3% better RMSE** and **65.3% better MAE** than ARIMA
- Top features: rolling_mean_7 (43%), lag_1 (35.4%), lag_7 (7.9%)

---

## 📦 Part 4 — Clustering: Customer Segmentation

**Goal:** Discover natural customer groups using unsupervised algorithms.  
**Dataset:** 93,058 unique customers × 5 RFM-style features  
**Features:** total_orders, avg_order_value, total_freight, avg_delivery_days, unique_categories

| Algorithm | Clusters | Silhouette | Notes |
|---|---|---|---|
| K-Means (k=3) | 3 | 0.484 | Most business-actionable |
| DBSCAN (eps=0.5, min_samples=20) | 5 | **0.704** | 2,332 noise points (2.5%) |
| Agglomerative (Ward) | 3 | 0.676 | Balanced: 93.9% / 3.0% / 3.1% |

**K-Means customer segments:**

| Segment | Size | Avg Order Value | Avg Delivery Days |
|---|---|---|---|
| 💰 Budget Buyers | 71,585 (76.9%) | R$93 | 8.9 days |
| ⭐ Premium Buyers | 18,662 (20.1%) | R$254 | 23.2 days |
| 🛒 Mid-Tier Buyers | 2,811 (3.0%) | R$104 | 11.7 days |

**Key findings:**
- K-Means recommended for business use despite lower silhouette — most interpretable segments
- Agglomerative complete linkage degenerated (99.7% in one cluster) → Ward linkage selected
- PCA used for 2D visualization only — clustering performed on full scaled feature set
- 20K customer sample used for hyperparameter search (32s vs 22min full dataset)

---

## ⚙️ Setup & Reproduction

**Requirements:**

```bash
conda create -n capstone python=3.11 -y
conda activate capstone
pip install pandas numpy scikit-learn matplotlib seaborn jupyter ipykernel tensorflow statsmodels pmdarima
python -m ipykernel install --user --name capstone --display-name "Python (capstone)"
```

**Dataset setup:**

1. Download from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
2. Extract all 9 CSVs into a `data/` folder at the project root
3. Open any notebook and select the `Python (capstone)` kernel
4. Run all cells top to bottom (Kernel → Restart & Run All)

**Required CSVs:**
```
data/
├── olist_orders_dataset.csv
├── olist_order_items_dataset.csv
├── olist_customers_dataset.csv
├── olist_sellers_dataset.csv
├── olist_products_dataset.csv
├── olist_order_payments_dataset.csv
├── olist_order_reviews_dataset.csv
├── olist_geolocation_dataset.csv
└── product_category_name_translation.csv
```

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11.15 | Core language |
| pandas | 3.0.2 | Data manipulation |
| scikit-learn | 1.8.0 | ML models, pipelines, metrics |
| TensorFlow/Keras | 2.21.0 | Neural networks |
| statsmodels | 0.14.6 | ARIMA time series |
| matplotlib / seaborn | 3.10.9 / 0.13.2 | Visualizations |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.