# Univariate Analysis — EDA Practice

Part of my **Data Learning Hub** series where I practice core data science concepts from scratch.

This repo contains my hands-on practice with **Univariate Analysis** — the first and most important step in any Exploratory Data Analysis (EDA) workflow.

---

## What is Univariate Analysis?

Univariate analysis means analyzing **one variable at a time** before drawing any comparisons or building models. It helps you understand:

- The **shape** of a distribution (normal, skewed, bimodal?)
- The **centre** (mean, median, mode — and which one to trust)
- The **spread** (std deviation, IQR, variance)
- **Outliers** (IQR method, Z-score method)
- Whether the data needs **transformation** before modelling

---

## Datasets Practiced On

| Dataset | Type | Key Learning |
|---|---|---|
| **Titanic** | Mixed (num + cat) | `fare` has extreme right skew (4.79), kurtosis 33.4 — log transform needed |
| **Tips** | Mixed | `total_bill` and `tip` both right-skewed; categorical columns well-balanced |
| **Penguins** | Mixed + missing values | `bill_length_mm` is bimodal — revealed subgroups by species |
| **Diamonds** | Large (54K rows) | `price` extreme skew; log transform reduces skewness from ~1.6 to ~0.2 |

---

## Repo Structure

```
eda-univariate-analysis/
│
├── notebooks/
│   ├── UVA_Titanic.ipynb                   # Titanic dataset — fare, age, sex, pclass
│   └── UnivariateAnalysis_Num_and_cat.ipynb # Tips, Penguins, Diamonds
│
├── scripts/
│   └── univariate_analysis_practice.py     # Reusable functions: univariate_numerical(), univariate_categorical()
│
└── visualizations/
    ├── 01_tips_*.png                        # Tips dataset plots
    ├── 02_penguins_*.png                    # Penguins dataset plots (including bimodal investigation)
    └── 03_diamonds_*.png                    # Diamonds dataset plots + transformation comparison
```

---

## Key Concepts Covered

**Numerical Variables**
- Central tendency: mean, median, mode — and when mean lies
- Spread: standard deviation, variance, IQR, coefficient of variation
- Shape: skewness and kurtosis interpretation
- Outlier detection: IQR method (1.5×IQR fence) and Z-score method (|z| > 3)
- Visualizations: histogram, KDE, box plot, violin plot, ECDF
- Transformations: log1p, sqrt — when and why to use them

**Categorical Variables**
- Frequency tables with counts and proportions
- Cardinality classification (low / medium / high)
- Imbalance detection
- Visualizations: bar chart, donut chart

---

## Key Findings from Practice

- **`fare` (Titanic):** Skewness = 4.79, Kurtosis = 33.4 — one of the most extreme distributions in a beginner dataset. Mean (~32) is more than double the median (~14) due to a small number of very expensive 1st class tickets.
- **`bill_length_mm` (Penguins):** Appears bimodal overall but is actually two normal distributions — Adelie (~38mm) vs Chinstrap/Gentoo (~48mm). A reminder to always investigate bimodal shapes.
- **`price` (Diamonds):** Log transformation reduces skewness from 1.62 → ~0.2, making it suitable for linear models.

---

## Lessons Learned

1. Always do univariate **before** fixing null values — distributions tell you *how* to impute
2. Skewness > 1 means median is more reliable than mean
3. High kurtosis (>10) is a red flag for extreme outliers that will break models
4. Bimodal distributions usually mean hidden subgroups — investigate!
5. `df.corr()` is **bivariate**, not univariate — finish univariate first

---

## Tools Used

- Python 3.11
- pandas, numpy
- matplotlib, seaborn
- Jupyter Notebook

---

*Part of my Data Learning Hub practice series. Next up: Bivariate Analysis.*
