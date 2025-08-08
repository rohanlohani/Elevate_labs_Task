
# Task 2: Titanic Dataset - Exploratory Data Analysis (EDA)

## Overview
This project demonstrates Exploratory Data Analysis (EDA) on the Titanic dataset to understand its structure, feature distributions, and survival-related patterns using statistical and visual techniques.

---

## Contents

- `titanic.py` — Python script performing the EDA (summary stats, plots, inferences)
- `tested.csv` — The Titanic dataset used for the analysis
- `screenshots/` — Folder containing generated visualizations
- `README.md` — This documentation file

---

## EDA Steps Performed

### 1. Generated Summary Statistics
Used `describe()` to calculate:
- Mean, median, standard deviation
- Quartiles and range

**Columns analyzed**: `Age`, `Fare`, `SibSp`, `Parch`

### 2. Created Histograms and Boxplots
- **Histograms**: Visualized distributions of numeric columns
- **Boxplots**: Identified potential outliers and spread of data

### 3. Correlation Matrix and Pairplot
- Created correlation matrix to assess linear relationships
- Plotted:
  - Heatmap using Seaborn
  - Pairplot grouped by survival status to explore feature interactions

### 4. Analyzed Patterns, Trends, and Anomalies
Observed:
- Skewness in `Fare` and `Age`
- Class imbalance in `Pclass` and `Survived`
- Clear survival advantages for higher fare and first-class passengers

### 5. Made Feature-Level Inferences
- First-class and higher-fare passengers had higher survival rates
- Most passengers were young adults traveling alone in third class
- `SibSp` and `Parch` were mostly 0 — indicating solo travelers

---

## Screenshots

The following visualizations are included in the `screenshots/` folder:
- Histograms for all numeric columns
- Boxplots for all numeric columns
- Correlation heatmap
- Pairplot (scatter matrix) grouped by survival

---

## How to Run

### 1. Install Required Libraries
```bash
pip install pandas matplotlib seaborn
```

### 2. Run the Script
Ensure `titanic.py` and `tested.csv` are in the same directory, then run:
```bash
python titanic.py
```

### 3. View Results
- Visualizations will be displayed or saved (depending on script configuration)
- Screenshots will appear in the `screenshots/` folder

---

## What i Learn

- How to use descriptive statistics and visualization for data understanding
- How to detect:
  - Patterns and trends
  - Skewness
  - Outliers
  - Correlations between features
- Why EDA is essential before model development

---

## Submission Includes

- `titanic.py` — EDA code
- `tested.csv` — Dataset
- `screenshots/` — Output visuals
- `README.md` — Documentation

Please refer to the provided screenshots for visual insights that support the inferences.
