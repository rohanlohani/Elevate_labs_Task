# Task 1: Data Cleaning & Preprocessing

## Overview
This project demonstrates the essential steps involved in cleaning and preprocessing the Titanic dataset to prepare it for machine learning tasks.

## Contents
- `titanic.py`: Python script performing data cleaning, handling missing values, encoding categorical features, normalization, outlier visualization, and data splitting.
- `tested.csv`: The Titanic dataset used for the task.

## Data Cleaning and Preprocessing Steps
1. **Handling Missing Values:**  
   - Filled missing values in `Age` and `Fare` using median.  
   - Filled missing values in `Embarked` using mode.  
   - Dropped the `Cabin` column due to a large number of missing values.

2. **Encoding Categorical Variables:**  
   - Converted `Sex` and `Embarked` into numerical form using one-hot encoding.

3. **Normalization:**  
   - Standardized the `Age` and `Fare` columns to have zero mean and unit variance using `StandardScaler`.

4. **Outlier Detection:**  
   - Visualized `Age` and `Fare` using boxplots.  
   - Removed extreme outliers in `Fare` based on visualization.

5. **Data Preparation:**  
   - Split the data into input features and target variable.  
   - Created training and testing sets for model development.

## How to Run
1. Ensure all required Python libraries are installed:

pip install pandas numpy matplotlib seaborn scikit-learn

2. Run the script:  titanic.py

## Notes
- This notebook/script covers foundational data preprocessing techniques necessary before applying machine learning models.
- Model training and evaluation are not included in this task.

## Outlier Detection and Visualization

### Fare Boxplot

![Fare Boxplot](images/boxplotforfare.png)

The boxplot shows that most fares are clustered near the median with some outliers at the higher end.

### Age Boxplot

![Age Boxplot](images/boxplotforage.png)

The age distribution is centered but has some outliers representing very young and elderly passengers.
