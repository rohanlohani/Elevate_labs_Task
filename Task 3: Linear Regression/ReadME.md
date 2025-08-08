
# 🏠 Task 3 - Linear Regression on Housing Dataset

This project implements simple and multiple linear regression using the Boston Housing dataset.

## 📊 Dataset

The dataset contains 13 independent variables related to housing and 1 target variable `MEDV` (Median value of owner-occupied homes in $1000's).

**Source**: [Boston Housing Dataset](https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction)

## 🔧 Tools Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## 📌 Task Objectives

- Import and preprocess dataset
- Perform train-test split
- Fit Linear Regression model using `sklearn`
- Evaluate with MAE, MSE, RMSE, R² score
- Plot actual vs predicted prices
- Interpret coefficients

## 📈 Evaluation Metrics

- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of Determination

## 🧪 How to Run

```bash
pip install pandas scikit-learn matplotlib seaborn
python linear_regression.py
```

## 📷 Output Example

### 📊 Actual vs Predicted Plot

This scatter plot compares the actual house prices (from the test set) with the prices predicted by the linear regression model.  
Ideally, the points should lie close to the diagonal (where actual = predicted), which would indicate perfect predictions.  
In this plot, most points follow a linear trend, suggesting that the model performs reasonably well.  
However, there are some outliers and deviations, especially at the lower and higher ends of the price range, indicating underfitting or unmodeled complexity.  
This visualization helps in identifying model accuracy and where predictions may need improvement.

## 🧠 What I Learned

- Regression modeling with scikit-learn
- Evaluating model performance using multiple metrics
- Visualizing model predictions
- Interpreting linear regression coefficients

