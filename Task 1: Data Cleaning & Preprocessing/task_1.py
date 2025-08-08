import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
data = pd.read_csv('tested.csv')

# Handle missing values for Age, Fare, and Embarked columns
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Fare'].fillna(data['Fare'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Drop the Cabin column (too many missing values)
data.drop('Cabin', axis=1, inplace=True, errors='ignore')

# Convert categorical variables into numeric
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# Show first few rows and info about data
print(data.head())
print(data.info())
print(data.isnull().sum())

# Standardize 'Age' and 'Fare' columns
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data[['Age', 'Fare']] = scaler.fit_transform(data[['Age', 'Fare']])

# Plot boxplots to visualize outliers in Age and Fare
sns.boxplot(x=data['Age'])
plt.show()

sns.boxplot(x=data['Fare'])
plt.show()

# Remove outliers in Fare (if any above a chosen threshold)
data = data[data['Fare'] < 300]

# Drop non-feature columns and separate input features and label
X = data.drop(['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1)
y = data['Survived']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
