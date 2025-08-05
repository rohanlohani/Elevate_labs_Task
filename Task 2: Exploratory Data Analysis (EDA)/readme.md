Titanic Dataset - Exploratory Data Analysis (EDA)
Overview
This project performs Exploratory Data Analysis (EDA) on the Titanic dataset to uncover patterns, trends, and relationships in the data. Using tools like Pandas, Matplotlib, and Seaborn, the analysis provides key statistical summaries and insightful visualizations that set the foundation for future machine learning or analytical tasks.

Dataset
Source: Titanic dataset (tested.csv)

Columns include:
PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked

Main Analysis Steps
Summary Statistics:
Calculated mean, median, min, max, and standard deviation for numerical features to understand central tendencies and variances.

Histograms:
Plotted for all numerical columns (Age, Fare, SibSp, Parch, etc.) to visualize distribution, skewness, and outliers.

Boxplots:
Used to highlight medians, interquartile ranges, and outliers for each numeric variable, making it easy to spot unusual data points.

Correlation Matrix & Heatmap:
Computed correlations to detect linear relationships, especially between features and the target (Survived). Visualized with a heatmap for easier interpretation.

Pairplot:
Created scatter and density plots for pairs of key features (Age, Fare, Pclass, Survived), colored by survival outcome, to assess feature interactions and class separation.

Key Insights
Most passengers were in third class and traveled solo.

Young adults (20-40) were the largest age group aboard.

Majority paid low fares, but first-class paid much more and had higher survival rates.

Survival rate overall was below 50%.

Some features are skewed (Fare, family sizes), and outliers are present.

Running the Analysis
Place the tested.csv file in your working directory or update the path in the script.

Run the code using:

bash
python titanic.py
The script will output statistics to the console and display/save visualizations.

Requirements
Python 3.x

pandas

matplotlib

seaborn

Install requirements (if needed) with:

bash
pip install pandas matplotlib seaborn
Visualization Outputs
Histograms: Show distributions for each numeric feature.

Boxplots: Display medians, ranges, and outliers.

Pairplot: Visualizes relationships and survival clusters.

Correlation Heatmap: Easily identifies feature relationships.

This EDA equips you with knowledge about data quality, distributions, and preliminary relationships—vital for feature engineering or machine learning!
