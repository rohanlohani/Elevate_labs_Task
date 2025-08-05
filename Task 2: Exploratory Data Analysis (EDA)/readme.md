Titanic Dataset - Exploratory Data Analysis (EDA)
Objective
Perform EDA (Exploratory Data Analysis) on the Titanic dataset to understand its structure, distributions, and feature relationships using statistics and visualizations.

Tools Used
Python

Pandas

Matplotlib

Seaborn

Steps Performed
Generated Summary Statistics:
Used describe() to view mean, median, standard deviation, etc., for all numeric columns (Age, Fare, SibSp, Parch).

Created Histograms and Boxplots:
Visualized distributions and potential outliers for each numeric feature using histograms and boxplots.

Correlation Matrix and Pairplot:

Calculated correlation matrix to identify linear relationships between features.

Plotted a heatmap for visual insight into correlations.

Used pairplot to examine feature relationships and survival outcome visually.

Analyzed Patterns, Trends, and Anomalies:

Noted skewed distributions, outliers in fares and ages, class imbalances, and patterns in survival by class and fare.

Made Feature-Level Inferences:

Higher-class and higher-fare passengers tended to have better survival rates.

Most passengers were young adults, third-class, and typically traveled alone.

Family size features (SibSp, Parch) are heavily skewed toward zero.

Screenshots
Visualizations included in this repo:

Histograms for all numeric columns

Boxplots for all numeric columns

Pairplot (scatter matrix) for selected features grouped by survival

Correlation heatmap

Structure
text
.
├── titanic.py              # Python script for analysis
├── tested.csv              # Titanic dataset
├── screenshots/            # Folder with plot images (histograms, boxplots, etc.)
└── README.md
How To Run
Make sure you have Python 3, Pandas, and Seaborn/Matplotlib installed.
Install with:

text
pip install pandas matplotlib seaborn
Place tested.csv and titanic.py in the same folder.

Run:

text
python titanic.py
View the output plots/screenshots and review the inferences.

What You'll Learn
Data visualization and descriptive statistics for understanding datasets.

How to spot patterns, trends, outliers, skewness, and correlations visually.

How feature-level visual EDA supports the entire ML workflow.

Submission
All code, dataset, screenshots, and this README are included to document and justify findings.

Please refer to the provided screenshots for key visual insight as required.
