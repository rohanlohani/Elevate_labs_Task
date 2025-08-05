Task 2: Exploratory Data Analysis (EDA)
Overview
This project showcases Exploratory Data Analysis (EDA) on the Titanic dataset to understand its structure, feature distributions, relationships, and survival patterns using visual and statistical techniques.

Contents
titanic.py: Python script that performs all EDA steps including statistical summary, plotting histograms, boxplots, correlation heatmap, and pairplot.

tested.csv: The Titanic dataset used for analysis.

screenshots/: Folder containing visualizations generated during EDA.

Exploratory Data Analysis Steps
Summary Statistics

Used describe() to analyze key statistics (mean, std, min, max, quartiles) for Age, Fare, SibSp, and Parch.

Distribution Visualization

Histograms created for all numeric columns to examine the data distribution.

Boxplots used to detect potential outliers and spread of values.

Correlation Analysis

Generated a correlation matrix to examine linear relationships among numeric features.

Created a heatmap using Seaborn to visually highlight high and low correlations.

Developed a pairplot colored by survival to show inter-feature relationships and survival patterns.

Pattern & Anomaly Detection

Observed skewness in Fare and Age, and class imbalance in Pclass and Survived.

Identified strong associations between higher survival and higher Fare, as well as lower passenger class.

Key Feature-Level Inferences

Survival Rates: Higher among first-class passengers and those who paid more.

Demographics: Majority were young adults traveling alone in third class.

Family Features: SibSp and Parch were mostly 0, indicating solo travelers.

How to Run
Install required Python libraries:

bash
Copy
Edit
pip install pandas matplotlib seaborn
Run the analysis script:

bash
Copy
Edit
python titanic.py
Review the output and plots saved in the screenshots/ folder.

Notes
This script provides deep visual and statistical insights into the Titanic dataset.

It serves as a foundation for further feature engineering and machine learning model development.

No model training is performed in this task — focus is solely on understanding the data.

Output Visualizations
Histograms
Histograms show that:

Fare is right-skewed with many low values.

Age distribution is bell-shaped but incomplete due to missing data.

Boxplots
Boxplots helped detect:

Fare outliers: Some passengers paid exceptionally high fares.

Age outliers: A few very young and old passengers.

Correlation Heatmap
Revealed positive correlation between Pclass and Fare, and negative correlation between Pclass and Survived.

Pairplot
Illustrated how features like Fare, Pclass, and Sex relate to survival visually and clearly.
