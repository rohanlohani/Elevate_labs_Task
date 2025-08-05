Titanic Dataset - Exploratory Data Analysis (EDA)
📌 Objective
Perform Exploratory Data Analysis (EDA) on the Titanic dataset to understand its structure, distributions, and feature relationships using descriptive statistics and visualizations.

🛠️ Tools Used
Python 3

Pandas

Matplotlib

Seaborn

🔍 Steps Performed
1. Generated Summary Statistics
Used df.describe() to compute:

Mean, median, standard deviation

Minimum and maximum values

Quartiles for all numeric columns (Age, Fare, SibSp, Parch)

2. Created Histograms and Boxplots
Histograms: Visualized the distribution of each numeric feature.

Boxplots: Identified potential outliers and spread of data.

3. Correlation Matrix and Pairplot
Computed correlation matrix to find linear relationships between variables.

Plotted:

Heatmap using Seaborn for quick correlation interpretation.

Pairplot to visualize distributions and feature relationships, colored by survival status.

4. Analyzed Patterns, Trends, and Anomalies
Observed skewed distributions in Fare and Age.

Detected class imbalance in passenger classes and survival status.

Identified that younger, higher-class passengers had better survival rates.

5. Made Feature-Level Inferences
Survival Likelihood:

Higher among first-class passengers and those who paid higher fares.

Demographics:

Most passengers were young adults in third class, often traveling alone.

Family Size:

SibSp and Parch are heavily skewed toward zero, indicating most passengers traveled solo.

📊 Screenshots
The following visualizations are included in the screenshots/ folder:

Histograms for numeric features

Boxplots for numeric features

Correlation heatmap

Pairplot (scatter matrix) showing survival relationships

📁 Repository Structure
bash
Copy
Edit
.
├── titanic.py              # Python script for performing EDA
├── tested.csv              # Titanic dataset file
├── screenshots/            # Folder containing generated plots
└── README.md               # Project documentation (this file)
▶️ How To Run
Install dependencies (if not already installed):

bash
Copy
Edit
pip install pandas matplotlib seaborn
Run the script:

bash
Copy
Edit
python titanic.py
View Outputs:

Plots will be displayed or saved (as configured).

Screenshots are available in the screenshots/ folder.

📚 What You'll Learn
How to apply descriptive statistics for data summarization.

Visual techniques to detect:

Skewness

Outliers

Feature correlations

The importance of visual EDA in building machine learning pipelines.

📦 Submission
All the following are included in the repository:

Python code (titanic.py)

Dataset (tested.csv)

Visual screenshots (screenshots/)

Documentation (README.md)

Refer to the screenshots/ folder for key visuals and insights derived from the data.
