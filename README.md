# Titanic Dataset – Exploratory Data Analysis and Data Cleaning

## Project Overview

This project focuses on exploring, cleaning, and preparing the Titanic dataset using Python and its data analysis libraries. The objective is to understand the dataset, identify and handle missing values, analyze passenger survival patterns, engineer new features, and create meaningful visualizations.

This project is completed as part of the **SkillMe Data Science Virtual Internship – Week 1 and Week 2** under the project **InsightHub: Interactive Data Intelligence & ML Dashboard**.

## Objectives

- Load and explore the Titanic dataset.
- Understand dataset columns and data types.
- Identify and handle missing values.
- Analyze passenger survival rates based on passenger class and gender.
- Create visualizations to understand data patterns.
- Perform feature engineering to create new meaningful columns.
- Convert categorical variables into numerical values.
- Detect and analyze outliers.
- Save the cleaned dataset for future analysis and modeling.

## Technologies and Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Google Colab

## Dataset Information

The Titanic dataset contains information about passengers aboard the Titanic, including their age, gender, passenger class, fare, family details, and survival status.

- Total records: 891
- Data source: Seaborn Titanic dataset

## Week 1: Exploratory Data Analysis (EDA)

The following operations were performed:

1. Loaded the Titanic dataset using Seaborn.
2. Inspected the dataset using `head()`, `info()`, and `describe()`.
3. Checked missing values using `isnull().sum()`.
4. Analyzed survival rates using `groupby()`.
5. Created a bar chart of survival rates by passenger class.
6. Created a histogram of passenger age distribution.

### Week 1 Key Findings

- Total passengers analyzed: 891.
- First-class passengers had a survival rate of 62.96%.
- Female passengers had a survival rate of 74.20%.
- Missing values were identified in the dataset for further cleaning.

### Week 1 Visualizations

- Survival Rate by Passenger Class.
- Distribution of Passenger Ages.
- Additional exploratory data visualizations.

## Week 2: Data Cleaning and Feature Engineering

The second week focused on cleaning the dataset and preparing it for further analysis and machine learning.

### Data Cleaning

- Filled missing age values using the median.
- Filled missing embarked values using the mode.
- Removed the `deck` column because it contained a large number of missing values.
- Removed the redundant `embark_town` column.
- Created a reusable `clean_data(df)` function to preserve the original dataset.

### Feature Engineering

Created two new features:

- `family_size`: Calculated using `sibsp + parch + 1`.
- `is_alone`: Identifies whether a passenger was travelling alone.

### Categorical Encoding

Used `pandas.get_dummies()` to convert categorical variables into numerical columns:

- `sex_female`
- `sex_male`
- `embarked_C`
- `embarked_Q`
- `embarked_S`

### Outlier Detection

- Created a boxplot to identify outliers in the `fare` column.
- Observed several unusually high fare values.
- Retained the outliers because they may represent genuine passenger ticket fares.

### Cleaned Dataset

The cleaned dataset was saved as:

`data/titanic_clean.csv`

It contains the cleaned values, engineered features, and encoded categorical columns for future analysis and modeling.

## Project Structure

```text
Titanic-Project/
│
├── data/
│   └── titanic_clean.csv
│
├── Titanic.ipynb
├── README.md
├── Output1.jpeg
├── Output2.png
├── Output3.png
└── Output4.png
```

## How to Run the Project

1. Download the `Titanic.ipynb` notebook from this repository.
2. Open [Google Colab](https://colab.research.google.com/).
3. Select File → Upload notebook.
4. Upload the downloaded notebook.
5. Select Runtime → Run all.
6. Explore the outputs, analysis, and visualizations.

## Conclusion

This project helped develop practical skills in data exploration, data cleaning, feature engineering, missing value handling, categorical encoding, outlier detection, and data visualization using Python.

Through Week 1 and Week 2, I gained hands-on experience in transforming raw data into a cleaned and structured dataset, providing a foundation for future data analysis and machine learning tasks.
