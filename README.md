# Titanic Dataset – Exploratory Data Analysis, Data Cleaning and Machine Learning

## Project Overview

This project focuses on exploring, cleaning, preparing, and building machine learning models using the Titanic dataset with Python and its data science libraries. The objective is to understand the dataset, identify and handle missing values, analyze passenger survival patterns, engineer new features, create meaningful visualizations, and develop machine learning models to predict passenger survival.

This project is completed as part of the **SkillMe Data Science Virtual Internship – Weeks 1, 2, and 3** under the project **InsightHub: Interactive Data Intelligence & ML Dashboard**.

## Objectives

- Load and explore the Titanic dataset.
- Understand dataset columns and data types.
- Identify and handle missing values.
- Analyze passenger survival rates based on passenger class and gender.
- Create visualizations to understand data patterns.
- Perform feature engineering to create new meaningful columns.
- Convert categorical variables into numerical values.
- Detect and analyze outliers.
- Train and evaluate machine learning classification models.
- Compare model performance using evaluation metrics.
- Analyze feature importance.
- Save and reload the best-performing model.

## Technologies and Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Google Colab

## Dataset Information

The Titanic dataset contains information about passengers aboard the Titanic, including their age, gender, passenger class, fare, family details, and survival status.

- Total records: 891
- Data source: Seaborn Titanic dataset

---

## Week 1: Exploratory Data Analysis (EDA)

The following operations were performed:

1. Loaded the Titanic dataset using Seaborn.
2. Inspected the dataset using `head()`, `info()`, and `describe()`.
3. Checked missing values using `isnull().sum()`.
4. Analyzed survival rates using `groupby()`.
5. Analyzed survival rates based on passenger class and gender.
6. Created a bar chart of survival rates by passenger class.
7. Created a histogram of passenger age distribution.
8. Generated a correlation heatmap to understand relationships between numerical variables.

### Week 1 Key Findings

- Total passengers analyzed: 891.
- First-class passengers had a survival rate of 62.96%.
- Female passengers had a survival rate of 74.20%.
- Missing values were identified in the dataset for further cleaning.

### Week 1 Visualizations

- Survival Rate by Passenger Class.
- Distribution of Passenger Ages.
- Correlation Heatmap of Titanic Dataset.

---

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

---

## Week 3: Train a Prediction Model with Scikit-learn

The third week focused on developing machine learning models using the cleaned Titanic dataset to predict passenger survival.

### Data Preparation

- Loaded the cleaned dataset from `data/titanic_clean.csv`.
- Selected 11 relevant input features, including `pclass`, `age`, `fare`, `family_size`, `is_alone`, and encoded categorical columns.
- Selected `survived` as the target variable.
- Split the dataset into training and testing sets using `train_test_split()`.
- Used an 80:20 training and testing ratio with `random_state=42`.

### Model Training

Trained two machine learning classification models:

**1. Logistic Regression**

- Created and trained a Logistic Regression classifier.
- Generated predictions on the test dataset.
- Evaluated the model using accuracy, precision, recall, and a confusion matrix.

**2. Random Forest Classifier**

- Created and trained a Random Forest classifier with 100 estimators.
- Generated predictions on the same test dataset.
- Evaluated the model using accuracy, precision, recall, and a confusion matrix.

### Model Performance Comparison

Both models were compared using accuracy, precision, and recall.

| Evaluation Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 79.89% | **82.68%** |
| Precision | 77.14% | **80.28%** |
| Recall | 72.97% | **77.03%** |

### Feature Importance Analysis

- Extracted feature importance scores using `feature_importances_` from the trained Random Forest model.
- Sorted the features based on their importance scores.
- Created a bar chart to visualize the contribution of different features to the model's predictions.

### Model Saving and Reloading

- Selected Random Forest as the best-performing model based on the evaluation results.
- Saved the trained model using `joblib.dump()`.
- Reloaded the saved model using `joblib.load()`.
- Generated predictions using the reloaded model.
- Verified that the reloaded model achieved an accuracy of 82.68% on the test dataset.

### Week 3 Conclusion

Random Forest performed better than Logistic Regression across all three evaluation metrics. It achieved 82.68% accuracy, 80.28% precision, and 77.03% recall, compared to Logistic Regression's 79.89% accuracy, 77.14% precision, and 72.97% recall.

Therefore, Random Forest was selected as the best-performing model and saved for future use.

---

## Project Structure

```text
Titanic-Project/
│
├── data/
│   └── titanic_clean.csv
│
├── README.md
├── Titanic.ipynb
│
├── fare_outlier_detection.png
├── logistic_regression_confusion_matrix.png
├── missing_values_after_cleaning.png
├── passenger_age_distribution.png
├── random_forest_confusion_matrix.png
├── random_forest_feature_importance_barplot.png
├── survival_analysis_after_cleaning.png
├── survival_rate_by_class.png
├── survival_rate_by_passenger_class.png
├── titanic_categorical_encoding.png
├── titanic_correlation_heatmap.png
├── titanic_dataset_first_five_rows.png
└── titanic_feature_engineering.png
```

## How to Run the Project

1. Download the `Titanic.ipynb` notebook from this repository.
2. Open [Google Colab](https://colab.research.google.com/).
3. Select File → Upload notebook.
4. Upload the downloaded notebook.
5. Make sure the cleaned dataset is available at `data/titanic_clean.csv`.
6. Select Runtime → Run all.
7. Explore the outputs, analysis, visualizations, model evaluations, and predictions.

To use the saved model separately, load it using Joblib:

```python
import joblib

model = joblib.load('best_titanic_model.pkl')
```

## Overall Conclusion

This project provided practical experience in the complete data science workflow, from exploratory data analysis and data cleaning to feature engineering and machine learning model development.

Through Weeks 1, 2, and 3, I gained hands-on experience with Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, and Joblib. I also learned how to train and evaluate classification models, compare their performance, interpret feature importance, and save trained models for future use.

The Random Forest Classifier achieved the best performance with 82.68% accuracy on the test dataset, making it the selected model for Titanic passenger survival prediction.

This project strengthened my understanding of data analysis, data preprocessing, machine learning, and model evaluation using Python.
