# Titanic Dataset – Exploratory Data Analysis, Data Cleaning, Machine Learning and Interactive Dashboard

## Project Overview

This project focuses on exploring, cleaning, preparing, visualizing, and building machine learning models using the Titanic dataset with Python and its data science libraries.

The objective is to understand the dataset, identify and handle missing values, analyze passenger survival patterns, engineer new features, create meaningful visualizations, and develop machine learning models to predict passenger survival.

An interactive Streamlit dashboard was also developed to display data insights, visualize passenger information, and generate passenger survival predictions using the trained Random Forest model.

This project is completed as part of the **SkillMe Data Science Virtual Internship – Weeks 1, 2, 3, and 4** under the project **InsightHub: Interactive Data Intelligence & ML Dashboard**.

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
- Develop an interactive dashboard using Streamlit.
- Display dataset statistics and visualizations.
- Generate passenger survival predictions using the trained model.
- Test the prediction system using different passenger profiles.

## Technologies and Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Google Colab
- Visual Studio Code

## Dataset Information

The Titanic dataset contains information about passengers aboard the Titanic, including their age, gender, passenger class, fare, family details, and survival status.

- Total records: 891
- Data source: Seaborn Titanic dataset
- Target variable: `survived`

---

# Week 1: Exploratory Data Analysis (EDA)

The first week focused on understanding the Titanic dataset and identifying important patterns and relationships.

### Activities Performed

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
- Male passengers had a survival rate of 18.89%.
- Missing values were identified in the dataset for further cleaning.

### Week 1 Visualizations

- Survival Rate by Passenger Class.
- Distribution of Passenger Ages.
- Correlation Heatmap of Titanic Dataset.

---

# Week 2: Data Cleaning and Feature Engineering

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

# Week 3: Train a Prediction Model with Scikit-learn

The third week focused on developing machine learning models using the cleaned Titanic dataset to predict passenger survival.

### Data Preparation

- Loaded the cleaned dataset from `data/titanic_clean.csv`.
- Selected 11 relevant input features, including `pclass`, `age`, `fare`, `family_size`, `is_alone`, and encoded categorical columns.
- Selected `survived` as the target variable.
- Split the dataset into training and testing sets using `train_test_split()`.
- Used an 80:20 training and testing ratio with `random_state=42`.
- Training records: 712.
- Testing records: 179.

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

# Week 4: Build an Interactive Streamlit Dashboard

The fourth week focused on developing an interactive web-based dashboard using Streamlit and integrating the trained Random Forest model for passenger survival prediction.

### Dashboard Development

Developed an interactive dashboard using Python and Streamlit to present Titanic dataset insights and machine learning predictions.

### Dashboard Features

**1. Model Information**

- Displays the selected machine learning model.
- Shows the Random Forest model's test accuracy of 82.68%.
- Provides information about the selected model.

**2. Titanic Data Analysis**

Displays important dataset statistics:

- Total Passengers: 891.
- Survived Passengers: 342.
- Overall Survival Rate: 38.38%.

**3. Data Visualizations**

Created dashboard visualizations, including:

- Survival Rate by Passenger Class.
- Passenger Age Distribution.

**4. Dataset Preview**

- Displays the cleaned Titanic dataset.
- Allows users to explore passenger records and their features.

**5. Passenger Survival Prediction**

Integrated the saved Random Forest model into the Streamlit application to predict passenger survival.

Users can enter passenger details through input widgets, including:

- Passenger Class.
- Gender.
- Passenger Age.
- Number of Siblings/Spouses.
- Number of Parents/Children.
- Port of Embarkation.
- Ticket Fare.

The dashboard processes the input, performs the required feature engineering, and generates a prediction using the trained model.

**6. Prediction Results**

The dashboard displays:

- Predicted survival outcome.
- Predicted survival probability.
- Probability of survival.
- Probability of not surviving.

The prediction is an estimate generated by the trained machine learning model, not a guarantee of an actual outcome.

### Prediction Testing

Tested the prediction functionality using three different passenger profiles.

| Test Case | Passenger Details | Prediction | Survival Probability |
|---|---|---|---:|
| 1 | Female, 1st Class, Age 29 | Survive | 88% |
| 2 | Male, 3rd Class, Age 22 | Not Survive | 3% |
| 3 | Female, 2nd Class, Age 35 | Survive | 79% |

All three test cases successfully generated prediction results.

### Week 4 Conclusion

The Streamlit dashboard successfully integrates data analysis, visualizations, and machine learning predictions into a single interactive application.

Users can explore Titanic dataset statistics, view visualizations, preview passenger records, and predict survival outcomes by entering passenger information.

The trained Random Forest model was successfully loaded into the dashboard and tested with three different passenger profiles.

---

## Project Structure

```text
Titanic-Project/
│
├── data/
│   └── titanic_clean.csv
│
├── screenshots/
│   ├── dashboard_overview.png
│   ├── dataset_preview.png
│   ├── prediction_form.png
│   ├── prediction_test_1.png
│   ├── prediction_test_2.png
│   ├── prediction_test_3.png
│   ├── fare_outlier_detection.png
│   ├── logistic_regression_confusion_matrix.png
│   ├── missing_values_after_cleaning.png
│   ├── passenger_age_distribution.png
│   ├── random_forest_confusion_matrix.png
│   ├── random_forest_feature_importance_barplot.png
│   ├── survival_analysis_after_cleaning.png
│   ├── survival_rate_by_class.png
│   ├── survival_rate_by_passenger_class.png
│   ├── titanic_categorical_encoding.png
│   ├── titanic_correlation_heatmap.png
│   ├── titanic_dataset_first_five_rows.png
│   └── titanic_feature_engineering.png
│
├── README.md
├── Titanic.ipynb
├── app.py
├── best_titanic_model.pkl
└── requirements.txt
```

**Note:** The `screenshots` folder contains the dashboard screenshots and EDA/ML visualization images. The `venv` folder is excluded from GitHub.

# How to Run the Project

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project Folder

```bash
cd Titanic_Streamlit_Dashboard
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit Application

```bash
streamlit run app.py
```

### 7. Open the Dashboard

Open the following URL in your browser:

```text
http://localhost:8501
```

The interactive Titanic Survival Prediction Dashboard will be displayed.

### Running the Jupyter Notebook

1. Download the `Titanic.ipynb` notebook from the repository.
2. Open [Google Colab](https://colab.research.google.com/).
3. Select File → Upload notebook.
4. Upload the downloaded notebook.
5. Make sure the cleaned dataset is available at `data/titanic_clean.csv`.
6. Select Runtime → Run all.
7. Explore the outputs, analysis, visualizations, model evaluations, and predictions.

### Loading the Saved Model Separately

The trained Random Forest model can also be loaded using Joblib:

```python
import joblib

model = joblib.load('best_titanic_model.pkl')
```

---

# Overall Conclusion

This project provided practical experience in the complete data science and machine learning workflow, from exploratory data analysis and data cleaning to feature engineering, model development, and interactive dashboard creation.

Across the four weeks, I gained hands-on experience with Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Joblib, and Streamlit.

I learned how to explore datasets, handle missing values, engineer meaningful features, visualize data patterns, train and evaluate classification models, compare model performance, analyze feature importance, and save trained models for future use.

The Random Forest Classifier achieved the best performance with **82.68% accuracy**, making it the selected model for Titanic passenger survival prediction.

Finally, the trained model was integrated into an interactive Streamlit dashboard, where users can explore passenger statistics, view visualizations, and generate survival predictions using different passenger details.

This project strengthened my understanding of data analysis, data preprocessing, machine learning, model evaluation, and interactive data visualization using Python.

---

## Project Status

**Completed – SkillMe Data Science Virtual Internship (Weeks 1–4)**
