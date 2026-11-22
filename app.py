
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

# --------------------------------------------------
# 2. DASHBOARD HEADER
# --------------------------------------------------

st.title("🚢 Titanic Survival Prediction Dashboard")

st.markdown("""
Explore the Titanic dataset, understand passenger survival
patterns, and predict survival chances using a Machine
Learning model.
""")

st.divider()

st.subheader("Welcome!")

st.write("""
This interactive dashboard presents data analysis,
visualizations, and real-time survival predictions.
""")

# --------------------------------------------------
# 3. LOAD CLEANED DATASET
# --------------------------------------------------

DATA_PATH = os.path.join("data", "titanic_clean.csv")
MODEL_PATH = "best_titanic_model.pkl"

if not os.path.exists(DATA_PATH):
    st.error("Cleaned dataset not found! Please add titanic_clean.csv to the data folder.")
    st.stop()

if os.path.getsize(DATA_PATH) == 0:
    st.error("The Titanic CSV file is empty. Please replace it with the correct cleaned dataset.")
    st.stop()

try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

if df.empty:
    st.error("The dataset contains no records.")
    st.stop()

required_columns = ["survived", "pclass", "age"]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()

# --------------------------------------------------
# 4. DATA ANALYSIS
# --------------------------------------------------

st.divider()

st.header("📊 Titanic Data Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Total Passengers",
        value=len(df)
    )

with col2:
    st.metric(
        label="Survived Passengers",
        value=int(df["survived"].sum())
    )

with col3:
    survival_rate = df["survived"].mean() * 100

    st.metric(
        label="Survival Rate",
        value=f"{survival_rate:.2f}%"
    )

# --------------------------------------------------
# 5. DATA VISUALIZATIONS
# --------------------------------------------------

st.divider()

st.header("📈 Data Visualizations")

chart1, chart2 = st.columns(2)

# Chart 1: Survival Rate by Passenger Class

with chart1:

    st.subheader("Survival Rate by Passenger Class")

    survival_by_class = (
        df.groupby("pclass")["survived"].mean() * 100
    )

    fig1, ax1 = plt.subplots(figsize=(7, 4))

    survival_by_class.plot(
        kind="bar",
        ax=ax1
    )

    ax1.set_title("Survival Rate by Passenger Class")
    ax1.set_xlabel("Passenger Class")
    ax1.set_ylabel("Survival Rate (%)")

    class_labels = {
        1: "1st Class",
        2: "2nd Class",
        3: "3rd Class"
    }

    ax1.set_xticks(range(len(survival_by_class)))
    ax1.set_xticklabels(
        [class_labels.get(int(x), str(x)) for x in survival_by_class.index],
        rotation=0
    )

    ax1.grid(axis="y", alpha=0.3)

    fig1.tight_layout()

    st.pyplot(fig1)

    plt.close(fig1)

# Chart 2: Passenger Age Distribution

with chart2:

    st.subheader("Passenger Age Distribution")

    fig2, ax2 = plt.subplots(figsize=(7, 4))

    sns.histplot(
        df["age"].dropna(),
        bins=30,
        kde=True,
        ax=ax2
    )

    ax2.set_title("Distribution of Passenger Ages")
    ax2.set_xlabel("Age")
    ax2.set_ylabel("Number of Passengers")

    ax2.grid(axis="y", alpha=0.3)

    fig2.tight_layout()

    st.pyplot(fig2)

    plt.close(fig2)

# --------------------------------------------------
# 6. DATASET PREVIEW
# --------------------------------------------------

st.divider()

st.header("📋 Dataset Preview")

st.write("First five rows of the cleaned Titanic dataset:")

st.dataframe(
    df.head(),
    use_container_width=True
)

# --------------------------------------------------
# 7. LOAD TRAINED MACHINE LEARNING MODEL
# --------------------------------------------------

st.divider()

st.header("🤖 Machine Learning Model")

if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found! Please add best_titanic_model.pkl to the project folder.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Unable to load the trained model: {e}")
    st.stop()

# Expected feature order used during Week 3 training

FEATURE_COLUMNS = [
    "pclass",
    "sex_male",
    "age",
    "sibsp",
    "parch",
    "fare",
    "family_size",
    "is_alone",
    "embarked_C",
    "embarked_Q",
    "embarked_S"
]

# Verify model compatibility

if hasattr(model, "feature_names_in_"):
    if list(model.feature_names_in_) != FEATURE_COLUMNS:
        st.error(
            "The model feature names do not match the dashboard input. "
            "Please check the Week 3 training feature order."
        )
        st.stop()

# --------------------------------------------------
# 8. SIDEBAR MODEL INFORMATION
# --------------------------------------------------

st.sidebar.title("📌 Model Information")

st.sidebar.metric(
    "Algorithm",
    "Random Forest"
)

st.sidebar.metric(
    "Test Accuracy",
    "82.68%"
)

st.sidebar.markdown("---")

st.sidebar.subheader("Important Features")

st.sidebar.markdown("""
- Passenger class
- Gender
- Age
- Ticket fare
- Family size
- Siblings and spouses
- Parents and children
""")

st.sidebar.markdown("---")

st.sidebar.info("""
The Random Forest model uses passenger information
to estimate survival probability.
""")

# --------------------------------------------------
# 9. INTERACTIVE PASSENGER INPUT
# --------------------------------------------------

st.divider()

st.header("🔮 Passenger Survival Prediction")

st.write("""
Enter passenger details below to predict the survival
outcome using the trained Random Forest model.
""")

st.subheader("Enter Passenger Information")

col1, col2 = st.columns(2)

with col1:

    passenger_class = st.selectbox(
        "Passenger Class",
        [1, 2, 3],
        format_func=lambda x: {
            1: "1st Class",
            2: "2nd Class",
            3: "3rd Class"
        }[x]
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    age = st.slider(
        "Passenger Age",
        min_value=0,
        max_value=80,
        value=29
    )

    fare = st.slider(
        "Ticket Fare",
        min_value=0.0,
        max_value=512.0,
        value=32.0,
        step=0.5
    )

with col2:

    sibsp = st.number_input(
        "Number of Siblings/Spouses",
        min_value=0,
        max_value=8,
        value=0,
        step=1
    )

    parch = st.number_input(
        "Number of Parents/Children",
        min_value=0,
        max_value=6,
        value=0,
        step=1
    )

    embarked = st.selectbox(
        "Port of Embarkation",
        ["S", "C", "Q"],
        format_func=lambda x: {
            "S": "Southampton",
            "C": "Cherbourg",
            "Q": "Queenstown"
        }[x]
    )

# --------------------------------------------------
# 10. FEATURE ENGINEERING FOR INPUT
# --------------------------------------------------

family_size = sibsp + parch + 1

is_alone = 1 if family_size == 1 else 0

input_data = pd.DataFrame([{
    "pclass": passenger_class,
    "sex_male": 1 if gender == "Male" else 0,
    "age": age,
    "sibsp": sibsp,
    "parch": parch,
    "fare": fare,
    "family_size": family_size,
    "is_alone": is_alone,
    "embarked_C": 1 if embarked == "C" else 0,
    "embarked_Q": 1 if embarked == "Q" else 0,
    "embarked_S": 1 if embarked == "S" else 0
}])

# --------------------------------------------------
# 11. DISPLAY PASSENGER DETAILS
# --------------------------------------------------

st.divider()

st.subheader("Passenger Details")

detail1, detail2, detail3 = st.columns(3)

with detail1:
    st.metric("Family Size", family_size)

with detail2:
    st.metric("Passenger Age", age)

with detail3:
    st.metric("Ticket Fare", f"${fare:.2f}")

# --------------------------------------------------
# 12. SURVIVAL PREDICTION
# --------------------------------------------------

st.divider()

if st.button("🔍 Predict Survival", type="primary", use_container_width=True):

    try:

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        # Get probability corresponding to survival class 1

        survival_class_index = list(model.classes_).index(1)

        survival_probability = (
            probabilities[survival_class_index] * 100
        )

        st.subheader("Prediction Result")

        if prediction == 1:

            st.success(
                "🎉 Prediction: Passenger is likely to SURVIVE!"
            )

        else:

            st.error(
                "Prediction: Passenger is likely NOT TO SURVIVE."
            )

        st.metric(
            label="Predicted Survival Probability",
            value=f"{survival_probability:.2f}%"
        )

        st.progress(
            float(survival_probability / 100)
        )

        st.caption("""
        This prediction is generated using a trained Random Forest
        classification model. It is an estimate, not a guarantee
        of the actual outcome.
        """)

        # Display both probabilities

        st.subheader("Prediction Probability")

        prob_col1, prob_col2 = st.columns(2)

        with prob_col1:

            death_index = list(model.classes_).index(0)

            st.metric(
                "Did Not Survive",
                f"{probabilities[death_index] * 100:.2f}%"
            )

        with prob_col2:

            st.metric(
                "Survived",
                f"{survival_probability:.2f}%"
            )

    except Exception as e:

        st.error(f"Prediction error: {e}")

# --------------------------------------------------
# 13. FOOTER
# --------------------------------------------------

st.divider()

st.caption("""
Titanic Survival Prediction Dashboard |
Built using Python, Pandas, Streamlit, Seaborn,
Matplotlib and Scikit-learn.
""")
