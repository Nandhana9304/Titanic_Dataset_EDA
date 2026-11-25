
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="InsightHub | Titanic Dashboard",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# 2. CUSTOM CSS
# ==================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background-color: #F5F9F6;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #17452D;
    font-weight: 700;
}

section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #DCE9DF;
}

.stButton > button {
    background: linear-gradient(90deg, #166534, #16A34A);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    min-height: 45px;
}

.stButton > button:hover {
    background: #14532D;
    color: white;
    border: none;
}

div[data-testid="stForm"] {
    background-color: white;
    border: 1px solid #DCE9DF;
    border-radius: 14px;
    padding: 20px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #DCE9DF;
    border-radius: 10px;
}

hr {
    border-color: #DCE9DF;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. FILE PATHS
# ==================================================

DATA_PATH = os.path.join("data", "titanic_clean.csv")
MODEL_PATH = "best_titanic_model.pkl"


# ==================================================
# 4. LOAD DATASET AND MODEL
# ==================================================

@st.cache_data
def load_dataset(path):
    return pd.read_csv(path)


@st.cache_resource
def load_model(path):
    return joblib.load(path)


if not os.path.exists(DATA_PATH):
    st.error("Dataset not found. Check data/titanic_clean.csv.")
    st.stop()

if os.path.getsize(DATA_PATH) == 0:
    st.error("The cleaned dataset file is empty.")
    st.stop()

try:
    df = load_dataset(DATA_PATH)
except Exception as e:
    st.error(f"Dataset loading error: {e}")
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

if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Check best_titanic_model.pkl.")
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()


# ==================================================
# 5. MODEL FEATURES
# ==================================================

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

if hasattr(model, "feature_names_in_"):
    if list(model.feature_names_in_) != FEATURE_COLUMNS:
        st.error("Model feature names do not match dashboard features.")
        st.stop()


# ==================================================
# 6. SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown(
        "<h2 style='color:#166534;'>🚢 InsightHub</h2>",
        unsafe_allow_html=True
    )

    st.caption("TITANIC SURVIVAL ANALYTICS")

    st.divider()

    st.markdown("### 📌 Model Information")

    st.markdown("**Algorithm**")
    st.markdown("## Random Forest")

    st.markdown("**Test Accuracy**")
    st.markdown("## 82.68%")

    st.divider()

    st.markdown("### ⭐ Important Features")

    st.markdown("""
    - Passenger class
    - Gender
    - Age
    - Ticket fare
    - Family size
    - Siblings and spouses
    - Parents and children
    - Port of embarkation
    """)

    st.divider()

    st.info(
        "The Random Forest model analyzes passenger information "
        "to estimate survival probability."
    )

    st.caption("SkillMe Data Science Virtual Internship")


# ==================================================
# 7. DASHBOARD HERO HEADER
# ==================================================

st.html("""
<div style="
    background: linear-gradient(120deg, #14532D, #15803D);
    padding: 38px;
    border-radius: 18px;
    color: white;
    margin-bottom: 30px;
    box-shadow: 0 5px 18px rgba(20,83,45,0.12);
">
    <h1 style="
        color: white;
        font-size: 34px;
        margin-bottom: 12px;
    ">
        🚢 Titanic Survival Prediction Dashboard
    </h1>

    <p style="
        color: #E5F5E9;
        font-size: 16px;
        margin-bottom: 0;
    ">
        Explore passenger survival patterns, analyze historical data,
        and predict survival chances using Machine Learning.
    </p>
</div>
""")


# ==================================================
# 8. DASHBOARD OVERVIEW
# ==================================================

total_passengers = len(df)
survived_passengers = int(df["survived"].sum())
not_survived = total_passengers - survived_passengers
survival_rate = df["survived"].mean() * 100

st.markdown("## 📊 Dashboard Overview")
st.caption("Key statistics from the Titanic passenger dataset.")

col1, col2, col3, col4 = st.columns(4)

metrics = [
    (col1, "👥", "Total Passengers", f"{total_passengers:,}"),
    (col2, "💚", "Survived Passengers", f"{survived_passengers:,}"),
    (col3, "📈", "Survival Rate", f"{survival_rate:.2f}%"),
    (col4, "🤖", "Model Accuracy", "82.68%")
]

for column, icon, label, value in metrics:

    with column:

        st.html(f"""
        <div style="
            background: white;
            border: 1px solid #DCE9DF;
            border-radius: 15px;
            padding: 22px;
            min-height: 145px;
            box-shadow: 0 3px 12px rgba(0,0,0,0.035);
        ">
            <div style="font-size: 26px; margin-bottom: 10px;">
                {icon}
            </div>

            <div style="
                color: #64756A;
                font-size: 14px;
                font-weight: 500;
            ">
                {label}
            </div>

            <div style="
                color: #14532D;
                font-size: 29px;
                font-weight: 800;
                margin-top: 8px;
            ">
                {value}
            </div>
        </div>
        """)


st.write("")


# ==================================================
# 9. DATA VISUALIZATIONS
# ==================================================

st.markdown("## 📈 Data Visualizations")
st.caption("Explore survival patterns and passenger age distribution.")

chart1, chart2 = st.columns(2)

# Chart 1: Survival by Passenger Class

with chart1:

    with st.container(border=True):

        st.markdown("### Survival by Passenger Class")

        survival_by_class = (
            df.groupby("pclass")["survived"].mean() * 100
        )

        labels = ["1st Class", "2nd Class", "3rd Class"]

        values = [
            survival_by_class.get(1, 0),
            survival_by_class.get(2, 0),
            survival_by_class.get(3, 0)
        ]

        fig1, ax1 = plt.subplots(figsize=(7, 4))

        bars = ax1.bar(
            labels,
            values,
            color=["#166534", "#4ADE80", "#A7F3D0"],
            width=0.58
        )

        ax1.set_ylabel("Survival Rate (%)")
        ax1.set_ylim(0, 100)
        ax1.grid(axis="y", alpha=0.2)
        ax1.set_axisbelow(True)

        for bar in bars:
            height = bar.get_height()

            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                height + 2,
                f"{height:.1f}%",
                ha="center",
                fontsize=10,
                fontweight="bold"
            )

        for spine in ax1.spines.values():
            spine.set_visible(False)

        fig1.tight_layout()
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)


# Chart 2: Passenger Age Distribution

with chart2:

    with st.container(border=True):

        st.markdown("### Passenger Age Distribution")

        fig2, ax2 = plt.subplots(figsize=(7, 4))

        sns.histplot(
            df["age"].dropna(),
            bins=30,
            kde=True,
            color="#16A34A",
            edgecolor="white",
            ax=ax2
        )

        ax2.set_xlabel("Passenger Age")
        ax2.set_ylabel("Number of Passengers")
        ax2.grid(axis="y", alpha=0.2)
        ax2.set_axisbelow(True)

        for spine in ax2.spines.values():
            spine.set_visible(False)

        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)


# ==================================================
# 10. PASSENGER SURVIVAL DISTRIBUTION
# ==================================================

st.write("")

st.markdown("## 🧭 Passenger Survival Distribution")

st.caption("Overview of passengers who survived and did not survive.")

with st.container(border=True):

    col1, col2 = st.columns([1, 1.5])

    with col1:

        st.markdown("### 💚 Survival Statistics")

        st.metric(
            label="Survived Passengers",
            value=f"{survived_passengers:,}",
            delta=f"{survived_passengers / total_passengers * 100:.2f}%"
        )

        st.metric(
            label="Did Not Survive",
            value=f"{not_survived:,}",
            delta=f"{not_survived / total_passengers * 100:.2f}%",
            delta_color="inverse"
        )

    with col2:

        fig, ax = plt.subplots(figsize=(5, 4))

        values = [survived_passengers, not_survived]

        labels = ["Survived", "Did Not Survive"]

        colors = ["#16A34A", "#FCA5A5"]

        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            counterclock=False,
            pctdistance=0.75,
            wedgeprops={
                "width": 0.38,
                "edgecolor": "white",
                "linewidth": 3
            },
            textprops={"fontsize": 10}
        )

        ax.text(
            0, 0,
            f"{total_passengers}\nPassengers",
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color="#14532D"
        )

        ax.set_title(
            "Passenger Survival Overview",
            fontsize=13,
            fontweight="bold",
            pad=15
        )

        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)

        plt.close(fig)


# ==================================================
# 11. DATASET EXPLORER
# ==================================================

st.write("")

with st.container(border=True):

    st.markdown("## 📋 Dataset Explorer")

    st.caption(
        "Explore the cleaned Titanic dataset used for analysis and prediction."
    )

    search_text = st.text_input(
        "🔍 Search Dataset",
        placeholder="Enter a value to search..."
    )

    if search_text:

        filtered_df = df[
            df.astype(str).apply(
                lambda column: column.str.contains(
                    search_text,
                    case=False,
                    na=False
                )
            ).any(axis=1)
        ]

    else:
        filtered_df = df

    st.dataframe(
        filtered_df.head(100),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        f"Showing {min(len(filtered_df), 100)} of "
        f"{len(filtered_df)} matching records."
    )


# ==================================================
# 12. PASSENGER PREDICTION FORM
# ==================================================

st.write("")

st.markdown("## 🔮 Passenger Survival Prediction")

st.caption(
    "Enter passenger details to estimate the probability of survival."
)

with st.container(border=True):

    st.markdown("### 👤 Enter Passenger Information")

    with st.form("passenger_prediction_form"):

        input_col1, input_col2 = st.columns(2)

        with input_col1:

            passenger_class = st.selectbox(
                "Passenger Class",
                [1, 2, 3],
                index=None,
                placeholder="Select passenger class",
                format_func=lambda x: {
                    1: "1st Class",
                    2: "2nd Class",
                    3: "3rd Class"
                }[x]
            )

            gender = st.selectbox(
                "Gender",
                ["Female", "Male"],
                index=None,
                placeholder="Select gender"
            )

            age = st.number_input(
                "Passenger Age",
                min_value=0,
                max_value=80,
                value=None,
                placeholder="Enter passenger age"
            )

            fare = st.number_input(
                "Ticket Fare",
                min_value=0.0,
                max_value=512.0,
                value=None,
                placeholder="Enter ticket fare"
            )

        with input_col2:

            sibsp = st.number_input(
                "Number of Siblings/Spouses",
                min_value=0,
                max_value=8,
                value=None,
                placeholder="Enter number"
            )

            parch = st.number_input(
                "Number of Parents/Children",
                min_value=0,
                max_value=6,
                value=None,
                placeholder="Enter number"
            )

            embarked = st.selectbox(
                "Port of Embarkation",
                ["S", "C", "Q"],
                index=None,
                placeholder="Select port",
                format_func=lambda x: {
                    "S": "Southampton",
                    "C": "Cherbourg",
                    "Q": "Queenstown"
                }[x]
            )

        st.write("")

        predict_button = st.form_submit_button(
            "🔍 Predict Passenger Survival",
            use_container_width=True
        )


# ==================================================
# 13. FEATURE ENGINEERING AND PREDICTION
# ==================================================

if predict_button:

    if any(value is None for value in [
        passenger_class, gender, age, fare,
        sibsp, parch, embarked
    ]):

        st.warning("Please enter all passenger details before predicting.")

    else:

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
        }], columns=FEATURE_COLUMNS)

        try:

            prediction = model.predict(input_data)[0]

            probabilities = model.predict_proba(input_data)[0]

            survival_index = list(model.classes_).index(1)
            death_index = list(model.classes_).index(0)

            survival_probability = probabilities[survival_index] * 100
            death_probability = probabilities[death_index] * 100

            st.markdown("## 🎯 Prediction Result")

            with st.container(border=True):

                if prediction == 1:
                    st.success("🎉 Passenger is likely to SURVIVE!")
                else:
                    st.error("Passenger is likely NOT TO SURVIVE.")

                result_col1, result_col2 = st.columns(2)

                with result_col1:
                    st.metric(
                        "Survival Probability",
                        f"{survival_probability:.2f}%"
                    )

                with result_col2:
                    st.metric(
                        "Non-Survival Probability",
                        f"{death_probability:.2f}%"
                    )

                st.progress(survival_probability / 100)

                st.caption(
                    "This prediction is an estimate generated by "
                    "the trained Random Forest model."
                )

        except Exception as e:
            st.error(f"Prediction error: {e}")
# ==================================================
# 13. PREDICTION RESULT
# ==================================================

if predict_button:

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
    }], columns=FEATURE_COLUMNS)

    try:

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        survival_index = list(model.classes_).index(1)
        death_index = list(model.classes_).index(0)

        survival_probability = probabilities[survival_index] * 100
        death_probability = probabilities[death_index] * 100

        st.write("")
        st.markdown("## 🎯 Prediction Result")

        with st.container(border=True):

            if prediction == 1:
                st.success("🎉 Passenger is likely to SURVIVE!")
            else:
                st.error("Passenger is likely NOT TO SURVIVE.")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric(
                    "💚 Survival Probability",
                    f"{survival_probability:.2f}%"
                )

            with result_col2:
                st.metric(
                    "❌ Non-Survival Probability",
                    f"{death_probability:.2f}%"
                )

            st.markdown("### Survival Probability")
            st.progress(float(survival_probability / 100))

            st.caption(
                "This prediction is an estimate generated by the "
                "trained Random Forest model, not a guarantee."
            )

            st.divider()

            st.markdown("### Passenger Summary")

            summary_col1, summary_col2, summary_col3 = st.columns(3)

            with summary_col1:
                st.metric("Passenger Class", passenger_class)

            with summary_col2:
                st.metric("Passenger Age", age)

            with summary_col3:
                st.metric("Family Size", family_size)

            st.markdown("### Prediction Probability Breakdown")

            probability_df = pd.DataFrame({
                "Outcome": ["Did Not Survive", "Survived"],
                "Probability": [
                    death_probability,
                    survival_probability
                ]
            })

            fig4, ax4 = plt.subplots(figsize=(8, 2.5))

            bars = ax4.barh(
                probability_df["Outcome"],
                probability_df["Probability"],
                color=["#FCA5A5", "#16A34A"],
                height=0.5
            )

            ax4.set_xlim(0, 110)
            ax4.set_xlabel("Probability (%)")
            ax4.grid(axis="x", alpha=0.2)
            ax4.set_axisbelow(True)

            for bar in bars:
                width = bar.get_width()

                ax4.text(
                    width + 1,
                    bar.get_y() + bar.get_height() / 2,
                    f"{width:.2f}%",
                    va="center",
                    fontsize=10,
                    fontweight="bold"
                )

            for spine in ax4.spines.values():
                spine.set_visible(False)

            fig4.tight_layout()
            st.pyplot(fig4, use_container_width=True)
            plt.close(fig4)

    except Exception as e:
        st.error(f"Prediction error: {e}")


# ==================================================
# 14. FOOTER
# ==================================================

st.divider()

st.html("""
<div style="
    text-align: center;
    color: #718078;
    font-size: 13px;
    padding: 22px;
">
    <b>InsightHub | Titanic Survival Prediction Dashboard</b>
    <br><br>
    Developed using Python, Pandas, Streamlit,
    Matplotlib, Seaborn and Scikit-learn.
    <br><br>
    SkillMe Data Science Virtual Internship
</div>
""")
