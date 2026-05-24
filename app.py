import streamlit as st
import requests

API_URL = "https://mlproject-ehb3.onrender.com/predict"

st.set_page_config(
    page_title="Student Performance Predictor",
    layout="centered",
    page_icon="🎓"
)

# ---------- HEADER ----------
st.markdown(
    """
    <h1 style='text-align: center; color: #4A90E2;'>
        🎓 Student Performance Predictor
    </h1>
    <p style='text-align: center; color: gray;'>
        Predict student math score using ML + FastAPI
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ---------- FORM ----------
with st.form(key="prediction_form"):

    st.subheader("👤 Student Details")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ("Select", "male", "female"))

        parental_level_of_education = st.selectbox(
            "Parental Education",
            ("Select", "associate's degree", "bachelor's degree",
             "high school", "master's degree", "some college", "some high school")
        )

        test_preparation_course = st.selectbox(
            "Test Prep Course",
            ("Select", "none", "completed")
        )

    with col2:
        ethnicity = st.selectbox(
            "Race/Ethnicity",
            ("Select", "group A", "group B", "group C", "group D", "group E")
        )

        lunch = st.selectbox(
            "Lunch Type",
            ("Select", "free/reduced", "standard")
        )

    st.markdown("---")
    st.subheader("📊 Scores")

    col3, col4 = st.columns(2)

    with col3:
        writing_score = st.number_input(
            "Writing Score",
            min_value=0, max_value=100, value=70
        )

    with col4:
        reading_score = st.number_input(
            "Reading Score",
            min_value=0, max_value=100, value=70
        )

    st.markdown("---")

    submit_button = st.form_submit_button(
        label="🚀 Predict Score",
        type="primary"
    )

# ---------- SESSION ----------
if "results" not in st.session_state:
    st.session_state.results = None

# ---------- PREDICTION ----------
if submit_button:

    if "Select" in [
        gender,
        ethnicity,
        parental_level_of_education,
        lunch,
        test_preparation_course
    ]:
        st.error("⚠️ Please fill all fields correctly.")
    else:
        payload = {
            "gender": gender,
            "race_ethnicity": ethnicity,
            "parental_level_of_education": parental_level_of_education,
            "lunch": lunch,
            "test_preparation_course": test_preparation_course,
            "reading_score": float(reading_score),
            "writing_score": float(writing_score),
        }

        try:
            with st.spinner("Predicting student performance..."):
                response = requests.post(API_URL, json=payload, timeout=30)
            
            st.write("Status Code:", response.status_code)
            st.write("Response Text:", response.text)

            if response.status_code == 200:
                result = response.json()
                st.session_state.results = result["predicted_math_score"]
            else:
                st.error("❌ Prediction failed. Try again.")

        except Exception as e:
            st.error(f"Error: {str(e)}")

# ---------- OUTPUT ----------
if st.session_state.results is not None:

    st.markdown("---")

    st.success("Prediction completed successfully 🎉")

    st.metric(
        label="Predicted Math Score",
        value=f"{st.session_state.results:.2f}"
    )