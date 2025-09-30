import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

st.set_page_config(
    page_title="Student Exam Performance Indicator",
    layout="centered"
)

st.title("Student Exam Performance Indicator")
st.markdown("---")
st.header("Predict Student Math Score")

with st.form(key='prediction_form'):
    
    st.subheader("Student Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox('Gender', ('Select your Gender', 'male', 'female'))
        
        parental_level_of_education = st.selectbox(
            'Parental Education',
            ("Select Parent Education", "associate's degree", "bachelor's degree", 
             "high school", "master's degree", "some college", "some high school")
        )
        
        test_preparation_course = st.selectbox('Test Prep Course', ('Select Test_course', 'none', 'completed'))

    with col2:
        ethnicity = st.selectbox(
            'Race or Ethnicity',
            ('Select Ethnicity', 'group A', 'group B', 'group C', 'group D', 'group E')
        )
        
        lunch = st.selectbox('Lunch Type', ('Select Lunch Type', 'free/reduced', 'standard'))

    st.markdown("---")
    st.subheader("Subject Scores (Out of 100)")
    
    score_col1, score_col2 = st.columns(2)
    
    with score_col1:
        score_for_writing_input = st.number_input(
            'Writing Score',
            min_value=0, max_value=100, value=70, step=1
        )
    
    with score_col2:
        score_for_reading_input = st.number_input(
            'Reading Score',
            min_value=0, max_value=100, value=70, step=1
        )
    
    st.markdown("---")
    submit_button = st.form_submit_button(label='Predict Math Score', type='primary')

if 'results' not in st.session_state:
    st.session_state.results = None

if submit_button:
    if any(val.startswith('Select') for val in [gender, ethnicity, parental_level_of_education, lunch, test_preparation_course]):
        st.error("Please select a value for all required fields before predicting.")
    else:
        try:
            data = CustomData(
                gender=gender,
                race_ethnicity=ethnicity,
                parental_level_of_education=parental_level_of_education,
                lunch=lunch,
                test_preparation_course=test_preparation_course,
                reading_score=float(score_for_writing_input),
                writing_score=float(score_for_reading_input)
            )
            
            pred_df = data.get_data_as_data_frame()
            
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            
            st.session_state.results = results[0]
            
        except Exception as e:
            st.error("An error occurred during prediction. Please check server logs.")

if st.session_state.results is not None:
    st.markdown("---")
    st.subheader("Prediction Result")
    st.metric(label="Predicted Math Score", value=f"{st.session_state.results:.2f}")

st.markdown("""
<style>
    .stSelectbox, .stNumberInput {
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)
