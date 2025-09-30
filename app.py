import streamlit as st
import pandas as pd
import numpy as np
# NOTE: In a real application, you would ensure these dependencies are installed:
# from sklearn.preprocessing import StandardScaler
# from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# --- Mock Classes for Demonstration ---
# In a real environment, you would use your actual CustomData and PredictPipeline classes.

class CustomData:
    """A mock class to structure incoming form data into a DataFrame."""
    def __init__(self):
        # The fields expected by the prediction pipeline
        self.column_names = [
            'gender', 'race_ethnicity', 'parental_level_of_education',
            'lunch', 'test_preparation_course', 'reading_score', 'writing_score'
        ]
        self.data = {}

    def set_data(self, gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score, writing_score):
        """Sets the data attributes based on inputs."""
        self.data = {
            'gender': gender,
            'race_ethnicity': race_ethnicity,
            'parental_level_of_education': parental_level_of_education,
            'lunch': lunch,
            'test_preparation_course': test_preparation_course,
            'reading_score': reading_score,
            'writing_score': writing_score
        }

    def get_data_as_data_frame(self):
        """Converts the collected data into a Pandas DataFrame."""
        df = pd.DataFrame([self.data], columns=self.column_names)
        return df

class PredictPipeline:
    """A mock class to simulate the prediction process."""
    def __init__(self):
        # In a real app, this would load the model and preprocessor (e.g., via pickle)
        # self.model = load_object(MODEL_PATH)
        # self.preprocessor = load_object(PREPROCESSOR_PATH)
        pass

    def predict(self, features):
        """  
        It calculates an arbitrary 'Math Score' based on Reading and Writing 
        scores, as the actual model logic is unavailable here.
        """
        
        # Calculate a mock result based on the average of the two input scores
        # This replaces the logic that would normally use the model.
        avg_score = (features['reading_score'][0] + features['writing_score'][0]) / 2
        
        # Add a small random element for realism in the mock result
        mock_prediction = round(avg_score + np.random.uniform(-5, 5), 2)
        
        # Ensure the mock score is between 0 and 100
        result = [max(0, min(100, mock_prediction))]
        return result

# --- Streamlit Application Logic ---

st.set_page_config(
    page_title="Student Exam Performance Indicator",
    layout="centered"
)

st.title("Student Exam Performance Indicator")
st.markdown("---")

st.header("Predict Student Math Score")

# Create an instance of the CustomData class to hold and process inputs
data_processor = CustomData()

# Use st.form for grouping widgets and submitting inputs at once
with st.form(key='prediction_form'):
    
    # Use columns to lay out categorical inputs side-by-side for a simpler look
    col1, col2 = st.columns(2)
    
    with col1:
        # 1. Gender
        gender = st.selectbox(
            'Gender',
            ('Select your Gender', 'male', 'female'),
            index=0,
            key='gender_select',
        )
        
        # 3. Parental Level of Education
        parental_level_of_education = st.selectbox(
            'Parental Education',
            (
                "Select Parent Education",
                "associate's degree", 
                "bachelor's degree", 
                "high school", 
                "master's degree", 
                "some college", 
                "some high school"
            ),
            index=0,
            key='parental_education_select',
        )
        
        # 5. Test preparation Course
        test_preparation_course = st.selectbox(
            'Test Prep Course',
            ('Select Test_course', 'none', 'completed'),
            index=0,
            key='test_course_select',
        )

    with col2:
        # 2. Race or Ethnicity
        ethnicity = st.selectbox(
            'Race or Ethnicity',
            ('Select Ethnicity', 'group A', 'group B', 'group C', 'group D', 'group E'),
            index=0,
            key='ethnicity_select',
        )
        
        # 4. Lunch Type
        lunch = st.selectbox(
            'Lunch Type',
            ('Select Lunch Type', 'free/reduced', 'standard'),
            index=0,
            key='lunch_select',
        )
    
    st.markdown("---")
    st.subheader("Student Subject Scores (Out of 100)")
    
    # Use columns for numerical inputs
    score_col1, score_col2 = st.columns(2)
    
    with score_col1:
        # Input 1: Score used for the 'writing_score' parameter in CustomData
        score_for_writing = st.number_input(
            'Writing Score',
            min_value=0, max_value=100, value=70, step=1,
            key='writing_input',
        )
    
    with score_col2:
        # Input 2: Score used for the 'reading_score' parameter in CustomData
        score_for_reading = st.number_input(
            'Reading Score',
            min_value=0, max_value=100, value=70, step=1,
            key='reading_input',
        )
    
    # Submission Button (centered)
    st.markdown("---")
    col_submit_1, col_submit_2, col_submit_3 = st.columns([1, 2, 1])
    with col_submit_2:
        submit_button = st.form_submit_button(label='Predict Math Score', type='primary', use_container_width=True)

# Initializing result outside the if block
results = None 

# Handle form submission
if submit_button:
    # Input validation (check if placeholder values are selected)
    if any(val in (gender, ethnicity, parental_level_of_education, lunch, test_preparation_course) 
           for val in ("Select your Gender", "Select Ethnicity", "Select Parent Education", "Select Lunch Type", "Select Test_course")):
        st.error("Please select a value for all required categorical fields before predicting.")
    else:
        try:
            # 1. Populate the CustomData object using the correct mapping 
            data_processor.set_data(
                gender=gender,
                race_ethnicity=ethnicity,
                parental_level_of_education=parental_level_of_education,
                lunch=lunch,
                test_preparation_course=test_preparation_course,
                reading_score=float(score_for_reading),
                writing_score=float(score_for_writing)
            )
            
            # 2. Get the DataFrame
            pred_df = data_processor.get_data_as_data_frame()
            
            # 3. Initialize and run the prediction pipeline
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            
            # 4. Display the result using a simple metric
            if results is not None:
                st.markdown("---")
                
                # Display result in a prominent metric block
                st.success("Prediction Completed:")
                st.metric(label="Predicted Math Score", value=f"{results[0]:.2f}")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.stop() # Stop execution after error
