# 📊 Student Performance Prediction — End-to-End ML Project

An end-to-end Machine Learning application that predicts a student's **Math Score** based on demographic, academic, and preparation-related features.

The project demonstrates a complete ML workflow including **EDA, data ingestion, preprocessing, multiple regression models, cross-validation, hyperparameter tuning, model selection, prediction pipelines, FastAPI APIs, and Streamlit UI**.

---

## 🎯 Problem Statement

Student academic performance can be influenced by several demographic and academic factors such as:

- Gender
- Race/ethnicity
- Parental level of education
- Lunch type
- Test preparation course
- Reading score
- Writing score

The objective of this project is to build a regression model that predicts a student's **Math Score** using these features.

---

## ✨ Key Features

- End-to-end modular ML pipeline
- Exploratory Data Analysis (EDA)
- Automated train-test split
- Numerical and categorical preprocessing
- Missing value handling
- One-hot encoding
- Feature scaling
- Comparison of multiple regression algorithms
- 5-fold cross-validation
- Hyperparameter tuning using `GridSearchCV`
- CV-based model selection
- Final evaluation on an untouched test set
- Serialized model and preprocessing pipeline
- FastAPI prediction API
- Streamlit interactive frontend
- Exception handling and logging
- Deployment-ready project structure

---

# 🏗️ System Architecture

```text
                         Dataset
                            │
                            ▼
                     Exploratory Data
                        Analysis
                            │
                            ▼
                    Data Ingestion
                            │
                            ▼
                  Train / Test Split
                     /          \
                    /            \
             Training Data      Test Data
                    │                │
                    ▼                │
             Data Transformation     │
                    │                │
                    ▼                │
          Multiple ML Algorithms     │
                    │                │
                    ▼                │
           5-Fold Cross Validation   │
                    │                │
                    ▼                │
           Hyperparameter Tuning     │
                    │                │
                    ▼                │
             Compare CV Scores       │
                    │                │
                    ▼                │
              Select Best Model      │
                    │                │
                    ▼                │
            Refit on Training Data   │
                    │                │
                    └───────┐        │
                            ▼        ▼
                         Final Test
                         Evaluation
                            │
                            ▼
                       Saved Model
                            │
                            ▼
                  FastAPI + Streamlit
                            │
                            ▼
                       Prediction
```

---

# 📈 Machine Learning Workflow

## 1. Exploratory Data Analysis

The dataset is analyzed using a separate Jupyter notebook before model development.

EDA includes:

- Dataset structure and data types
- Missing value analysis
- Duplicate record analysis
- Numerical feature distributions
- Outlier analysis using boxplots
- Categorical feature distributions
- Correlation analysis
- Feature vs target analysis
- Categorical features vs Math Score analysis

The EDA notebook is available under:

```text
notebooks/
└── EDA.ipynb
```

---

## 2. Data Ingestion

The data ingestion component:

- Reads the raw CSV dataset
- Creates the required artifact directory
- Stores the raw dataset
- Splits the dataset into training and testing sets
- Saves the train and test datasets as CSV files

The project uses an **80/20 train-test split** with a fixed random state for reproducibility.

```text
1000+ records
      │
      ├── 80% Training Data
      │
      └── 20% Test Data
```

---

## 3. Data Transformation

Numerical and categorical features are processed separately.

### Numerical Features

```text
writing_score
reading_score
```

Processing:

```text
Missing Value Imputation
        ↓
Median
        ↓
StandardScaler
```

### Categorical Features

```text
gender
race_ethnicity
parental_level_of_education
lunch
test_preparation_course
```

Processing:

```text
Missing Value Imputation
        ↓
Most Frequent Value
        ↓
OneHotEncoder
        ↓
StandardScaler
```

The preprocessing object is serialized and stored as:

```text
artifacts/preprocessor.pkl
```

---

# 🤖 Model Training

Seven regression algorithms are evaluated:

1. Linear Regression
2. Random Forest Regressor
3. Decision Tree Regressor
4. Gradient Boosting Regressor
5. XGBoost Regressor
6. CatBoost Regressor
7. AdaBoost Regressor

---

# 🔍 Model Selection & Hyperparameter Tuning

The model selection process uses **5-fold cross-validation** on the training data.

The test set is kept untouched during model selection.

### Workflow

```text
Training Data
     │
     ▼
Baseline Model
     │
     ▼
5-Fold Cross Validation
     │
     ▼
GridSearchCV
     │
     ▼
Best Hyperparameters
     │
     ▼
Best CV R² for each model
     │
     ▼
Compare all models
     │
     ▼
Select model with highest CV R²
     │
     ▼
Refit selected model on full training data
     │
     ▼
Evaluate once on test data
```

This prevents the test set from being used for model selection.

---

# 📊 Model Performance

The current model evaluation produced the following results:

| Model                 | Baseline CV R² | Best CV R² |
| --------------------- | -------------: | ---------: |
| Random Forest         |         0.8332 |     0.8384 |
| Decision Tree         |         0.6850 |     0.7044 |
| Gradient Boosting     |         0.8484 |     0.8531 |
| **Linear Regression** |     **0.8686** | **0.8686** |
| XGBoost               |         0.8066 |     0.8326 |
| CatBoost              |         0.8444 |     0.8552 |
| AdaBoost              |         0.8244 |     0.8264 |

### Selected Model

**Linear Regression**

```text
Best CV R²: 0.8686
Final Test R²: 0.8804
```

Linear Regression achieved the highest cross-validation performance among the evaluated models, so it was selected as the final model.

The final model was then retrained using the complete training dataset and evaluated on the previously unseen test dataset.

---

# 🔮 Prediction Pipeline

The prediction pipeline contains two main components.

### `CustomData`

Converts user-provided input into a Pandas DataFrame with the same feature structure used during training.

### `PredictPipeline`

Loads:

```text
model.pkl
preprocessor.pkl
```

and performs:

```text
Input Data
    ↓
Preprocessing
    ↓
Model Prediction
    ↓
Predicted Math Score
```

This ensures that prediction-time preprocessing is consistent with the preprocessing used during training.

---

# 🌐 FastAPI Backend

The project provides a REST API using FastAPI.

## Health Check

### Endpoint

```http
GET /
```

### Response

```json
{
  "message": "API is running"
}
```

---

## Prediction

### Endpoint

```http
POST /predict
```

### Example Request

```json
{
  "gender": "male",
  "race_ethnicity": "group B",
  "parental_level_of_education": "bachelor's degree",
  "lunch": "standard",
  "test_preparation_course": "completed",
  "reading_score": 72,
  "writing_score": 70
}
```

### Example Response

```json
{
  "predicted_math_score": 78.45
}
```

---

# 🛠️ Tech Stack

### Programming

- Python

### Data Analysis

- Pandas
- NumPy
- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- XGBoost
- CatBoost

### ML Techniques

- Data preprocessing
- One-hot encoding
- Feature scaling
- Missing value imputation
- Regression
- Cross-validation
- Hyperparameter tuning
- GridSearchCV
- R² evaluation

### Backend

- FastAPI

### Frontend

- Streamlit

### Deployment / Environment

- Docker
- Render
- Git/GitHub

---

# 🚀 Running the Project Locally

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/MLproject.git
cd MLproject
```

---

## 2. Create a Virtual Environment

### macOS / Linux

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Train the Model

Run the complete training pipeline:

```bash
python -m src.pipeline.train_pipeline
```

This will:

- Ingest the dataset
- Create train/test datasets
- Transform the features
- Train multiple regression models
- Perform 5-fold cross-validation
- Perform hyperparameter tuning
- Select the best model based on CV R²
- Retrain the selected model
- Evaluate it on the test dataset
- Save the model and preprocessor

---

## 5. Start the FastAPI Backend

```bash
uvicorn src.api.main:app --reload
```

The API will be available locally at:

```text
http://127.0.0.1:8000
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

---

## 6. Start the Streamlit Application

Open another terminal:

```bash
streamlit run app.py
```

---

# 👩‍💻 Author

**Hiteshi Kukreja**

Computer Engineer | AI/ML Enthusiast | Data & ML

---

## ⭐ Feedback

If you find this project useful, feel free to ⭐ the repository or explore the implementation.
