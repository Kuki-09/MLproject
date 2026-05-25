# 📊 Student Performance Indicator (ML End-to-End Project)

An end-to-end Machine Learning system that predicts a student's Math score based on demographic and academic features. The project includes a full ML pipeline, model training with multiple algorithms, hyperparameter tuning, and deployment-ready APIs using FastAPI and Streamlit.

---

## 🧠 Problem Statement

Students' academic performance depends on multiple factors such as:

- Parental education
- Test preparation
- Gender & ethnicity
- Reading & writing skills

This project predicts a student's **math score** using these features.

---

## 🏗️ System Architecture

```
User (Browser)
     ↓
Streamlit Frontend (Streamlit Cloud)
     ↓
FastAPI Backend on Render (/predict)
     ↓
Predict Pipeline (Preprocessing + Model)
     ↓
Trained ML Model (.pkl)
     ↓
Prediction Output
```

---

## ⚙️ Tech Stack

### 🧑‍💻 Programming

- Python

### 📊 Data & ML

- Pandas
- NumPy
- Scikit-learn
- XGBoost
- CatBoost
- GridSearchCV

### 🧠 ML Pipeline

- Data Ingestion
- Data Transformation (Pipeline + ColumnTransformer)
- Model Training (7 algorithms comparison)
- Hyperparameter tuning

### 🌐 Deployment

- **FastAPI** — Backend API, deployed on **Render**
- **Streamlit** — Frontend UI, deployed on **Streamlit Cloud**

---

## 📊 ML Workflow

### Data Ingestion

- Load dataset
- Train-test split

### Data Transformation

- Missing value handling
- OneHotEncoding
- Standardization
- Saved as `preprocessor.pkl`

### Model Training

Trained **7 regression models:**

- Linear Regression
- Random Forest
- Decision Tree
- Gradient Boosting
- XGBoost
- CatBoost
- AdaBoost

- Hyperparameter tuning using **GridSearchCV**
- Best model saved as `model.pkl`

### Prediction Pipeline

- Loads saved model + preprocessor
- Transforms input data
- Returns prediction

---

## 🚀 API Endpoints (FastAPI)

### 🟢 Health Check

```
GET /
```

**Response:**

```json
{ "message": "API is running" }
```

### 🔵 Prediction

```
POST /predict
```

**Input:**

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

**Output:**

```json
{
  "predicted_math_score": 78.45
}
```

---

## 🖥️ Streamlit UI

**Features:**

- Interactive form input
- Real-time prediction
- Clean UI with dropdowns and sliders
- API-based communication with FastAPI backend

---

## 🧪 Model Performance

- **Best Model:** Linear Regression
- **R² Score:** ~0.88
- Hyperparameter tuning improved performance significantly

---

## ☁️ Deployment

### Backend — Render

The FastAPI backend is deployed on **[Render](https://render.com)**.

- Render automatically builds and runs the FastAPI app from the repository
- Uses `uvicorn` as the ASGI server
- **Start Command:**
  ```bash
  uvicorn src.api.main:app --host 0.0.0.0 --port 10000
  ```

### Frontend — Streamlit Cloud

The Streamlit frontend is deployed on **[Streamlit Cloud](https://streamlit.io/cloud)**.

- Connects directly to the GitHub repository
- Reads `requirements.txt` for dependencies
- The API base URL in `app.py` should point to the deployed Render backend URL

---

## 🛠️ How to Run Locally

### 1️⃣ Clone repository

```bash
git clone https://github.com/your-username/MLproject.git
cd MLproject
```

### 2️⃣ Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Train model

```bash
python -m src.pipeline.train_pipeline
```

### 5️⃣ Run FastAPI backend

```bash
uvicorn src.api.main:app --reload
```

### 6️⃣ Run Streamlit frontend

```bash
streamlit run app.py
```

---

## 📌 Key Highlights

✔ End-to-end ML pipeline  
✔ Multiple model comparison  
✔ Hyperparameter tuning  
✔ Modular code structure  
✔ FastAPI backend — deployed on **Render**  
✔ Streamlit frontend — deployed on **Streamlit Cloud**  
✔ Production-ready architecture

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and feel free to fork it!
