# 🏠 House Price Prediction System

### Docker • Pipelines • MLflow

---

## 🚀 Overview

This project is an **end-to-end machine learning system** for predicting house prices using structured data.

It is designed with **production-oriented architecture**, combining:

* Modular ML pipelines
* Experiment tracking with MLflow
* Interactive UI using Streamlit
* Containerization using Docker

> This is not just a model — it is a **deployable ML system**.

---

## 🎯 Objectives

* Build a **scalable ML pipeline** for house price prediction
* Track experiments and model performance using MLflow
* Provide a **user-friendly interface** for predictions
* Ensure **reproducibility and portability** using Docker

---

## 🧠 System Architecture

```
Data → Ingestion → Validation → Transformation → Training → Evaluation → Model
                                                                  ↓
                                                         Prediction Pipeline
                                                                  ↓
                                                            Streamlit UI
```

---

## 📦 Project Structure

```
house-price-ml/
│
├── app/
│   ├── streamlit_app.py
│   └── pages/
│       ├── 1_Predict.py
│       └── 2_Model_Info.py
│
├── artifacts/
│   └── models/
│       ├── model.pkl
│       └── preprocessor.pkl
│
├── configs/
│   ├── config.yaml
│   ├── model.yaml
│   └── paths.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── src/
│   └── house_price/
│       ├── pipeline/
│       ├── components/
│       ├── config/
│       └── utils/
│
├── mlruns/                 # MLflow tracking
├── Dockerfile
├── Makefile
├── requirements.txt
├── main.py
└── README.md
```

---

## ⚙️ Pipelines

### 🔹 Training Pipeline

Handles the full ML lifecycle:

* Data Ingestion
* Data Validation
* Data Transformation
* Model Training
* Model Evaluation
* Model Persistence

Run:

```
python main.py
```

---

### 🔹 Prediction Pipeline

* Loads trained model and preprocessor
* Accepts new input data
* Returns prediction

Used in:

* Streamlit app
* Future API layer

---

## 📊 MLflow Integration

This project uses MLflow for experiment tracking.

### Tracks:

* Parameters
* Metrics (RMSE, MAE, R²)
* Model artifacts

### Run MLflow UI:

```
mlflow ui
```

Open:

```
http://localhost:5000
```

---

## 🖥️ Streamlit App

Interactive UI built using Streamlit.

### Features:

* Input house details
* Predict price instantly
* View model information

### Run app:

```
streamlit run app/streamlit_app.py
```

---

## 🐳 Docker Setup

Containerized using Docker for reproducibility.

### Build image:

```
docker build -t house-price-app .
```

### Run container:

```
docker run -p 8501:8501 house-price-app
```

### Access app:

```
http://localhost:8501
```

---

## 🛠️ Makefile Commands

```
make install     # install dependencies
make train       # run training pipeline
make app         # launch Streamlit app
make test        # run tests
make lint        # code linting
make format      # format code
```

---

## 📈 Model Details

* Algorithms:

  * Linear Regression
  * Random Forest Regressor

* Evaluation Metrics:

  * RMSE (primary)
  * MAE
  * R² Score

---

## ⚠️ Known Limitations

* No real-time API (yet)
* No batch prediction support
* No monitoring or logging system
* Limited feature engineering

---

## 🚀 Future Improvements

* Add FastAPI backend
* Deploy to cloud (AWS / GCP / Render)
* Add batch prediction (CSV upload)
* Add model versioning & registry
* Add monitoring and alerting

---

## 💡 Key Takeaways

* End-to-end ML system (not just a notebook)
* Pipeline-driven architecture
* Experiment tracking with MLflow
* UI + Docker packaging

---

## 🏃‍➡️ How to Run

* pip install -r requirements.txt
* python main.py
Run app
* streamlit run app/streamlit_app.py

Build Image and Run Container
* docker build -t house-price-app .
* docker run -p 8501:8501 house-price-app
