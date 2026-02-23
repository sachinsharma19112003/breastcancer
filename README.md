Sachin 👍 ye content bilkul sahi hai — aap ise direct `README.md` me paste kar sakte ho.

Lekin main isko thoda **aur professional + GitHub-ready format** me polish karke de raha hoon (badges + structure improve karke). Ye version recruiters ko aur impress karega 👇

---

# 🩺 Breast Cancer Prediction Web App using KNN

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![ML](https://img.shields.io/badge/Machine%20Learning-KNN-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Project Overview

This project is a Machine Learning–based web application that predicts whether a breast tumor is **Benign (Non-Cancerous)** or **Malignant (Cancerous)** using the K-Nearest Neighbors (KNN) algorithm.

The application is built and deployed using **Streamlit**, providing an interactive interface where users can dynamically adjust the value of **K** and evaluate model performance in real-time.

This project demonstrates end-to-end ML workflow:

* Data preprocessing
* Feature scaling
* Model training
* Performance evaluation
* Visualization
* Web deployment

---

## 🎯 Problem Statement

Breast cancer is one of the most common cancers worldwide. Early detection significantly increases survival rates.

The objective of this project is to develop a classification model capable of predicting tumor diagnosis using structured medical features.

---

## 📊 Dataset Description

The dataset contains multiple features computed from digitized images of breast mass cell nuclei.

### 🔎 Important Features:

* Radius
* Texture
* Perimeter
* Area
* Smoothness
* Compactness
* Concavity
* Symmetry
* Fractal Dimension

### 🎯 Target Variable:

* **0 → Benign**
* **1 → Malignant**

### 🧹 Data Preprocessing Steps:

* Removed unnecessary columns (e.g., ID)
* Applied Label Encoding to diagnosis column
* Feature scaling using StandardScaler

---

## ⚙️ Machine Learning Workflow

### 1️⃣ Data Loading

Dataset loaded using Pandas.

### 2️⃣ Data Cleaning

Removed irrelevant columns.

### 3️⃣ Encoding

Converted categorical diagnosis values to numerical format.

### 4️⃣ Train-Test Split

* 80% Training Data
* 20% Testing Data

### 5️⃣ Feature Scaling

Applied StandardScaler because KNN is distance-based.

### 6️⃣ Model Training

Used Scikit-learn’s `KNeighborsClassifier`.

### 7️⃣ Model Evaluation

Evaluated using:

* Accuracy Score
* Confusion Matrix
* Classification Report (Precision, Recall, F1-Score)

### 8️⃣ Visualization

Plotted **Accuracy vs K** graph to identify optimal K value.

### 9️⃣ Model Saving

Saved trained model as:

```
breast.pkl
```

---

## 🚀 Application Features

✅ Interactive sidebar to select K value (1–20)
✅ Real-time model training
✅ Accuracy display
✅ Confusion Matrix output
✅ Detailed classification report
✅ Accuracy vs K graph visualization
✅ Model persistence using Pickle
✅ Clean Streamlit web interface

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Streamlit
* Pickle

---

## 📂 Project Structure

```
├── app.py
├── breast-cancer.csv
├── breast.pkl
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run the Project

### 1️⃣ Clone Repository

```
git clone gh repo clone sachinsharma19112003/breastcancer
```

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run Streamlit App

```
streamlit run app.py
```

---

## 📈 Model Performance

The model achieves high accuracy in predicting tumor diagnosis.
Performance varies based on selected K value.

The Accuracy vs K visualization helps determine the optimal K for best classification results.

---

## 🌐 Deployment Options

The project can be deployed on:

* Streamlit Cloud
* Render
* Railway

---

## 👨‍💻 Author

**Sachin Sharma**
MCA Student | Machine Learning Enthusiast
Aspiring Data Scientist 🚀


