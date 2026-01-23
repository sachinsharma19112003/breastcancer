import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------
# App Title
# -----------------------------
st.title("Breast Cancer Prediction - KNN Model")
st.write("Machine Learning Model Deployment using Streamlit")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("breast-cancer.csv")
    df = df.drop(columns=["id"], errors="ignore")
    return df

df = load_data()
st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Label Encoding
# -----------------------------
le = LabelEncoder()
df["diagnosis"] = le.fit_transform(df["diagnosis"])

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=45
)

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# K Selection
# -----------------------------
st.sidebar.header("Model Settings")
k_value = st.sidebar.slider("Select K value", 1, 20, 5)

# -----------------------------
# Train Model
# -----------------------------
knn = KNeighborsClassifier(n_neighbors=k_value)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

# -----------------------------
# Model Performance
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
st.subheader("Model Accuracy")
st.success(f"Accuracy: {accuracy:.2f}")

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
st.write(cm)

st.subheader("Classification Report")
st.text(classification_report(
    y_test,
    y_pred,
    target_names=["Benign", "Malignant"]
))

# -----------------------------
# Accuracy vs K Graph
# -----------------------------
accuracy_list = []

for k in range(1, 21):
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train, y_train)
    y_pred_temp = knn_temp.predict(X_test)
    accuracy_list.append(accuracy_score(y_test, y_pred_temp))

st.subheader("KNN Accuracy vs K")

fig, ax = plt.subplots()
ax.plot(range(1, 21), accuracy_list, marker="o")
ax.set_xlabel("Value of K")
ax.set_ylabel("Accuracy")
ax.set_title("KNN Accuracy vs K")

st.pyplot(fig)

# -----------------------------
# Save Model
# -----------------------------
with open("breast.pkl", "wb") as file:
    pickle.dump(knn, file)

st.info("Model saved successfully as breast.pkl")
