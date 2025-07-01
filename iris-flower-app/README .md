
# 🌸 Iris Flower Prediction Web App

A simple and interactive machine learning web app built with **Streamlit** to predict the species of an Iris flower using its measurements. This app is deployed from **Google Colab** using **Ngrok** and is beginner-friendly.

---

## 📊 Features

- Predict flower species: **Setosa**, **Versicolor**, or **Virginica**
- Interactive sliders for user input
- Probability scores displayed as a bar chart
- Hosted from Google Colab using `pyngrok`

---

## 🧠 Model Details

- **Algorithm:** Random Forest Classifier  
- **Dataset:** Scikit-learn's Iris dataset  
- **Inputs:**  
  - Sepal Length  
  - Sepal Width  
  - Petal Length  
  - Petal Width  
- **Output:** Predicted Species with Confidence Scores

---

## 📁 Folder Structure

```
iris-flower-app/
│
├── app.py               # Streamlit app file
├── train_model.py       # Script to train and save model
├── iris_model.pkl       # Trained model
└── README.md            # Project documentation
```

---

## 🚀 How to Run in Google Colab

### ✅ Step 1: Install required packages
```python
!pip install streamlit pyngrok scikit-learn matplotlib joblib
```

### ✅ Step 2: Train the model and save it
```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, 'iris_model.pkl')
```

### ✅ Step 3: Create the Streamlit app
```python
app_code = \"\"\" 
import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('iris_model.pkl')
st.title("🌼 Iris Flower Classifier")

sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)
probability = model.predict_proba(input_data)

species = ['Setosa', 'Versicolor', 'Virginica']
st.subheader("Prediction")
st.success(f"Predicted species: {species[prediction[0]]}")

st.subheader("Confidence")
st.bar_chart(probability[0])
\"\"\"
with open("app.py", "w") as f:
    f.write(app_code)
```

### ✅ Step 4: Add Ngrok Auth Token
Get your token from https://dashboard.ngrok.com/get-started/your-authtoken
```python
from pyngrok import conf, ngrok
conf.get_default().auth_token = "PASTE_YOUR_NGROK_TOKEN"
```

### ✅ Step 5: Launch the Streamlit app
```python
!streamlit run app.py &>/content/log.txt & 
public_url = ngrok.connect(port='8501')
print("App is live at:", public_url)
```

---

## 📷 Screenshots

| 🌼 Input Sliders | 📈 Prediction Output |
|------------------|----------------------|
| ![sliders](https://i.imgur.com/ZuwFG0U.png) | ![output](https://i.imgur.com/fA0lhA3.png) |

---

## ✍️ Author

- **Name:** Tanushree Rathod
- **Built with ❤️ using Streamlit in Google Colab**

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).
