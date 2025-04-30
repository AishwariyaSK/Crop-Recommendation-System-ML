# 🌱 Crop Recommendation System

## 📌 Project Overview
The **Crop Recommendation System** is a machine learning-based solution that suggests the most suitable crops for a given location based on environmental factors such as soil type, climate conditions, and water availability. The project includes data preprocessing, feature transformation, model training, and a **Streamlit web app** for easy user interaction.

### 🔹 Dataset
seethapathy, bangaru kamatchi; R, Parvathi (2020), 
“AGRO-METEOROLOGICAL DATA OF INDIAN STATE TAMIL NADU”, 
Mendeley Data, V1, doi: 10.17632/zyyb98msjc.1

### 🔹 Preprocessing & Feature Engineering
- **Feature Transformation**
- **Feature Engineering**
- **Text Processing**
- **Data Augmentation**

### 🔹 Models & Training
The dataset was trained on multiple machine learning models with **GridSearchCV** for hyperparameter tuning:
- **Random Forest (RF)**: **0.998 accuracy** (Best Model ✅)
- **XGBoost (XGB)**
- **Gradient Boosting (GB)**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**

The **Random Forest model** was chosen for deployment due to its highest accuracy and robustness.

---
## ⚙️ Installation & Setup

### 🔹 Prerequisites
Ensure you have **Python 3.8+** installed along with the following dependencies:

```bash
pip install -r requirements.txt
```

### 🔹 Clone the Repository
```bash
git clone https://github.com/AishwariyaSK/Crop-Recommendation-System-ML.git
cd Crop-Recommendation-System-ML
```

### 🔹 Running the Streamlit App
```bash
streamlit run app.py
```

---
## 📂 Project Structure
| File / Directory | Description |
|-----------------|-------------|
| `app.py` | Main **Streamlit** application file |
| `models/` | Contains trained **ML models**, encoders, and scalers |
| `AgriData.csv/` | Dataset used for training |
| `CRS_FINAL.ipynb/` | Jupyter notebooks for **EDA & Model Training** |
| `requirements.txt` | List of required Python packages |

---
## 🚀 Future Enhancements
- Integration with **real-time weather API** for better recommendations.
- More advanced **deep learning models** for increased accuracy.
- Deployment as a **web service API**.

---
## 👩‍💻 Author
[AishwariyaSK](https://github.com/AishwariyaSK) 🚀

Feel free to **⭐ star** the repository if you find this useful! 😊

