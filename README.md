# **Spectral Data Prediction**  

### Project Overview

This project aims to predict spectral data using multiple machine learning models,
including Random Forest, XGBoost, and Convolutional Neural Networks (CNNs). 
The workflow involves data preprocessing, dimensionality reduction (PCA), model training,
evaluation, and comparison to determine the most effective approach.

## **Installation**  

### **1. Clone the repository:**  
```bash
https://github.com/bhayani-krupa/-mycotoxin_prediction
```

### **2. Create a virtual environment (optional but recommended):**  
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### **3. Install dependencies:**  
```bash
pip install -r requirements.txt
```


## **Running the Application**  

### **Run the Streamlit App:**  
```bash
streamlit run app.py
```
📌 **Features:**  
✅ Upload spectral data (CSV).  
✅ Perform PCA & predict DON concentration using XGBoost.  
✅ Visualize results.  
✅ Download predictions as CSV.  

### **Run the Jupyter Notebook for Training & Analysis:**  
```bash
jupyter notebook
```
- Open `untitled.ipynb` for **data preprocessing, PCA, model training, and evaluation**.  

---

## **Project Structure**  

```
📂 spectral-prediction  
│── app.py                # Streamlit web app  
│── untitled.ipynb  # Jupyter Notebook for training  
│── requirements.txt      # Dependencies  
│── models/  
│   ├── xgboost.pkl       # Trained XGBoost model  
│   ├── pca_model.pkl     # PCA transformation model  
│   ├── scaler.pkl        # Scaler for normalization  
│   ├── feature_names.pkl # Feature names used for training  
│── data/  
│   ├── TASK-ML-INTERN.csv.csv   # Example dataset  
│── README.md             # Project documentation  
```

---

## **Model Performance Summary**  

| Model             | MAE    | RMSE   | R² Score |
|------------------|--------|--------|----------|
| Random Forest    | 1782.61 | 3709.71 | 0.9508   |
| **Tuned XGBoost** | **1612.22** | **3383.59** | **0.9590**   |
| CNN Model        | 1847.93 | 4077.38 | 0.9405   |

---

## **Contributing**  
Contributions are welcome! Feel free to submit **issues or pull requests**. 🚀  

---

## **License**  
This project is licensed under the **MIT License**.  

---
