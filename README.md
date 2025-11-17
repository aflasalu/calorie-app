# 🔥 Calorie Burn Predictor (Streamlit + Machine Learning)

A simple and interactive **Calorie Burn Prediction App** built using **Python, Streamlit, and Machine Learning**.  
It predicts calories burned based on user inputs and also includes:

- Activity intensity classification  
- Activity type estimation using MET values  
- Workout duration recommendations  
- Optional lookup table for activities  
- Plot of predicted calories vs duration  
- BMI calculation  
- Model coefficients (if using a linear model)

---

## 🚀 Features

### ✅ Machine Learning Prediction  
Uses a trained ML model (`calorie_model.pkl`) + scaler (`scaler.pkl`) to estimate calories burned.

### ✅ Activity Lookup Table  
If `activities.csv` is included, the app can:
- Estimate MET  
- Identify closest activity  
- Recommend workouts by time needed to burn target calories  

### ✅ BMI Calculation  
Automatically computes your BMI and category.

### ✅ Visualization  
Shows a graph of predicted calories vs duration.

---

## 📦 Folder Structure

