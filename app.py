
# app.py - Calorie Burn Predictor (edited by assistant)
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Calorie Burn Predictor", page_icon="🔥", layout="centered")

# ---------- Helpers & constants ----------
FEATURES = ['Gender','Age','Height','Weight','Duration','Heart_Rate','Body_Temp']

def encode_gender_val(g):
    if isinstance(g, str):
        g = g.strip().lower()
        if g.startswith('m'): return 1.0
        if g.startswith('f'): return 0.0
    try:
        return float(g)
    except:
        return 0.5

def activity_estimate(df_activity, activity_name, weight_kg, duration_min):
    if df_activity is None:
        return None
    cols = [c for c in df_activity.columns]
    activity_col = None
    cal_per_kg_col = None
    for c in cols:
        if 'activity' in c.lower() or 'exercise' in c.lower() or 'sport' in c.lower():
            activity_col = c
        if 'calori' in c.lower() and 'kg' in c.lower():
            cal_per_kg_col = c
    if activity_col is None:
        activity_col = cols[0]
    if cal_per_kg_col is None:
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df_activity[c])]
        cal_per_kg_col = numeric_cols[-1] if numeric_cols else None
    if cal_per_kg_col is None:
        return None
    matches = df_activity[df_activity[activity_col].astype(str).str.contains(activity_name, case=False, na=False)]
    if matches.empty:
        row = df_activity.iloc[0]
    else:
        row = matches.iloc[0]
    try:
        cal_per_kg = float(row[cal_per_kg_col])
    except:
        return None
    calories_per_hour = cal_per_kg * weight_kg
    total = calories_per_hour * (duration_min / 60.0)
    return float(total)

# ---------- Load activity CSV if present ----------
ACTIVITY_DF = None
try:
    ACTIVITY_DF = pd.read_csv("activities.csv")
    # ensure MET column if present is numeric
    if "MET" in ACTIVITY_DF.columns:
        ACTIVITY_DF["MET"] = pd.to_numeric(ACTIVITY_DF["MET"], errors="coerce")
        ACTIVITY_DF = ACTIVITY_DF[ACTIVITY_DF["MET"].notna() & (ACTIVITY_DF["MET"] > 0)]
    st.sidebar.success("Activity lookup table loaded." )
except FileNotFoundError:
    st.sidebar.info("Add activities.csv to enable activity lookup & recommendations.")

# ---------- UI: Sidebar inputs ----------
st.sidebar.header("User inputs")
gender = st.sidebar.selectbox("Gender", ["Male","Female","Other"])
age = st.sidebar.slider("Age (years)", 10, 100, 25)
height = st.sidebar.slider("Height (cm)", 120, 220, 170)
weight = st.sidebar.slider("Weight (kg)", 30, 140, 70)
duration = st.sidebar.slider("Duration (minutes)", 5, 240, 30)
heart_rate = st.sidebar.slider("Average Heart Rate (bpm)", 40, 200, 120)
body_temp = st.sidebar.slider("Body Temperature (°C)", 34.0, 41.0, 37.0, step=0.1)

st.sidebar.markdown("---")
activity_choice = None
if ACTIVITY_DF is not None:
    act_col = ACTIVITY_DF.columns[0]
    activities = ACTIVITY_DF[act_col].astype(str).tolist()
    activity_choice = st.sidebar.selectbox("Select activity (optional)", [""] + activities)
else:
    activity_choice = st.sidebar.text_input("Activity (optional) - free text", "")

show_details = st.sidebar.checkbox("Show preprocessing & model details", value=False)

# ---------- Model & scaler paths ----------
MODEL_NAME = "calorie_model.pkl"
SCALER_NAME = "scaler.pkl"
model = None
scaler = None

# Attempt load from disk
def _load_pickle(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

model = _load_pickle(MODEL_NAME)
scaler = _load_pickle(SCALER_NAME)

if model is not None and scaler is not None:
    st.sidebar.success("Model & scaler loaded from app folder.")
else:
    st.sidebar.info("Model or scaler missing. Upload below or place files in app folder.")

# allow upload fallback
uploaded_model = st.sidebar.file_uploader("Upload calorie_model.pkl (if missing)", type=["pkl","sav"], key="m")
uploaded_scaler = st.sidebar.file_uploader("Upload scaler.pkl (if missing)", type=["pkl","sav"], key="s")
if uploaded_model is not None and uploaded_scaler is not None and (model is None or scaler is None):
    try:
        model = pickle.load(uploaded_model)
        scaler = pickle.load(uploaded_scaler)
        st.sidebar.success("Model and scaler loaded from uploaded files.")
    except Exception as e:
        st.sidebar.error("Failed to load uploaded pickles: " + str(e))

# ---------- Prepare input and predict ----------
def prepare_input_df(gender_val, age, height, weight, duration, heart_rate, body_temp):
    return pd.DataFrame([[gender_val, age, height, weight, duration, heart_rate, body_temp]], columns=FEATURES)

gender_val = encode_gender_val(gender)
input_df = prepare_input_df(gender_val, age, height, weight, duration, heart_rate, body_temp)

if show_details:
    st.write("Input dataframe:")
    st.dataframe(input_df)

# ML prediction block
ml_prediction = None
ml_error_msg = None
if model is None or scaler is None:
    st.info("ML model or scaler not available. Upload or place 'calorie_model.pkl' and 'scaler.pkl' in the app folder to enable ML prediction.")
else:
    try:
        try:
            input_scaled = scaler.transform(input_df)
        except Exception:
            input_scaled = scaler.transform(input_df.values)
        raw_pred = model.predict(input_scaled)
        if isinstance(raw_pred, (list, tuple, np.ndarray)):
            ml_prediction = float(np.array(raw_pred).ravel()[0])
        else:
            ml_prediction = float(raw_pred)
        if show_details:
            st.write("DEBUG: Scaled input (first row):", input_scaled[0].tolist() if hasattr(input_scaled, '__len__') else str(input_scaled))
            st.write("DEBUG: Raw model predict output:", raw_pred)
    except Exception as e:
        ml_prediction = None
        ml_error_msg = f"ML prediction failed: {type(e).__name__}: {e}"
        st.error(ml_error_msg)

# ---------- MET lookup estimate ----------
lookup_prediction = None
act_name = ""
if ACTIVITY_DF is not None:
    act_name = activity_choice if activity_choice != "" else ""
    try:
        if act_name:
            lookup_prediction = activity_estimate(ACTIVITY_DF, act_name, float(weight), duration)
    except Exception as e:
        lookup_prediction = None
        if show_details:
            st.write("DEBUG activity estimate error:", e)

# ---------- Display Results ----------
st.title("🔥 Calorie Burn Prediction")
st.markdown("Enter details and choose an activity (optional). App shows ML prediction and MET lookup estimate.")

st.header("Results")
if ml_prediction is not None:
    st.success(f"ML Estimate: **{ml_prediction:.2f} kcal**")
else:
    st.info("ML Estimate: model not available or failed.")

if lookup_prediction is not None:
    st.info(f"Lookup Estimate ({act_name}): **{lookup_prediction:.2f} kcal**")
else:
    if activity_choice == "":
        st.info("Lookup Estimate: no activity selected.")
    else:
        st.info("Lookup Estimate: activity lookup missing or failed.")

if ml_prediction is not None and lookup_prediction is not None:
    diff = ml_prediction - lookup_prediction
    st.write(f"Difference (ML - lookup): {diff:.1f} kcal")

# BMI calculation
try:
    h_m = height / 100.0
    bmi = weight / (h_m*h_m) if h_m>0 else None
except:
    bmi = None
if bmi is not None:
    if bmi < 18.5: cat = "Underweight"
    elif bmi < 25: cat = "Normal"
    elif bmi < 30: cat = "Overweight"
    else: cat = "Obese"
    st.write(f"**BMI:** {bmi:.1f} — *{cat}*")

# Plot predicted vs duration (if ML available)
if model is not None and scaler is not None and ml_prediction is not None:
    durations = list(range(5, 121, 5))
    rows = []
    base = input_df.copy()
    for d in durations:
        base.loc[0,'Duration'] = d
        try:
            scaled = scaler.transform(base)
        except Exception:
            scaled = scaler.values
        pred = float(model.predict(scaled)[0])
        rows.append(pred)
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(durations, rows, linewidth=2)
    ax.scatter([duration],[ml_prediction], color='red', zorder=5)
    ax.set_xlabel("Duration (minutes)")
    ax.set_ylabel("Predicted calories (kcal)")
    ax.set_title("Predicted calories vs duration")
    ax.grid(alpha=0.25)
    st.pyplot(fig)

# Activity recommendations (MET based)
st.subheader("Workout recommendations")
target_kcal = st.slider("Target calories to burn", min_value=100, max_value=1000, value=300, step=50)

if ACTIVITY_DF is not None and weight > 0:
    rec_df = ACTIVITY_DF.copy()
    rec_df["MET"] = pd.to_numeric(rec_df["MET"], errors="coerce")
    rec_df = rec_df.dropna(subset=["MET"])
    rec_df = rec_df[rec_df["MET"] > 0]
    rec_df["Minutes"] = target_kcal / (0.0175 * rec_df["MET"] * float(weight))
    rec_df = rec_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Minutes"])
    rec_df = rec_df.sort_values("Minutes")
    st.write("Estimated duration for each activity:")
    st.dataframe(rec_df[["Activity","MET","Minutes"]].round({"Minutes":1}).reset_index(drop=True))
    st.markdown("**Quick options (shortest time):**")
    for _, row in rec_df.head(3).iterrows():
        st.write(f"- **{row['Activity']}** → about **{row['Minutes']:.1f} min**")
    # Focus planner
    st.markdown("### Focus on a single activity")
    focus_activity = st.selectbox("Choose activity", rec_df["Activity"].tolist())
    if focus_activity:
        r = rec_df.loc[rec_df["Activity"] == focus_activity].iloc[0]
        st.info(f"To burn **{target_kcal} kcal** doing **{focus_activity}** (MET {r['MET']}), you need about **{r['Minutes']:.1f} minutes**.")
else:
    st.info("Add activities.csv to see workout recommendations.")

# Optional: show linear model coefficients if available
try:
    if hasattr(model, "coef_") and len(np.array(model.coef_).shape) == 1:
        coefs = np.array(model.coef_).ravel()
        coef_df = pd.DataFrame({"feature": FEATURES, "coef": coefs})
        coef_df = coef_df.reindex(coef_df['coef'].abs().sort_values(ascending=False).index)
        st.markdown("### Model coefficients (linear model)")
        st.table(coef_df.style.format({"coef":"{:.3f}"}))
except Exception:
    pass

st.markdown("---")
st.caption("Tip: If model/scaler are missing, upload them in the sidebar or place the .pkl files in the same folder as this app.")



