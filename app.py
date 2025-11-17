# app.py - Streamlit Calorie Burn Predictor (robust, with file-upload fallback + activity lookup)
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
st.write("DEBUG: app cwd =", os.getcwd())
st.write("DEBUG: files here =", os.listdir())

# Try to load activity lookup table (for activity classification + recommendations)
ACTIVITY_DF = None
try:
    ACTIVITY_DF = pd.read_csv("activities.csv")

    # ✅ Make sure MET is numeric and valid
    ACTIVITY_DF["MET"] = pd.to_numeric(ACTIVITY_DF["MET"], errors="coerce")
    ACTIVITY_DF = ACTIVITY_DF[ACTIVITY_DF["MET"].notna() & (ACTIVITY_DF["MET"] > 0)]

    st.sidebar.success("Activity lookup table found.")
except FileNotFoundError:
    ACTIVITY_DF = None
    st.sidebar.info("Add activities.csv to enable activity type classification & workout recommendations.")




st.set_page_config(page_title="Calorie Burn Predictor", page_icon="🔥", layout="centered")
st.title("🔥 Calorie Burn Prediction")
st.markdown("Enter details and choose an activity (optional). App will show ML prediction and a lookup estimate if available.")

# ---------- Helpers ----------
FEATURES = ['Gender','Age','Height','Weight','Duration','Heart_Rate','Body_Temp']

def encode_gender_val(g):
    if isinstance(g, str):
        g = g.strip().lower()
        if g.startswith('m'): return 1
        if g.startswith('f'): return 0
    try:
        return float(g)
    except:
        return 0.5

def find_activity_lookup():
    # try common filenames
    candidates = ["exercise.csv", "exercise_dataset.csv", "exercise_dataset (1).csv", "exercise_dataset.csv"]
    for c in candidates:
        if os.path.exists(c):
            try:
                return pd.read_csv(c)
            except Exception:
                pass
    return None

def activity_estimate(df_activity, activity_name, weight_kg, duration_min):
    # df_activity assumed to have 'Activity, Exercise or Sport (1 hour)' and 'Calories per kg' or similar
    cols = [c for c in df_activity.columns]
    # find activity column name heuristically
    activity_col = None
    cal_per_kg_col = None
    for c in cols:
        if 'activity' in c.lower() or 'exercise' in c.lower() or 'sport' in c.lower():
            activity_col = c
        if 'calori' in c.lower() and 'kg' in c.lower():
            cal_per_kg_col = c
    if activity_col is None:
        # fallback: first column
        activity_col = cols[0]
    if cal_per_kg_col is None:
        # fallback: try last numeric column
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df_activity[c])]
        cal_per_kg_col = numeric_cols[-1] if numeric_cols else None

    # find row matching activity_name
    if cal_per_kg_col is None:
        return None
    matches = df_activity[df_activity[activity_col].astype(str).str.contains(activity_name, case=False, na=False)]
    if matches.empty:
        # try fuzzy: take first row
        row = df_activity.iloc[0]
    else:
        row = matches.iloc[0]
    try:
        cal_per_kg = float(row[cal_per_kg_col])
    except:
        return None
    # dataset is per hour per kg; scale by weight and duration
    calories_per_hour = cal_per_kg * weight_kg
    total = calories_per_hour * (duration_min / 60.0)
    return float(total)

# ---------- Load model & scaler (or allow upload) ----------
def load_pickle_from_file(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

model = None
scaler = None
MODEL_NAME = "calorie_model.pkl"
SCALER_NAME = "scaler.pkl"

# --- Load model and scaler (robust, with error reporting) ---
model = None
scaler = None
MODEL_NAME = "calorie_model.pkl"
SCALER_NAME = "scaler.pkl"

model_path = os.path.join(os.getcwd(), MODEL_NAME)
scaler_path = os.path.join(os.getcwd(), SCALER_NAME)

if os.path.isfile(model_path) and os.path.isfile(scaler_path):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        st.success("✅ Model and scaler loaded successfully.")
    except Exception as e:
        model = None
        scaler = None
        st.error(f"Failed to load model or scaler: {type(e).__name__}: {e}")
        st.stop()
else:
    st.warning("Model or scaler not found in app folder. You can upload them below (upload both files).")


# If missing, allow upload
if model is None or scaler is None:
    st.warning("Model or scaler not found in app folder. You can upload them below (upload both files).")
    uploaded_model = st.file_uploader("Upload calorie_model.pkl (if missing)", type=["pkl","sav"], key="m")
    uploaded_scaler = st.file_uploader("Upload scaler.pkl (if missing)", type=["pkl","sav"], key="s")
    if uploaded_model is not None and uploaded_scaler is not None:
        try:
            model = pickle.load(uploaded_model)
            scaler = pickle.load(uploaded_scaler)
            st.success("Model and scaler loaded from uploaded files.")
        except Exception as e:
            st.error("Failed to load uploaded pickles: " + str(e))
            st.stop()
    else:
        st.info("If you already have the files in the folder, skip uploading. Otherwise upload both to proceed.")
        # do not stop here; user can still input but we stop prediction below if model missing

# ---------- Activity lookup dataset (optional) ----------
activity_df = find_activity_lookup()
if activity_df is not None:
    st.sidebar.success("Activity lookup table found.")
    # offer a sample of available activities
    act_col = activity_df.columns[0]
    activities = activity_df[act_col].astype(str).tolist()
else:
    activities = []

# ---------- Inputs ----------
st.sidebar.header("User inputs")
gender = st.sidebar.selectbox("Gender", ["Male","Female","Other"])
age = st.sidebar.slider("Age (years)", 10, 100, 25)
height = st.sidebar.slider("Height (cm)", 120, 220, 170)
weight = st.sidebar.slider("Weight (kg)", 30, 140, 70)
duration = st.sidebar.slider("Duration (minutes)", 5, 240, 30)
heart_rate = st.sidebar.slider("Average Heart Rate (bpm)", 60, 200, 120)
body_temp = st.sidebar.slider("Body Temperature (°C)", 34.0, 41.0, 37.0, step=0.1)

st.sidebar.markdown("---")
activity_choice = None
if activities:
    activity_choice = st.sidebar.selectbox("Select activity (optional)", [""] + activities)
else:
    activity_choice = st.sidebar.text_input("Activity (optional) - free text", "")

show_details = st.checkbox("Show preprocessing & model details", value=False)

# ---------- Prepare input DataFrame (matching FEATURE names) ----------
gender_val = encode_gender_val(gender)
input_df = pd.DataFrame([[gender_val, age, height, weight, duration, heart_rate, body_temp]], columns=FEATURES)

if show_details:
    st.write("Input dataframe:")
    st.dataframe(input_df)

# ---------- Predict using ML model (if loaded) ----------
ml_prediction = None
if model is None or scaler is None:
    st.error("Model and scaler are not loaded. Upload or place 'calorie_model.pkl' and 'scaler.pkl' in the app folder to enable ML prediction.")
else:
    try:
        # scaler may expect DataFrame with column names or numpy; ensure shape matches
        try:
            input_scaled = scaler.transform(input_df)  # prefer DataFrame so feature-name warning avoided
        except Exception:
            input_scaled = scaler.transform(input_df.values)
        ml_prediction = float(model.predict(input_scaled)[0])
    except Exception as e:
        st.error("Prediction failed: " + str(e))
        ml_prediction = None

# ---------- Lookup estimate if activity table present ----------
lookup_prediction = None
if activity_df is not None:
    act_name = activity_choice if activity_choice != "" else ""
    weight_kg = float(weight)
    if act_name:
        lookup_prediction = activity_estimate(activity_df, act_name, weight_kg, duration)

# ---------- Display results ----------
st.header("Results")
if ml_prediction is not None:
    st.success(f"ML Estimate: **{ml_prediction:.2f} kcal**")
else:
    st.info("ML Estimate: model not available")

if lookup_prediction is not None:
    st.info(f"Lookup Estimate ({act_name}): **{lookup_prediction:.2f} kcal**")

# Show combined note
if ml_prediction is not None and lookup_prediction is not None:
    diff = ml_prediction - lookup_prediction
    st.write(f"Difference (ML - lookup): {diff:.1f} kcal")

# ---------- BMI + duration plot ----------
# BMI
try:
    h_m = height / 100.0
    bmi = weight / (h_m*h_m) if h_m>0 else None
except:
    bmi = None

if bmi:
    if bmi < 18.5: cat = "Underweight"
    elif bmi < 25: cat = "Normal"
    elif bmi < 30: cat = "Overweight"
    else: cat = "Obese"
    st.write(f"**BMI:** {bmi:.1f} — *{cat}*")

# Duration vs predicted calories plot: vary duration keeping other inputs fixed
if model is not None and scaler is not None:
    durations = list(range(5, 121, 5))
    rows = []
    base = input_df.copy()
    for d in durations:
        base.loc[0,'Duration'] = d
        try:
            scaled = scaler.transform(base)
        except:
            scaled = scaler.values
        pred = float(model.predict(scaled)[0])
        rows.append(pred)
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(durations, rows, linewidth=2)
    if ml_prediction is not None:
        ax.scatter([duration],[ml_prediction], color='red', zorder=5)
        ax.annotate(f"{ml_prediction:.0f} kcal", xy=(duration, ml_prediction), xytext=(duration+3, ml_prediction+10),
                    arrowprops=dict(arrowstyle="->", alpha=0.6), fontsize=9)
    ax.set_xlabel("Duration (minutes)")
    ax.set_ylabel("Predicted calories (kcal)")
    ax.set_title("Predicted calories vs duration")
    ax.grid(alpha=0.25)
    st.pyplot(fig)

# Intensity label
if ml_prediction is not None:
    if ml_prediction < 150: intensity = "Light"
    elif ml_prediction < 400: intensity = "Moderate"
    else: intensity = "Intense"
    st.info(f"Activity intensity estimate: **{intensity}**")
# ---------- Activity Type Estimate (Option A) ----------
if ACTIVITY_DF is not None and ml_prediction is not None and duration > 0 and weight > 0:
    # kcal/min from model
    kcal_per_min = ml_prediction / duration
    # MET estimate using standard formula: kcal/min ≈ 0.0175 * MET * weight_kg
    met_est = kcal_per_min / (0.0175 * weight)

   # ---------- Activity Type Estimate (Option A) ----------
if ACTIVITY_DF is not None and ml_prediction is not None and duration > 0 and weight > 0:
    # kcal/min from model
    kcal_per_min = ml_prediction / duration
    # MET estimate using standard formula: kcal/min ≈ 0.0175 * MET * weight_kg
    met_est = kcal_per_min / (0.0175 * weight)

    # Find closest activity by MET (use POSITION, not index label)
    diff = (ACTIVITY_DF["MET"] - met_est).abs()
    closest_pos = int(diff.to_numpy().argmin())   # 0, 1, 2, ... position
    act_row = ACTIVITY_DF.iloc[closest_pos]

    st.subheader("Estimated activity type")

    # Ensure we have scalar values
    met_est_val = float(met_est)
    met_match_val = float(act_row["MET"])
    activity_name = str(act_row["Activity"])

    st.write(
        f"Estimated MET: **{met_est_val:.1f}**  \n"
        f"Closest match: **{activity_name}** (MET {met_match_val:.1f})"
    )



# ---------- Workout Recommendations (Option C) ----------
st.subheader("Workout recommendations")
target_kcal = st.slider(
    "Target calories to burn",
    min_value=100,
    max_value=1000,
    value=300,
    step=50,
    help="Estimated duration for each activity to reach this target."
)

if ACTIVITY_DF is not None and weight > 0:
    rec_df = ACTIVITY_DF.copy()

    # ✅ Extra safety: ensure MET is numeric and valid
    rec_df["MET"] = pd.to_numeric(rec_df["MET"], errors="coerce")
    rec_df = rec_df[rec_df["MET"].notna() & (rec_df["MET"] > 0)]

    if not rec_df.empty:
        # Minutes needed per activity to hit target_kcal
        rec_df["Minutes_needed"] = target_kcal / (0.0175 * rec_df["MET"] * weight)

        # Remove any infinite / NaN minutes
        rec_df = rec_df.replace([np.inf, -np.inf], np.nan)
        rec_df = rec_df.dropna(subset=["Minutes_needed"])

        rec_df = rec_df.sort_values("Minutes_needed")

        st.write("Estimated duration for each activity:")
        st.dataframe(
            rec_df[["Activity", "MET", "Minutes_needed"]]
            .rename(columns={"Minutes_needed": "Minutes"})
            .round({"Minutes": 1})
        )

        st.markdown("**Quick options (shortest time):**")
        for _, row in rec_df.head(3).iterrows():
            st.write(
                f"- **{row['Activity']}** → about **{row['Minutes_needed']:.1f} min**"
            )
    else:
        st.info("No valid MET values found in activities.csv.")
elif ACTIVITY_DF is None:
    st.info("Add activities.csv to see workout recommendations.")



# ---------- Optional: show linear model coefficients if available ----------

try:
    if hasattr(model, "coef_") and len(model.coef_.shape) == 1:
        coefs = model.coef_.ravel()
        coef_df = pd.DataFrame({"feature": FEATURES, "coef": coefs})
        coef_df = coef_df.reindex(coef_df['coef'].abs().sort_values(ascending=False).index)
        st.markdown("### Model coefficients (linear model)")
        st.table(coef_df.style.format({"coef":"{:.3f}"}))
except Exception:
    pass

st.markdown("---")
st.caption("Tip: If model/scaler are missing, upload them in the sidebar or place the .pkl files in the same folder as this app.")
