# ==============================
# Imports
# ==============================
from matplotlib.figure import Figure
import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
import urllib.parse

# ---------- Page setup ----------
st.set_page_config(page_title="AI Health Risk Predictor", layout="wide")

# =======================
# Function to set background
# =======================
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .card {{
            background: rgba(255, 255, 255, 0.85);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            margin-bottom: 20px;
        }}
        .stButton>button {{
            background: linear-gradient(90deg, #42a5f5, #1e88e5);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 10px 20px;
            font-size: 16px;
            font-weight: bold;
            transition: 0.3s;
        }}
        .stButton>button:hover {{
            background: linear-gradient(90deg, #1e88e5, #1565c0);
            transform: scale(1.05);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# =======================
# Apply background
# =======================
set_background("web-back.png")

st.set_page_config(
    page_title="Disease Prediction App",
    page_icon="icon.ico",  # your .ico file
    layout="wide"
)

# ---------- Helper for safe loading ----------
def load_pickle(path:str, label:str):
    try:
        obj = joblib.load(path)
        return obj
    except Exception as e:
        st.error(f"❌ Could not load **{label}** from:\n`{path}`\n\n**Error:** {e}")
        st.stop()

# ---------- Load your models/scalers ----------
HEART_MODEL_PATH  = r"D:\disease_pred.py\mediscope.ai\heart_disease_model.pkl"
HEART_SCALER_PATH = r"D:\disease_pred.py\mediscope.ai\heart_scaler.pkl"
heart_model  = load_pickle(HEART_MODEL_PATH,  "Heart model")
heart_scaler = load_pickle(HEART_SCALER_PATH, "Heart scaler")

LUNG_MODEL_PATH   = r"D:\disease_pred.py\mediscope.ai\lungs_cancer_model.pkl"
LUNG_SCALER_PATH  = r"D:\disease_pred.py\mediscope.ai\lung_scaler.pkl"
lung_model  = load_pickle(LUNG_MODEL_PATH,  "Lung model")
lung_scaler = load_pickle(LUNG_SCALER_PATH, "Lung scaler")

DIAB_MODEL_PATH   = r"D:\disease_pred.py\mediscope.ai\diabetes_model.pkl"
DIAB_SCALER_PATH  = r"D:\disease_pred.py\mediscope.ai\Diabetes_scaler.pkl"
diab_model  = load_pickle(DIAB_MODEL_PATH,  "Diabetes model")
diab_scaler = load_pickle(DIAB_SCALER_PATH, "Diabetes scaler")


st.title("🩺 Mediscope AI - Disease Prediction")

search_query = st.text_input("🔍 Search for a disease (Heart Disease, Lung Cancer, Diabetes)")

# Convert to lowercase for flexibility
search_query = search_query.strip().lower()

if search_query in ["heart disease", "heart", "cardio"]:
    disease_choice = "Heart Disease"
elif search_query in ["lung cancer", "lung", "cancer"]:
    disease_choice = "Lung Cancer"
elif search_query in ["diabetes", "sugar"]:
    disease_choice = "Diabetes"
else:
    disease_choice = None

gender_map = {"Male": 1, "Female": 0, "Other": 2}
smoking_map = {
    "never": 3, "former": 1, "current": 0,
    "No Info": 2, "ever": 4, "not current": 5
}
numeric_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
# =======================
# Navigation Bar
# =======================
selected = option_menu(
    menu_title=None,
    options=["🏠 Home", "ℹ️ About Us", "📞 Contact Us", "📘 More Info"],
    icons=["house", "info-circle", "telephone", "book"],
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#339cb1"},
        "icon": {"color": "#0072ff", "font-size": "20px"},
        "nav-link": {"font-size": "16px", "text-align": "center", "margin":"0px"},
        "nav-link-selected": {"background-color": "#0072ff", "color": "white"},
    }
)

# =======================
# Main Sections
# =======================
if selected == "🏠 Home":
    st.markdown("<h1 style='text-align:center'>⚕️ Mediscope AI Health Risk Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center'>Predict risks for <b>Heart Disease</b>, <b>Lung Cancer</b>, <b>Diabetes</b>, and <b>Covid-19</b>.</p>", unsafe_allow_html=True)
    st.image("D:\disease_pred.py\Mediscope AI.png", use_container_width=True)

    # Sub-menu for diseases
    st.subheader("🔍 Choose a Predictor")
    choice = st.radio("", ["Heart Disease", "Lung Cancer", "Diabetes", "Covid-19"], horizontal=True)

    # === Your existing prediction blocks remain the same ===
    # paste your Heart / Lung / Diabetes / Covid prediction code here under respective if blocks
    # Example:
    # ---------- HEART ----------
    if choice == "Heart Disease":
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("💗 Heart Disease Risk Assessment")

        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age (years)", 20, 100, 45)
            sex = st.selectbox("Sex", [0,1], format_func=lambda x: "Male" if x==1 else "Female")
            cp = st.selectbox("Chest Pain Type (cp)", [0,1,2,3])
            trestbps = st.number_input("Resting BP (mm Hg)", 80, 220, 120)
            chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
        with col2:
            fbs = st.selectbox("Fasting Sugar ≥120 mg/dl", [0,1])
            restecg = st.selectbox("Resting ECG", [0,1,2])
            thalach = st.number_input("Max Heart Rate", 60, 220, 150)
            exang = st.selectbox("Exercise-Induced Angina", [0,1])
            oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
        with col3:
            slope = st.selectbox("ST Slope", [0,1,2])
            ca = st.selectbox("Major Vessels (0–3)", [0,1,2,3])
            thal = st.selectbox("Thalassemia (thal)", [0,1,2,3])

        if st.button("🔍 Predict"):
            sex_val = 1 if sex == "Male" else 0
            input_data = [age, sex_val, chol, trestbps, thalach, cp]
            
            # Prepare input for prediction
            X = np.array([[
                age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                exang, oldpeak, slope, ca, thal
            ]])
            X_scaled = heart_scaler.transform(X)
            pred = heart_model.predict(X_scaled)[0]
            prob = heart_model.predict_proba(X_scaled)[0][1]
            
            # Show result
            if pred == 1:
                st.error(f"⚠️ High Risk of Heart Disease (Probability: {prob*100:.2f}%)")
            else:
                st.success(f"✅ Low Risk of Heart Disease (Probability: {prob*100:.2f}%)")
            
            # Show input summary
            st.subheader("📊 Your Input Summary")
            st.write(f"- Age: {age}")
            st.write(f"- Sex: {sex}")
            st.write(f"- Cholesterol: {chol}")
            st.write(f"- Blood Pressure: {trestbps}")
            st.write(f"- Max Heart Rate: {thalach}")
            st.write(f"- Chest Pain Type: {cp}")
            
            # Recommendations
            st.subheader("💡 Health Recommendations")
            if pred == 1:
                st.write("- Maintain a healthy diet (low cholesterol, low salt).")
                st.write("- Regular exercise (at least 30 mins/day).")
                st.write("- Consult a cardiologist for further tests.")
            else:
                st.write("- Keep up a healthy lifestyle.")
                st.write("- Do regular check-ups every 6-12 months.")

                # Graph
            # 📊 Responsive Probability Pie Chart
            fig, ax = plt.subplots()
            ax.pie([prob, 1-prob], labels=["Risk", "No Risk"], autopct='%1.1f%%', startangle=90, colors=["red", "green"])
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(fig)

            # 📊 Radar Chart: Patient vs Healthy Range
            import plotly.graph_objects as go  # <-- Add this import
            input_data = {"Age": age, "BP": trestbps, "Cholesterol": chol, "MaxHR": thalach}
            healthy_ranges = {"Age": 40, "BP": 120, "Cholesterol": 180, "MaxHR": 160}

            radar_df = pd.DataFrame({
                "Metric": list(input_data.keys()),
                "Patient": list(input_data.values()),
                "Healthy": list(healthy_ranges.values())
            })
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_df["Patient"], theta=radar_df["Metric"], fill="toself", name="Patient"
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_df["Healthy"], theta=radar_df["Metric"], fill="toself", name="Healthy"
            ))
            fig_radar.update_layout(title="Patient vs Healthy Comparison", polar=dict(radialaxis=dict(visible=True)))
            st.plotly_chart(fig_radar, use_container_width=True)

        
        # ---------- LUNG ----------
        # =========================
        # Input Form
        # =========================
    elif choice == "Lung Cancer":
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🫁 Lung Cancer Risk Assessment")
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                gender = st.selectbox("Gender", ["Female", "Male"])
                age = st.number_input("Age", min_value=1, max_value=120, value=50)

                smoking = st.selectbox("Smoking", ["No", "Yes"])
                yellow_fingers = st.selectbox("Yellow Fingers", ["No", "Yes"])
                anxiety = st.selectbox("Anxiety", ["No", "Yes"])
                peer_pressure = st.selectbox("Peer Pressure", ["No", "Yes"])
                chronic_disease = st.selectbox("Chronic Disease", ["No", "Yes"])

            with col2:
                fatigue = st.selectbox("Fatigue", ["No", "Yes"])
                allergy = st.selectbox("Allergy", ["No", "Yes"])
                wheezing = st.selectbox("Wheezing", ["No", "Yes"])
                alcohol = st.selectbox("Alcohol Consuming", ["No", "Yes"])
                coughing = st.selectbox("Coughing", ["No", "Yes"])
                short_breath = st.selectbox("Shortness of Breath", ["No", "Yes"])
                swallowing = st.selectbox("Swallowing Difficulty", ["No", "Yes"])
                chest_pain = st.selectbox("Chest Pain", ["No", "Yes"])

            submitted = st.form_submit_button("🔍 Predict")

        # =========================
        # Prediction Logic
        # =========================
        if submitted:
            # Encode values
            gender_val = 1 if gender == "Male" else 0
            def encode(val): return 2 if val == "Yes" else 1

            features = [
                gender_val, age,
                encode(smoking), encode(yellow_fingers), encode(anxiety), encode(peer_pressure),
                encode(chronic_disease), encode(fatigue), encode(allergy), encode(wheezing),
                encode(alcohol), encode(coughing), encode(short_breath), encode(swallowing), encode(chest_pain)
            ]

            X = np.array([features])
            Xs = lung_scaler.transform(X)

            pred = lung_model.predict(Xs)[0]
            proba = lung_model.predict_proba(Xs)[0]

            result = "Cancer" if pred == 1 else "No Cancer"

            # =========================
            # Result Box
            # =========================
            if result == "Cancer":
                st.markdown(f"<div class='result-box cancer'>⚠️ Prediction: {result}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='result-box no-cancer'>✅ Prediction: {result}</div>", unsafe_allow_html=True)

            # =========================
            # Probability Bars
            # =========================
            st.write("### Probability")
            st.markdown(f"""
            <div>
                <div>No Cancer: {(proba[0]*100):.1f}%</div>
                <div class="bar"><div class="fill-green" style="width:{proba[0]*100}%"></div></div>
                <div>Cancer: {(proba[1]*100):.1f}%</div>
                <div class="bar"><div class="fill-red" style="width:{proba[1]*100}%"></div></div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if pred == 1:
                st.error(f"⚠️ High Risk of Lung Cancer ({proba[1]*100:.2f}%)")
            else:
                st.success(f"✅ Low Risk of Lung Cancer ({(1-proba[1])*100:.2f}%)")

                # 📊 Probability Bar Chart
                import plotly.express as px
                fig_bar = px.bar(x=["No Risk", "Risk"],
                                y=[1-proba[1], proba[1]],
                                title="Lung Cancer Risk Probability",
                                color=["green", "red"])
                st.plotly_chart(fig_bar, use_container_width=True)

                # 📊 Radar Chart
                import plotly.graph_objects as go  # <-- Add this import to fix the error
                input_data = {"Age": age, "Smoking": smoking, "Fatigue": fatigue}
                healthy_ranges = {"Age": 40, "Smoking": 0, "Fatigue": 0}

                radar_df = pd.DataFrame({
                    "Metric": list(input_data.keys()),
                    "Patient": list(input_data.values()),
                    "Healthy": list(healthy_ranges.values())
                })

                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=radar_df["Patient"], theta=radar_df["Metric"], fill="toself", name="Patient"
                ))
                fig_radar.add_trace(go.Scatterpolar(
                    r=radar_df["Healthy"], theta=radar_df["Metric"], fill="toself", name="Healthy"
                ))
                fig_radar.update_layout(title="Patient vs Healthy Comparison", polar=dict(radialaxis=dict(visible=True)))
                st.plotly_chart(fig_radar, use_container_width=True)

    # ---------- DIABETES ----------
    elif choice== "Diabetes":
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🩸 Diabetes Risk Assessment")

        st.sidebar.header("🔧 Patient Details")
        gender = st.selectbox("Gender", list(gender_map.keys()))
        age = st.selectbox("Age", list(range(0, 121)))
        hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        smoking_history = st.selectbox("Smoking History", list(smoking_map.keys()))
        bmi = st.selectbox("BMI", [round(x * 0.1, 1) for x in range(100, 601)])
        hba1c = st.selectbox("HbA1c Level", [round(x * 0.1, 1) for x in range(30, 151)])
        glucose = st.selectbox("Blood Glucose Level", [x for x in range(50, 301)])

        # Button
    if st.sidebar.button("🔍 Predict Diabetes Risk"):
        # Prepare input
        input_data = pd.DataFrame([{
            "gender": gender_map[gender],
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "smoking_history": smoking_map[smoking_history],
            "bmi": bmi,
            "HbA1c_level": hba1c,
            "blood_glucose_level": glucose
        }])
        input_data[numeric_cols] = diab_scaler.transform(input_data[numeric_cols])

        # Prediction
        prediction = diab_model.predict(input_data)[0]
        probability = diab_model.predict_proba(input_data)[:, 1][0]

        # ----------------------
        # Display Results
        # ----------------------
        st.markdown("---")
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📊 Prediction Result")
            if prediction == 1:
                st.error(f"⚠️ High risk of Diabetes")
            else:
                st.success(f"✅ Low risk of Diabetes")

            st.markdown(f"**Probability of Diabetes:** `{probability:.2f}`")

        with col2:
            # Pie Chart of Probability
            fig, ax = plt.subplots()
            ax.pie(
                [probability, 1-probability],
                labels=["Diabetic", "Non-Diabetic"],
                autopct="%1.1f%%",
                colors=["#ff6b6b", "#51cf66"],
                startangle=90
            )
            ax.axis("equal")
            st.pyplot(fig)
        # ----------------------
        # Feature Importance
        # ----------------------
        st.markdown("---")
        st.subheader("🔎 Feature Importance (Model Insights)")
        importances = diab_model.feature_importances_
        features = X.columns if 'X' in locals() else input_data.columns  # fallback if X not available
        importance_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)

        st.bar_chart(importance_df.set_index("Feature"))

        st.markdown(
            """
            ℹ️ **Note:** Higher feature importance means the feature contributes more to the model's prediction.  
            Typically, `HbA1c` and `blood glucose` play strong roles in diabetes risk.
            """
        )
    # ---------- COVID-19 ----------
    elif choice == "Covid-19":
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🦠 Covid-19 Prediction")

        # Load Model & Scaler
        covid_model = load_pickle(r"D:\disease_pred.py\covid_model.pkl", "Covid-19 model")
        covid_scaler = load_pickle(r"D:\disease_pred.py\covid_scaler.pkl", "Covid-19 scaler")

        # ================== Custom CSS for Stunning UI ==================
        st.markdown("""
            <style>
                .title {
                    font-size:40px; font-weight:bold; text-align:center; color:#2C3E50;
                }
                .subtitle {
                    font-size:20px; text-align:center; color:white; margin-bottom:30px;
                }
                .stButton button {
                    background: linear-gradient(45deg, #1abc9c, #16a085);
                    color:white; font-size:18px; font-weight:bold; border-radius:12px;
                    padding:10px 20px;
                }
                .result-card {
                    background-color: #f1f2f6; 
                    padding: 25px; 
                    border-radius: 15px; 
                    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
                    margin-top: 20px;
                }
                .positive {
                    color: red; font-weight:bold; font-size:22px;
                }
                .negative {
                    color: orange; font-weight:bold; font-size:22px;
                }
            </style>
        """, unsafe_allow_html=True)

        # ================== Title ==================
        st.markdown("<div class='title'>🦠 Covid-19 Prediction System</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>Enter your medical parameters to predict the risk of Covid-19</div>", unsafe_allow_html=True)

        # ================== Input Form ==================
        with st.form("covid_form"):
            st.subheader("🔹 Enter Patient Information")
            col1, col2, col3 = st.columns(3)

            with col1:
                age = st.number_input("Age", 1, 100, 30)
                leukocytes = st.number_input("Leukocytes", 0.0, 20000.0, 6000.0)
                neutrophilsP = st.number_input("Neutrophils %", 0.0, 100.0, 60.0)
                lymphocytesP = st.number_input("Lymphocytes %", 0.0, 100.0, 30.0)
                monocytesP = st.number_input("Monocytes %", 0.0, 100.0, 5.0)
                eosinophilsP = st.number_input("Eosinophils %", 0.0, 100.0, 2.0)
                basophilsP = st.number_input("Basophils %", 0.0, 100.0, 1.0)

            with col2:
                neutrophils = st.number_input("Neutrophils", 0.0, 20000.0, 3500.0)
                lymphocytes = st.number_input("Lymphocytes", 0.0, 20000.0, 2500.0)
                monocytes = st.number_input("Monocytes", 0.0, 20000.0, 300.0)
                eosinophils = st.number_input("Eosinophils", 0.0, 20000.0, 150.0)
                basophils = st.number_input("Basophils", 0.0, 20000.0, 50.0)
                redbloodcells = st.number_input("Red Blood Cells", 2.0, 8.0, 5.0)
                mcv = st.number_input("MCV", 50.0, 120.0, 90.0)

            with col3:
                mch = st.number_input("MCH", 20.0, 40.0, 30.0)
                mchc = st.number_input("MCHC", 25.0, 38.0, 33.0)
                rdwP = st.number_input("RDW %", 10.0, 20.0, 13.0)
                hemoglobin = st.number_input("Hemoglobin", 5.0, 20.0, 14.0)
                hematocritP = st.number_input("Hematocrit %", 20.0, 60.0, 42.0)
                platelets = st.number_input("Platelets", 50000.0, 600000.0, 250000.0)
                mpv = st.number_input("MPV", 5.0, 15.0, 10.0)

            click = st.form_submit_button("🔍 Predict Now")

        # ================== Prediction ==================
        if click:
            features = np.array([[age, leukocytes, neutrophilsP, lymphocytesP, monocytesP,
                                eosinophilsP, basophilsP, neutrophils, lymphocytes, monocytes,
                                eosinophils, basophils, redbloodcells, mcv, mch, mchc,
                                rdwP, hemoglobin, hematocritP, platelets, mpv]])
            
            scaled_features = covid_scaler.transform(features)
            prediction = covid_model.predict(scaled_features)[0]
            probability = covid_model.predict_proba(scaled_features)[0][1] * 100 if hasattr(covid_model, "predict_proba") else 0

            # ================== Result Display ==================
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            if prediction == 1:
                st.markdown("<p class='positive'>⚠️ Result: Positive for Covid-19</p>", unsafe_allow_html=True)
                st.write(f"🔴 Probability of infection: **{probability:.2f}%**")
                st.subheader("📝 Recommendations:")
                st.write("""
                - Avoid public gatherings  
                - Wear a mask at all times  
                - Stay in isolation  
                - Drink fluids & maintain nutrition  
                - Consult a doctor immediately  
                """)
            else:
                st.markdown("<p class='negative'>✅ Result: Negative for Covid-19</p>", unsafe_allow_html=True)
                st.write(f"🟢 Probability of being healthy: **{100 - probability:.2f}%**")
                st.subheader("📝 Recommendations:")
                st.write("""
                - Maintain a balanced diet  
                - Exercise regularly  
                - Get enough sleep  
                - Stay hydrated  
                - Wash hands frequently & maintain hygiene  
                """)
            st.markdown("</div>", unsafe_allow_html=True)

            # ================== Seaborn Visualization ==================
            st.subheader("📊 Your Parameters vs Normal Ranges")

            data = {
                "Parameter": ["Hemoglobin", "Platelets", "Leukocytes", "Neutrophils %", "Lymphocytes %"],
                "Your Value": [hemoglobin, platelets, leukocytes, neutrophilsP, lymphocytesP],
                "Normal Range Low": [13, 150000, 4000, 40, 20],
                "Normal Range High": [17, 400000, 11000, 70, 40]
            }
            df = pd.DataFrame(data)

            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=df, x="Parameter", y="Your Value", color="skyblue", ax=ax)
            for i in range(len(df)):
                ax.plot([i-0.4, i+0.4], [df["Normal Range Low"][i], df["Normal Range Low"][i]], "g-", linewidth=2)
                ax.plot([i-0.4, i+0.4], [df["Normal Range High"][i], df["Normal Range High"][i]], "r-", linewidth=2)
            plt.ylabel("Value")
            plt.title("Your Values vs Normal Ranges")
            st.pyplot(fig)

        st.markdown("</div>", unsafe_allow_html=True)
        
elif selected == "ℹ️ About Us":
    st.title("ℹ️ About Us")
    st.write("""
    💡 Mediscope AI is an intelligent healthcare assistant designed to 
    predict risks of **Heart Disease, Lung Cancer, Diabetes, and Covid-19** using 
    advanced Machine Learning models.
    """)
    st.markdown("""
        <style>
            .title {
                font-size:40px;
                font-weight:700;
                color:#2E86C1;
                text-align:center;
                margin-bottom:20px;
            }
            .subtitle {
                font-size:22px;
                color:#de1259;
                font-weight:600;
                margin-top:20px;
            }
            .content {
                font-size:18px;
                line-height:1.6;
                color:#od140c;
                text-align:justify;
            }
            .highlight {
                background-color:#353794;
                padding:15px;
                border-radius:15px;
                margin-top:10px;
                font-size:18px;
                line-height:1.6;
                color:#e4e8ed;
                max-height:500px;
                overflow-y:auto;
            }
        </style>
    """, unsafe_allow_html=True)

    # Title
    st.markdown("<div class='title'>About Us</div>", unsafe_allow_html=True)

    # Main Content
    st.markdown("""
    <div class="content">
    Welcome to <b>Mediscope AI</b>, an AI-powered healthcare platform dedicated to making health insights more 
    accessible, accurate, and preventive.  

    Our mission is simple: <b>empower people to take control of their health before it becomes critical.</b> 
    We believe early detection saves lives, and that’s why we are building advanced 
    <b>AI-driven prediction tools</b> for conditions like <b>heart disease, cardiovascular disorders (CVD), diabetes, and lung cancer.</b>
    These conditions account for millions of deaths worldwide each year, and many of them can be managed—or even prevented—if detected early. 
    With our AI technology, users can get an early warning and take proactive steps toward better health.
    </div>
    """, unsafe_allow_html=True)

    # What We Do Section
    st.markdown("<div class='subtitle'>🔹 What We Do</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="highlight content">
    - Provide AI-based prediction tools that analyze medical and lifestyle data to give risk assessments.  
    - Offer easy-to-understand reports and recommendations to support preventive healthcare.  
    - Continuously expand our tools with more disease models for comprehensive health monitoring.  
    </div>
    """, unsafe_allow_html=True)

    # Vision Section
    st.markdown("<div class='subtitle'>🔹 Our Vision</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="highlight content">
    We aim to transform healthcare from <b>reactive to proactive.</b>  
    In the future, we are integrating a smart <b>AI healthcare assistant</b> to guide users, answer 
    health-related queries, and assist with wellness decisions—making healthcare 
    <b>personalized, accessible, and reliable.</b>
    Our approach combines the best of technology, compassion, and medical knowledge to help create a healthier tomorrow.
    </div>
    """, unsafe_allow_html=True)

    # Why Choose Us Section
    st.markdown("<div class='subtitle'>🔹 Why Choose Us</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="highlight content">
    - Backed by <b>cutting-edge AI & machine learning models.</b>  
    - Focused on <b>accuracy, transparency, and ease of use.</b>  
    - Built with <b>privacy & data security</b> at the core.  
    - Committed to making healthcare inclusive and accessible for everyone.
    - <b>AI-Driven Accuracy</b> Our platform uses advanced machine learning and deep learning models 
    trained on medical datasets to deliver reliable and precise health risk predictions.
    - <b>Seamless User Experience</b> We prioritize user experience, ensuring our platform is intuitive and easy to navigate.
    - <b>Future-Ready Platform</b> Built with scalability in mind, HealthPredict AI is constantly evolving.
    From multi-disease predictions to AI-powered healthcare assistants and integration with wearable devices,
    our vision is to create a comprehensive digital health companion.
    </div>
    """, unsafe_allow_html=True)

    # Closing Statement
    st.markdown("""
    <div class="content" style="margin-top:20px; font-style:italic; text-align:center;">
    At <b>HealthPredict AI</b>, we’re not just building prediction tools—we’re building 
    <b>trust, care, and smarter healthcare solutions for a healthier tomorrow.</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    .team-row {
        display: flex;
        justify-content: center;
        gap: 32px;
        flex-wrap: wrap;
        margin-top: 32px;
        margin-bottom: 32px;
    }
    .team-card {
        background: #Fbbd3fc;
        border-radius: 24px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.12);
        padding: 32px 24px 24px 24px;
        width: 280px;
        text-align: center;
        transition: box-shadow 0.2s, transform 0.2s;
        margin-bottom: 16px;
    }
    .team-card:hover {
        box-shadow: 0 16px 48px rgba(0,0,0,0.18);
        transform: translateY(-6px) scale(1.03);
    }
    .team-img {
        width: 110px;
        height: 110px;
        object-fit: cover;
        border-radius: 50%;
        margin-bottom: 18px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.10);
        border: 4px solid #1e20e8;
    }
    .team-name {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 4px;
        color: #1e20e8;
        text-align: center;
    }
    .team-role {
        font-size: 15px;
        font-weight: 600;
        color: #1ec0e9;
        margin-bottom: 10px;
        letter-spacing: 1px;
    }
    .team-desc {
        font-size: 14px;
        color: #101413;
        margin-bottom: 15px;
    }
    .skills {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 6px;
        margin-bottom: 10px;
    }
    .skill-tag {
        background-color: #1a20e8;
        color: white;
        font-size: 12px;
        padding: 5px 14px;
        border-radius: 15px;
        margin: 2px;
        font-weight: 500;
        letter-spacing: 0.5px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center; color:#1f0b11;'>Meet Our Team</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color:#2e2142; font-weight:bold; font-size:18px;'>Passionate students dedicated to revolutionizing healthcare through AI technology and innovation.</p>", unsafe_allow_html=True)

    team = [
        {
            "name": "Avinash Panda",
            "role": "Front-End Developer",
            "desc": "Specialized in creating user-friendly interfaces and seamless user experiences.",
            "img": "member1.jpg",
            "skills": ["HTML", "CSS", "JavaScript"]
        },
        {
            "name": "Lalatendu Kumar Sahu",
            "role": "Machine Learning Engineer",
            "desc": "Focused on making complex medical data accessible and user-friendly.",
            "img": "lalatendu.jpg",
            "skills": ["Machine Learning", "Deep Learning", "NLP"]
        },
        {
            "name": "Debabrata Sahu",
            "role": "Data Engineer",
            "desc": "Ensured secure data handling and efficient model deployment.",
            "img": "member3.png",
            "skills": ["Excel", "Machine Learning", "Data Preprocessing"]
        },
        {
            "name": "Deepak Kumar Jena",
            "role": "Front-End Developer",
            "desc": "Ensure aesthetic and functional user interfaces. Also responsible for user experience research and testing.",
            "img": "member4.png",
            "skills": ["HTML", "CSS", "JavaScript"]
        },
        {
            "name": "Member 5",
            "role": "AI Researcher",
            "desc": "Working on advanced algorithms for healthcare-focused AI applications.",
            "img": "member5.JPG",
            "skills": ["Python", "AI Research", "NLP"]
        }
    ]

    # Team cards in a row
    cols = st.columns(len(team))
    for idx, member in enumerate(team):
        with cols[idx]:
            img_path = member["img"]
            import os
            if not os.path.isfile(img_path):
                # Use a built-in placeholder image (base64 for a transparent PNG)
                default_base64 = "iVBORw0KGgoAAAANSUhEUgAAAJYAAACWCAMAAADzJ3zIAAAABlBMVEUAAAD///+l2Z/dAAAAAXRSTlMAQObYZgAAAGtJREFUeF7twTEBAAAAwqD1T20ND6AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwG4B8AABR2QW7gAAAABJRU5ErkJggg=="
                st.markdown(
                    f'<img src="data:image/png;base64,{default_base64}" style="width:110px;height:110px;border-radius:50%;margin-bottom:18px;border:4px solid #1a20e8;">',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<img src="data:image/jpg;base64,{base64.b64encode(open(img_path, "rb").read()).decode()}" style="width:110px;height:110px;border-radius:50%;margin-bottom:18px;border:4px solid #1a20e8;">',
                    unsafe_allow_html=True
                )
            st.markdown(f"<div style='font-size:20px;font-weight:bold;color:#1a20e8;margin-bottom:4px'>{member['name']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:15px;font-weight:600;color:#1a20e8;margin-bottom:10px;letter-spacing:1px'>{member['role'].upper()}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:14px;font-weight:bold;color:#101413;margin-bottom:15px'>{member['desc']}</div>", unsafe_allow_html=True)
            st.markdown(
                "<div style='display:flex;flex-wrap:wrap;justify-content:center;gap:6px;margin-bottom:10px'>" +
                "".join([f"<span style='background-color:#1a20e8;color:white;font-size:12px;padding:5px 14px;border-radius:15px;margin:2px;font-weight:500;letter-spacing:0.5px'>{skill}</span>" for skill in member['skills']]) +
                "</div>",
                unsafe_allow_html=True
            )

elif selected == "📞 Contact Us":
    st.title("📞 Contact Us")
    st.write("""
    📧 Email: smediscopeai@gmail.com
             
    🏢 Address: Kalam Institute of Technology, Berhampur, Odisha, India

    - **Get in Touch:** Have questions, feedback, or collaboration ideas? We’d like to hear from you.
    Whether you are a student, researcher, healthcare professional, or just curious about our project, reach out to us.

    - **Collaborations & Partnerships:** We welcome partnerships with medical professionals, educational institutions, and research organizations to improve our AI-driven healthcare system.
    """)

elif selected == "📘 More Info":
    st.title("📘 More Info")
    st.write("""
    🔹 Developed By - Btech 7th Sem Students
    """)
    # Add custom CSS for styling
    st.markdown(
        """
        <style>
        .section-title {
            font-size:60px;
            font-weight:700;
            color:#111dcf;
            text-align:center;
            margin-bottom:20px;
        }
        .sub-title {
            font-size:22px;
            font-weight:600;
            color:#111dcf;
            margin-top:25px;
            margin-bottom:10px;
        }
        .para {
            font-size:16px;
            line-height:1.6;
            color:#2F2F2F;
            text-align:justify;
        }
        ul {
            font-size:16px;
            line-height:1.6;
            color:#2F2F2F;
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.markdown('<p class="section-title">More Information</p>', unsafe_allow_html=True)

    st.markdown(
        """
        <p class="para">
        Our project demonstrates the practical application of <b>Artificial Intelligence (AI)</b> 
        and <b>Machine Learning (ML)</b> in the healthcare sector. 
        It focuses on <b>disease prediction</b> for conditions such as 
        <b>Heart Disease, Lung Cancer, and Diabetes</b>. 
        The system is designed to provide users with quick, accurate, and 
        easy-to-understand results, supported by professional health recommendations.
        </p>
        """, unsafe_allow_html=True
    )

    st.markdown('<p class="sub-title">Key Features</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <ul>
            <li>✔️ AI-powered predictions using trained ML models</li>
            <li>✔️ Interactive and user-friendly interface</li>
            <li>✔️ Personalized health recommendations</li>
            <li>✔️ Informative visualizations for better clarity</li>
            <li>✔️ Fast and secure prediction workflow</li>
        </ul>
        """, unsafe_allow_html=True
    )

    st.markdown('<p class="sub-title">Disclaimer</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <ul>
            This platform is developed as a student final-year project for educational purposes.
            The predictions provided are not a substitute for medical advice. Always consult 
            a qualified healthcare professional for diagnosis and treatment.
        </ul>
        """, unsafe_allow_html=True
    )

    st.markdown('<p class="sub-title">Impact & Use Cases</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <p class="para">
        This project acts as an <b>awareness and support tool</b> for early disease detection. 
        It is not a replacement for medical diagnosis but can assist in:
        </p>
        <ul>
            <li>📌 Early health risk assessment for individuals</li>
            <li>📌 Academic learning and research in AI-driven healthcare</li>
            <li>📌 Supporting doctors with AI insights (future integration)</li>
        </ul>
        """, unsafe_allow_html=True
    )

    st.markdown('<p class="sub-title">Future Scope</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <p class="para">
        Our vision is to create accessible, AI-powered, and user-friendly healthcare prediction tools 
        that promote early diagnosis, better awareness, and healthier living.
        </p>
        """, unsafe_allow_html=True
    )
