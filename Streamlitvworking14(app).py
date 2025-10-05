import os
import json
import gspread
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
from google.oauth2.service_account import Credentials

# === SETTINGS ===
MODEL_PATH = "model2.pkl"
SHEET_URL = "https://docs.google.com/spreadsheets/d/1kuQP6jLVoZtMCaHXkWyn6LjvoNTKQEFynV-AKBVjPDs/edit?usp=sharing"
WORKSHEET_NAME = "IDS"

# === Load Model ===
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Model file '{MODEL_PATH}' not found.")
    st.stop()

# === Function: Authenticate Google Sheets (from GitHub Secret) ===
def authenticate_google_sheets():
    try:
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.file",
            "https://www.googleapis.com/auth/drive",
        ]

        # ✅ Load the service account JSON from environment variable (GitHub Secret)
        service_account_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        if not service_account_json:
            raise ValueError(
                "Environment variable 'GOOGLE_SERVICE_ACCOUNT_JSON' not found. "
                "Ensure it’s added as a GitHub Secret and passed in your workflow."
            )

        # Parse the JSON string into a Python dictionary
        creds_dict = json.loads(service_account_json)

        # Create credentials directly from the dict
        creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        st.error(f"Error authenticating with Google Sheets: {str(e)}")
        st.stop()

# === Function: Get Row by ID from Google Sheet ===
def get_row_by_id_from_google_sheet(sheet_url, worksheet_name, user_id):
    try:
        client = authenticate_google_sheets()
        sheet = client.open_by_url(sheet_url)
        worksheet = sheet.worksheet(worksheet_name)
        data = pd.DataFrame(worksheet.get_all_records())

        user_row = data[data["ID"].astype(str) == str(user_id)]
        if user_row.empty:
            return None, f"No record found for ID {user_id}"

        return user_row.iloc[0], None

    except Exception as e:
        return None, f"Error accessing Google Sheets: {str(e)}"

# === Function to Clear Cache ===
def clear_cache():
    st.cache_data.clear()

# === Streamlit UI ===
st.set_page_config(page_title="Substance Use Risk Prediction", layout="centered")

st.markdown(
    """ 
    <style>
    .main {background-color: #f5f7fa; padding: 2rem;}
    h1 {color: #2c3e50; font-family: 'Segoe UI', sans-serif;}
    .stButton>button {background-color: #4CAF50; color:white; border-radius:12px; padding:10px 24px; font-size:16px;}
    </style>
""",
    unsafe_allow_html=True,
)

st.title("Substance Use Risk Prediction")

# Step 1: Ask for user ID
user_id = st.text_input("Enter your ID:", placeholder="e.g., 12345")

# Step 2: Show button only if ID is entered
if user_id:
    if st.button("View your results"):
        clear_cache()
        row, error_msg = get_row_by_id_from_google_sheet(SHEET_URL, WORKSHEET_NAME, user_id)

        if row is not None:
            try:
                # Safe lookups
                fields = [
                    "Peer pressure score",
                    "Age",
                    "Gender",
                    "Confidence Level",
                    "Earned Recognition",
                    "Impulsivness",
                    "Exclusion Anxiety",
                    "People Pleaser",
                    "Income level",
                ]
                missing = [f for f in fields if pd.isna(row.get(f, None))]
                if missing:
                    st.error("Missing data in fields: " + ", ".join(missing))
                    st.stop()

                income_map = {"Low": 0, "Medium": 1, "High": 2}
                input_data = pd.DataFrame(
                    {
                        "ID": [user_id],
                        "Peer pressure score": [float(row["Peer pressure score"]) / 100],
                        "Age": [int(float(row["Age"]))],
                        "Gender": [str(row["Gender"])],
                        "Confidence Level": [float(row["Confidence Level"])],
                        "Earned Recognition": [float(row["Earned Recognition"])],
                        "Impulsivness": [float(row["Impulsivness"])],
                        "Exclusion Anxiety": [float(row["Exclusion Anxiety"])],
                        "People Pleaser": [float(row["People Pleaser"])],
                        "Income level": [
                            float(row["Income level"])
                            if isinstance(row["Income level"], (int, float))
                            else income_map.get(str(row["Income level"]).capitalize(), 1)
                        ],
                    }
                )

                # Predict
                prediction = model.predict(input_data)[0]
                risk_map = {"low": 25, "medium": 50, "high": 75, "very high": 100}
                risk_value = risk_map.get(str(prediction).lower(), 0)

                # Gauge Chart
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=risk_value,
                        number={"suffix": "%", "font": {"size": 36}},
                        title={"text": "Predicted Risk Level", "font": {"size": 24}},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#e74c3c"},
                            "steps": [
                                {"range": [0, 25], "color": "#2ecc71"},
                                {"range": [25, 50], "color": "#f1c40f"},
                                {"range": [50, 75], "color": "#e67e22"},
                                {"range": [75, 100], "color": "#e74c3c"},
                            ],
                            "threshold": {
                                "line": {"color": "#34495e", "width": 4},
                                "thickness": 0.75,
                                "value": risk_value,
                            },
                        },
                    )
                )

                fig.update_layout(
                    margin={"t": 0, "b": 0, "l": 0, "r": 0},
                    height=300,
                    paper_bgcolor="#f5f7fa",
                )

                st.plotly_chart(fig, use_container_width=True)
                st.success(f"Predicted Risk Level: **{str(prediction).capitalize()}** ({risk_value}%)")

            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.error(error_msg)

        # === Traits UI (modern layout) ===
        st.subheader("Your Psychological Traits")
        col1, col2 = st.columns(2)

        with col1:
            peer_pressure = st.slider("Peer Pressure Score (%)", 0, 100, int(float(peer_val)))
            confidence = st.slider("Confidence Level (%)", 0, 15, int(float(conf_val)))
            earned_recognition = st.slider("Earned Recognition (%)", 0, 10, int(float(earned_val)))

        with col2:
            impulsiveness = st.slider("Impulsiveness (%)", 0, 15, int(float(impulsive_val)))
            exclusion_anxiety = st.slider("Exclusion Anxiety (%)", 0, 25, int(float(excl_val)))
            people_pleaser = st.slider("People Pleaser (%)", 0, 25, int(float(pp_val)))

        col3 = st.columns(1)[0]
        with col3:
            age = st.number_input("Age", min_value=12, max_value=25, value=int(float(age_val)))

        # === Personalized Tips Section ===
        if 'risk_value' in locals() and risk_value > 50:
            st.markdown("---")
            st.subheader("💡 Personalized Tips to Handle Peer Pressure")

            trait_tips = {
                "Peer pressure score": [
                    "Pause and remind yourself of your values before responding to pressure.",
                    "Practice saying 'no' in different scenarios to build confidence.",
                    "Choose activities that align with your goals to reduce exposure to pressure."
                ],
                "Confidence Level": [
                    "List your past achievements to remind yourself of your strengths.",
                    "Set small, realistic goals and celebrate when you achieve them.",
                    "Practice positive self-talk to build confidence in tough situations."
                ],
                "Earned Recognition": [
                    "Remind yourself that recognition comes from consistent effort, not risky behavior.",
                    "Seek validation from within, not just from others.",
                    "Surround yourself with people who value you for who you are, not what you do."
                ],
                "Impulsivness": [
                    "Pause and count to 10 before making a quick decision.",
                    "Write down pros and cons before acting on an impulse.",
                    "Avoid environments where you’re more likely to make impulsive choices."
                ],
                "Exclusion Anxiety": [
                    "Remind yourself that true friends accept you as you are.",
                    "Engage in activities that boost self-esteem outside of peer validation.",
                    "Challenge negative thoughts about being excluded with positive affirmations."
                ],
                "People Pleaser": [
                    "Practice setting small boundaries, like politely declining small requests.",
                    "Remember that saying 'no' doesn’t make you a bad friend.",
                    "Focus on your needs as much as others’ to maintain balance."
                ]
            }

            trait_values = {
                "Peer pressure score": peer_pressure,
                "Confidence Level": confidence,
                "Earned Recognition": earned_recognition,
                "Impulsiveness": impulsiveness,
                "Exclusion Anxiety": exclusion_anxiety,
                "People Pleaser": people_pleaser
            }

            for trait, value in trait_values.items():
                if value > 50:
                    st.markdown(f"**{trait}:**")
                    for tip in trait_tips.get(trait, []):
                        st.markdown(f"- {tip}")
