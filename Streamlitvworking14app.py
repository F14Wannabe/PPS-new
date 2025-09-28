import os
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

# === SETTINGS ===
FOLDER_PATH = os.path.expanduser("~/Downloads/folder_input/")  # change as needed
MODEL_PATH = "model2.pkl"
SHEET_URL = "Yhttps://docs.google.com/spreadsheets/d/1kuQP6jLVoZtMCaHXkWyn6LjvoNTKQEFynV-AKBVjPDs/edit?usp=sharing"  # Replace with your Google Sheet URL
WORKSHEET_NAME = "IDS"  # Replace with your actual worksheet name

# === Load Model ===
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Model file '{MODEL_PATH}' not found.")
    st.stop()

# === Function: Authenticate Google Sheets ===
def authenticate_google_sheets():
    try:
        # Define the scope of the API
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/spreadsheets",
                 "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]

        # Get the file path of the JSON key in the current working directory (where the script is)
        json_keyfile = os.path.join(os.getcwd(), 'service_account_file.json')

        # Ensure that the JSON key exists in the current directory
        if not os.path.exists(json_keyfile):
            raise FileNotFoundError(f"Service account JSON file not found at {json_keyfile}")

        # Authenticate with the service account JSON file
        creds = Credentials.from_service_account_file(json_keyfile, scopes=scope)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        st.error(f"Error authenticating with Google Sheets: {str(e)}")
        st.stop()

# === Function: Get Row by ID from Google Sheet ===
def get_row_by_id_from_google_sheet(sheet_url, worksheet_name, user_id):
    try:
        # Authenticate and get the Google Sheets client
        client = authenticate_google_sheets()

        # Open the Google Sheet by URL
        sheet = client.open_by_url(sheet_url)

        # Select the specific worksheet
        worksheet = sheet.worksheet(worksheet_name)

        # Get all data from the sheet
        data = pd.DataFrame(worksheet.get_all_records())

        # Find the row corresponding to the user_id
        user_row = data[data["ID"].astype(str) == str(user_id)]

        if user_row.empty:
            return None, f"No record found for ID {user_id}"

        return user_row.iloc[0], None

    except Exception as e:
        return None, f"Error accessing Google Sheets: {str(e)}"

# === Function to Clear Cache ===
def clear_cache():
    # Clear the cache to reload the latest data
    st.cache_data.clear()  # For Streamlit versions >= 1.10.0

# === Streamlit UI ===
st.set_page_config(page_title="Substance Use Risk Prediction", layout="centered")

# Modern styling
st.markdown(""" 
    <style>
    .main {background-color: #f5f7fa; padding: 2rem;}
    h1 {color: #2c3e50; font-family: 'Segoe UI', sans-serif;}
    .stButton>button {background-color: #4CAF50; color:white; border-radius:12px; padding:10px 24px; font-size:16px;}
    .stSlider>div>div>div>div {color: #2c3e50;}
    </style>
""", unsafe_allow_html=True)

st.title("Substance Use Risk Prediction")

# Step 1: Ask for user ID
user_id = st.text_input("Enter your ID:", placeholder="e.g., 12345")

# Step 2: Show button only if ID is entered
if user_id:
    if st.button("View your results"):
        clear_cache()  # Clear the cache to reload the latest data

        # Fetch user data from Google Sheet
        row = None
        found_row, error_msg = get_row_by_id_from_google_sheet(SHEET_URL, WORKSHEET_NAME, user_id)

        if found_row is not None:
            row = found_row
            try:
                # === Safe lookups with detailed error handling ===
                errors = []

                peer_val = row.get("Peer pressure score", None)
                if pd.isna(peer_val):
                    errors.append("Peer pressure score is missing or invalid.")

                age_val = row.get("Age", None)
                if pd.isna(age_val):
                    errors.append("Age is missing or invalid.")

                gender_val = row.get("Gender", None)
                if pd.isna(gender_val):
                    errors.append("Gender is missing or invalid.")

                conf_val = row.get("Confidence Level", None)
                if pd.isna(conf_val):
                    errors.append("Confidence Level is missing or invalid.")

                earned_val = row.get("Earned Recognition", None)
                if pd.isna(earned_val):
                    errors.append("Earned Recognition is missing or invalid.")

                impulsive_val = row.get("Impulsivness", None)
                if pd.isna(impulsive_val):
                    errors.append("Impulsiveness is missing or invalid.")

                excl_val = row.get("Exclusion Anxiety", None)
                if pd.isna(excl_val):
                    errors.append("Exclusion Anxiety is missing or invalid.")

                pp_val = row.get("People Pleaser", None)
                if pd.isna(pp_val):
                    errors.append("People Pleaser score is missing or invalid.")

                income_raw = row.get("Income level", None)
                if pd.isna(income_raw):
                    errors.append("Income level is missing or invalid.")

                # If any errors found, display them
                if errors:
                    st.error("The following data is missing or invalid:")
                    for err in errors:
                        st.write(f"- {err}")
                    st.stop()

                # Prepare input data if no issues with values
                income_map = {"Low": 0, "Medium": 1, "High": 2}
                input_data = pd.DataFrame({
                    "ID": [user_id],
                    "Peer pressure score": [float(peer_val) / 100],
                    "Age": [int(float(age_val))],
                    "Gender": [str(gender_val)],
                    "Confidence Level": [float(conf_val)],
                    "Earned Recognition": [float(earned_val)],
                    "Impulsivness": [float(impulsive_val)],
                    "Exclusion Anxiety": [float(excl_val)],
                    "People Pleaser": [float(pp_val)],
                    "Income level": [float(income_raw) if isinstance(income_raw, (int, float))
                                     else income_map.get(str(income_raw).capitalize(), 1)]
                })
                # Make prediction
                prediction = model.predict(input_data)[0]

                # Map categorical prediction into numeric for gauge
                risk_map = {"low": 25, "medium": 50, "high": 75, "very high": 100}
                risk_value = risk_map.get(str(prediction).lower(), 0)

                # === Gauge Chart ===
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_value,
                    number={'suffix': "%", 'font': {'size': 36}},
                    title={'text': "Predicted Risk Level", 'font': {'size': 24}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#e74c3c"},
                        'steps': [
                            {'range': [0, 25], 'color': "#2ecc71"},
                            {'range': [25, 50], 'color': "#f1c40f"},
                            {'range': [50, 75], 'color': "#e67e22"},
                            {'range': [75, 100], 'color': "#e74c3c"}
                        ],
                        'threshold': {
                            'line': {'color': "#34495e", 'width': 4},
                            'thickness': 0.75,
                            'value': risk_value
                        }
                    }
                ))

                fig.update_layout(
                    margin={'t': 0, 'b': 0, 'l': 0, 'r': 0},
                    height=300,
                    paper_bgcolor="#f5f7fa"
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

        # === Personalized Tips Section (moved to bottom) ===
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

