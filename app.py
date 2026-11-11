import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="IPL Score Predictor",
    page_icon="🏏",
    layout="centered"
)

# --------------------------------------------------
# 2. LOAD SAVED MODELS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "ipl_score_prediction.pkl"

if not MODEL_PATH.exists():
    st.error("Saved model file not found. Check your models folder.")
    st.stop()

artifacts = joblib.load(MODEL_PATH)

venue_model = artifacts['venue_model']
original_venue_encoder = artifacts['original_venue_encoder']
stage_models = artifacts['stage_models']
stage_encoder = artifacts['stage_encoder']
reduced_features = artifacts['reduced_features']

# --------------------------------------------------
# 3. APPLICATION TITLE
# --------------------------------------------------

st.title("🏏 IPL First Innings Score Predictor")

st.write(
    "Predict the final score of an IPL innings "
    "using machine learning."
)

st.divider()

# --------------------------------------------------
# 4. MATCH INFORMATION INPUTS
# --------------------------------------------------

st.header("Enter Match Details")

col1, col2 = st.columns(2)

with col1:
    current_runs = st.number_input(
        "Current Score",
        min_value=0,
        max_value=500,
        value=100,
        step=1,
        key="current_runs"
    )

with col2:
    wickets = st.number_input(
        "Wickets Lost",
        min_value=0,
        max_value=10,
        value=3,
        step=1,
        key="wickets"
    )

col3, col4 = st.columns(2)

with col3:
    overs = st.number_input(
        "Completed Overs",
        min_value=0,
        max_value=20,
        value=10,
        step=1,
        key="overs"
    )

with col4:
    balls = st.number_input(
        "Balls in Current Over",
        min_value=0,
        max_value=5,
        value=1,
        step=1,
        key="balls"
    )

# Convert overs and balls into legal balls
legal_balls = int(overs * 6 + balls)

venue = st.text_input(
    "Stadium / Venue",
    value="Narendra Modi Stadium, Ahmedabad",
    key="venue"
)

st.divider()

# --------------------------------------------------
# 5. RECENT RUNS INPUTS
# --------------------------------------------------

st.subheader("Recent Batting Performance")

col1, col2, col3 = st.columns(3)

with col1:
    runs_last_6_balls = st.number_input(
        "Last 6 Balls",
        min_value=0,
        max_value=100,
        value=10,
        key="runs_last_6"
    )

with col2:
    runs_last_12_balls = st.number_input(
        "Last 12 Balls",
        min_value=0,
        max_value=150,
        value=20,
        key="runs_last_12"
    )

with col3:
    runs_last_30_balls = st.number_input(
        "Last 30 Balls",
        min_value=0,
        max_value=250,
        value=45,
        key="runs_last_30"
    )

st.divider()

# --------------------------------------------------
# 6. PREDICTION BUTTON
# --------------------------------------------------

predict_button = st.button(
    "Predict Final Score 🏏",
    type="primary",
    use_container_width=True
)

# --------------------------------------------------
# 7. PREDICTION LOGIC
# --------------------------------------------------

if predict_button:

    # Input validation

    if legal_balls == 0:
        st.error("At least one legal ball must be bowled.")

    elif overs == 20 and balls != 0:
        st.error("An innings cannot exceed 20 overs.")

    elif not venue.strip():
        st.error("Please enter a venue.")

    elif runs_last_6_balls > runs_last_12_balls:
        st.error("Last 6 balls cannot have more runs than last 12 balls.")

    elif runs_last_12_balls > runs_last_30_balls:
        st.error("Last 12 balls cannot have more runs than last 30 balls.")

    elif runs_last_30_balls > current_runs:
        st.error("Recent runs cannot exceed the current score.")

    else:

        # Calculate current run rate

        current_run_rate = (
            current_runs / legal_balls
        ) * 6

        # Select the appropriate model

        if legal_balls <= 60:

            model = venue_model
            encoder = original_venue_encoder
            stage_name = "Overs 1-10"

        elif legal_balls <= 90:

            model = stage_models['Overs 11-15']
            encoder = stage_encoder
            stage_name = "Overs 11-15"

        else:

            model = stage_models['Overs 16-20']
            encoder = stage_encoder
            stage_name = "Overs 16-20"

        # Prepare numerical features

        numerical_data = pd.DataFrame([{
            'legal_balls': legal_balls,
            'current_runs': current_runs,
            'wickets': wickets,
            'current_run_rate': current_run_rate,
            'runs_last_30_balls': runs_last_30_balls,
            'runs_last_6_balls': runs_last_6_balls,
            'runs_last_12_balls': runs_last_12_balls
        }])[reduced_features]

        # Encode venue

        venue_encoded = encoder.transform(
            pd.DataFrame({
                'venue': [venue]
            })
        )

        # Combine numerical and venue features

        X = np.concatenate(
            [
                numerical_data.values,
                venue_encoded
            ],
            axis=1
        )

        # Generate prediction

        if legal_balls == 120:

            predicted_score = current_runs

        else:

            predicted_score = model.predict(X)[0]

            # Final score cannot be below current score

            predicted_score = max(
                predicted_score,
                current_runs
            )

        predicted_score = round(predicted_score)

        # --------------------------------------------------
        # 8. DISPLAY RESULTS
        # --------------------------------------------------

        st.success("Prediction completed successfully!")

        st.subheader("Predicted Final Score")

        st.metric(
            label="Expected Final Score",
            value=f"{predicted_score} runs",
            delta=f"{predicted_score - current_runs} runs from current score"
        )

        st.divider()

        st.subheader("Match Summary")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                "Current Score",
                f"{current_runs}/{wickets}"
            )

            st.metric(
                "Legal Balls Bowled",
                f"{legal_balls}/120"
            )

        with result_col2:
            st.metric(
                "Current Run Rate",
                f"{current_run_rate:.2f}"
            )

            st.metric(
                "Remaining Balls",
                120 - legal_balls
            )

        st.write(f"**Venue:** {venue}")
        st.write(f"**Model Used:** {stage_name}")