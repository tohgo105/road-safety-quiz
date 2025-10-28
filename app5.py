import streamlit as st
import joblib
import pandas as pd
import random
from quiz_data import quiz_data

# Load the model and label encoders created on Kaggle
model = joblib.load("accident_model.pkl")
le_dict = joblib.load("label_encoders.pkl")
expected_features = model.feature_name_

# Initializing streamlit session state
if "step" not in st.session_state:
    st.session_state.step = 0
if "answers" not in st.session_state:
    st.session_state.answers = []
if "start" not in st.session_state:
    st.session_state.start = False
if "quiz_indices" not in st.session_state:
    st.session_state.quiz_indices = []

# "Road Safety Quiz!"" Start Page and Start Button Settings
if not st.session_state.start:
    st.markdown("""
    # 🚗 Welcome to the Road Safety Quiz!

    In this quiz, you'll be presented with **3 questions**.

    For each question, you'll see information about two different road conditions — **Road A** and **Road B** — including:

    - Road type (urban, rural, highway)
    - Number of lanes
    - Curvature of the road
    - Lighting conditions
    - Weather conditions
    - Time of day

    👉 Your task: **Choose the safer road** based on your intuition!
    
    After answering all questions, you'll get a detailed result with risk scores and correct answers.
    """)
    if st.button("Start Quiz"):
        st.session_state.start = True
        st.session_state.step = 0
        st.session_state.answers = []
        st.session_state.quiz_indices = random.sample(range(len(quiz_data)), 3)
        st.rerun()
    st.stop()

# Screen after completing all questions
if st.session_state.step >= len(st.session_state.quiz_indices):
    st.subheader("Quiz Completed!")
    correct = sum([1 for a in st.session_state.answers if a["correct"]])
    total = len(st.session_state.answers)
    st.success(f"✅ You got {correct} out of {total} correct!")

    # Show answer review
    for i, ans in enumerate(st.session_state.answers):
        st.markdown(f"**Question {i+1}**")
        st.write(f"Your choice: {ans['user_choice']}")
        st.write(f"Correct answer: {ans['correct_answer']}")

        # Compare Road A/B and safety risks in a table format
        df_display = pd.DataFrame({
            "Feature": list(ans["road_1"].keys()) + ["risk"],
            "Road A": list(ans["road_1"].values()) + [round(ans["risk_1"], 2)],
            "Road B": list(ans["road_2"].values()) + [round(ans["risk_2"], 2)]
        })
        st.table(df_display)

        if ans["correct"]:
            st.success("✅ Correct")
        else:
            st.error("❌ Incorrect")
        st.markdown("---")

    # "Play Again" button settings
    if st.button("Play Again"):
        st.session_state.start = False
        st.session_state.step = 0
        st.session_state.answers = []
        st.session_state.quiz_indices = []
        st.rerun()
    st.stop()

# Get current quiz question
i = st.session_state.step
quiz = quiz_data[st.session_state.quiz_indices[i]]
road_1 = quiz["road_1"]
road_2 = quiz["road_2"]

st.subheader(f"Question {i+1} of {len(st.session_state.quiz_indices)} — Which road is safer?")
col1, col2 = st.columns(2)

user_choice = None

with col1:
    st.markdown("**Road A**")
    for key, value in road_1.items():
        st.write(f"{key}: {value}")
    if st.button("Choose Road A", key=f"road_a_{i}"):
        user_choice = "Road A"

with col2:
    st.markdown("**Road B**")
    for key, value in road_2.items():
        st.write(f"{key}: {value}")
    if st.button("Choose Road B", key=f"road_b_{i}"):
        user_choice = "Road B"

if user_choice is None:
    st.stop()

# Converting the data into a format suitable for model input
df_1 = pd.DataFrame([road_1])
df_2 = pd.DataFrame([road_2])

# Preprocess features
for col in le_dict:
    df_1[col] = le_dict[col].transform(df_1[col])
    df_2[col] = le_dict[col].transform(df_2[col])

for col in expected_features:
    if col not in df_1.columns:
        df_1[col] = 0
    if col not in df_2.columns:
        df_2[col] = 0

# Align the columns with expected_features
df_1 = df_1[expected_features]
df_2 = df_2[expected_features]

# Get predicted values
risk_1 = model.predict(df_1)[0]
risk_2 = model.predict(df_2)[0]
correct_answer = "Road A" if risk_1 < risk_2 else "Road B"
correct = (user_choice == correct_answer)

# Record user answers
st.session_state.answers.append({
    "user_choice": user_choice,
    "correct_answer": correct_answer,
    "correct": correct,
    "risk_1": risk_1,
    "risk_2": risk_2,
    "road_1": road_1,
    "road_2": road_2
})

# Proceed to next question
st.session_state.step += 1
st.rerun()