# Road Safety Quiz
This is a simple quiz app made with Streamlit.
It shows two roads and asks which one is safer.
The prediction is done using a LightGBM model trained on Kaggle.

## Features:
- 3 random questions per session
- Predicts accident risk based on road conditions
- Shows correct answers and risk scores after the quiz

## What I learned:
- How to use Streamlit session state
- How to load a model and make predictions
- How to deploy on Streamlit Cloud

## Demo
- Live App: https://road-safety-quiz.streamlit.app

## Files
- `app.py` – Main Streamlit application
- `quiz_data.py` – Contains quiz question data
- `accident_model.pkl` – Pre-trained LightGBM model
- `label_encoders.pkl` – Saved LabelEncoders
- `requirements.txt` – App dependencies
