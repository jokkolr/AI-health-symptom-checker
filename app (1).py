
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import gradio as gr

# Load and clean dataset
df = pd.read_csv("Training.csv")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.fillna(0)

# Prepare features and labels
X = df.drop(["prognosis"], axis=1)
y = df["prognosis"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train model
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X, y_encoded)

# Prediction function
def predict_disease(symptoms):
    symptoms = [s.strip().lower().replace(" ", "_") for s in symptoms.split(",")]
    input_data = [0] * len(X.columns)
    for symptom in symptoms:
        if symptom in X.columns:
            input_data[X.columns.get_loc(symptom)] = 1
    prediction = model.predict([input_data])[0]
    disease_name = le.inverse_transform([prediction])[0]
    return f"🩺 Predicted Disease: {disease_name}"

# Gradio Interface
demo = gr.Interface(
    fn=predict_disease,
    inputs=gr.Textbox(label="Enter symptoms (comma separated)", placeholder="e.g. fever, cough, fatigue"),
    outputs=gr.Textbox(label="Predicted Disease"),
    title="AI Health Symptom Checker 🧬",
    description="Enter your symptoms to get an AI-based possible diagnosis. (Educational use only)",
    examples=[
        ["fever, cough, fatigue"],
        ["itching, skin_rash, nodal_skin_eruptions"],
        ["chest_pain, fast_heart_rate, breathlessness"]
    ]
)

demo.launch()
