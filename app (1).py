import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import gradio as gr

# ------------------------
# Load and prepare dataset
# ------------------------
df = pd.read_csv("Training.csv")

# Drop unnamed columns if present
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Fill missing values
df = df.fillna(0)

# Features and labels
X = df.drop("prognosis", axis=1)
y = df["prognosis"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y_encoded)

# Prepare list of all symptoms for CheckboxGroup
symptoms_list = list(X.columns)

# ------------------------
# Prediction function
# ------------------------
def predict_disease(selected_symptoms):
    input_data = [1 if symptom in selected_symptoms else 0 for symptom in X.columns]
    prediction = model.predict([input_data])[0]
    return f"🩺 Predicted Disease: {le.inverse_transform([prediction])[0]}"

# ------------------------
# Gradio interface
# ------------------------
if __name__ == "__main__":
    # Automatically create valid examples from dataset columns
    auto_examples = []
    for col in symptoms_list[:10]:  # take first 10 symptoms for examples
        auto_examples.append([col])

    demo = gr.Interface(
        fn=predict_disease,
        inputs=gr.CheckboxGroup(
            choices=symptoms_list,
            label="Select your symptoms",
        ),
        outputs=gr.Textbox(label="Predicted Disease"),
        title="AI Health Symptom Checker 🧬",
        description="Select your symptoms to get an AI-based possible diagnosis. (Educational use only)",
        examples=auto_examples,
        flagging_mode="never"  # updated for Gradio >=3.40
    )
    demo.launch()
