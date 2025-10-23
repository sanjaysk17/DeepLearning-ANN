import gradio as gr
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# === Load saved model and scaler ===
model = load_model("customer_churn_ann.h5")          # Your trained ANN
scaler = joblib.load("scaler.pkl")                   # Make sure this is a StandardScaler object saved via joblib

# === Prediction function ===
def predict_churn(CreditScore, Gender, Age, Tenure, Balance, NumOfProducts,
                  HasCrCard, IsActiveMember, EstimatedSalary,
                  Geography_France, Geography_Germany, Geography_Spain):

    # Map Gender to 0/1
    Gender = 1 if Gender == "Male" else 0

    # Prepare input as DataFrame
    input_data = pd.DataFrame([[CreditScore, Gender, Age, Tenure, Balance,
                                NumOfProducts, HasCrCard, IsActiveMember,
                                EstimatedSalary, Geography_France, Geography_Germany, Geography_Spain]],
                              columns=['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
                                       'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                                       'EstimatedSalary', 'Geography_France', 'Geography_Germany', 'Geography_Spain'])

    # Scale numeric input
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0][0]
    result = "🚨 Customer Likely to Exit" if prediction >= 0.5 else "✅ Customer Likely to Stay"
    confidence = f"{prediction * 100:.2f}%" if prediction >= 0.5 else f"{(1 - prediction) * 100:.2f}%"

    return f"{result}\nConfidence: {confidence}"

# === Gradio Interface ===
iface = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Number(label="Credit Score", value=650),
        gr.Radio(["Male", "Female"], label="Gender"),
        gr.Number(label="Age", value=35),
        gr.Number(label="Tenure", value=5),
        gr.Number(label="Balance", value=100000.0),
        gr.Number(label="Num Of Products", value=1),
        gr.Radio([0, 1], label="Has Credit Card (0=No, 1=Yes)", value=1),
        gr.Radio([0, 1], label="Is Active Member (0=No, 1=Yes)", value=1),
        gr.Number(label="Estimated Salary", value=80000.0),
        gr.Radio([0, 1], label="Geography France", value=1),
        gr.Radio([0, 1], label="Geography Germany", value=0),
        gr.Radio([0, 1], label="Geography Spain", value=0),
    ],
    outputs="text",
    title="Customer Churn Prediction (Pretrained ANN)",
    description="Enter customer details to predict churn using the saved ANN model."
)

# Launch app
iface.launch()
