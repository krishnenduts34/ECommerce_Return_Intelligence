import joblib

model = joblib.load("model/return_prediction_model.pkl")

print("Model loaded successfully!")
print("Model type:", type(model))