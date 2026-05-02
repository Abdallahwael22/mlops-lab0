import os
import mlflow
import pandas as pd 

def main():
    # --- HARDCODED MLFLOW AUTHENTICATION ---
    os.environ["MLFLOW_TRACKING_USERNAME"] = "abdallahwael082"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "0ce0281adf60193dd06e4ca2190e4624021a1d69"
    # ---------------------------------------

    # 1. Connect to the MLflow server
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        "https://dagshub.com/abdallahwael082/mlops-lab0.mlflow",
    )
    mlflow.set_tracking_uri(tracking_uri)
    
    # 2. THE MAGIC LINE: Load by Alias instead of Run ID!
    model_uri = "models:/titanic custom model@Production"
    
    print(f"Downloading Production model from DagsHub ({model_uri})...")
    
    # 3. Load the model
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    
    # 4. Create the sample passenger
    sample_passenger = pd.DataFrame([{
        "Pclass": 3,
        "Name": "Mr. John Doe",
        "Sex": "male",
        "Age": 30.0,
        "SibSp": 0,
        "Parch": 0,
        "Ticket": "A/5 21171",
        "Fare": 8.05,
        "Cabin": None,
        "Embarked": "S"
    }])
    
    # 5. Predict
    prediction = loaded_model.predict(sample_passenger)
    
    result = "Survived 🚢" if prediction[0] == 1 else "Did not survive 🌊"
    print(f"\n=== Prediction Results ===")
    print(f"Outcome: {result}")

if __name__ == "__main__":
    main()