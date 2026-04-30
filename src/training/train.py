import os
import joblib
import pickle
import numpy as np
import optuna
import mlflow
from src.training.process_data import FamilyNameExtractor, preprocess_data
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from src.logger import ExecutorLogger

def train_optimize_model(X_train, y_train, model_name: str, models_dir: str = "models", logger: ExecutorLogger = None):
    if logger is None:
        logger= ExecutorLogger(level="INFO")
    
    os.makedirs(models_dir, exist_ok=True)
    preprocessor = preprocess_data(logger)
    
    def objective(trial):
        if model_name == "random_forest":
            n_estimators=trial.suggest_int("n_estimators",50,300)
            max_depth=trial.suggest_int("max_depth",4,20)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        elif model_name == "logistic_regression":
            c=trial.suggest_float("C", 0.01, 10.0, log=True)
            model = LogisticRegression(C=c, random_state=42)
        else:
            raise ValueError("Unsupported model name")  
        pipeline=Pipeline(steps=[("name_extractor", FamilyNameExtractor()),
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        
        scores_dict=cross_validate(pipeline, X_train, y_train, cv=5, scoring="accuracy")
        score = scores_dict['test_score'].mean()
        return score
    logger.info(f"Starting Optuna hyperparameter tuning for {model_name}...")
    study=optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)
    logger.info(f"Best Optuna parameters found: {study.best_params}")
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_score", study.best_value)
    
    if model_name == "random_forest":
        best_model = RandomForestClassifier(**study.best_params, random_state=42)
    elif model_name == "logistic_regression":
        best_model = LogisticRegression(**study.best_params, random_state=42)
    best_pipeline=Pipeline(steps=[("name_extractor", FamilyNameExtractor()),
            ("preprocessor", preprocessor),
            ("model", best_model)
        ])   
    best_pipeline.fit(X_train, y_train)
    model_path = os.path.join(models_dir, f"{model_name.replace(' ', '_')}_model.pkl")
    joblib.dump(best_pipeline, model_path)
    logger.info(f"Best {model_name} model saved to {model_path}")
    mlflow.sklearn.log_model(best_pipeline, artifact_path="models")
    return best_pipeline