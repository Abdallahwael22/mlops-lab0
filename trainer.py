import argparse
import os
import mlflow
import dagshub
from src.logger import ExecutorLogger
from src.training.download_data import download_titanic_data
from src.training.process_data import load_split_data, preprocess_data
from src.training.train import train_optimize_model
from src.training.evaluate import evaluate_model_with_skore
import hydra

from omegaconf import DictConfig
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

@hydra.main(version_base="1.3",config_path="conf", config_name="config")
def main(cfg : DictConfig):
    selected_model = cfg.model.model
    data_path = cfg.paths.data_dir # Not used in the current code, but can be passed to functions if needed
    models_path = cfg.paths.models_dir # Not used in the current code, but can be passed to functions if needed
    reports_path = cfg.paths.reports_dir # Not used in the current code, but can be passed to functions if needed

    log = ExecutorLogger(logs_path="training_runs", level="INFO")
    log.info(f"Starting Titanic Pipeline for: {selected_model}")
    os.environ["MLFLOW_TRACKING_USERNAME"] = "abdallahwael082"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "0ce0281adf60193dd06e4ca2190e4624021a1d69"
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        "https://dagshub.com/abdallahwael082/mlops-lab0.mlflow",
    )
    mlflow.set_tracking_uri(tracking_uri)
    log.info(f"Using MLflow tracking server: {tracking_uri}")
    mlflow.set_experiment("titanic_project")
    
    try:
        with mlflow.start_run(run_name=f"train_{selected_model}"):
            
            mlflow.log_param("model_type", selected_model)
            log.info(" Phase 1: Data Acquisition ")
            download_titanic_data(download_path="data", logger=log)
            
            log.info(" Phase 2: Data Processing ")
            X_train, X_valid, y_train, y_valid = load_split_data(file_path="data/", logger=log)
            
            log.info(" Phase 3: Model Tuning & Training ")
            best_pipeline = train_optimize_model(
                X_train=X_train, 
                y_train=y_train, 
                model_name=selected_model, 
                models_dir="models", 
                logger=log
            )
            
            log.info(" Phase 4: Evaluation & Tracking ")
            evaluate_model_with_skore(
                X=X_valid, 
                y=y_valid, 
                pipeline=best_pipeline, 
                model_name=selected_model, 
                reports_dir="reports", 
                logger=log
            )
            
            log.info("=== Pipeline Execution Complete ===")
            log.info("Run 'skore UI reports/titanic_skore_project' to view your metrics.")

    except Exception as e:
        log.critical(f"Pipeline execution halted due to an error: {e}")

if __name__ == "__main__":
    main()