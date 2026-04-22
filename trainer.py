import argparse
from src.logger import ExecutorLogger
from src.training.download_data import download_titanic_data
from src.training.process_data import load_split_data, preprocess_data
from src.training.train import train_optimize_model
from src.training.evaluate import evaluate_model_with_skore

def main():
    parser = argparse.ArgumentParser(description="Titanic MLOps Pipeline with Optuna")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["random forest", "logistic regression"], 
        default="random forest",
        help="Choose which model to tune, train, and evaluate."
    )
    args = parser.parse_args()
    selected_model = args.model

    log = ExecutorLogger(logs_path="training_runs", level="INFO")
    
    log.info(f"Starting Titanic Pipeline for: {selected_model}")
    
    try:
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