import skore
import os
from src.logger import ExecutorLogger

def evaluate_model_with_skore(X, y, pipeline, model_name: str, reports_dir: str = "reports", logger: ExecutorLogger = None):
    """Uses skore to cross-validate and track metrics for a specific pipeline."""
    if logger is None:
        logger = ExecutorLogger()

    logger.info(f"Setting up skore project in {reports_dir}")
    os.makedirs(reports_dir, exist_ok=True)
    
    project_path = os.path.join(reports_dir, "titanic_skore_project")
    project = skore.Project(project_path)
    logger.info(f"Running skore cross-validation for optimized {model_name}")
    report = skore.CrossValidationReport(pipeline, X, y)
    project.put(f"{model_name}_optimized_report", report)
    logger.info(f"Metrics for {model_name} tracked successfully in skore project.")