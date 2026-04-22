import os
from venv import logger
from dotenv import load_dotenv
load_dotenv()
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
from src.logger import ExecutorLogger



def download_titanic_data(download_path, logger) -> str:

    logger.info("Downloading Titanic dataset from Kaggle...")
    api=KaggleApi()
    api.authenticate()
    os.makedirs(download_path, exist_ok=True)
    api.competition_download_files('titanic', path=download_path)
    zip_file_path = os.path.join(download_path, 'titanic.zip')
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(download_path)
    os.remove(zip_file_path)
    os.remove(os.path.join(download_path, 'gender_submission.csv'))
    logger.info(f"Titanic dataset downloaded to {download_path}")
    return download_path


if __name__ == "__main__":
    logger = ExecutorLogger(level="INFO")
    download_path = "data/raw/titanic"
    download_titanic_data(download_path, logger)