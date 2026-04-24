import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer   
from sklearn.impute import SimpleImputer
from src.logger import ExecutorLogger
import pandas as pd
""" SOURCE = os.path.join("data", "raw")
DESTINATION = os.path.join("data", "processed") """
class FamilyNameExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        X_out["Family_Name"]=X_out["Name"].apply(lambda name: name.split(",")[0].strip())
        X_out.drop(columns=["Name"], inplace=True)
        return X_out
    
    
def preprocess_data(logger):
    logger.info("Data Processing started")
    
    numeric_features = ['Age', 'Fare', 'SibSp', 'Parch']
    categorical_features = ['Pclass', 'Sex', 'Embarked']
    string_features = ['Family_Name']
    
    numeric_process=SimpleImputer(strategy='median')

    categorical_process= Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    string_process= Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    preprocessor= ColumnTransformer(transformers=[
        ('num', numeric_process, numeric_features),
        ('cat', categorical_process, categorical_features),
        ('str', string_process, string_features)
    ])  
    return preprocessor


def load_split_data(file_path: str, logger):
    if logger is None:
        logger = ExecutorLogger(level="INFO")
    logger.info("Data Processing started")
    df = pd.read_csv(os.path.join(file_path, f"train.csv"))
    X = df.drop(columns=['Survived', 'PassengerId', 'Ticket', 'Cabin'])
    y = df['Survived']
    logger.info(f"Data loaded successfully. Features shape: {X.shape}")
    X_train, X_valid, y_train, y_valid  = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    return X_train, X_valid, y_train, y_valid
