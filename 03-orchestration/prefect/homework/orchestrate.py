import pathlib
import pickle
import pandas as pd
import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import mlflow
from typing import Tuple
from prefect import flow, task
from scipy.sparse import csr_matrix


@task(retries=3, retry_delay_seconds=2)
def read_data(filename: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    df = pd.read_parquet(filename)

    print(f"Read {len(df)} records from {filename}")

    # df.tpep_dropoff_datetime = pd.to_datetime(df.tpep_dropoff_datetime)
    # df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime)

    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    print(f"Adjusted dataframe has {len(df)} records.")

    return df


@task
def add_features(
    df_train: pd.DataFrame,
) -> Tuple[csr_matrix, np.ndarray, DictVectorizer]:
    """Add features to the model"""
    # df_train["PU_DO"] = df_train["PULocationID"] + "_" + df_train["DOLocationID"]
    # df_val["PU_DO"] = df_val["PULocationID"] + "_" + df_val["DOLocationID"]

    categorical = ['PULocationID', 'DOLocationID']
    numerical = ["trip_distance"]

    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    # val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    # X_val = dv.transform(val_dicts)

    y_train = df_train["duration"].values
    # y_val = df_val["duration"].values
    return X_train, y_train, dv


@task(log_prints=True)
def train_best_model(
    X_train: scipy.sparse._csr.csr_matrix,
    # X_val: scipy.sparse._csr.csr_matrix,
    y_train: np.ndarray,
    # y_val: np.ndarray,
    dv: sklearn.feature_extraction.DictVectorizer,
) -> None:
    """Train a linear regression model and log artifacts"""

    with mlflow.start_run():

        mlflow.log_param("train-data-path", "./data/yellow_tripdata_2023-03.parquet")

        lr = LinearRegression()
        lr.fit(X_train, y_train)

        print("Intercept:", lr.intercept_)

        pathlib.Path("models").mkdir(exist_ok=True)
        with open("models/lin_reg.bin", "wb") as f_out:
            pickle.dump((dv, lr), f_out)

        mlflow.log_artifact("models/lin_reg.bin", artifact_path="lin_reg")
        mlflow.sklearn.log_model(lr, artifact_path="models_mlflow")
    return None


@flow
def main_flow(
    train_path: str = "./data/yellow_tripdata_2023-03.parquet",
    # val_path: str = "./data/green_tripdata_2023-02.parquet",
) -> None:
    """The main training pipeline"""

    # MLflow settings
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-experiment")

    # Load
    df_train = read_data(train_path)
    # df_val = read_data(val_path)

    # Transform
    X_train, y_train, dv = add_features(df_train)

    # Train
    train_best_model(X_train, y_train, dv)


if __name__ == "__main__":
    main_flow()
