import os
import pickle
import click
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)
    
mlflow.set_experiment("my-homework")


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))


    with mlflow.start_run():
        mlflow.set_tag("developer", "cristian")
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        print(f"Validation RMSE: {rmse}")

        # Log the RMSE metric
        mlflow.log_metric("rmse", rmse)

        # Log the trained model
        mlflow.sklearn.log_model(rf, artifact_path="artifacts")
        print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")



if __name__ == '__main__':
    run_train()