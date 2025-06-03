import os
import pickle
import click
import mlflow

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run():
        # new_params = {}
        # for param in RF_PARAMS:
        #     new_params[param] = int(params[param])
        new_params = {
            'n_estimators': min(200, int(params.get('n_estimators', 100))),
            'max_depth': min(20, int(params.get('max_depth', 10))),
            'min_samples_split': int(params.get('min_samples_split', 2)),
            'min_samples_leaf': int(params.get('min_samples_leaf', 1)),
            'random_state': int(params.get('random_state', 42)),
        }

        rf = RandomForestRegressor(**new_params)
        rf.fit(X_train, y_train)

        # Evaluate model on the validation and test sets
        val_rmse = root_mean_squared_error(y_val, rf.predict(X_val))
        mlflow.log_metric("val_rmse", val_rmse)
        test_rmse = root_mean_squared_error(y_test, rf.predict(X_test))
        mlflow.log_metric("test_rmse", test_rmse)

        # Log model for registry
        mlflow.sklearn.log_model(rf, artifact_path="model")


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
def run_register_model(data_path: str, top_n: int):

    client = MlflowClient()

    print("Getting top hyperparameter runs from experiment:", HPO_EXPERIMENT_NAME, flush=True)

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )
    print(f"Retraining and evaluating top {top_n} models...", flush=True)
    for i,run in enumerate(runs):
        print(f"--> Training model {i+1}/{top_n}, run_id: {run.info.run_id}", flush=True)
        try:
            train_and_log_model(data_path=data_path, params=run.data.params)
        except Exception as e:
            print(f"!!! Error training model {i+1}: {e}", flush=True)

    # Select the model with the lowest test RMSE
    print("Selecting best model based on test RMSE...", flush=True)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs( 
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.test_rmse ASC"]
    )[0]

    run_id = best_run.info.run_id
    test_rmse = best_run.data.metrics["test_rmse"]
    print(f"Best model: run_id={run_id}, test_rmse={test_rmse}", flush=True)

    # Register the best model
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri=model_uri, name="random-forest-best-model")
    print(f"Registered model from run ID {best_run.info.run_id} with test RMSE: {best_run.data.metrics['test_rmse']:.3f}")


if __name__ == '__main__':
    run_register_model()


def register_model_main():
    data_path = os.environ.get("DATA_PATH", "/opt/airflow/output")
    top_n = int(os.environ.get("TOP_N", 5))
    run_register_model.main(args=[f"--data_path={data_path}", f"--top_n={top_n}"])