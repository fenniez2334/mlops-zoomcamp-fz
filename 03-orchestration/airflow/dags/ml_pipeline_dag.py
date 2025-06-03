from airflow import DAG
try:
    from airflow.operators.python_operator import PythonOperator  # Airflow 3.0+
except ImportError:
    from airflow.operators.python import PythonOperator  # Airflow <3.0
    
from datetime import datetime
import sys
import os

# Add 'scripts' directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

from preprocess_data import preprocess_main
from train import train_main
from register_model import register_main

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

with DAG(
    dag_id='mlops_training_pipeline',
    default_args=default_args,
    schedule_interval=None,
    catchup=False
) as dag:

    preprocess = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_main
    )

    train = PythonOperator(
        task_id='train_model',
        python_callable=train_main
    )

    register = PythonOperator(
        task_id='register_model',
        python_callable=register_main
    )

    preprocess >> train >> register
