import pickle
import pandas as pd
import numpy as np
import sys


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# Q2. Preparing the output
def prepare_output(df, y_pred, year, month, output_file):
    """
    Prepare the output DataFrame with ride_id and predicted_duration.
    """

    df_result = pd.DataFrame()
    df_result['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result['predicted_duration'] = y_pred

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )


def run():
    taxi_type = sys.argv[1] # 'yellow'
    year = int(sys.argv[2]) # 2023
    month = int(sys.argv[3]) # 3

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/{taxi_type}_{year:04d}-{month:02d}.parquet'
    df = read_data(input_file)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    # Q1: the standard deviation of the predicted duration for this dataset?
    std_dev = np.std(y_pred)
    print(f"Standard deviation of predicted durations: {std_dev:.2f}")
    # Q5: the mean of the predicted duration for this dataset?
    mean_duration = y_pred.mean()
    print(f"Mean predicted durations: {mean_duration:.2f}")


    prepare_output(df, y_pred, year, month, output_file)

if __name__ == '__main__':
    run()