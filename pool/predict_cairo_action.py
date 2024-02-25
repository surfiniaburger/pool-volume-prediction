from giza_actions.action import Action, action
from giza_actions.task import task
from giza_datasets import DatasetsLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from giza_actions.model import GizaModel
import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Generate random input data similar to X_val
def generate_random_input_data(num_samples):
    # Define the columns
    columns = ['day', 'pool_id'] + ['blockchain_' + str(i) for i in range(10)] + ['token_pair_' + str(i) for i in range(10)] + ['day_of_week', 'month', 'year']

    # Generate random data
    random_data = []
    for _ in range(num_samples):
        # Generate random date
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2024, 12, 31)
        random_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))

        # Generate random pool ID
        pool_id = '0x' + ''.join(random.choices('0123456789abcdef', k=40))

        # Generate random blockchain and token pair values
        blockchain_values = [random.choice([True, False]) for _ in range(10)]
        token_pair_values = [random.choice([True, False]) for _ in range(10)]

        # Generate random day of week, month, and year
        day_of_week = random.randint(0, 6)
        month = random.randint(1, 12)
        year = random.randint(2020, 2024)

        # Append to random data list
        random_data.append([random_date, pool_id] + blockchain_values + token_pair_values + [day_of_week, month, year])

    # Create DataFrame
    df_random = pd.DataFrame(random_data, columns=columns)

    return df_random




window_size = 30
loader = DatasetsLoader()
# Load data from Polar into a DataFrame
df_polar = loader.load('balancer-daily-trade-volume')

# Extracting data from the Polar DataFrame
data = {
    'day': df_polar['day'],
    'pool_id': df_polar['pool_id'],
    'blockchain': df_polar['blockchain'],
    'token_pair': df_polar['token_pair'],
    'trading_volume_usd': df_polar['trading_volume_usd']
}

# Creating a new Pandas DataFrame
df_pandas = pd.DataFrame(data)

# Perform one-hot encoding for categorical variables
df_encoded = pd.get_dummies(df_pandas, columns=['blockchain', 'token_pair'])

# Initialize StandardScaler
standard_scaler = StandardScaler()

# Perform Standardization on numerical features
df_encoded[['trading_volume_usd']] = standard_scaler.fit_transform(df_encoded[['trading_volume_usd']])

# Convert 'day' column to datetime format
df_encoded['day'] = pd.to_datetime(df_encoded['day'])

# Extract relevant features: day of the week, month, and year
df_encoded['day_of_week'] = df_encoded['day'].dt.dayofweek
df_encoded['month'] = df_encoded['day'].dt.month
df_encoded['year'] = df_encoded['day'].dt.year

# Calculate the total number of data points
total_data_points = df_encoded.shape[0]

# Calculate the total number of sequences
total_sequences = total_data_points - window_size + 1

# Select only necessary columns from the DataFrame
selected_columns = ['trading_volume_usd', 'blockchain_arbitrum', 'blockchain_avalanche_c', 'blockchain_base', 'blockchain_ethereum', 'blockchain_gnosis', 'blockchain_optimism', 'blockchain_polygon', 'token_pair_wstETH-wUSDM', 'token_pair_xSNXa-YFI', 'token_pair_yCURVE-YFI', 'day_of_week', 'month', 'year']
df_selected = df_encoded[selected_columns]

# Slide a window of this length across your time-series data
sequences_input = []
sequences_target = []

for i in range(total_sequences):
    # Extract the historical data points as the input sequence
    input_sequence = df_selected.iloc[i : i + window_size].values
    sequences_input.append(input_sequence)
    
    # Extract the next data point as the target for prediction
    target = df_selected.iloc[i + window_size - 1, 2]
    sequences_target.append(target)

# Convert lists to numpy arrays
sequences_input = np.array(sequences_input)
sequences_target = np.array(sequences_target)

# Reshape the target sequences to match the shape of the input sequences
sequences_target = sequences_target.reshape(-1, 1)

sequences_input = sequences_input.astype(np.float32)
sequences_target = sequences_target.astype(np.float32)

@task(name="Prepare Datasets")
def prepare_datasets():
    print("Prepare dataset...")

    # Splitting into training and testing sets (80% train, 20% test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(df_encoded.drop(columns=['trading_volume_usd']), df_encoded['trading_volume_usd'], test_size=0.2, random_state=42)

    # Splitting the training set into training and validation sets (80% train, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    print("✅ Datasets prepared successfully")

    return X_train, y_train, X_test, y_test, X_val, y_val




# Update with your actual model ID and version ID
MODEL_ID = 377  # Replace with your model ID
VERSION_ID = 1   # Replace with your version ID

@task(name=f'Pool Volume Forecasting Task')
def forecast_pool_volumes(input_data, model_id, version_id):
    # Load the GizaModel for pool volume forecasting
    model = GizaModel(id=model_id, version=version_id)

    # Perform prediction using the loaded model
    result, request_id = model.predict(input_feed={"input": input_data}, verifiable=True, output_dtype='Tensor<FP16x16>')

    return result, request_id

@action(name=f'Execute Pool Volume Forecasting', log_prints=True)
def execution():
    # Use validation data for forecasting
    X_train, y_train, X_test, y_test, X_val, y_val = prepare_datasets()

    # Perform pool volume forecasting using the validation data
    result, request_id = forecast_pool_volumes(X_val, MODEL_ID, VERSION_ID)

    # Print or handle the prediction result and request ID
    print("Forecasted Pool Volumes: ", result)
    print("Request ID: ", request_id)

    # Number of random samples to generate
    num_samples = 10  # Adjust as needed

    # Generate random input data
    random_input_data = generate_random_input_data(num_samples)

    # Perform pool volume forecasting using the validation data
    result, request_id = forecast_pool_volumes(random_input_data, MODEL_ID, VERSION_ID)

    # Print or handle the prediction result and request ID
    print("Forecasted Pool Volumes (random): ", result)
    print("Request ID: ", request_id)

    return result, request_id

if __name__ == "__main__":
    action_deploy = Action(entrypoint=execution, name="pool-volume-prediction-with-cairo-action")
    action_deploy.serve(name="pool-volume-prediction-with-cairo-deployment")
