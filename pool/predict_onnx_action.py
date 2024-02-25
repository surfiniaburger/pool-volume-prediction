from giza_actions.action import Action, action
from giza_actions.task import task
from giza_datasets import DatasetsLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from giza_actions.model import GizaModel

@task(name="Prepare Datasets")
def prepare_datasets():
    print("Prepare dataset...")
    
    # Load data from Polar into a DataFrame
    loader = DatasetsLoader()
    df_polar = loader.load('balancer-daily-trade-volume')

    # Extracting data from the Polar DataFrame
    data = {
        'day': df_polar['day'],
        'pool_id': df_polar['pool_id'],
        'blockchain': df_polar['blockchain'],
        'token_pair': df_polar['token_pair'],
        'pool_volume': df_polar['pool_volume']
    }

    # Creating a new Pandas DataFrame
    df_pandas = pd.DataFrame(data)

    # Perform one-hot encoding for categorical variables
    df_encoded = pd.get_dummies(df_pandas, columns=['blockchain', 'token_pair'])

    # Initialize StandardScaler
    standard_scaler = StandardScaler()

    # Perform Standardization on numerical features
    df_encoded[['pool_volume']] = standard_scaler.fit_transform(df_encoded[['pool_volume']])

    # Convert 'day' column to datetime format
    df_encoded['day'] = pd.to_datetime(df_encoded['day'])

    # Extract relevant features: day of the week, month, and year
    df_encoded['day_of_week'] = df_encoded['day'].dt.dayofweek
    df_encoded['month'] = df_encoded['day'].dt.month
    df_encoded['year'] = df_encoded['day'].dt.year

    # Calculate the total number of data points
    total_data_points = df_encoded.shape[0]

    # Calculate the total number of sequences
    window_size = 30
    total_sequences = total_data_points - window_size + 1

    # Select only necessary columns from the DataFrame
    selected_columns = ['pool_volume', 'blockchain_arbitrum', 'blockchain_avalanche_c', 'blockchain_base', 'blockchain_ethereum', 'blockchain_gnosis', 'blockchain_optimism', 'blockchain_polygon', 'token_pair_wstETH-wUSDM', 'token_pair_xSNXa-YFI', 'token_pair_yCURVE-YFI', 'day_of_week', 'month', 'year']
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

    # Splitting into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(sequences_input, sequences_target, test_size=0.2, random_state=42)

    # Splitting the training set into training and validation sets (80% train, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(sequences_input, sequences_target, test_size=0.2, random_state=42)

    print("✅ Datasets prepared successfully")

    return X_val, y_val



@task(name="Prediction with ONNX")
def prediction(X_val):
    model = GizaModel(model_path="./wavenet.onnx")

    result = model.predict(input_feed={"onnx::Gemm_0": X_val}, verifiable=False)

    return result

@action(name="Execution: Prediction with ONNX", log_prints=True)
def execution():
    # Prepare datasets
    X_val, y_val = prepare_datasets()

    # Perform prediction with ONNX
    result = prediction(X_val)
    print(f"Predicted Pool Volumes: {result}")

    return result

if __name__ == "__main__":
    action_deploy = Action(entrypoint=execution, name="pool-volume-prediction-action")
    action_deploy.serve(name="pool-volume-prediction-deployment")
