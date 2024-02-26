from giza_actions.action import Action, action
from giza_actions.task import task
from giza_datasets import DatasetsLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from giza_actions.model import GizaModel


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



@task(name="Prediction with ONNX")
def prediction(X_val):
    model = GizaModel(model_path="./wavenet.onnx")

    result = model.predict(input_feed={"input_1": X_val.reshape(1, 30, 13)}, verifiable=False)

    return result

@action(name="Execution: Prediction with ONNX", log_prints=True)
def execution():
    # Prepare datasets
    X_train, y_train, X_test, y_test, X_val, y_val = prepare_datasets()

    selected_columns = ['blockchain_arbitrum', 'blockchain_avalanche_c', 'blockchain_base', 'blockchain_ethereum', 'blockchain_gnosis', 'blockchain_optimism', 'blockchain_polygon', 'token_pair_wstETH-wUSDM', 'token_pair_xSNXa-YFI', 'token_pair_yCURVE-YFI', 'day_of_week', 'month', 'year']
    X_val_selected = X_val[selected_columns]

    # Convert the DataFrame to a NumPy array and ensure it is of type float32
    X_val_array = X_val_selected.astype(np.float32).values

    print("Shape of X_val_array before reshaping:", X_val_array.shape)

    # Define the window size
    window_size = 30

    # Initialize an empty list to store the reshaped samples
    reshaped_samples = []

    # Slide a window of size window_size over the original data
    for i in range(len(X_val_array) - window_size + 1):
        # Extract a window of size window_size
        window = X_val_array[i : i + window_size]
        # Append the window to the list of reshaped samples
        reshaped_samples.append(window)

    # Convert the list of reshaped samples to a numpy array
    X_val_reshaped = np.array(reshaped_samples)

    # Ensure the reshaped array has the correct shape
    print("Shape of X_val_reshaped:", X_val_reshaped.shape)


    # Ensure the array has the correct shape (number of samples, number of features), reshape it to add dimension
    # This will conert it from a 2D array to a 3D array
    X_val_reshaped = np.expand_dims(X_val_reshaped, axis=0)

    # Perform prediction with ONNX
    result = prediction(X_val_reshaped)
    print(f"Predicted Pool Volumes: {result}")

    return result

if __name__ == "__main__":
    action_deploy = Action(entrypoint=execution, name="pool-volume-prediction-action")
    action_deploy.serve(name="pool-volume-prediction-deployment")
