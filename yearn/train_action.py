import numpy as np
import torch
import torch.nn as nn
import torch.onnx
import torch.optim as optim
from giza_actions.action import Action, action
from giza_actions.task import task
from giza_datasets import DatasetsHub, DatasetsLoader
from torch.utils.data import DataLoader, TensorDataset
import os
import certifi
import polars as pl
loader = DatasetsLoader()
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

os.environ['SSL_CERT_FILE'] = certifi.where()


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Neural Network Definition
input_size = 3  # 14x14
hidden_size1 = 64
hidden_size2 = 32
num_epochs = 50
output_size = 1
batch_size = 128
learning_rate = 0.001
num_models = 5  # Number of models in the ensemble

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

@task(name="Prepare Datasets")
def prepare_datasets():
    print("Prepare dataset...")

    # Assuming 'loader' is a custom module for loading datasets
    df = loader.load('yearn-individual-deposits')

    # Assuming your Polar DataFrame is already loaded and named 'df'
    # Extract features and target variable
    features = df[['evt_block_time', 'evt_block_number', 'token_decimals']]
    target = df['value']

    # Encode 'token_contract_address' as categorical
    unique_addresses = sorted(set(df['token_contract_address']))
    address_to_label = {addr: i for i, addr in enumerate(unique_addresses)}
    encoded_addresses = [address_to_label[addr] for addr in df['token_contract_address']]

    # Create a new DataFrame with the encoded 'token_contract_address' column
    features = pl.DataFrame({
          'evt_block_number': features['evt_block_number'],
          'token_decimals': features['token_decimals'],
          'token_contract_address': encoded_addresses
    })

    # Convert DataFrame to Pandas DataFrame and then to PyTorch tensors
    x_data = torch.tensor(features.to_pandas().values.astype("float32"))
    y_data = torch.tensor(target.to_pandas().values.astype("float32"))

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    print("✅ Datasets prepared successfully")

    return x_train, y_train, x_test, y_test

@task(name="Create Loaders")
def create_data_loaders(x_train, y_train, x_test, y_test):
    print("Create loaders...")

    # Load the dataset
    df = loader.load('yearn-individual-deposits')

    # Extract features and target variable
    features = df[['evt_block_number', 'token_decimals']]
    target = df['value']

    # Encode 'token_contract_address' as categorical
    unique_addresses = sorted(set(df['token_contract_address']))
    address_to_label = {addr: i for i, addr in enumerate(unique_addresses)}
    encoded_addresses = [address_to_label[addr] for addr in df['token_contract_address']]

    # Create a new DataFrame with the encoded 'token_contract_address' column
    features = pl.DataFrame({
        'evt_block_number': features['evt_block_number'],
        'token_decimals': features['token_decimals'],
        'token_contract_address': encoded_addresses
    })

    # Convert DataFrame to PyTorch tensors
    x_data = torch.tensor(features.to_pandas().values.astype("float32"))
    y_data = torch.tensor(target.to_pandas().values.astype("float32"))

    # Define batch size
    batch_size = 256

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    # Define DataLoader and TensorDataset
    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False
    )

    print("✅ Loaders created!")

    return train_loader, test_loader

@task(name="Train model")
def train_model(train_loader):
    """
    Train multiple instances of the neural network model (ensemble).

    Returns:
        models (list): List of trained neural network models.
    """
    models = []
    for _ in range(num_models):
        model = NeuralNet(input_size=3, hidden_size1=64, hidden_size2=32, output_size=1)
        model.to(device)

        # Define Loss function and Optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        print("Train model...")
        for epoch in range(num_epochs):
            model.train()  # Set model to training mode
            running_loss = 0.0

            for i, (data, labels) in enumerate(train_loader):
                data = data.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if (i + 1) % 100 == 0:
                    print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}")
                    running_loss = 0.0

        models.append(model)
        print("✅ Model trained successfully")
    return models

@task(name="Test model")
def test_model(model, test_loader, threshold=0.5):
    print("Test model...")
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        y_true = []
        y_pred = []
        for data, labels in test_loader:
            # Move data to device and reshape if necessary
            data = data.to(device)
            if len(data.shape) > 2:  # If data is not flattened
                data = data.view(data.size(0), -1)
            labels = labels.to(device)
            # Forward pass
            outputs = model(data)
            predicted_probs = torch.sigmoid(outputs)
            predicted_labels = (predicted_probs > threshold).float()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted_labels.cpu().numpy())

        y_true_binary = [1 if label > threshold else 0 for label in y_true]
        y_pred_binary = [1 if label > threshold else 0 for label in y_pred]

        precision = precision_score(y_true_binary, y_pred_binary)
        recall = recall_score(y_true_binary, y_pred_binary)
        f1 = f1_score(y_true_binary, y_pred_binary)

        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

    return precision, recall, f1

@task(name="Evaluate ensemble")
def evaluate_ensemble(models, test_loader):
    predictions = []
    for model in models:
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(device)
                labels = labels.to(device)
                outputs = model(data)
                predictions.extend(outputs.cpu().numpy())
    predictions = np.array(predictions)
    return np.mean(predictions, axis=0)

@task(name="Convert To ONNX")
def convert_to_onnx(model, onnx_file_path):
    """
    Convert the trained PyTorch model to ONNX format and save it.

    Args:
        model (NeuralNet): Trained PyTorch model.
        onnx_file_path (str): File path to save the ONNX model.

    Returns:
        None
    """
    # Create a dummy input
    dummy_input = torch.randn(1, input_size).to(device)

    # Export the model to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
    )

    print(f"Model has been converted to ONNX and saved as {onnx_file_path}")

@action(name="Action: Convert To ONNX", log_prints=True)
def execution():
    # Prepare datasets
    x_train, y_train, x_test, y_test = prepare_datasets()

    # Create data loaders
    train_loader, test_loader = create_data_loaders(x_train, y_train, x_test, y_test)

    # Train the model
    model = train_model(train_loader)

    # Test the model
    test_model(model, train_loader)

    # Convert to ONNX
    onnx_file_path = "yearn_model.onnx"
    convert_to_onnx(model, onnx_file_path)

    # Evaluate the ensemble
    evaluate_ensemble(model, test_loader)
    ensemble_predictions = evaluate_ensemble(model, test_loader)
    ensemble_rmse = np.sqrt(np.mean((ensemble_predictions - y_test.numpy())**2))
    print(f"Ensemble RMSE: {ensemble_rmse}")

# Create an Action object and serve it
if __name__ == "__main__":
    action_deploy = Action(entrypoint=execution, name="pytorch-yearn-action")
    action_deploy.serve(name="pytorch-yearn-deployment")
