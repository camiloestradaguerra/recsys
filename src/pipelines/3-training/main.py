"""
Model Training Component

This module orchestrates the training of the deep neural network for restaurant
recommendations. It implements:
- Data loading and splitting
- Training loop with early stopping
- Validation monitoring
- Model checkpointing
- Feature importance analysis

The training process uses the best hyperparameters found through Bayesian optimization
and includes the critical fix for location/time filtering.

Author: Equipo ADX  
Date: 2025-11-13
"""

import argparse
import sys
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import boto3
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import yaml
import mlflow
import mlflow.pytorch

from model import EstablishmentDNN, LocationTimeFilter

# Import S3DataManager from cleaning_data pipeline
sys.path.insert(0, str(Path(__file__).parent.parent / "0-cleaning_data"))
from main import S3DataManager

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",
    colorize=True
)


class EstablishmentDataset(Dataset):
    """
    PyTorch Dataset for establishment recommendations.
    
    This dataset wraps feature matrices and target labels, providing an interface
    for PyTorch's DataLoader. It's designed to handle the high cardinality of
    establishments efficiently by using integer labels and lazy loading.
    
    Parameters
    ----------
    features : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    targets : np.ndarray  
        Target labels of shape (n_samples,), encoded as integers.
        
    Examples
    --------
    >>> dataset = EstablishmentDataset(X_train, y_train)
    >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        """Initialize dataset with features and targets."""
        # Convert to float32 to handle any object dtype issues
        features_float = features.astype(np.float32)
        targets_int = targets.astype(np.int64)

        self.features = torch.FloatTensor(features_float)
        self.targets = torch.LongTensor(targets_int)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Feature vector and target label.
        """
        return self.features[idx], self.targets[idx]


def load_config(config_path: str) -> Dict:
    """
    Load training configuration from YAML file.
    
    The configuration file contains hyperparameters, data splits, and feature
    definitions. This centralized configuration makes it easy to experiment
    with different settings.
    
    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
        
    Returns
    -------
    Dict
        Configuration dictionary with model, training, and data settings.
        
    Examples
    --------
    >>> config = load_config('config.yaml')
    >>> batch_size = config['training']['batch_size']
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration loaded from {config_path}")
    return config


def prepare_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    train_split: float = 0.7,
    val_split: float = 0.15,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare and split data for training.
    
    This function extracts features and targets, then performs stratified splitting
    to ensure each set has representative samples of all establishments. Stratification
    is crucial when dealing with imbalanced classes (some establishments are much
    more popular than others).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with engineered features.
    feature_cols : List[str]
        Names of columns to use as features.
    target_col : str
        Name of the target column (encoded establishment IDs).
    train_split : float, optional
        Proportion of data for training. Default is 0.7.
    val_split : float, optional
        Proportion of data for validation. Default is 0.15.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.
        
    Returns
    -------
    Tuple[np.ndarray, ...]
        Six arrays: X_train, X_val, X_test, y_train, y_val, y_test.
        
    Examples
    --------
    >>> X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(
    ...     df, feature_cols, 'establecimiento_encoded'
    ... )
    
    Notes
    -----
    The function performs two sequential splits:
    1. Split into train and temp (train_split)
    2. Split temp into val and test (proportionally)
    
    This ensures the splits sum to 1.0 and each set has the desired size.
    """
    logger.info("Preparing data splits...")

    # Extract features and target
    X = df[feature_cols].astype('float32').values
    y = df[target_col].astype('int64').values
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    logger.info(f"Number of unique establishments: {len(np.unique(y))}")
    
    # First split: train and temp
    # Note: Using random split instead of stratified due to sparse classes
    # With 1089 establishments and 2000 samples, many classes have only 1-2 samples
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        train_size=train_split,
        random_state=random_state
    )
    
    # Second split: val and test
    val_proportion = val_split / (1 - train_split)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        train_size=val_proportion,
        random_state=random_state
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Val set: {X_val.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """
    Train the model for one epoch.
    
    An epoch is one complete pass through the training data. This function
    processes data in batches, computes loss, and updates model parameters.
    
    Parameters
    ----------
    model : nn.Module
        The neural network model to train.
    train_loader : DataLoader
        DataLoader providing training batches.
    criterion : nn.Module
        Loss function (typically CrossEntropyLoss).
    optimizer : optim.Optimizer
        Optimization algorithm (typically Adam).
    device : torch.device
        Device to run computations on (CPU or CUDA).
        
    Returns
    -------
    float
        Average training loss across all batches.
        
    Notes
    -----
    The function uses gradient clipping to prevent exploding gradients, which
    can occur in deep networks. The clip value of 1.0 is a reasonable default
    but can be tuned if training becomes unstable.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for features, targets in tqdm(train_loader, desc="Training", leave=False):
        features = features.to(device)
        targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate the model on a dataset.
    
    This function computes loss and accuracy without updating model parameters.
    It's used during validation and testing to monitor performance.
    
    Parameters
    ----------
    model : nn.Module
        The model to evaluate.
    data_loader : DataLoader
        DataLoader providing evaluation batches.
    criterion : nn.Module
        Loss function.
    device : torch.device
        Device to run computations on.
        
    Returns
    -------
    Tuple[float, float]
        Average loss and accuracy (0-1 scale).
        
    Notes
    -----
    Accuracy is computed as the proportion of samples where the predicted
    class (argmax of output) matches the true class. For recommendation
    systems, we often care more about top-k accuracy, but top-1 accuracy
    provides a useful single-number metric.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, targets in tqdm(data_loader, desc="Evaluating", leave=False):
            features = features.to(device)
            targets = targets.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            
            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int,
    patience: int,
    model_path: Path
) -> Dict[str, List[float]]:
    """
    Train the model with early stopping.
    
    This is the main training loop that orchestrates multiple epochs of training
    and validation. It implements early stopping to prevent overfitting: if
    validation loss doesn't improve for `patience` epochs, training stops.
    
    Parameters
    ----------
    model : nn.Module
        The model to train.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    criterion : nn.Module
        Loss function.
    optimizer : optim.Optimizer
        Optimizer.
    device : torch.device
        Computation device.
    epochs : int
        Maximum number of epochs to train.
    patience : int
        Number of epochs to wait for improvement before stopping.
    model_path : Path
        Path where the best model will be saved.
        
    Returns
    -------
    Dict[str, List[float]]
        Training history with keys 'train_loss', 'val_loss', 'val_acc'.
        
    Examples
    --------
    >>> history = train_model(
    ...     model, train_loader, val_loader,
    ...     criterion, optimizer, device,
    ...     epochs=50, patience=10, model_path=Path('model.pth')
    ... )
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(history['train_loss'], label='Train')
    >>> plt.plot(history['val_loss'], label='Val')
    
    Notes
    -----
    The function saves two checkpoints:
    - Best model (lowest validation loss)
    - Latest model (end of training)
    
    This allows resuming training or using the best model for inference.
    
    AWS SageMaker Note
    ------------------
    In SageMaker, replace model_path with:
        model_path = Path(os.environ['SM_MODEL_DIR']) / 'model.pth'
    This ensures the model is saved to the correct location for artifact collection.
    """
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    logger.info(f"Starting training for up to {epochs} epochs...")
    
    for epoch in range(epochs):
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate on validation set
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Store metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.4f} - "
            f"Val Loss: {val_loss:.4f} - "
            f"Val Acc: {val_acc:.4f}"
        )
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, model_path)
            logger.success(f"New best model saved with val_loss: {val_loss:.4f}")
        else:
            epochs_without_improvement += 1
            logger.info(f"No improvement for {epochs_without_improvement} epochs")
            
        # Early stopping check
        if epochs_without_improvement >= patience:
            logger.warning(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    logger.success("Training completed!")
    return history

def load_pickle_from_s3(s3_path: str):
    """
    Load a pickle file directly from S3.
    """
    s3 = boto3.client('s3')
    bucket, key = s3_path.replace("s3://", "").split("/", 1)

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        s3.download_fileobj(bucket, key, tmp_file)
        tmp_file_path = tmp_file.name

    try:
        return joblib.load(tmp_file_path)
    finally:
        os.remove(tmp_file_path)


def upload_to_s3(local_path: Path, bucket: str, s3_key: str):
    """
    Upload a file to an S3 bucket.
    """
    s3 = boto3.client("s3")
    s3.upload_file(str(local_path), bucket, s3_key)
    logger.success(f"File uploaded to s3://{bucket}/{s3_key}")


def run_training_pipeline(
    input_path: str,
    model_path: str,
    encoders_path: str,
    config_path: str,
    s3_bucket: str,
    s3_output_prefix: str
) -> None:
    """
    Execute the complete training pipeline.
    
    This orchestrates all training steps:
    1. Load configuration and data
    2. Prepare train/val/test splits
    3. Create data loaders
    4. Initialize model
    5. Train with early stopping
    6. Save model and artifacts
    
    Parameters
    ----------
    input_path : str
        Path to engineered features (parquet file).
    model_path : str
        Path where trained model will be saved.
    encoders_path : str
        Path to label encoders (from feature engineering).
    config_path : str
        Path to training configuration YAML.
        
    Raises
    ------
    FileNotFoundError
        If input files don't exist.
    ValueError
        If configuration is invalid.
        
    Examples
    --------
    >>> run_training_pipeline(
    ...     input_path='data/03-features/features.parquet',
    ...     model_path='models/dnn_model.pth',
    ...     encoders_path='models/encoders.pkl',
    ...     config_path='config.yaml'
    ... )
    
    Notes
    -----
    The function automatically detects GPU availability and uses it if present.
    On AWS SageMaker with GPU instances (p3.2xlarge), this significantly
    speeds up training.
    """
    # Set up MLflow
    mlflow.set_experiment("recsys_v3_training")

    with mlflow.start_run():
        # Load configuration
        config = load_config(config_path)

        # Log config parameters
        mlflow.log_params({
            "batch_size": config['training']['batch_size'],
            "learning_rate": config['training']['learning_rate'],
            "epochs": config['training']['epochs'],
            "hidden_dim1": config['model']['hidden_dim1'],
            "hidden_dim2": config['model']['hidden_dim2'],
            "hidden_dim3": config['model']['hidden_dim3'],
            "dropout_rate": config['model']['dropout_rate']
        })

        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        mlflow.log_param("device", str(device))

        # Load data from S3
        logger.info(f"Loading data from {input_path}")
        s3_manager = S3DataManager()
        s3_path_parts = input_path.replace("s3://", "").split("/", 1)
        bucket_input = s3_path_parts[0]
        prefix = s3_path_parts[1] if len(s3_path_parts) > 1 else ""

        logger.info(f"Searching for newest 'df_features' in s3://{bucket_input}/{prefix}")
        newest_file_path = s3_manager.get_newest_file_by_date(
            bucket_name=bucket_input,
            prefix=prefix,
            starts_with="df_features"
        )
        if not newest_file_path:
            raise FileNotFoundError(f"No se encontró archivo 'df_features' en S3: s3://{bucket_input}/{prefix}")

        df = pd.read_parquet(newest_file_path)
        logger.info(f"Loaded {len(df)} records with {df.shape[1]} columns")

    logger.info(f"Loaded {len(df)} records with {df.shape[1]} columns")
    
    # Load encoders
    encoder_dict = load_pickle_from_s3(encoders_path)
    establishment_encoder = encoder_dict['establishment_encoder']
    num_establishments = len(establishment_encoder.classes_)
    logger.info(f"Number of establishments: {num_establishments}")
    
    # Define feature columns (all except target and original columns)
    exclude_cols = [
        'establecimiento', 'establecimiento_encoded',
        'hora_inicio', 'hora_fin', 'diaid',
        'id_persona', 'especialidad', 'localizacion_externa',
        'monto', 'neteo_mensual', 'neteo_diario',
        'estado_civil', 'genero', 'rol', 'segmento_comercial',
        'ciudad', 'zona', 'region', 'cadena', 'franja_horaria', 'edad_grupo'
    ]
    feature_cols = [col for col in df.columns if col not in exclude_cols and '_encoded' not in col or col.endswith('_encoded')]
    logger.info(f"Using {len(feature_cols)} features")
    
    # Prepare data splits
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(
        df,
        feature_cols,
        target_col='establecimiento_encoded',
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split'],
        random_state=config['data']['random_state']
    )
    
    # Create datasets and loaders
    train_dataset = EstablishmentDataset(X_train, y_train)
    val_dataset = EstablishmentDataset(X_val, y_val)
    test_dataset = EstablishmentDataset(X_test, y_test)
    batch_size = config['training']['batch_size']

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = EstablishmentDNN(
        input_dim=input_dim,
        num_establishments=num_establishments,
        hidden_dim1=config['model']['hidden_dim1'],
        hidden_dim2=config['model']['hidden_dim2'],
        hidden_dim3=config['model']['hidden_dim3'],
        dropout_rate=config['model']['dropout_rate']
    ).to(device)
    
    logger.info(f"Model initialized with {input_dim} input features")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Train model
    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=config['training']['epochs'],
        patience=config['training']['early_stopping_patience'],
        model_path=model_file
    )

    # Save all artifacts locally + S3
    model_dir = model_file.parent

    # --- Modelo ---
    torch.save({'model_state_dict': model.state_dict()}, model_dir / 'dnn_model.pth')
    s3_model_key = os.path.join(s3_output_prefix, 'dnn_model.pth').replace("\\", "/")
    upload_to_s3(local_path=str(model_dir / 'dnn_model.pth'), bucket=s3_bucket, s3_key=s3_model_key)
    logger.success(f"Modelo enviado a S3: s3://{s3_bucket}/{s3_model_key}")

    # --- Feature columns ---
    feature_cols_path = model_dir / 'feature_columns.pkl'
    joblib.dump(feature_cols, feature_cols_path)
    s3_feature_columns_key = os.path.join(s3_output_prefix, 'feature_columns.pkl').replace("\\", "/")
    upload_to_s3(local_path=str(feature_cols_path), bucket=s3_bucket, s3_key=s3_feature_columns_key)
    logger.success(f"Feature columns enviadas a S3: s3://{s3_bucket}/{s3_feature_columns_key}")

    # --- Training history ---
    history_path = model_dir / 'training_history.pkl'
    joblib.dump(history, history_path)
    s3_history_key = os.path.join(s3_output_prefix, 'training_history.pkl').replace("\\", "/")
    upload_to_s3(local_path=str(history_path), bucket=s3_bucket, s3_key=s3_history_key)
    logger.success(f"Training history enviado a S3: s3://{s3_bucket}/{s3_history_key}")

    # --- LocationTimeFilter ---
    location_filter = LocationTimeFilter.from_dataframe(df)
    location_filter_path = model_dir / 'location_filter.pkl'
    joblib.dump(location_filter, location_filter_path)
    s3_location_filter_key = os.path.join(s3_output_prefix, 'location_filter.pkl').replace("\\", "/")
    upload_to_s3(local_path=str(location_filter_path), bucket=s3_bucket, s3_key=s3_location_filter_key)
    logger.success(f"Location filter enviado a S3: s3://{s3_bucket}/{s3_location_filter_key}")    


    # Evaluate on test set
    logger.info("Evaluating on test set...")
    model.load_state_dict(torch.load(model_file)['model_state_dict'])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    logger.success(f"Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.4f}")
    
    logger.success("Training pipeline completed successfully!")


def main():
    """
    Parse command-line arguments and run training pipeline.
    
    Command-line Arguments
    ----------------------
    --input_path : str, required
        Path to engineered features parquet file.
    --model_path : str, required
        Path where trained model will be saved.
    --encoders_path : str, required
        Path to label encoders from feature engineering.
    --config_path : str, required
        Path to training configuration YAML.
        
    Examples
    --------
    $ python main.py \\
        --input_path data/03-features/features.parquet \\
        --model_path models/dnn_model.pth \\
        --encoders_path models/label_encoders.pkl \\
        --config_path config.yaml
    """
    parser = argparse.ArgumentParser(
        description="Train deep neural network for restaurant recommendations",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to engineered features"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path where model will be saved"
    )
    
    parser.add_argument(
        "--encoders_path",
        type=str,
        required=True,
        help="Path to label encoders"
    )
    
    parser.add_argument(
        "--s3_bucket",
        type=str,
        required=False,
        default=None,
        help="S3 bucket where the trained model will be uploaded"
    )

    parser.add_argument(
        "--s3_output_prefix",
        type=str,
        required=False,
        default=None,
        help="Prefix in S3 where model artifact will be stored"
    )
    
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to training configuration"
    )
    
    args = parser.parse_args()
    
    run_training_pipeline(
        input_path=args.input_path,
        model_path=args.model_path,
        encoders_path=args.encoders_path,
        config_path=args.config_path,
        s3_bucket=args.s3_bucket,
        s3_output_prefix=args.s3_output_prefix
    )


if __name__ == "__main__":
    main()
