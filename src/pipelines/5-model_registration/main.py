"""
Model Registration Component

Registers the trained model in MLflow with metrics and metadata.

Author: Equipo ADX
Date: 2025-11-13
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import boto3
import joblib
import mlflow
import mlflow.pytorch
import torch
from loguru import logger
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent / '3-training'))
from model import EstablishmentDNN

load_dotenv()

logger.remove()
logger.add(sys.stderr, format="<green>{time}</green> | <level>{level}</level> | <level>{message}</level>", level="INFO")


def download_from_s3(s3_path: str) -> str:
    """Download a file from S3 to a local temp file and return its path."""
    s3 = boto3.client('s3')
    bucket, key = s3_path.replace("s3://", "").split("/", 1)
    tmp_dir = tempfile.mkdtemp()
    local_path = os.path.join(tmp_dir, os.path.basename(key))
    s3.download_file(bucket, key, local_path)
    return local_path


def upload_to_s3(local_path: str, s3_uri: str):
    """Upload a local file to S3."""
    s3 = boto3.client('s3')
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    s3.upload_file(local_path, bucket, key)
    logger.success(f"Modelo subido a S3 en: {s3_uri}")


def register_model(model_path, encoders_path, metrics_path, feature_cols_path, model_name):
    """Register model in MLflow and upload to S3."""
    logger.info("Starting model registration...")

    # Download all files from S3
    model_path = download_from_s3(model_path)
    encoders_path = download_from_s3(encoders_path)
    metrics_path = download_from_s3(metrics_path)
    feature_cols_path = download_from_s3(feature_cols_path)

    # Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')

    # Load encoders and feature columns
    encoder_dict = joblib.load(encoders_path)
    feature_cols = joblib.load(feature_cols_path)

    num_establishments = len(encoder_dict['establishment_encoder'].classes_)
    input_dim = len(feature_cols)

    # Initialize model
    model = EstablishmentDNN(
        input_dim=input_dim,
        num_establishments=num_establishments
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Set MLflow tracking URI
    mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI', 'mlruns')
    mlflow.set_tracking_uri(mlflow_uri)

    # Start MLflow run
    with mlflow.start_run(run_name=f"{model_name}_registration") as run:
        mlflow.log_params({
            'batch_size': 32,
            'learning_rate': 0.0001967641848109,
            'hidden_dim1': 1024,
            'hidden_dim2': 256,
            'hidden_dim3': 256,
            'dropout_rate': 0.1429465700244763,
            'weight_decay': 0.00008261871088,
            'input_dim': input_dim,
            'num_establishments': num_establishments
        })

        mlflow.log_metrics(metrics)
        mlflow.pytorch.log_model(model, "model")
        mlflow.log_artifact(encoders_path, "encoders")
        mlflow.log_artifact(metrics_path, "metrics")
        mlflow.log_artifact(feature_cols_path, "features")

        mlflow.set_tags({
            'model_type': 'deep_neural_network',
            'framework': 'pytorch',
            'problem_type': 'multi-class_classification',
            'use_case': 'restaurant_recommendation',
            'git_commit': os.environ.get('GITHUB_SHA', 'local'),
            'git_branch': os.environ.get('GITHUB_REF_NAME', 'local')
        })

        model_uri = f"runs:/{run.info.run_id}/model"
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        logger.success(f"Model registered: {model_name} version {registered_model.version}")
        logger.info(f"Run ID: {run.info.run_id}")

        # Save the model as .pth and upload it to S3
        local_pth_path = os.path.join(tempfile.gettempdir(), "dnn_model_registered.pth")
        torch.save({'model_state_dict': model.state_dict()}, local_pth_path)

        s3_output_path = "s3://dcelip-dev-artifacts-s3/mlops/model_artifacts/dnn_model_registered.pth"
        upload_to_s3(local_pth_path, s3_output_path)
        logger.info(f"Modelo guardado en S3: {s3_output_path}")
        
    return run.info.run_id


def main():
    parser = argparse.ArgumentParser(description="Register model in MLflow and upload to S3")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--encoders_path", type=str, required=True)
    parser.add_argument("--metrics_path", type=str, required=True)
    parser.add_argument("--feature_cols_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="recsys_dnn")
    args = parser.parse_args()

    register_model(
        model_path=args.model_path,
        encoders_path=args.encoders_path,
        metrics_path=args.metrics_path,
        feature_cols_path=args.feature_cols_path,
        model_name=args.model_name
    )


if __name__ == "__main__":
    main()