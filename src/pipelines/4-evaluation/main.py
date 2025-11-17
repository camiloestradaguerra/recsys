"""
Model Evaluation Component

Evaluates the trained model and generates metrics reports including NDCG@k scores.

Author: Equipo ADX
Date: 2025-11-13
"""
import argparse
import json
import sys
from pathlib import Path

import boto3
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics import ndcg_score
from torch.utils.data import DataLoader, TensorDataset

# ------------------------------------------------------------
# Import S3DataManager from cleaning_data pipeline
# ------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "0-cleaning_data"))
from main import S3DataManager

# ------------------------------------------------------------
# Import model definition
# ------------------------------------------------------------
sys.path.append(str(Path(__file__).parent.parent / "3-training"))
from model import EstablishmentDNN


logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time}</green> | <level>{level}</level> | <level>{message}</level>",
    level="INFO"
)


# ---------------------------------------------------------------------
# Helper: download any S3 file
# ---------------------------------------------------------------------
def download_from_s3(s3_uri, local_path):
    s3 = boto3.client("s3")
    bucket = s3_uri.replace("s3://", "").split("/")[0]
    key = "/".join(s3_uri.replace("s3://", "").split("/")[1:])
    s3.download_file(bucket, key, local_path)
    return local_path


# ---------------------------------------------------------------------
# Helper: upload any local file to S3
# ---------------------------------------------------------------------
def upload_to_s3(local_path, s3_uri):
    s3 = boto3.client("s3")
    bucket = s3_uri.replace("s3://", "").split("/")[0]
    key = "/".join(s3_uri.replace("s3://", "").split("/")[1:])
    s3.upload_file(local_path, bucket, key)
    logger.success(f"Archivo subido a {s3_uri}")


# ---------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------
def evaluate_model(model_path, data_path, encoders_path, feature_cols_path, output_path, k_values=[3, 5, 10]):
    logger.info("Starting model evaluation...")

    # ------------------------------------------------------------
    # Load latest df_features from S3
    # ------------------------------------------------------------
    logger.info(f"Loading data from {data_path}")

    s3_manager = S3DataManager()
    bucket_input = data_path.replace("s3://", "").split("/")[0]
    prefix = "/".join(data_path.replace("s3://", "").split("/")[1:])

    newest_file_path = s3_manager.get_newest_file_by_date(
        bucket_name=bucket_input,
        prefix=prefix,
        starts_with="df_features"
    )

    if not newest_file_path:
        raise FileNotFoundError(f"No se encontró archivo df_features en {data_path}")

    logger.info(f"Newest features file: {newest_file_path}")

    # Download dataset
    local_data = download_from_s3(newest_file_path, "/tmp/data.parquet")
    df = pd.read_parquet(local_data)
    logger.info(f"Loaded {len(df)} rows and {df.shape[1]} columns")

    # ------------------------------------------------------------
    # Download artifacts
    # ------------------------------------------------------------
    local_model = download_from_s3(model_path, "/tmp/model.pth")
    local_encoders = download_from_s3(encoders_path, "/tmp/encoders.pkl")
    local_features = download_from_s3(feature_cols_path, "/tmp/feature_columns.pkl")

    # ------------------------------------------------------------
    # Load model checkpoint
    # ------------------------------------------------------------
    checkpoint = torch.load(local_model, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------
    # Prepare input data
    # ------------------------------------------------------------
    feature_cols = joblib.load(local_features)
    X = df[feature_cols].astype("float32").values
    y = df["establecimiento_encoded"].astype("int64").values

    # ------------------------------------------------------------
    # Load encoders
    # ------------------------------------------------------------
    encoder_dict = joblib.load(local_encoders)
    num_establishments = len(encoder_dict["establishment_encoder"].classes_)

    # ------------------------------------------------------------
    # Build model and load weights
    # ------------------------------------------------------------
    model = EstablishmentDNN(
        input_dim=X.shape[1],
        num_establishments=num_establishments
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------
    with torch.no_grad():
        outputs = model(torch.FloatTensor(X).to(device))
        probs = F.softmax(outputs, dim=1).cpu().numpy()

    predictions = np.argmax(probs, axis=1)
    accuracy = (predictions == y).mean()

    logger.success(f"Accuracy: {accuracy:.4f}")

    # ------------------------------------------------------------
    # NDCG metrics
    # ------------------------------------------------------------
    metrics = {"accuracy": float(accuracy)}

    y_true = np.zeros_like(probs)
    y_true[np.arange(len(y)), y] = 1

    for k in k_values:
        ndcg = ndcg_score(y_true, probs, k=k)
        metrics[f"ndcg_at_{k}"] = float(ndcg)
        logger.success(f"NDCG@{k}: {ndcg:.4f}")

    # ------------------------------------------------------------
    # Save metrics.json
    # ------------------------------------------------------------
    local_metrics_path = "/tmp/metrics.json"
    with open(local_metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Ensure output path ends with /metrics.json
    if output_path.endswith("/"):
        output_path = output_path + "metrics.json"
    elif not output_path.endswith(".json"):
        output_path = output_path + "/metrics.json"

    upload_to_s3(local_metrics_path, output_path)

    logger.success(f"Metrics saved to: {output_path}")
    return metrics


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate recommendation model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--encoders_path", type=str, required=True)
    parser.add_argument("--feature_cols_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        data_path=args.data_path,
        encoders_path=args.encoders_path,
        feature_cols_path=args.feature_cols_path,
        output_path=args.output_path
    )


if __name__ == "__main__":
    main()