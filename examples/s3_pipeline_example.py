"""
Ejemplo de Pipeline con Integración S3

Este script demuestra cómo modificar el pipeline principal para usar S3
con organización automática por fechas.

Author: Equipo ADX
"""

import sys
from pathlib import Path
from datetime import datetime
from loguru import logger

# Importar utilidades de S3 y configuración
from src.utils.s3_manager import get_s3_manager_from_env
from src.utils.config_manager import Config

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",
    colorize=True
)


def run_pipeline_with_s3():
    """
    Ejecuta el pipeline completo usando S3 para almacenamiento.

    Este ejemplo muestra:
    1. Lectura de datos raw desde S3
    2. Procesamiento de datos
    3. Escritura de resultados en S3 con carpetas por fecha
    4. Guardado de modelos y artefactos
    """

    logger.info("=" * 80)
    logger.info("PIPELINE RECSYS V3 - MODO S3")
    logger.info("=" * 80)

    # Paso 0: Verificar configuración
    logger.info("\n[PASO 0] Verificando configuración...")
    if not Config.is_s3_mode():
        logger.error("STORAGE_MODE debe ser 's3'. Configura en .env: STORAGE_MODE=s3")
        sys.exit(1)

    logger.info(f"✓ Storage mode: {Config.STORAGE_MODE}")
    logger.info(f"✓ S3 Bucket: {Config.AWS_S3_BUCKET_NAME}")
    logger.info(f"✓ AWS Region: {Config.AWS_REGION}")
    logger.info(f"✓ Use date folders: {Config.USE_DATE_FOLDERS}")

    # Crear S3Manager
    s3 = get_s3_manager_from_env()

    # Fecha actual para logging
    today = datetime.now().strftime(Config.DATE_FOLDER_FORMAT)
    logger.info(f"✓ Date folder: {today}")

    # -------------------------------------------------------------------------
    # PASO 1: Data Sampling
    # -------------------------------------------------------------------------
    logger.info("\n[PASO 1/6] Data Sampling...")

    # Leer datos raw desde S3
    raw_data_key = Config.get_raw_data_path()
    logger.info(f"Reading raw data from S3: {raw_data_key}")

    try:
        df_raw = s3.read_parquet(raw_data_key)
        logger.success(f"✓ Loaded {len(df_raw):,} rows from S3")
    except FileNotFoundError:
        logger.error(f"❌ Raw data not found in S3: {raw_data_key}")
        logger.info("Upload raw data first:")
        logger.info(f"  aws s3 cp data/01-raw/df_extendida_clean.parquet s3://{Config.AWS_S3_BUCKET_NAME}/{raw_data_key}")
        sys.exit(1)

    # Muestreo
    sample_size = Config.SAMPLE_SIZE
    logger.info(f"Sampling {sample_size} rows...")
    df_sampled = df_raw.sample(n=min(sample_size, len(df_raw)), random_state=Config.RANDOM_STATE)

    # Guardar en S3 con carpeta por fecha
    sampled_key = Config.get_sampled_data_path()
    s3_path = s3.write_parquet(
        df=df_sampled,
        s3_key=sampled_key,
        use_date_folder=Config.USE_DATE_FOLDERS
    )
    logger.success(f"✓ Sampled data saved to: {s3_path}")

    # -------------------------------------------------------------------------
    # PASO 2: Feature Engineering
    # -------------------------------------------------------------------------
    logger.info("\n[PASO 2/6] Feature Engineering...")

    # Aquí iría tu lógica de feature engineering
    # Por simplicidad, usamos el mismo dataframe
    df_features = df_sampled.copy()
    logger.info(f"Generated {len(df_features.columns)} features")

    # Guardar features en S3
    features_key = Config.get_features_path()
    s3_path = s3.write_parquet(
        df=df_features,
        s3_key=features_key,
        use_date_folder=Config.USE_DATE_FOLDERS
    )
    logger.success(f"✓ Features saved to: {s3_path}")

    # Guardar encoders (ejemplo dummy)
    label_encoders = {"ciudad": {}, "especialidad": {}}  # Tu lógica aquí
    encoders_key = Config.get_encoders_path()
    s3_path = s3.write_pickle(
        obj=label_encoders,
        s3_key=encoders_key,
        use_date_folder=Config.USE_DATE_FOLDERS
    )
    logger.success(f"✓ Encoders saved to: {s3_path}")

    # -------------------------------------------------------------------------
    # PASO 3: Data Validation
    # -------------------------------------------------------------------------
    logger.info("\n[PASO 3/6] Data Validation...")

    # Aquí iría tu lógica de validación con Evidently
    logger.info("Running data quality checks...")

    # validation_report_key = Config.get_validation_report_path()
    # s3.write_...(validation_report, validation_report_key, use_date_folder=True)
    logger.success("✓ Validation completed")

    # -------------------------------------------------------------------------
    # PASO 4: Model Training
    # -------------------------------------------------------------------------
    logger.info("\n[PASO 4/6] Model Training...")

    # Aquí iría tu lógica de entrenamiento
    logger.info("Training DNN model...")

    # Ejemplo: crear checkpoint dummy
    checkpoint = {
        'model_state_dict': {},  # Tu modelo aquí
        'optimizer_state_dict': {},
        'epoch': 50,
        'val_loss': 0.25,
        'val_acc': 0.78
    }

    # Guardar modelo en S3 con carpeta por fecha
    model_key = Config.get_model_path()
    s3_path = s3.write_pytorch_model(
        checkpoint=checkpoint,
        s3_key=model_key,
        use_date_folder=Config.USE_DATE_FOLDERS
    )
    logger.success(f"✓ Model saved to: {s3_path}")

    # Guardar otros artefactos
    feature_columns = list(df_features.columns)
    feature_columns_key = Config.get_feature_columns_path()
    s3.write_pickle(feature_columns, feature_columns_key, use_date_folder=Config.USE_DATE_FOLDERS)

    location_filter = {}  # Tu filtro aquí
    location_filter_key = Config.get_location_filter_path()
    s3.write_pickle(location_filter, location_filter_key, use_date_folder=Config.USE_DATE_FOLDERS)

    logger.success("✓ All artifacts saved")

    # -------------------------------------------------------------------------
    # PASO 5: Model Evaluation
    # -------------------------------------------------------------------------
    logger.info("\n[PASO 5/6] Model Evaluation...")

    # Aquí iría tu lógica de evaluación
    metrics = {
        "accuracy": 0.78,
        "ndcg_at_3": 0.82,
        "ndcg_at_5": 0.85,
        "ndcg_at_10": 0.88
    }

    # Guardar métricas en S3 como pickle
    metrics_key = Config.get_metrics_path()
    s3.write_pickle(metrics, metrics_key.replace('.json', '.pkl'), use_date_folder=Config.USE_DATE_FOLDERS)

    logger.success(f"✓ Metrics: {metrics}")

    # -------------------------------------------------------------------------
    # PASO 6: Model Registration (MLflow)
    # -------------------------------------------------------------------------
    logger.info("\n[PASO 6/6] Model Registration...")

    # Aquí iría tu lógica de registro en MLflow
    logger.info(f"MLflow Tracking URI: {Config.MLFLOW_TRACKING_URI}")
    logger.info(f"Experiment: {Config.MLFLOW_EXPERIMENT_NAME}")

    # import mlflow
    # mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    # mlflow.log_params({...})
    # mlflow.log_metrics(metrics)
    # mlflow.pytorch.log_model(model, "model")

    logger.success("✓ Model registered in MLflow")

    # -------------------------------------------------------------------------
    # Resumen Final
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETADO EXITOSAMENTE")
    logger.info("=" * 80)

    logger.info("\nArchivos generados en S3:")
    logger.info(f"  📁 Sampled data:  s3://{Config.AWS_S3_BUCKET_NAME}/data/sampled/{today}/")
    logger.info(f"  📁 Features:      s3://{Config.AWS_S3_BUCKET_NAME}/data/features/{today}/")
    logger.info(f"  📁 Models:        s3://{Config.AWS_S3_BUCKET_NAME}/models/{today}/")
    logger.info(f"  📁 Reports:       s3://{Config.AWS_S3_BUCKET_NAME}/reports/{today}/")

    logger.info("\nPara listar archivos generados:")
    logger.info(f"  aws s3 ls s3://{Config.AWS_S3_BUCKET_NAME}/models/{today}/ --recursive")

    logger.info("\nPara descargar el modelo:")
    logger.info(f"  aws s3 cp s3://{Config.AWS_S3_BUCKET_NAME}/models/{today}/dnn_model.pth ./models/")


def verify_s3_files():
    """Verifica que los archivos necesarios existan en S3."""
    logger.info("\n[VERIFICACIÓN] Verificando archivos en S3...")

    s3 = get_s3_manager_from_env()

    # Verificar raw data
    raw_data_key = Config.get_raw_data_path()
    if s3.file_exists(raw_data_key):
        logger.success(f"✓ Raw data exists: {raw_data_key}")
    else:
        logger.warning(f"⚠ Raw data not found: {raw_data_key}")
        logger.info("  Upload with: aws s3 cp data/01-raw/df_extendida_clean.parquet s3://...")

    # Listar archivos recientes
    today = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"\nArchivos de hoy ({today}):")

    for prefix in ["data/sampled", "data/features", "models", "reports"]:
        files = s3.list_files(prefix=f"{prefix}/{today}/")
        if files:
            logger.info(f"\n  {prefix}/{today}/:")
            for file in files:
                logger.info(f"    - {file}")
        else:
            logger.info(f"\n  {prefix}/{today}/: (vacío)")


def download_latest_model():
    """Descarga el modelo más reciente desde S3."""
    logger.info("\n[DESCARGA] Descargando modelo más reciente...")

    s3 = get_s3_manager_from_env()

    # Obtener el modelo más reciente
    latest_model_key = s3.get_latest_dated_file(
        base_path=Config.S3_MODELS_PATH,
        filename="dnn_model.pth"
    )

    if latest_model_key:
        logger.success(f"✓ Found latest model: {latest_model_key}")

        # Descargar
        checkpoint = s3.read_pytorch_model(latest_model_key)

        # Guardar localmente
        import torch
        local_path = Path("models/dnn_model_latest.pth")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, local_path)

        logger.success(f"✓ Model downloaded to: {local_path}")
    else:
        logger.warning("⚠ No model found in S3")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RecSys V3 Pipeline with S3")
    parser.add_argument(
        "--mode",
        choices=["run", "verify", "download"],
        default="run",
        help="Mode: run pipeline, verify files, or download latest model"
    )

    args = parser.parse_args()

    if args.mode == "run":
        run_pipeline_with_s3()
    elif args.mode == "verify":
        verify_s3_files()
    elif args.mode == "download":
        download_latest_model()
