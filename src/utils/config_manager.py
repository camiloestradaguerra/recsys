"""
Configuration Manager Module for RecSys V3

Gestiona la configuración del proyecto, incluyendo rutas locales vs S3,
variables de entorno y settings de la aplicación.

Author: Equipo ADX
"""

import os
from pathlib import Path
from typing import Optional, Union
from dotenv import load_dotenv
from loguru import logger


# Cargar variables de entorno
load_dotenv()


class Config:
    """
    Configuración centralizada para RecSys V3.

    Proporciona acceso a:
    - Configuración de AWS S3
    - Rutas de datos (local o S3)
    - Configuración de MLflow
    - Configuración de API
    - Configuración de pipeline
    """

    # -----------------------------------------------------------------------------
    # AWS S3 Configuration
    # -----------------------------------------------------------------------------
    AWS_S3_BUCKET_NAME: str = os.getenv("AWS_S3_BUCKET_NAME", "recsys-v3-bucket")
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")

    # S3 Paths
    S3_RAW_DATA_PATH: str = os.getenv("S3_RAW_DATA_PATH", "data/raw")
    S3_SAMPLED_DATA_PATH: str = os.getenv("S3_SAMPLED_DATA_PATH", "data/sampled")
    S3_FEATURES_PATH: str = os.getenv("S3_FEATURES_PATH", "data/features")
    S3_MODELS_PATH: str = os.getenv("S3_MODELS_PATH", "models")
    S3_REPORTS_PATH: str = os.getenv("S3_REPORTS_PATH", "reports")
    S3_VALIDATION_PATH: str = os.getenv("S3_VALIDATION_PATH", "data/validation")
    S3_MLFLOW_ARTIFACTS_PATH: str = os.getenv("S3_MLFLOW_ARTIFACTS_PATH", "mlflow/artifacts")

    S3_RAW_DATA_FILENAME: str = os.getenv("S3_RAW_DATA_FILENAME", "df_extendida_clean.parquet")

    # -----------------------------------------------------------------------------
    # Storage Mode
    # -----------------------------------------------------------------------------
    STORAGE_MODE: str = os.getenv("STORAGE_MODE", "local")  # "local" or "s3"

    # Local paths (usado cuando STORAGE_MODE=local)
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    LOCAL_DATA_DIR = PROJECT_ROOT / os.getenv("LOCAL_DATA_DIR", "data")
    LOCAL_MODELS_DIR = PROJECT_ROOT / os.getenv("LOCAL_MODELS_DIR", "models")
    LOCAL_REPORTS_DIR = PROJECT_ROOT / os.getenv("LOCAL_REPORTS_DIR", "reports")

    # -----------------------------------------------------------------------------
    # MLflow Configuration
    # -----------------------------------------------------------------------------
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "recsys_v3_training")
    MLFLOW_BACKEND_STORE_URI: Optional[str] = os.getenv("MLFLOW_BACKEND_STORE_URI")
    MLFLOW_ARTIFACT_ROOT: Optional[str] = os.getenv("MLFLOW_ARTIFACT_ROOT")

    # -----------------------------------------------------------------------------
    # API Configuration
    # -----------------------------------------------------------------------------
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8001"))
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    CORS_ORIGINS: list[str] = os.getenv("CORS_ORIGINS", "").split(",")

    # -----------------------------------------------------------------------------
    # Model Configuration
    # -----------------------------------------------------------------------------
    MODEL_NAME: str = os.getenv("MODEL_NAME", "recsys_dnn")
    MODEL_VERSION: str = os.getenv("MODEL_VERSION", "latest")
    MODEL_S3_PATH: str = os.getenv("MODEL_S3_PATH", "models/dnn_model.pth")
    LABEL_ENCODERS_S3_PATH: str = os.getenv("LABEL_ENCODERS_S3_PATH", "models/label_encoders.pkl")
    FEATURE_COLUMNS_S3_PATH: str = os.getenv("FEATURE_COLUMNS_S3_PATH", "models/feature_columns.pkl")
    LOCATION_FILTER_S3_PATH: str = os.getenv("LOCATION_FILTER_S3_PATH", "models/location_filter.pkl")

    # -----------------------------------------------------------------------------
    # Pipeline Configuration
    # -----------------------------------------------------------------------------
    SAMPLE_SIZE: int = int(os.getenv("SAMPLE_SIZE", "2000"))
    RANDOM_STATE: int = int(os.getenv("RANDOM_STATE", "42"))
    USE_DATE_FOLDERS: bool = os.getenv("USE_DATE_FOLDERS", "true").lower() == "true"
    DATE_FOLDER_FORMAT: str = os.getenv("DATE_FOLDER_FORMAT", "%Y-%m-%d")
    USE_TIMESTAMP: bool = os.getenv("USE_TIMESTAMP", "false").lower() == "true"

    # -----------------------------------------------------------------------------
    # Training Configuration
    # -----------------------------------------------------------------------------
    EPOCHS: int = int(os.getenv("EPOCHS", "50"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "0.0001967641848109"))
    EARLY_STOPPING_PATIENCE: int = int(os.getenv("EARLY_STOPPING_PATIENCE", "10"))
    DEVICE: str = os.getenv("DEVICE", "cpu")

    # -----------------------------------------------------------------------------
    # Feature Flags
    # -----------------------------------------------------------------------------
    ENABLE_MODEL_CACHE: bool = os.getenv("ENABLE_MODEL_CACHE", "true").lower() == "true"
    ENABLE_DETAILED_LOGGING: bool = os.getenv("ENABLE_DETAILED_LOGGING", "false").lower() == "true"
    ENABLE_DATA_VALIDATION: bool = os.getenv("ENABLE_DATA_VALIDATION", "true").lower() == "true"
    ENABLE_MLFLOW_AUTOLOG: bool = os.getenv("ENABLE_MLFLOW_AUTOLOG", "true").lower() == "true"
    ENABLE_DRIFT_DETECTION: bool = os.getenv("ENABLE_DRIFT_DETECTION", "true").lower() == "true"

    @classmethod
    def is_s3_mode(cls) -> bool:
        """Verifica si el modo de storage es S3."""
        return cls.STORAGE_MODE.lower() == "s3"

    @classmethod
    def is_local_mode(cls) -> bool:
        """Verifica si el modo de storage es local."""
        return cls.STORAGE_MODE.lower() == "local"

    @classmethod
    def get_raw_data_path(cls) -> str:
        """
        Obtiene la ruta del archivo de datos raw según el modo de storage.

        Returns
        -------
        str
            Ruta local o S3 key del archivo raw
        """
        if cls.is_s3_mode():
            return f"{cls.S3_RAW_DATA_PATH}/{cls.S3_RAW_DATA_FILENAME}"
        else:
            return str(cls.LOCAL_DATA_DIR / "01-raw" / cls.S3_RAW_DATA_FILENAME)

    @classmethod
    def get_sampled_data_path(cls, filename: str = "sampled_data.parquet") -> str:
        """Obtiene la ruta del archivo de datos muestreados."""
        if cls.is_s3_mode():
            return f"{cls.S3_SAMPLED_DATA_PATH}/{filename}"
        else:
            return str(cls.LOCAL_DATA_DIR / "02-sampled" / filename)

    @classmethod
    def get_features_path(cls, filename: str = "features.parquet") -> str:
        """Obtiene la ruta del archivo de features."""
        if cls.is_s3_mode():
            return f"{cls.S3_FEATURES_PATH}/{filename}"
        else:
            return str(cls.LOCAL_DATA_DIR / "03-features" / filename)

    @classmethod
    def get_model_path(cls, filename: str = "dnn_model.pth") -> str:
        """Obtiene la ruta del archivo de modelo."""
        if cls.is_s3_mode():
            return f"{cls.S3_MODELS_PATH}/{filename}"
        else:
            return str(cls.LOCAL_MODELS_DIR / filename)

    @classmethod
    def get_encoders_path(cls, filename: str = "label_encoders.pkl") -> str:
        """Obtiene la ruta de los encoders."""
        if cls.is_s3_mode():
            return f"{cls.S3_MODELS_PATH}/{filename}"
        else:
            return str(cls.LOCAL_MODELS_DIR / filename)

    @classmethod
    def get_feature_columns_path(cls, filename: str = "feature_columns.pkl") -> str:
        """Obtiene la ruta de las columnas de features."""
        if cls.is_s3_mode():
            return f"{cls.S3_MODELS_PATH}/{filename}"
        else:
            return str(cls.LOCAL_MODELS_DIR / filename)

    @classmethod
    def get_location_filter_path(cls, filename: str = "location_filter.pkl") -> str:
        """Obtiene la ruta del filtro de localización."""
        if cls.is_s3_mode():
            return f"{cls.S3_MODELS_PATH}/{filename}"
        else:
            return str(cls.LOCAL_MODELS_DIR / filename)

    @classmethod
    def get_metrics_path(cls, filename: str = "metrics.json") -> str:
        """Obtiene la ruta del archivo de métricas."""
        if cls.is_s3_mode():
            return f"{cls.S3_REPORTS_PATH}/{filename}"
        else:
            return str(cls.LOCAL_REPORTS_DIR / filename)

    @classmethod
    def get_validation_report_path(cls, filename: str = "data_validation.html") -> str:
        """Obtiene la ruta del reporte de validación."""
        if cls.is_s3_mode():
            return f"{cls.S3_VALIDATION_PATH}/{filename}"
        else:
            return str(cls.LOCAL_REPORTS_DIR / filename)

    @classmethod
    def validate_config(cls) -> None:
        """
        Valida la configuración y muestra warnings si hay problemas.
        """
        if cls.is_s3_mode():
            if not cls.AWS_S3_BUCKET_NAME:
                logger.warning("AWS_S3_BUCKET_NAME no está configurado")

            if not cls.AWS_ACCESS_KEY_ID or not cls.AWS_SECRET_ACCESS_KEY:
                logger.info("No se encontraron credenciales de AWS. Se intentará usar IAM Role.")

        logger.info(f"Configuración cargada:")
        logger.info(f"  - Storage mode: {cls.STORAGE_MODE}")
        logger.info(f"  - S3 Bucket: {cls.AWS_S3_BUCKET_NAME if cls.is_s3_mode() else 'N/A'}")
        logger.info(f"  - MLflow tracking: {cls.MLFLOW_TRACKING_URI}")
        logger.info(f"  - Environment: {cls.ENVIRONMENT}")


# Validar configuración al importar
Config.validate_config()
