"""
S3 Manager Module for RecSys V3

Este módulo proporciona funcionalidades para interactuar con AWS S3,
incluyendo lectura y escritura de archivos con organización por fechas.

Author: Equipo ADX
"""

import os
import io
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Any
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import pandas as pd
import joblib
import torch
from loguru import logger


class S3Manager:
    """
    Gestor de operaciones con AWS S3 para el proyecto RecSys V3.

    Proporciona métodos para:
    - Leer/escribir archivos Parquet
    - Leer/escribir modelos PyTorch
    - Leer/escribir objetos pickle
    - Organización automática por fecha
    - Gestión de carpetas con timestamp

    Examples
    --------
    >>> s3_manager = S3Manager(bucket_name="my-recsys-bucket")
    >>> df = s3_manager.read_parquet("data/raw/dataset.parquet")
    >>> s3_manager.write_parquet(df, "data/processed/output.parquet", use_date_folder=True)
    """

    def __init__(
        self,
        bucket_name: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: Optional[str] = None,
    ):
        """
        Inicializa el S3Manager.

        Parameters
        ----------
        bucket_name : str
            Nombre del bucket de S3
        aws_access_key_id : Optional[str]
            AWS Access Key ID. Si no se proporciona, se obtiene de variables de entorno
        aws_secret_access_key : Optional[str]
            AWS Secret Access Key. Si no se proporciona, se obtiene de variables de entorno
        region_name : Optional[str]
            Región de AWS. Por defecto 'us-east-1'
        """
        self.bucket_name = bucket_name
        self.region_name = region_name or os.getenv("AWS_REGION", "us-east-1")

        # Configurar credenciales
        if aws_access_key_id and aws_secret_access_key:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=self.region_name
            )
        else:
            # Usar credenciales del entorno o IAM role
            self.s3_client = boto3.client('s3', region_name=self.region_name)

        logger.info(f"S3Manager inicializado para bucket: {bucket_name}")

    def _get_s3_key_with_date(self, base_key: str, date_format: str = "%Y-%m-%d") -> str:
        """
        Genera una key de S3 con una carpeta de fecha.

        Parameters
        ----------
        base_key : str
            Ruta base del archivo (ej: "data/output/file.parquet")
        date_format : str
            Formato de fecha para la carpeta. Por defecto "%Y-%m-%d"

        Returns
        -------
        str
            Key con carpeta de fecha (ej: "data/output/2025-11-14/file.parquet")
        """
        path = Path(base_key)
        date_folder = datetime.now().strftime(date_format)

        # Insertar carpeta de fecha antes del nombre del archivo
        new_path = path.parent / date_folder / path.name
        return str(new_path)

    def read_parquet(self, s3_key: str) -> pd.DataFrame:
        """
        Lee un archivo Parquet desde S3.

        Parameters
        ----------
        s3_key : str
            Ruta del archivo en S3 (ej: "data/raw/dataset.parquet")

        Returns
        -------
        pd.DataFrame
            DataFrame con los datos leídos

        Raises
        ------
        FileNotFoundError
            Si el archivo no existe en S3
        """
        try:
            logger.info(f"Leyendo parquet desde s3://{self.bucket_name}/{s3_key}")

            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            df = pd.read_parquet(io.BytesIO(obj['Body'].read()))

            logger.success(f"Parquet leído exitosamente: {len(df)} filas")
            return df

        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise FileNotFoundError(f"Archivo no encontrado en S3: s3://{self.bucket_name}/{s3_key}")
            raise

    def write_parquet(
        self,
        df: pd.DataFrame,
        s3_key: str,
        use_date_folder: bool = True,
        date_format: str = "%Y-%m-%d",
        **parquet_kwargs
    ) -> str:
        """
        Escribe un DataFrame como Parquet en S3.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame a escribir
        s3_key : str
            Ruta base del archivo en S3
        use_date_folder : bool
            Si True, crea una carpeta con la fecha actual
        date_format : str
            Formato de fecha para la carpeta
        **parquet_kwargs
            Argumentos adicionales para pd.DataFrame.to_parquet

        Returns
        -------
        str
            Ruta completa del archivo en S3
        """
        if use_date_folder:
            s3_key = self._get_s3_key_with_date(s3_key, date_format)

        try:
            logger.info(f"Escribiendo parquet a s3://{self.bucket_name}/{s3_key}")

            buffer = io.BytesIO()
            df.to_parquet(buffer, index=False, **parquet_kwargs)
            buffer.seek(0)

            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=buffer.getvalue()
            )

            logger.success(f"Parquet escrito exitosamente: {len(df)} filas")
            return f"s3://{self.bucket_name}/{s3_key}"

        except Exception as e:
            logger.error(f"Error escribiendo parquet: {e}")
            raise

    def read_pickle(self, s3_key: str) -> Any:
        """
        Lee un archivo pickle desde S3.

        Parameters
        ----------
        s3_key : str
            Ruta del archivo en S3

        Returns
        -------
        Any
            Objeto deserializado
        """
        try:
            logger.info(f"Leyendo pickle desde s3://{self.bucket_name}/{s3_key}")

            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            data = joblib.load(io.BytesIO(obj['Body'].read()))

            logger.success("Pickle leído exitosamente")
            return data

        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise FileNotFoundError(f"Archivo no encontrado en S3: s3://{self.bucket_name}/{s3_key}")
            raise

    def write_pickle(
        self,
        obj: Any,
        s3_key: str,
        use_date_folder: bool = True,
        date_format: str = "%Y-%m-%d"
    ) -> str:
        """
        Escribe un objeto como pickle en S3.

        Parameters
        ----------
        obj : Any
            Objeto a serializar
        s3_key : str
            Ruta base del archivo en S3
        use_date_folder : bool
            Si True, crea una carpeta con la fecha actual
        date_format : str
            Formato de fecha para la carpeta

        Returns
        -------
        str
            Ruta completa del archivo en S3
        """
        if use_date_folder:
            s3_key = self._get_s3_key_with_date(s3_key, date_format)

        try:
            logger.info(f"Escribiendo pickle a s3://{self.bucket_name}/{s3_key}")

            buffer = io.BytesIO()
            joblib.dump(obj, buffer)
            buffer.seek(0)

            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=buffer.getvalue()
            )

            logger.success("Pickle escrito exitosamente")
            return f"s3://{self.bucket_name}/{s3_key}"

        except Exception as e:
            logger.error(f"Error escribiendo pickle: {e}")
            raise

    def read_pytorch_model(self, s3_key: str, map_location: str = 'cpu') -> dict:
        """
        Lee un modelo PyTorch desde S3.

        Parameters
        ----------
        s3_key : str
            Ruta del archivo en S3
        map_location : str
            Dispositivo donde cargar el modelo

        Returns
        -------
        dict
            Checkpoint del modelo
        """
        try:
            logger.info(f"Leyendo modelo PyTorch desde s3://{self.bucket_name}/{s3_key}")

            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            buffer = io.BytesIO(obj['Body'].read())
            checkpoint = torch.load(buffer, map_location=map_location)

            logger.success("Modelo PyTorch leído exitosamente")
            return checkpoint

        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise FileNotFoundError(f"Archivo no encontrado en S3: s3://{self.bucket_name}/{s3_key}")
            raise

    def write_pytorch_model(
        self,
        checkpoint: dict,
        s3_key: str,
        use_date_folder: bool = True,
        date_format: str = "%Y-%m-%d"
    ) -> str:
        """
        Escribe un modelo PyTorch en S3.

        Parameters
        ----------
        checkpoint : dict
            Checkpoint del modelo a guardar
        s3_key : str
            Ruta base del archivo en S3
        use_date_folder : bool
            Si True, crea una carpeta con la fecha actual
        date_format : str
            Formato de fecha para la carpeta

        Returns
        -------
        str
            Ruta completa del archivo en S3
        """
        if use_date_folder:
            s3_key = self._get_s3_key_with_date(s3_key, date_format)

        try:
            logger.info(f"Escribiendo modelo PyTorch a s3://{self.bucket_name}/{s3_key}")

            buffer = io.BytesIO()
            torch.save(checkpoint, buffer)
            buffer.seek(0)

            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=buffer.getvalue()
            )

            logger.success("Modelo PyTorch escrito exitosamente")
            return f"s3://{self.bucket_name}/{s3_key}"

        except Exception as e:
            logger.error(f"Error escribiendo modelo PyTorch: {e}")
            raise

    def list_files(self, prefix: str = "") -> list[str]:
        """
        Lista archivos en S3 con un prefijo dado.

        Parameters
        ----------
        prefix : str
            Prefijo para filtrar archivos

        Returns
        -------
        list[str]
            Lista de keys de archivos
        """
        try:
            logger.info(f"Listando archivos en s3://{self.bucket_name}/{prefix}")

            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )

            if 'Contents' not in response:
                return []

            files = [obj['Key'] for obj in response['Contents']]
            logger.info(f"Encontrados {len(files)} archivos")

            return files

        except ClientError as e:
            logger.error(f"Error listando archivos: {e}")
            raise

    def file_exists(self, s3_key: str) -> bool:
        """
        Verifica si un archivo existe en S3.

        Parameters
        ----------
        s3_key : str
            Ruta del archivo en S3

        Returns
        -------
        bool
            True si el archivo existe, False en caso contrario
        """
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError:
            return False

    def get_latest_dated_file(self, base_path: str, filename: str) -> Optional[str]:
        """
        Obtiene el archivo más reciente de una estructura con carpetas de fecha.

        Parameters
        ----------
        base_path : str
            Ruta base donde buscar (ej: "data/output/")
        filename : str
            Nombre del archivo a buscar

        Returns
        -------
        Optional[str]
            Key del archivo más reciente, o None si no se encuentra
        """
        files = self.list_files(prefix=base_path)

        # Filtrar archivos que terminen con el nombre buscado
        matching_files = [f for f in files if f.endswith(filename)]

        if not matching_files:
            return None

        # Ordenar por fecha (asumiendo formato ISO en el path)
        matching_files.sort(reverse=True)

        return matching_files[0]


def get_s3_manager_from_env() -> S3Manager:
    """
    Crea un S3Manager usando variables de entorno.

    Variables de entorno requeridas:
    - AWS_S3_BUCKET_NAME
    - AWS_ACCESS_KEY_ID (opcional, puede usar IAM role)
    - AWS_SECRET_ACCESS_KEY (opcional, puede usar IAM role)
    - AWS_REGION (opcional, por defecto 'us-east-1')

    Returns
    -------
    S3Manager
        Instancia configurada de S3Manager

    Raises
    ------
    ValueError
        Si no se encuentra AWS_S3_BUCKET_NAME
    """
    bucket_name = os.getenv("AWS_S3_BUCKET_NAME")
    if not bucket_name:
        raise ValueError("AWS_S3_BUCKET_NAME no está definido en las variables de entorno")

    return S3Manager(
        bucket_name=bucket_name,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION")
    )
