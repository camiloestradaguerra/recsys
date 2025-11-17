# RecSys V3 - Sistema de Recomendaciones Inteligente

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.108.0-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![AWS S3](https://img.shields.io/badge/AWS-S3-orange.svg)](https://aws.amazon.com/s3/)
[![Azure DevOps](https://img.shields.io/badge/Azure-DevOps-blue.svg)](https://azure.microsoft.com/services/devops/)

Sistema de recomendaciones personalizado basado en Deep Learning con filtrado contextual por ubicación y tiempo. Incluye pipeline MLOps completo con integración de AWS S3 y Azure DevOps para CI/CD automatizado.

---

## Tabla de Contenidos

- [Overview](#overview)
- [Características Principales](#características-principales)
- [Quick Start](#quick-start)
- [Configuración de AWS S3](#configuración-de-aws-s3)
- [Configuración de Azure DevOps](#configuración-de-azure-devops)
- [Arquitectura](#arquitectura)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Pipeline de ML](#pipeline-de-ml)
- [Uso del S3Manager](#uso-del-s3manager)
- [Modelo y Features](#modelo-y-features)
- [API Usage](#api-usage)
- [Testing](#testing)
- [Deployment](#deployment)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Configuración de Producción](#configuración-de-producción)
- [Contributing](#contributing)

---

## Overview

Pipeline MLOps end-to-end que implementa un sistema de recomendaciones usando Deep Neural Networks con capacidades críticas de filtrado por ubicación y tiempo. El sistema procesa datos transaccionales, genera features, entrena modelos y sirve recomendaciones vía FastAPI.

### Características Críticas

El modelo implementa el `LocationTimeFilter` (src/pipelines/3-training/model.py:296-361) que garantiza:

- **Restricciones geográficas**: Solo recomienda establecimientos en la ciudad del usuario
- **Restricciones temporales**: Solo recomienda establecimientos abiertos a la hora solicitada

---

## Características Principales

### Machine Learning
- **Deep Neural Network** (PyTorch) con arquitectura 1024→256→256
- **53 features** ingenieras (temporales, usuario, ubicación, interacción)
- **Location-Time Filter** para recomendaciones contextuales
- **MLflow** para experiment tracking y model registry
- **Evidently** para data validation y drift detection

### Infraestructura & DevOps
- **AWS S3** para almacenamiento escalable con organización por fecha
- **Azure DevOps** con pipelines completos de CI/CD
- **FastAPI** con documentación interactiva
- **Streamlit** dashboard para monitoreo

### Calidad de Código
- **Testing automatizado** (pytest + coverage)
- **Security scanning** (Bandit + Safety)
- **Type hints** en todo el código
- **Pydantic validation** para API

---

## Quick Start

### Opción 1: Setup Automático con S3

```bash
# Clonar repositorio
git clone https://github.com/tu-org/recsys_v3.git
cd recsys_v3

# Ejecutar script de configuración automática
bash scripts/setup_s3.sh
```

Este script configurará:
- Virtual environment y dependencias
- Variables de entorno (.env)
- AWS S3 bucket y estructura
- Verificación de conexión

### Opción 2: Setup Manual Local

```bash
# Clonar repositorio
git clone https://github.com/tu-org/recsys_v3.git
cd recsys_v3

# Configurar entorno
make setup-env
source .venv/bin/activate

# Instalar dependencias
make install

# Ejecutar pipeline completo
make run-pipeline

# Iniciar API
make run-api
# Visitar http://localhost:8001/docs
```

### Opción 3: Con AWS S3

```bash
# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales AWS

# Modo S3
export STORAGE_MODE=s3
python examples/s3_pipeline_example.py --mode run

# Verificar archivos generados
python examples/s3_pipeline_example.py --mode verify
```

---

## Configuración de AWS S3

### Prerequisitos

- Python 3.11+ instalado
- AWS CLI configurado
- Cuenta de AWS con permisos de S3

### 1. Variables de Entorno

Crear archivo `.env` basado en `.env.example`:

```bash
cp .env.example .env
```

Configurar las siguientes variables esenciales:

```bash
# Modo de almacenamiento
STORAGE_MODE=s3

# AWS Configuration
AWS_S3_BUCKET_NAME=tu-bucket-recsys-v3
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
AWS_REGION=us-east-1

# Usar carpetas con fecha
USE_DATE_FOLDERS=true
DATE_FOLDER_FORMAT=%Y-%m-%d
```

### 2. Crear Bucket en AWS

```bash
# Crear bucket
aws s3 mb s3://tu-bucket-recsys-v3 --region us-east-1

# Verificar
aws s3 ls s3://tu-bucket-recsys-v3/

# Configurar versionado (opcional)
aws s3api put-bucket-versioning \
  --bucket tu-bucket-recsys-v3 \
  --versioning-configuration Status=Enabled
```

### 3. Crear Estructura de Carpetas

```bash
# Crear estructura básica
aws s3api put-object --bucket tu-bucket-recsys-v3 --key data/raw/
aws s3api put-object --bucket tu-bucket-recsys-v3 --key data/sampled/
aws s3api put-object --bucket tu-bucket-recsys-v3 --key data/features/
aws s3api put-object --bucket tu-bucket-recsys-v3 --key models/
aws s3api put-object --bucket tu-bucket-recsys-v3 --key reports/
```

### 4. Subir Datos Raw

```bash
# Subir archivo de datos raw (242MB)
aws s3 cp data/01-raw/df_extendida_clean.parquet \
  s3://tu-bucket-recsys-v3/data/raw/df_extendida_clean.parquet

# Verificar subida
aws s3 ls s3://tu-bucket-recsys-v3/data/raw/
```

### 5. Configurar Permisos IAM

Crear una política IAM con estos permisos:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::tu-bucket-recsys-v3",
        "arn:aws:s3:::tu-bucket-recsys-v3/*"
      ]
    }
  ]
}
```

### 6. Verificar Configuración

```bash
# Probar conexión S3
python -c "
from src.utils.s3_manager import get_s3_manager_from_env
s3 = get_s3_manager_from_env()
print('Archivos:', s3.list_files('data/'))
"
```

---

## Configuración de Azure DevOps

### Prerequisitos

- Cuenta de Azure DevOps activa
- Proyecto creado en Azure DevOps
- Bucket de S3 configurado en AWS
- Credenciales de AWS

### 1. Crear Proyecto en Azure DevOps

1. Ve a `https://dev.azure.com/tu-organizacion`
2. Click en **New Project**
3. Configura:
   - **Project name:** RecSys-V3
   - **Visibility:** Private
   - **Version control:** Git

### 2. Conectar Repositorio

#### Opción A: Importar desde GitHub

1. Ve a **Repos** → **Files**
2. Click en **Import repository**
3. Ingresa la URL del repo: `https://github.com/tu-org/recsys_v3.git`
4. Click en **Import**

#### Opción B: Push desde Git local

```bash
cd recsys_v3
git remote add azure https://tu-org@dev.azure.com/tu-org/RecSys-V3/_git/RecSys-V3
git push azure master
```

### 3. Configurar Service Connections

#### AWS Service Connection

1. Ve a **Project Settings** → **Service connections**
2. Click en **New service connection**
3. Selecciona **AWS for Azure Pipelines**
4. Configura:
   - **Connection name:** `aws-recsys-connection`
   - **Access Key ID:** `<tu-aws-access-key-id>`
   - **Secret Access Key:** `<tu-aws-secret-access-key>`
   - **Region:** `us-east-1`
5. Click en **Verify and save**

### 4. Crear Variable Groups

#### Variable Group: `recsys-v3-common`

1. Ve a **Pipelines** → **Library**
2. Click en **+ Variable group**
3. Configura:
   - **Variable group name:** `recsys-v3-common`

4. Agrega las siguientes variables:

| Variable | Value | Secret |
|----------|-------|--------|
| `pythonVersion` | `3.11` | No |
| `projectName` | `recsys_v3` | No |
| `ENVIRONMENT` | `production` | No |
| `LOG_LEVEL` | `INFO` | No |

5. Click en **Save**

#### Variable Group: `recsys-v3-aws-credentials`

1. Click en **+ Variable group**
2. Configura:
   - **Variable group name:** `recsys-v3-aws-credentials`

3. Agrega las siguientes variables:

| Variable | Value | Secret |
|----------|-------|--------|
| `AWS_S3_BUCKET_NAME` | `recsys-v3-bucket` | No |
| `AWS_REGION` | `us-east-1` | No |
| `AWS_ACCESS_KEY_ID` | `<tu-access-key-id>` | **Sí** |
| `AWS_SECRET_ACCESS_KEY` | `<tu-secret-access-key>` | **Sí** |
| `awsServiceConnection` | `aws-recsys-connection` | No |

4. Click en **Save**

#### Variable Group: `recsys-v3-mlflow`

| Variable | Value | Secret |
|----------|-------|--------|
| `MLFLOW_TRACKING_URI` | `https://tu-mlflow-server.com` | No |
| `MLFLOW_EXPERIMENT_NAME` | `recsys_v3_training` | No |

### 5. Crear Pipelines

#### Pipeline Principal (CI/CD)

1. Ve a **Pipelines** → **Pipelines**
2. Click en **New pipeline**
3. Selecciona **Azure Repos Git**
4. Selecciona tu repositorio
5. Selecciona **Existing Azure Pipelines YAML file**
6. Path: `/azure-pipelines.yml`
7. Click en **Continue** → **Run**

**Stages:**
- Build & Test
- Security Scan
- Deploy Dev
- Deploy Prod

#### Pipeline de Entrenamiento

1. Click en **New pipeline**
2. Path: `/azure-pipelines-training.yml`
3. Click en **Run**

**Características:**
- Programado: Diario a las 2 AM UTC
- Lee datos desde S3
- Guarda modelos en S3 con carpetas por fecha (YYYY-MM-DD)
- Registra en MLflow

**Stages:**
1. Setup & Data Validation
2. Model Training
3. Model Evaluation
4. Model Registration
5. Notifications

### 6. Crear Environments

#### Environment: `development`

1. Ve a **Pipelines** → **Environments**
2. Click en **New environment**
3. Configura:
   - **Name:** `development`
   - **Description:** Ambiente de desarrollo
4. Click en **Create**

#### Environment: `production`

1. Click en **New environment**
2. Configura:
   - **Name:** `production`
   - **Description:** Ambiente de producción

3. **Agregar Aprobaciones:**
   - Click en los 3 puntos → **Approvals and checks**
   - Click en **Approvals**
   - Agrega usuarios que deben aprobar deployments
   - Configura timeout: 7 días
   - Click en **Create**

---

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                        AWS S3 Bucket                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │   Raw    │  │ Sampled  │  │ Features │  │  Models  │      │
│  │   Data   │  │   Data   │  │   Data   │  │ (dated)  │      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ S3Manager
                              │
┌─────────────────────────────┴───────────────────────────────────┐
│                    ML Pipeline (Python)                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ Sampling │→│ Features │→│ Training │→│Evaluation│      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
│                       ↓                         ↓                │
│                  ┌──────────┐            ┌──────────┐          │
│                  │Validation│            │ MLflow   │          │
│                  └──────────┘            └──────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ FastAPI
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Production API                             │
│              Recommendations Endpoint + Health Check            │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────┴───────────────────────────────────┐
│                    Azure DevOps Pipelines                       │
│  ┌──────────┐  ┌──────────┐                                    │
│  │  CI/CD   │  │ Training │                                    │
│  │ Pipeline │  │ Pipeline │                                    │
│  └──────────┘  └──────────┘                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Estructura del Proyecto

```
recsys_v3/
│
├── src/
│   ├── pipelines/              # Componentes del pipeline ML
│   │   ├── 1-data_sampling/
│   │   ├── 2-feature_engineering/
│   │   ├── 3-training/
│   │   ├── 4-evaluation/
│   │   ├── 5-model_registration/
│   │   ├── data_validation/
│   │   └── main_pipeline.py    # Orquestador del pipeline
│   │
│   └── utils/                  # Utilidades
│       ├── s3_manager.py       # Gestor de operaciones S3
│       └── config_manager.py   # Gestión de configuración
│
├── entrypoint/                 # API FastAPI
│   ├── main.py
│   ├── schemas.py
│   └── routers/
│       ├── health.py
│       └── recommendations.py
│
├── dashboard/                  # Dashboard Streamlit
│   └── app.py
│
├── examples/                   # Ejemplos de uso
│   └── s3_pipeline_example.py  # Pipeline completo con S3
│
├── scripts/                    # Scripts de utilidad
│   └── setup_s3.sh             # Setup automático de S3
│
├── tests/                      # Suite de tests
│   ├── test_sampling.py
│   ├── test_api.py
│   └── test_integration.py
│
├── data/                       # Datos (no en git)
│   ├── 01-raw/                 # Datos raw (7.5M filas)
│   ├── 02-sampled/             # Datos muestreados
│   └── 03-features/            # Features (53 columnas)
│
├── models/                     # Modelos entrenados (no en git)
├── reports/                    # Reportes y métricas
│
├── azure-pipelines.yml         # Pipeline CI/CD principal
├── azure-pipelines-training.yml # Pipeline de entrenamiento
│
├── .env.example                # Template de variables de entorno
│
├── requirements.txt            # Dependencias (+ boto3, s3fs)
├── Makefile                    # Automatización de tareas
│
└── README.md                   # Este archivo
```

---

## Pipeline de ML

### 1. Data Sampling

```bash
make run-sampling
# Input:  data/01-raw/df_extendida_clean.parquet (7,527,130 filas)
# Output: data/02-sampled/sampled_data.parquet (2,000 filas)
```

**Importante**: El tamaño de muestra de 2000 es para **desarrollo y testing**. Para producción usar dataset completo.

### 2. Feature Engineering

```bash
make run-features
```

**53 Features creadas:**
- **Temporales** (13): hora, día_semana, mes, cyclical encodings (sin/cos)
- **Usuario** (8): edad, género, estado_civil, antigüedad, patrones de consumo
- **Ubicación** (4): ciudad, zona, región, interacción hora-ciudad
- **Interacción** (7): frecuencia usuario-establecimiento, preferencias
- **Encoded** (21): Variables categóricas codificadas

### 3. Data Validation

```bash
python src/pipelines/data_validation/main.py \
  --input_path data/03-features/features.parquet \
  --output_path reports/data_validation.html
```

**Validaciones:**
- Análisis de valores faltantes
- Estadísticas de distribución
- Detección de outliers
- Matriz de correlación

### 4. Model Training

```bash
make run-training
```

**Arquitectura:**
- Input: 53 features
- Hidden Layer 1: 1024 + BatchNorm + ReLU + Dropout
- Hidden Layer 2: 256 + BatchNorm + ReLU + Dropout
- Hidden Layer 3: 256 + BatchNorm + ReLU + Dropout
- Output: num_establishments (softmax)

**Hyperparameters optimizados:**
```yaml
batch_size: 32
learning_rate: 0.0001967641848109
weight_decay: 0.00008261871088
dropout_rate: 0.1429465700244763
epochs: 50
```

### 5. Model Evaluation

```bash
make run-evaluation
```

**Métricas:**
- Accuracy (top-1): 70-80%
- NDCG@5: 0.75-0.85
- NDCG@10: 0.80-0.90

### 6. Model Registration

```bash
make run-registration
```

Registra el modelo en MLflow con todos los artefactos.

---

## Uso del S3Manager

### Importar y Crear Instancia

```python
from src.utils.s3_manager import get_s3_manager_from_env

# Crear S3 manager desde variables de entorno
s3 = get_s3_manager_from_env()

# O con credenciales explícitas
from src.utils.s3_manager import S3Manager
s3 = S3Manager(
    bucket_name="recsys-v3-bucket",
    aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
    aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    region_name="us-east-1"
)
```

### Leer Archivos desde S3

```python
# Leer Parquet
df = s3.read_parquet("data/raw/df_extendida_clean.parquet")

# Leer Pickle (encoders, filtros, etc.)
label_encoders = s3.read_pickle("models/2025-11-14/label_encoders.pkl")

# Leer Modelo PyTorch
checkpoint = s3.read_pytorch_model("models/2025-11-14/dnn_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
```

### Escribir Archivos en S3

```python
# Escribir Parquet con carpeta por fecha automática
s3_path = s3.write_parquet(
    df=sampled_df,
    s3_key="data/sampled/sampled_data.parquet",
    use_date_folder=True  # Crea: data/sampled/2025-11-14/sampled_data.parquet
)
print(f"Saved to: {s3_path}")

# Escribir sin carpeta por fecha
s3_path = s3.write_parquet(
    df=df_raw,
    s3_key="data/raw/df_extendida_clean.parquet",
    use_date_folder=False
)

# Escribir Pickle
s3.write_pickle(
    obj=label_encoders,
    s3_key="models/label_encoders.pkl",
    use_date_folder=True
)

# Escribir Modelo PyTorch
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': 50,
    'val_loss': 0.25
}
s3.write_pytorch_model(
    checkpoint=checkpoint,
    s3_key="models/dnn_model.pth",
    use_date_folder=True
)
```

### Utilidades

```python
# Verificar si existe un archivo
if s3.file_exists("models/dnn_model.pth"):
    print("Model exists")

# Listar archivos
files = s3.list_files(prefix="models/2025-11-14/")
for file in files:
    print(file)

# Obtener archivo más reciente
latest_model = s3.get_latest_dated_file(
    base_path="models/",
    filename="dnn_model.pth"
)
print(f"Latest model: {latest_model}")
```

### Estructura en S3

```
s3://tu-bucket/
├── data/
│   ├── raw/                    # Sin fecha (archivos base)
│   ├── sampled/YYYY-MM-DD/     # Con fecha
│   ├── features/YYYY-MM-DD/    # Con fecha
│   └── validation/YYYY-MM-DD/  # Con fecha
├── models/YYYY-MM-DD/          # Modelos organizados por fecha
├── reports/YYYY-MM-DD/         # Reportes y métricas
└── mlflow/artifacts/
```

---

## Modelo y Features

### LocationTimeFilter

Componente crítico que garantiza recomendaciones válidas:

```python
filtered_probs = location_filter.apply(
    predictions=model_output,
    user_ciudad="Quito",
    hora=14,
    establishment_names=all_names
)
# Retorna solo establecimientos en Quito abiertos a las 2 PM
```

### Performance Esperado

**Dataset de desarrollo (2000 muestras):**
- Accuracy: 70-80%
- NDCG@5: 0.75-0.85
- Training time: 10-30 min (CPU)

**Producción (7.5M filas):**
- Requiere GPU (recomendado: ml.p3.2xlarge)
- Training time: 2-4 horas

---

## API Usage

### Iniciar Servidor

```bash
# Desarrollo
make run-api

# Producción
uvicorn entrypoint.main:app --host 0.0.0.0 --port 8001
```

### Obtener Recomendaciones

```bash
curl -X POST "http://localhost:8001/recommendations/" \
  -H "Content-Type: application/json" \
  -d '{
    "id_persona": 21096.0,
    "ciudad": "Quito",
    "hora": 14,
    "k": 5
  }'
```

### Respuesta

```json
{
  "recommendations": [
    {
      "establecimiento": "ESTABLECIMIENTO A",
      "probability": 0.85,
      "ciudad": "Quito"
    }
  ],
  "filtered_by_location": true,
  "filtered_by_time": true,
  "used_real_user_data": true
}
```

### Documentación Interactiva

- **Swagger UI:** http://localhost:8001/docs
- **ReDoc:** http://localhost:8001/redoc

---

## Testing

### Tests Unitarios

```bash
# Ejecutar todos los tests
make test

# Con reporte de cobertura
make test-coverage

# Solo tests de API
pytest tests/test_api.py -v
```

### Verificar Integración S3

```bash
# Probar conexión
python -c "
from src.utils.s3_manager import get_s3_manager_from_env
s3 = get_s3_manager_from_env()
print('Archivos:', s3.list_files('data/'))
"

# Verificar archivos generados
python examples/s3_pipeline_example.py --mode verify

# Descargar último modelo
python examples/s3_pipeline_example.py --mode download
```

---

## Deployment

### Local

```bash
uvicorn entrypoint.main:app --reload --host 0.0.0.0 --port 8001
```

### Azure App Service

```bash
# Via Azure DevOps (automático)
git push origin master

# Manual deployment
az webapp up \
  --name recsys-v3-api-prod \
  --resource-group recsys-rg \
  --runtime "PYTHON:3.11"
```

### AWS SageMaker

El proyecto está listo para AWS SageMaker. Ver ejemplos en `.github/workflows/ml-pipeline.yml`.

---

## Monitoring

### Streamlit Dashboard

```bash
streamlit run dashboard/app.py
# Visitar http://localhost:8501
```

**Tres páginas:**

1. **Model Performance**
   - Métricas en tiempo real
   - Visualización del historial de entrenamiento
   - Feature importance

2. **Data Drift Detection**
   - Detección de drift usando Evidently
   - Comparación de distribuciones
   - Alertas de reentrenamiento

3. **System Health**
   - Monitoreo de API
   - Tiempo de respuesta
   - Logs de predicciones

---

## Troubleshooting

### Error: "AWS credentials not found"

```bash
# Configurar AWS CLI
aws configure

# O exportar variables
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
export AWS_REGION=us-east-1
```

### Error: "Bucket does not exist"

```bash
# Crear bucket
aws s3 mb s3://recsys-v3-bucket --region us-east-1

# Verificar
aws s3 ls s3://recsys-v3-bucket/
```

### Error: "Module not found"

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### CUDA out of memory

```yaml
# Editar src/pipelines/3-training/config.yaml
training:
  batch_size: 16  # Reducir de 32
```

### API no puede cargar modelo

```bash
# Verificar archivos requeridos
ls -la models/
# Requeridos: dnn_model.pth, label_encoders.pkl, feature_columns.pkl, location_filter.pkl
```

### Pipeline de Azure DevOps falla

1. Verificar Variable Groups configurados
2. Verificar Service Connections activos
3. Revisar logs del pipeline
4. Verificar permisos IAM en AWS

### S3 Performance Lento

**Soluciones:**
1. Usar región correcta (mismo región que tu computadora/servidor)
2. Usar compresión Snappy en Parquet
3. Implementar caché local para archivos frecuentes

---

## Configuración de Producción

**IMPORTANTE**: La configuración por defecto usa **2000 muestras** para desarrollo.

### Para Producción:

```python
# src/pipelines/main_pipeline.py
'sample_size': None  # Usar dataset completo (7.5M filas)

# O usar muestra grande
'sample_size': 100000  # Muestra representativa
```

**Consideraciones:**
- GPU recomendado (ml.p3.2xlarge en AWS)
- Stratified sampling para mejor balance
- Monitoreo de data drift
- Entrenamiento incremental para updates

---

## Contributing

1. Fork el repositorio
2. Crear feature branch: `git checkout -b feature/nueva-funcionalidad`
3. Hacer cambios siguiendo el estilo de código
4. Ejecutar tests: `make test`
5. Commit con mensajes descriptivos
6. Push al branch
7. Abrir Pull Request

---

## Security Best Practices

1. **Secrets Management**
   - Usar Variable Groups en Azure DevOps (encrypted)
   - Nunca commitear `.env`
   - Rotar credenciales regularmente

2. **IAM Permissions**
   - Permisos mínimos necesarios en AWS
   - Usar IAM Roles en EC2/ECS cuando sea posible

3. **Application Security**
   - Ejecutar aplicación con usuario no privilegiado
   - Validación de inputs con Pydantic
   - Scan de vulnerabilidades (Bandit, Safety)

---

## Comandos Útiles

### Pipeline Commands

```bash
# Ejecutar pipeline de entrenamiento
python src/pipelines/main_pipeline.py

# Ejecutar con S3
export STORAGE_MODE=s3
python examples/s3_pipeline_example.py --mode run

# Verificar archivos en S3
python examples/s3_pipeline_example.py --mode verify
```

### S3 Commands

```bash
# Upload file
aws s3 cp local-file.txt s3://bucket/path/

# Download file
aws s3 cp s3://bucket/path/file.txt ./

# List files
aws s3 ls s3://bucket/path/ --recursive

# Sync directory
aws s3 sync ./local-dir s3://bucket/remote-dir/
```

### Azure DevOps Commands

```bash
# Ejecutar pipeline manualmente
az pipelines run --name "RecSys V3 - Training"

# Ver status
az pipelines runs list --pipeline-name "RecSys V3 - Training"

# Descargar artifacts
az pipelines runs artifact download \
  --run-id <build-id> \
  --artifact-name training-reports \
  --path ./artifacts
```

---

## Changelog - Version 2.0.0

### Nuevas Funcionalidades

1. **Integración AWS S3**
   - Módulo S3Manager completo
   - Organización automática con carpetas por fecha
   - Soporte para Parquet, Pickle y modelos PyTorch

2. **Pipelines Azure DevOps**
   - Pipeline principal CI/CD
   - Pipeline de entrenamiento automatizado (diario 2 AM UTC)
   - Todos los artifacts se guardan en S3 con organización por fecha

3. **Configuración**
   - ConfigManager centralizado
   - Soporte para modo local y S3
   - Variables de entorno con .env

### Archivos Nuevos

- `src/utils/s3_manager.py`
- `src/utils/config_manager.py`
- `examples/s3_pipeline_example.py`
- `scripts/setup_s3.sh`
- `.env.example`
- `azure-pipelines.yml`
- `azure-pipelines-training.yml`

### Archivos Modificados

- `requirements.txt` (agregado boto3, s3fs)
- `README.md` (documentación completa integrada)

---

## License

Este proyecto está bajo la licencia MIT.

---

## Autor

**Equipo ADX**

---

## Acknowledgments

- MLOps best practices from School of DevOps
- Model architecture based on collaborative filtering research
- Feature engineering techniques from recommendation system literature
- Location filtering inspired by real production deployment challenges

---

**Versión:** 2.0.0 (con AWS S3 y Azure DevOps)
**Última actualización:** 2025-11-14
**Python:** 3.11+

---

**Nota Importante**: Este proyecto usa un **dataset muestreado de 2000 registros por defecto** para desarrollo y testing. Para deployment en producción con el dataset completo (7.5M filas), actualizar la configuración `sample_size` y provisionar recursos computacionales apropiados (GPU recomendado para entrenamiento).
