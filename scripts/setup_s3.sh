#!/bin/bash

# =============================================================================
# Script de Configuración Inicial para RecSys V3 con S3
# =============================================================================
# Autor: Equipo ADX
# Descripción: Configura el entorno y AWS S3 para RecSys V3
# Uso: bash scripts/setup_s3.sh
# =============================================================================

set -e # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}$1${NC}"
echo -e "${BLUE}========================================${NC}"
}

print_success() {
echo -e "${GREEN} $1${NC}"
}

print_error() {
echo -e "${RED} $1${NC}"
}

print_warning() {
echo -e "${YELLOW} $1${NC}"
}

print_info() {
echo -e "${BLUE} $1${NC}"
}

# Check if command exists
command_exists() {
command -v "$1" >/dev/null 2>&1
}

# =============================================================================
# PASO 0: Verificar Prerequisitos
# =============================================================================

print_header "PASO 0: Verificando prerequisitos"

# Check Python
if command_exists python3; then
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
print_success "Python instalado: $PYTHON_VERSION"
else
print_error "Python 3 no está instalado"
exit 1
fi

# Check AWS CLI
if command_exists aws; then
AWS_VERSION=$(aws --version | cut -d' ' -f1)
print_success "AWS CLI instalado: $AWS_VERSION"
else
print_error "AWS CLI no está instalado"
print_info "Instalar con: pip install awscli"
exit 1
fi

# Check Git
if command_exists git; then
GIT_VERSION=$(git --version | cut -d' ' -f3)
print_success "Git instalado: $GIT_VERSION"
else
print_error "Git no está instalado"
exit 1
fi

echo ""

# =============================================================================
# PASO 1: Configurar Virtual Environment
# =============================================================================

print_header "PASO 1: Configurando entorno virtual"

if [ ! -d "venv" ]; then
print_info "Creando virtual environment..."
python3 -m venv venv
print_success "Virtual environment creado"
else
print_warning "Virtual environment ya existe"
fi

# Activate venv
print_info "Activando virtual environment..."
source venv/bin/activate

# Upgrade pip
print_info "Actualizando pip..."
pip install --upgrade pip --quiet

# Install dependencies
print_info "Instalando dependencias..."
pip install -r requirements.txt --quiet

print_success "Dependencias instaladas"

echo ""

# =============================================================================
# PASO 2: Configurar .env
# =============================================================================

print_header "PASO 2: Configurando variables de entorno"

if [ ! -f ".env" ]; then
print_info "Creando archivo .env desde .env.example..."
cp .env.example .env
print_success "Archivo .env creado"
print_warning "IMPORTANTE: Edita .env con tus credenciales de AWS"
print_info "Variables a configurar:"
print_info " - AWS_S3_BUCKET_NAME"
print_info " - AWS_ACCESS_KEY_ID"
print_info " - AWS_SECRET_ACCESS_KEY"
print_info " - AWS_REGION"
echo ""
read -p "Presiona Enter para editar .env ahora, o Ctrl+C para salir..."
${EDITOR:-nano} .env
else
print_warning ".env ya existe, no se sobrescribirá"
fi

echo ""

# =============================================================================
# PASO 3: Verificar Configuración de AWS
# =============================================================================

print_header "PASO 3: Verificando configuración de AWS"

# Source .env
if [ -f ".env" ]; then
export $(cat .env | grep -v '^#' | xargs)
fi

# Check AWS credentials
print_info "Verificando credenciales de AWS..."
if aws sts get-caller-identity &> /dev/null; then
AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
AWS_USER=$(aws sts get-caller-identity --query Arn --output text)
print_success "Credenciales válidas"
print_info " Account: $AWS_ACCOUNT"
print_info " User: $AWS_USER"
else
print_error "Credenciales de AWS inválidas"
print_info "Configura AWS CLI con: aws configure"
exit 1
fi

echo ""

# =============================================================================
# PASO 4: Configurar S3 Bucket
# =============================================================================

print_header "PASO 4: Configurando bucket de S3"

# Get bucket name from .env
BUCKET_NAME=${AWS_S3_BUCKET_NAME:-""}

if [ -z "$BUCKET_NAME" ]; then
print_error "AWS_S3_BUCKET_NAME no está definido en .env"
read -p "Ingresa el nombre del bucket: " BUCKET_NAME
echo "AWS_S3_BUCKET_NAME=$BUCKET_NAME" >> .env
fi

print_info "Bucket: $BUCKET_NAME"

# Check if bucket exists
if aws s3 ls "s3://$BUCKET_NAME" 2>/dev/null; then
print_success "Bucket ya existe"
else
print_warning "Bucket no existe"
read -p "¿Crear bucket? (y/n): " CREATE_BUCKET
if [ "$CREATE_BUCKET" = "y" ]; then
REGION=${AWS_REGION:-us-east-1}
print_info "Creando bucket en región $REGION..."

if [ "$REGION" = "us-east-1" ]; then
aws s3 mb "s3://$BUCKET_NAME"
else
aws s3 mb "s3://$BUCKET_NAME" --region "$REGION"
fi

print_success "Bucket creado exitosamente"
else
print_warning "Bucket no creado. Configúralo manualmente."
exit 0
fi
fi

echo ""

# =============================================================================
# PASO 5: Crear Estructura de Carpetas en S3
# =============================================================================

print_header "PASO 5: Creando estructura de carpetas en S3"

print_info "Creando estructura de directorios..."

FOLDERS=(
"data/raw/"
"data/sampled/"
"data/features/"
"data/validation/"
"models/"
"reports/"
"mlflow/artifacts/"
)

for folder in "${FOLDERS[@]}"; do
print_info " Creando: $folder"
aws s3api put-object --bucket "$BUCKET_NAME" --key "$folder" 2>/dev/null || true
done

print_success "Estructura de carpetas creada"

# Verify
print_info "Verificando estructura..."
aws s3 ls "s3://$BUCKET_NAME/" --recursive | head -20

echo ""

# =============================================================================
# PASO 6: Subir Datos Raw (Opcional)
# =============================================================================

print_header "PASO 6: Subir datos raw a S3 (opcional)"

RAW_DATA_PATH="data/01-raw/df_extendida_clean.parquet"

if [ -f "$RAW_DATA_PATH" ]; then
FILE_SIZE=$(ls -lh "$RAW_DATA_PATH" | awk '{print $5}')
print_info "Archivo encontrado: $RAW_DATA_PATH ($FILE_SIZE)"

read -p "¿Subir archivo a S3? (y/n): " UPLOAD_RAW
if [ "$UPLOAD_RAW" = "y" ]; then
print_info "Subiendo archivo... (esto puede tomar varios minutos)"
aws s3 cp "$RAW_DATA_PATH" "s3://$BUCKET_NAME/data/raw/df_extendida_clean.parquet"
print_success "Archivo subido exitosamente"
else
print_warning "Archivo no subido"
fi
else
print_warning "Archivo raw no encontrado: $RAW_DATA_PATH"
print_info "Súbelo manualmente con:"
print_info " aws s3 cp $RAW_DATA_PATH s3://$BUCKET_NAME/data/raw/"
fi

echo ""

# =============================================================================
# PASO 7: Configurar Versionado (Opcional)
# =============================================================================

print_header "PASO 7: Configurar versionado de S3 (opcional)"

read -p "¿Habilitar versionado en el bucket? (y/n): " ENABLE_VERSIONING
if [ "$ENABLE_VERSIONING" = "y" ]; then
print_info "Habilitando versionado..."
aws s3api put-bucket-versioning \
--bucket "$BUCKET_NAME" \
--versioning-configuration Status=Enabled
print_success "Versionado habilitado"
else
print_warning "Versionado no habilitado"
fi

echo ""

# =============================================================================
# PASO 8: Verificar Configuración
# =============================================================================

print_header "PASO 8: Verificando configuración final"

# Test S3 connection with Python
print_info "Probando conexión S3 con Python..."

python3 << EOF
import sys
sys.path.insert(0, '.')

try:
from src.utils.s3_manager import get_s3_manager_from_env
from src.utils.config_manager import Config

# Verify config
print(f" Storage mode: {Config.STORAGE_MODE}")
print(f" S3 Bucket: {Config.AWS_S3_BUCKET_NAME}")
print(f" AWS Region: {Config.AWS_REGION}")

# Test S3 connection
s3 = get_s3_manager_from_env()
files = s3.list_files(prefix='data/')
print(f" S3 connection successful")
print(f" Files in bucket: {len(files)}")

except Exception as e:
print(f" Error: {e}")
sys.exit(1)
EOF

if [ $? -eq 0 ]; then
print_success "Configuración verificada correctamente"
else
print_error "Error en la configuración"
exit 1
fi

echo ""

# =============================================================================
# RESUMEN FINAL
# =============================================================================

print_header "CONFIGURACIÓN COMPLETADA"

echo ""
print_success "RecSys V3 está listo para usar con S3!"
echo ""

print_info "Próximos pasos:"
echo " 1. Ejecutar pipeline de prueba:"
echo " python examples/s3_pipeline_example.py --mode run"
echo ""
echo " 2. Verificar archivos generados:"
echo " python examples/s3_pipeline_example.py --mode verify"
echo ""
echo " 3. Ver archivos en S3:"
echo " aws s3 ls s3://$BUCKET_NAME/ --recursive"
echo ""
echo " 4. Iniciar API:"
echo " uvicorn entrypoint.main:app --reload --host 0.0.0.0 --port 8001"
echo ""

print_info "Documentación:"
echo " - Quick Start: docs/QUICK_START.md"
echo " - S3 Integration: docs/S3_INTEGRATION.md"
echo " - Azure DevOps: docs/AZURE_DEVOPS_SETUP.md"
echo ""

print_warning "Recuerda:"
echo " - No commitear el archivo .env al repositorio"
echo " - Mantener las credenciales de AWS seguras"
echo " - Configurar lifecycle policies en S3 para costos"
echo ""

print_success "¡Éxito! "
