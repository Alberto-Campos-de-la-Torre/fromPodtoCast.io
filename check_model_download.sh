#!/bin/bash
# Script para monitorear el progreso del git clone del modelo Whisper_Megi_IA

MODEL_DIR="/media/ttech-main/8464b39b-ba5b-49c5-9e05-010122b9874d/Whisper_Megi_IA"

echo "======================================================================"
echo "Monitoreo: Descarga de Whisper_Megi_IA"
echo "======================================================================"
echo ""

if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ El directorio aún no existe: $MODEL_DIR"
    echo "   El git clone todavía está iniciando..."
    exit 1
fi

echo "📂 Directorio encontrado: $MODEL_DIR"
echo ""

# Verificar archivos críticos
echo "Verificando archivos del modelo..."
files=(
    "config.json"
    "tokenizer_config.json"
    "preprocessor_config.json"
    "model.safetensors"
    "pytorch_model.bin"
)

for file in "${files[@]}"; do
    if [ -f "$MODEL_DIR/$file" ]; then
        size=$(du -h "$MODEL_DIR/$file" | cut -f1)
        echo "  ✓ $file ($size)"
    else
        echo "  ⏳ $file (descargando...)"
    fi
done

echo ""
echo "Tamaño total del directorio:"
du -sh "$MODEL_DIR"

echo ""
echo "Archivos grandes (>100MB):"
find "$MODEL_DIR" -type f -size +100M -exec du -h {} \; 2>/dev/null | sort -h

echo ""
echo "======================================================================"
