#!/bin/bash
# INICIO RÁPIDO - Re-procesamiento de Metadata

echo "════════════════════════════════════════════════════════════════"
echo "  🔄 SISTEMA DE RE-PROCESAMIENTO DE METADATA"
echo "     Sin necesidad de re-procesar audio"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Función para mostrar ayuda
show_help() {
    cat << EOF
OPCIONES DE USO:

1️⃣  EJECUTAR TEST
   Prueba rápida del sistema con metadata de ejemplo
   
   ./scripts/quickstart_reprocess.sh test

2️⃣  RE-PROCESAR UN ARCHIVO
   Re-procesa un archivo de metadata específico
   
   ./scripts/quickstart_reprocess.sh file <ruta/al/archivo.json>

3️⃣  RE-PROCESAR TODO UN DIRECTORIO
   Re-procesa todos los archivos de metadata encontrados
   
   ./scripts/quickstart_reprocess.sh all <directorio>

4️⃣  VER DOCUMENTACIÓN
   Muestra la documentación completa
   
   ./scripts/quickstart_reprocess.sh docs

5️⃣  VER EJEMPLOS
   Muestra ejemplos de uso detallados
   
   ./scripts/quickstart_reprocess.sh examples

EJEMPLOS:
   ./scripts/quickstart_reprocess.sh test
   ./scripts/quickstart_reprocess.sh file data/output/metadata/podcast_123.json
   ./scripts/quickstart_reprocess.sh all data/output
   ./scripts/quickstart_reprocess.sh docs

EOF
}

# Función para ejecutar test
run_test() {
    echo "🧪 Ejecutando test del sistema..."
    echo ""
    python scripts/test_reprocess.py
}

# Función para re-procesar archivo
reprocess_file() {
    local file=$1
    if [ -z "$file" ]; then
        echo "❌ Error: Debes especificar un archivo"
        echo "   Uso: $0 file <ruta/al/archivo.json>"
        exit 1
    fi
    
    if [ ! -f "$file" ]; then
        echo "❌ Error: El archivo no existe: $file"
        exit 1
    fi
    
    echo "📄 Re-procesando archivo: $file"
    echo ""
    python scripts/reprocess_metadata.py "$file"
}

# Función para re-procesar directorio
reprocess_all() {
    local dir=$1
    if [ -z "$dir" ]; then
        echo "❌ Error: Debes especificar un directorio"
        echo "   Uso: $0 all <directorio>"
        exit 1
    fi
    
    if [ ! -d "$dir" ]; then
        echo "❌ Error: El directorio no existe: $dir"
        exit 1
    fi
    
    echo "📁 Re-procesando todos los archivos en: $dir"
    echo ""
    python scripts/reprocess_metadata.py --all --data-dir "$dir"
}

# Función para mostrar documentación
show_docs() {
    echo "📚 Documentación disponible:"
    echo ""
    echo "1. Documentación completa:"
    echo "   cat docs/REPROCESS_METADATA.md | less"
    echo ""
    echo "2. Resumen de implementación:"
    echo "   cat docs/REPROCESS_SUMMARY.md | less"
    echo ""
    echo "3. Diagrama de flujo:"
    echo "   cat docs/REPROCESS_FLOWCHART.txt"
    echo ""
    echo "¿Qué deseas ver? (1/2/3):"
    read -r choice
    
    case $choice in
        1)
            less docs/REPROCESS_METADATA.md
            ;;
        2)
            less docs/REPROCESS_SUMMARY.md
            ;;
        3)
            cat docs/REPROCESS_FLOWCHART.txt
            ;;
        *)
            echo "Opción inválida"
            ;;
    esac
}

# Función para mostrar ejemplos
show_examples() {
    echo "📖 Mostrando ejemplos de uso..."
    echo ""
    cat scripts/example_reprocess.sh
}

# Main
case $1 in
    test)
        run_test
        ;;
    file)
        reprocess_file "$2"
        ;;
    all)
        reprocess_all "$2"
        ;;
    docs)
        show_docs
        ;;
    examples)
        show_examples
        ;;
    help|--help|-h|"")
        show_help
        ;;
    *)
        echo "❌ Opción desconocida: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
