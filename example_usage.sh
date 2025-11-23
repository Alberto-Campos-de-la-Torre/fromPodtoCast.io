#!/bin/bash
# Script de ejemplo para usar fromPodcast

# Ejemplo 1: Procesar un solo archivo de podcast
echo "Ejemplo 1: Procesar un archivo de podcast"
python main.py /ruta/a/tu/podcast.mp3 -o ./data/output

# Ejemplo 2: Procesar un directorio completo de podcasts
echo "Ejemplo 2: Procesar directorio de podcasts"
python main.py /ruta/a/directorio/podcasts -o ./data/output

# Ejemplo 3: Especificar archivo de metadata personalizado
echo "Ejemplo 3: Con metadata personalizado"
python main.py /ruta/a/podcast.mp3 -o ./data/output --metadata ./data/train_data.json

# Ejemplo 4: Usar configuración personalizada
echo "Ejemplo 4: Con configuración personalizada"
python main.py /ruta/a/podcast.mp3 -c ./config/custom_config.json -o ./data/output

