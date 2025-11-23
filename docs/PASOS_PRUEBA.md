# 📋 Pasos para Probar fromPodtoCast

## ✅ Paso 1: Instalar Dependencias

### Opción A: Si ya tienes PyTorch instalado (recomendado)
```bash
cd /home/ttech-main/fromPodtoCast
pip install -r requirements-minimal.txt
```

### Opción B: Instalación completa
```bash
cd /home/ttech-main/fromPodtoCast
pip install -r requirements.txt
```

### Verificar instalación:
```bash
python3 scripts/check_dependencies.py
```

---

## ✅ Paso 2: Ejecutar Prueba Rápida

Verifica que todos los módulos funcionen:
```bash
python3 scripts/test_example.py
```

Deberías ver: `🎉 ¡Todas las pruebas pasaron!`

---

## ✅ Paso 3: Preparar Archivo de Audio

### Opción A: Usar archivos existentes de Local-csm
```bash
# Copiar archivos de audio de prueba
cp /home/ttech-main/csm/Local-csm/data/audio/*.wav /home/ttech-main/fromPodtoCast/data/input/ 2>/dev/null

# Verificar que se copiaron
ls -lh /home/ttech-main/fromPodtoCast/data/input/
```

### Opción B: Usar tu propio archivo
```bash
# Copiar tu archivo de podcast
cp /ruta/a/tu/podcast.mp3 /home/ttech-main/fromPodtoCast/data/input/
```

---

## ✅ Paso 4: Procesar un Archivo de Prueba

### Procesar un archivo individual:
```bash
cd /home/ttech-main/fromPodtoCast

# Si tienes archivos en data/input/
python3 main.py data/input/audio20.wav -o data/output
```

### O procesar todos los archivos del directorio:
```bash
python3 main.py data/input/ -o data/output
```

### Lo que verás durante la ejecución:
```
============================================================
Procesando podcast: audio20.wav
============================================================

1. Segmentando audio...
   ✓ Generados X segmentos

2. Normalizando segmentos...
   Normalizando: 100%|████████| X/X [00:XX<00:00, X.XXit/s]
   ✓ Normalizados X segmentos

3. Transcribiendo segmentos...
   Transcribiendo: 100%|████████| X/X [00:XX<00:00, X.XXit/s]
   ✓ Transcritos X segmentos

4. Generando metadatos finales...
   ✓ Generados X registros de metadata

============================================================
✓ Metadata guardado en: data/output/metadata.json
  Total de registros: X
============================================================
```

---

## ✅ Paso 5: Verificar Resultados

### Ver la estructura generada:
```bash
tree data/output/ -L 3
# O si no tienes tree:
ls -R data/output/
```

### Ver el archivo de metadata:
```bash
# Ver primeras líneas
head -30 data/output/metadata.json

# Ver todo el archivo (si es pequeño)
cat data/output/metadata.json | python3 -m json.tool
```

### Verificar un segmento de audio:
```bash
# Ver información de un segmento
python3 -c "
import librosa
import soundfile as sf
audio, sr = librosa.load('data/output/normalized/audio20/audio20_segment_0000.wav')
print(f'Sample rate: {sr} Hz')
print(f'Duración: {len(audio)/sr:.2f} segundos')
print(f'Forma: {audio.shape}')
"
```

### Contar segmentos generados:
```bash
# Contar segmentos normalizados
find data/output/normalized -name "*.wav" | wc -l

# Ver tamaño total
du -sh data/output/
```

---

## ✅ Paso 6: Revisar Metadata Generado

El archivo `metadata.json` debería tener este formato:

```json
[
  {
    "text": "Transcripción del segmento...",
    "path": "/ruta/absoluta/al/archivo.wav",
    "speaker": 0,
    "speaker_label": "SPEAKER_00",
    "start": 0.0,
    "end": 12.5,
    "duration": 12.5,
    "language": "es",
    "podcast_id": "audio20"
  }
]
```

### Verificar que las transcripciones sean correctas:
```bash
# Extraer solo los textos
cat data/output/metadata.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
for i, item in enumerate(data[:5]):  # Primeros 5
    print(f'{i+1}. {item[\"text\"][:100]}...')
"
```

---

## 🔧 Solución de Problemas

### Error: "No module named 'librosa'"
```bash
pip install librosa soundfile
```

### Error: "ffmpeg not found"
```bash
sudo apt-get install ffmpeg
```

### Error: "CUDA out of memory"
Edita `config/config.json`:
```json
{
  "whisper_model": "tiny"  // Cambiar de "base" a "tiny"
}
```

### Audio sin transcripción
- Verifica que el audio tenga volumen suficiente
- Prueba con un modelo Whisper más grande: `"whisper_model": "small"`

### Procesamiento muy lento
- Usa modelo Whisper más pequeño: `"whisper_model": "tiny"`
- Procesa archivos más cortos primero

---

## 📊 Ejemplo Completo de Uso

```bash
# 1. Ir al directorio del proyecto
cd /home/ttech-main/fromPodtoCast

# 2. Instalar dependencias (si no están)
pip install -r requirements-minimal.txt

# 3. Copiar archivo de prueba
cp /home/ttech-main/csm/Local-csm/data/audio/audio20.wav data/input/

# 4. Procesar
python3 main.py data/input/audio20.wav -o data/output

# 5. Ver resultados
cat data/output/metadata.json | head -50
```

---

## 🎯 Próximos Pasos

Una vez que tengas el `metadata.json` generado:

1. **Para Sesame1b**: Usa directamente el archivo JSON
   ```bash
   python pretokenize.py --train_data data/output/metadata.json --val_data data/output/metadata.json --output tokenized_data.hdf5
   ```

2. **Para Kyutai TTS**: Puede requerir ajustes menores según el formato específico

3. **Mejorar calidad**: 
   - Ajusta `min_duration` y `max_duration` en `config/config.json`
   - Usa modelo Whisper más grande para mejor transcripción
   - Habilita diarización si tienes múltiples hablantes

---

## 📝 Notas Importantes

- El primer uso de Whisper descargará el modelo (puede tardar unos minutos)
- El procesamiento puede tardar según la duración del audio
- Los segmentos se guardan en `data/output/segments/` (temporales) y `data/output/normalized/` (finales)
- El archivo `metadata.json` contiene rutas absolutas a los archivos normalizados

