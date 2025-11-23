# Configuración de pyannote.audio

## 📋 Requisitos

Para usar la diarización avanzada de hablantes con `pyannote.audio`, necesitas:

1. **Instalar pyannote.audio**
2. **Token de Hugging Face**
3. **Aceptar términos de uso de los modelos**

## 🚀 Instalación Rápida

### Opción 1: Script Automático

```bash
cd /home/ttech-main/fromPodtoCast
source venv/bin/activate
./scripts/setup_pyannote.sh
```

### Opción 2: Instalación Manual

```bash
cd /home/ttech-main/fromPodtoCast
source venv/bin/activate
pip install pyannote.audio
```

## 🔑 Configuración de Hugging Face

### Paso 1: Crear cuenta y aceptar términos

1. Crea una cuenta en [Hugging Face](https://huggingface.co)
2. Acepta los términos de uso de estos modelos:
   - [speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - [embedding](https://huggingface.co/pyannote/embedding)

### Paso 2: Generar token

1. Ve a [Settings > Tokens](https://huggingface.co/settings/tokens)
2. Crea un nuevo token con permisos de lectura
3. Copia el token generado

### Paso 3: Configurar el token

#### Opción A: En config.json (Recomendado)

Edita `config/config.json`:

```json
{
  "use_diarization": true,
  "hf_token": "tu_token_aqui"
}
```

#### Opción B: Variable de entorno

```bash
export HUGGINGFACE_TOKEN=tu_token_aqui
```

O agrega a tu `.bashrc`:

```bash
echo 'export HUGGINGFACE_TOKEN=tu_token_aqui' >> ~/.bashrc
source ~/.bashrc
```

## ✅ Verificar Instalación

```bash
cd /home/ttech-main/fromPodtoCast
source venv/bin/activate
python3 -c "from pyannote.audio import Pipeline; print('✅ pyannote.audio instalado correctamente')"
```

## 🧪 Probar Diarización

```python
from src.speaker_diarizer import SpeakerDiarizer

# Con token
diarizer = SpeakerDiarizer(hf_token="tu_token")

# O sin token (puede fallar si el modelo es privado)
diarizer = SpeakerDiarizer()

# Diarizar un archivo
result = diarizer.diarize("audio.mp3")
print(result)
```

## ⚠️ Solución de Problemas

### Error: "Model not found" o "401 Unauthorized"

- Verifica que hayas aceptado los términos de uso en Hugging Face
- Verifica que el token sea correcto
- Asegúrate de que el token tenga permisos de lectura

### Error: "RuntimeError: operator torchvision::nms does not exist"

Este es un conflicto de versiones entre torch y torchvision. Soluciones:

1. **Usar entorno virtual separado** (recomendado):
```bash
python3 -m venv venv_pyannote
source venv_pyannote/bin/activate
pip install pyannote.audio torch torchvision --upgrade
```

2. **Actualizar torch y torchvision**:
```bash
pip install torch torchvision --upgrade
```

### Error: "CUDA out of memory"

- Usa CPU en lugar de GPU:
```json
{
  "device": "cpu"
}
```

- O procesa archivos más cortos

## 📝 Notas

- La diarización es **opcional**: el proyecto funciona sin ella
- Sin pyannote.audio, se usa un método simple que asigna el mismo speaker_id a todos los segmentos
- Con pyannote.audio, se identifican automáticamente diferentes hablantes
- El primer uso descargará los modelos (puede tardar varios minutos)

## 🔗 Referencias

- [pyannote.audio GitHub](https://github.com/pyannote/pyannote-audio)
- [Documentación oficial](https://github.com/pyannote/pyannote-audio#installation)
- [Modelos en Hugging Face](https://huggingface.co/pyannote)

