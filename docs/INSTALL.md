# Guía de Instalación - fromPodtoCast

## ⚠️ Conflictos de Dependencias

Si estás experimentando conflictos de dependencias (especialmente con `torch`, `moshi`, o `bitsandbytes`), sigue esta guía.

## Opciones de Instalación

### Opción 1: Entorno Virtual Dedicado (Recomendado)

Crea un entorno virtual separado para fromPodtoCast:

```bash
# Crear entorno virtual
python3 -m venv venv_frompodtocast
source venv_frompodtocast/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### Opción 2: Instalación Mínima (Si ya tienes PyTorch)

Si ya tienes PyTorch instalado en tu entorno y quieres evitar conflictos:

```bash
pip install -r requirements-minimal.txt
```

### Opción 3: Resolver Conflictos Manualmente

Si necesitas mantener todas las dependencias en el mismo entorno:

#### Para entornos con moshi 0.2.11:
```bash
# moshi requiere torch<2.8,>=2.2.0
pip install "torch>=2.2.0,<2.8.0" "torchaudio>=2.1.0,<2.8.0"
pip install "bitsandbytes>=0.45,<0.46"
pip install -r requirements-minimal.txt
```

#### Para entornos con torchvision 0.24.0+cu130:
```bash
# torchvision requiere torch==2.9.0
pip install torch==2.9.0 torchaudio==2.9.0
pip install -r requirements-minimal.txt
```

## Verificar Instalación

```bash
python scripts/check_dependencies.py
```

## Solución de Problemas

### Error: "moshi requires torch<2.8"
- **Solución**: Usa un entorno virtual separado o instala `torch<2.8.0`

### Error: "torchvision requires torch==2.9.0"
- **Solución**: Actualiza torch a 2.9.0 o desinstala torchvision si no es necesario

### Error: "bitsandbytes incompatible"
- **Solución**: Instala la versión compatible: `pip install "bitsandbytes>=0.45,<0.46"`

## Notas

- `fromPodtoCast` no requiere `moshi` directamente
- Los conflictos surgen si compartes el entorno con otros proyectos que usan `moshi`
- La mejor práctica es usar entornos virtuales separados por proyecto

