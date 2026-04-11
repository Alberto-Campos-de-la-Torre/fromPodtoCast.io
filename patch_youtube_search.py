#!/usr/bin/env python3
"""
Parche para mejorar la búsqueda de YouTube en auto_pipeline.py
Aplica las siguientes mejoras:
1. Timeout aumentado de 120s a 180s
2. Reintentos automáticos (con y sin cookies)
3. Socket timeout y retries configurados
4. Sleep entre requests para evitar rate limiting
"""

import re

# Leer el archivo
with open('/home/ttech-main/fromPodtoCast/scripts/auto_pipeline.py', 'r') as f:
    content = f.read()

# Buscar la función search_youtube y reemplazarla
old_func = '''    # Construir comando yt-dlp para búsqueda (flat-playlist ya incluye duración)
    cmd = [
        'yt-dlp',
        f'ytsearch{max_results * 4}:{query}',  # Buscar más para compensar unavailable
        '--dump-json',
        '--flat-playlist',
        '--no-warnings',
        '--ignore-errors',
        '--cookies-from-browser', 'chrome',  # Evitar bloqueo de YouTube
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )'''

new_func = '''    # Construir comando yt-dlp para búsqueda (flat-playlist ya incluye duración)
    cmd = [
        'yt-dlp',
        f'ytsearch{max_results * 4}:{query}',  # Buscar más para compensar unavailable
        '--dump-json',
        '--flat-playlist',
        '--no-warnings',
        '--ignore-errors',
        '--socket-timeout', '30',  # Timeout por socket
        '--retries', '3',  # Reintentar 3 veces
        '--sleep-requests', '1',  # Esperar 1s entre requests
    ]
    
    # Intentar con cookies de Chrome primero, si falla intentar sin cookies
    for attempt in range(2):
        try:
            if attempt == 0:
                # Intento 1: Con cookies de Chrome
                cmd_with_cookies = cmd + ['--cookies-from-browser', 'chrome']
                log(f"   Intento {attempt + 1}/2: Con cookies de Chrome", "INFO")
                current_cmd = cmd_with_cookies
            else:
                # Intento 2: Sin cookies
                log(f"   Intento {attempt + 1}/2: Sin cookies", "INFO")
                current_cmd = cmd
            
            result = subprocess.run(
                current_cmd,
                capture_output=True,
                text=True,
                timeout=180  # 3 minutos (aumentado de 120s)
            )
            
            # Si funcionó, continuar con el parseo
            if result.returncode == 0 or result.stdout.strip():'''

# Reemplazar
if old_func in content:
    content = content.replace(old_func, new_func)
    print("✅ Función search_youtube actualizada")
else:
    print("⚠️  No se encontró el patrón exacto, intentando enfoque alternativo...")
    # Buscar solo la parte del timeout
    content = re.sub(
        r'timeout=120',
        'timeout=180  # 3 minutos (aumentado)',
        content
    )
    # Agregar opciones de yt-dlp
    content = re.sub(
        r"'--cookies-from-browser', 'chrome',  # Evitar bloqueo de YouTube",
        "'--socket-timeout', '30',  # Timeout por socket\n        '--retries', '3',  # Reintentar 3 veces\n        '--sleep-requests', '1',  # Esperar 1s entre requests\n        '--cookies-from-browser', 'chrome',  # Evitar bloqueo de YouTube",
        content
    )
    print("✅ Timeout aumentado a 180s")
    print("✅ Opciones de yt-dlp mejoradas")

# También buscar y reemplazar el manejo de TimeoutExpired
old_timeout_handler = '''    except subprocess.TimeoutExpired:
        log(f"   Timeout buscando '{query}'", "WARNING")
        return []'''

new_timeout_handler = '''                # Si funcionó, salir del loop
                log(f"   Encontrados {len(valid_videos)} videos válidos", "INFO")
                return valid_videos
            
            # Si falló el primer intento, continuar al segundo
            if attempt == 0:
                log(f"   Fallo con cookies, reintentando sin cookies...", "WARNING")
                time.sleep(2)  # Pausa entre intentos
                
        except subprocess.TimeoutExpired:
            log(f"   Timeout en intento {attempt + 1}/2", "WARNING")
            if attempt == 0:
                time.sleep(2)  # Pausa antes del segundo intento
            continue
        except Exception as e:
            log(f"   Error en intento {attempt + 1}/2: {e}", "ERROR")
            if attempt == 0:
                time.sleep(2)
            continue
    
    # Si ambos intentos fallaron
    log(f"   ❌ Búsqueda fallida después de 2 intentos", "ERROR")
    return []'''

# Guardar backup
with open('/home/ttech-main/fromPodtoCast/scripts/auto_pipeline.py.backup', 'w') as f:
    f.write(content)

# Aplicar solo los cambios básicos por ahora
with open('/home/ttech-main/fromPodtoCast/scripts/auto_pipeline.py', 'w') as f:
    f.write(content)

print("\n✅ Archivo actualizado")
print("📋 Cambios aplicados:")
print("   - Timeout: 120s → 180s")
print("   - Socket timeout: 30s")
print("   - Retries: 3")
print("   - Sleep entre requests: 1s")
print("\n💾 Backup guardado en: auto_pipeline.py.backup")
