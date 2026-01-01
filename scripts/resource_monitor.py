#!/usr/bin/env python3
"""
Monitor de recursos del sistema para diagnóstico del pipeline.

Ejecutar en terminal separada mientras corre el pipeline para
detectar fugas de memoria y puntos de congelamiento.

Uso:
    python scripts/resource_monitor.py --output data/diagnostics/monitor.csv
"""
import os
import sys
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️  psutil no instalado. Instalar con: pip install psutil")


def get_gpu_info():
    """Obtiene información de las GPUs usando nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 6:
                        gpus.append({
                            'index': int(parts[0]),
                            'name': parts[1],
                            'memory_used_mb': int(parts[2]),
                            'memory_total_mb': int(parts[3]),
                            'utilization': int(parts[4]),
                            'temperature': int(parts[5])
                        })
            return gpus
    except Exception as e:
        pass
    return []


def get_system_info():
    """Obtiene información del sistema."""
    info = {
        'timestamp': datetime.now().isoformat(),
        'cpu_percent': 0,
        'ram_used_gb': 0,
        'ram_total_gb': 0,
        'ram_percent': 0,
        'swap_used_gb': 0,
        'swap_percent': 0,
    }
    
    if PSUTIL_AVAILABLE:
        # CPU
        info['cpu_percent'] = psutil.cpu_percent(interval=0.1)
        
        # RAM
        mem = psutil.virtual_memory()
        info['ram_used_gb'] = round(mem.used / (1024**3), 2)
        info['ram_total_gb'] = round(mem.total / (1024**3), 2)
        info['ram_percent'] = mem.percent
        
        # Swap
        swap = psutil.swap_memory()
        info['swap_used_gb'] = round(swap.used / (1024**3), 2)
        info['swap_percent'] = swap.percent
    
    return info


def get_python_processes():
    """Obtiene procesos Python activos (posibles workers del pipeline)."""
    processes = []
    if PSUTIL_AVAILABLE:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_percent', 'cpu_percent']):
            try:
                if 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if any(x in cmdline for x in ['pipeline', 'processor', 'main.py', 'whisper', 'pyannote']):
                        processes.append({
                            'pid': proc.info['pid'],
                            'cmdline': cmdline[:80],
                            'memory_percent': round(proc.info['memory_percent'], 2),
                            'cpu_percent': proc.info['cpu_percent']
                        })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    return processes


def format_gpu_status(gpus):
    """Formatea el estado de las GPUs para mostrar."""
    if not gpus:
        return "GPU: No disponible"
    
    lines = []
    for gpu in gpus:
        mem_percent = (gpu['memory_used_mb'] / gpu['memory_total_mb']) * 100
        lines.append(
            f"  GPU{gpu['index']}: {gpu['memory_used_mb']:,}MB/{gpu['memory_total_mb']:,}MB "
            f"({mem_percent:.1f}%) | {gpu['utilization']}% util | {gpu['temperature']}°C"
        )
    return '\n'.join(lines)


def monitor_loop(output_file: str, interval: float = 5.0, alert_threshold_gpu: int = 90,
                 alert_threshold_ram: int = 90):
    """Loop principal de monitoreo."""
    
    # Crear directorio de salida
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Escribir encabezado CSV
    with open(output_file, 'w') as f:
        f.write("timestamp,cpu_percent,ram_used_gb,ram_percent,swap_used_gb,")
        f.write("gpu0_mem_mb,gpu0_percent,gpu0_util,gpu0_temp,")
        f.write("gpu1_mem_mb,gpu1_percent,gpu1_util,gpu1_temp\n")
    
    print("=" * 70)
    print("  🔍 Monitor de Recursos - Pipeline Diagnóstico")
    print("=" * 70)
    print(f"  Archivo de salida: {output_file}")
    print(f"  Intervalo: {interval}s")
    print(f"  Alerta GPU: >{alert_threshold_gpu}% | Alerta RAM: >{alert_threshold_ram}%")
    print("-" * 70)
    print("  Presiona Ctrl+C para detener")
    print("=" * 70)
    print()
    
    iteration = 0
    max_gpu_mem = [0, 0]  # Tracking máximos por GPU
    max_ram = 0
    
    try:
        while True:
            iteration += 1
            sys_info = get_system_info()
            gpus = get_gpu_info()
            
            # Calcular datos de GPU
            gpu_data = []
            for i in range(2):  # Asumimos máximo 2 GPUs
                if i < len(gpus):
                    gpu = gpus[i]
                    mem_percent = (gpu['memory_used_mb'] / gpu['memory_total_mb']) * 100
                    gpu_data.append({
                        'mem_mb': gpu['memory_used_mb'],
                        'percent': mem_percent,
                        'util': gpu['utilization'],
                        'temp': gpu['temperature']
                    })
                    max_gpu_mem[i] = max(max_gpu_mem[i], gpu['memory_used_mb'])
                else:
                    gpu_data.append({'mem_mb': 0, 'percent': 0, 'util': 0, 'temp': 0})
            
            max_ram = max(max_ram, sys_info['ram_used_gb'])
            
            # Escribir a CSV
            with open(output_file, 'a') as f:
                f.write(f"{sys_info['timestamp']},{sys_info['cpu_percent']:.1f},")
                f.write(f"{sys_info['ram_used_gb']:.2f},{sys_info['ram_percent']:.1f},")
                f.write(f"{sys_info['swap_used_gb']:.2f},")
                for gd in gpu_data:
                    f.write(f"{gd['mem_mb']},{gd['percent']:.1f},{gd['util']},{gd['temp']},")
                f.write("\n")
            
            # Mostrar en consola
            now = datetime.now().strftime('%H:%M:%S')
            
            # Detectar alertas
            alerts = []
            for i, gd in enumerate(gpu_data):
                if gd['percent'] > alert_threshold_gpu:
                    alerts.append(f"🚨 GPU{i} MEM CRÍTICA: {gd['percent']:.1f}%")
            if sys_info['ram_percent'] > alert_threshold_ram:
                alerts.append(f"🚨 RAM CRÍTICA: {sys_info['ram_percent']:.1f}%")
            
            # Formato de salida
            print(f"\r[{now}] #{iteration:04d} | ", end='')
            print(f"CPU: {sys_info['cpu_percent']:5.1f}% | ", end='')
            print(f"RAM: {sys_info['ram_used_gb']:.1f}/{sys_info['ram_total_gb']:.0f}GB ({sys_info['ram_percent']:.0f}%) | ", end='')
            
            for i, gd in enumerate(gpu_data):
                if gd['mem_mb'] > 0:
                    print(f"GPU{i}: {gd['mem_mb']:,}MB ({gd['percent']:.0f}%) ", end='')
            
            print("", end='\r' if not alerts else '\n')
            
            if alerts:
                for alert in alerts:
                    print(f"  {alert}")
            
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("  📊 Resumen de Monitoreo")
        print("=" * 70)
        print(f"  Iteraciones: {iteration}")
        print(f"  RAM máxima: {max_ram:.2f} GB")
        for i, mem in enumerate(max_gpu_mem):
            if mem > 0:
                print(f"  GPU{i} memoria máxima: {mem:,} MB")
        print(f"\n  Datos guardados en: {output_file}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Monitor de recursos para diagnóstico del pipeline')
    parser.add_argument('--output', '-o', default='data/diagnostics/resource_log.csv',
                        help='Archivo de salida CSV')
    parser.add_argument('--interval', '-i', type=float, default=5.0,
                        help='Intervalo de muestreo en segundos (default: 5)')
    parser.add_argument('--alert-gpu', type=int, default=90,
                        help='Umbral de alerta para memoria GPU %% (default: 90)')
    parser.add_argument('--alert-ram', type=int, default=90,
                        help='Umbral de alerta para RAM %% (default: 90)')
    
    args = parser.parse_args()
    
    if not PSUTIL_AVAILABLE:
        print("Error: psutil es requerido. Instalar con: pip install psutil")
        sys.exit(1)
    
    monitor_loop(args.output, args.interval, args.alert_gpu, args.alert_ram)


if __name__ == '__main__':
    main()
