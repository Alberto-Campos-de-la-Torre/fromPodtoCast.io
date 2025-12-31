"""
Gestión centralizada de dispositivos GPU para el pipeline.

Distribuye la carga de trabajo entre múltiples GPUs:
- GPU 0 (RTX 5090): Workloads pesados (Ollama/LLM)
- GPU 1 (RTX 5070 Ti): Workloads ligeros (Whisper, Diarization)
"""
import torch
from typing import Dict, Optional
import logging


class GPUManager:
    """
    Gestión centralizada de dispositivos GPU.
    
    Permite configurar qué GPU usa cada componente del pipeline.
    """
    
    # Configuración por defecto: Heavy en GPU 0, Light en GPU 1
    DEFAULT_CONFIG = {
        'ollama': 0,        # Heavy LLM (qwen3:14b) - controlado vía systemd
        'whisper': 1,       # Transcription
        'diarization': 1,   # Speaker identification
        'embeddings': 1,    # Voice embeddings
    }
    
    def __init__(self, config: Optional[Dict] = None, logger: Optional[logging.Logger] = None):
        """
        Inicializa el gestor de GPUs.
        
        Args:
            config: Configuración de GPU del config.json (gpu_config section)
            logger: Logger para mensajes
        """
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        
        # Detectar GPUs disponibles
        self.device_count = torch.cuda.device_count()
        self.devices = {}
        
        if self.device_count > 0:
            for i in range(self.device_count):
                self.devices[i] = {
                    'name': torch.cuda.get_device_name(i),
                    'memory_total': torch.cuda.get_device_properties(i).total_memory,
                }
            self._log_gpu_info()
        else:
            self.logger.warning("No se detectaron GPUs CUDA disponibles")
            self.enabled = False
    
    def _log_gpu_info(self):
        """Registra información de las GPUs detectadas."""
        self.logger.info(f"GPUs detectadas: {self.device_count}")
        for idx, info in self.devices.items():
            mem_gb = info['memory_total'] / (1024**3)
            self.logger.info(f"  GPU {idx}: {info['name']} ({mem_gb:.1f} GB)")
    
    def get_device(self, component: str) -> str:
        """
        Obtiene el dispositivo PyTorch para un componente.
        
        Args:
            component: Nombre del componente ('whisper', 'diarization', 'embeddings')
            
        Returns:
            String del dispositivo (ej: 'cuda:1', 'cuda:0', 'cpu')
        """
        if not self.enabled or self.device_count == 0:
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Obtener GPU del config, o usar default
        gpu_idx = self.config.get(f'{component}_gpu', self.DEFAULT_CONFIG.get(component, 0))
        
        # Validar que el índice existe
        if gpu_idx >= self.device_count:
            self.logger.warning(f"GPU {gpu_idx} no existe, usando GPU 0")
            gpu_idx = 0
        
        return f'cuda:{gpu_idx}'
    
    def get_device_for_whisper(self) -> str:
        """Obtiene el dispositivo para Whisper."""
        return self.get_device('whisper')
    
    def get_device_for_diarization(self) -> str:
        """Obtiene el dispositivo para diarización."""
        return self.get_device('diarization')
    
    def get_device_for_embeddings(self) -> str:
        """Obtiene el dispositivo para embeddings."""
        return self.get_device('embeddings')
    
    def get_memory_info(self) -> Dict:
        """
        Obtiene información de memoria de todas las GPUs.
        
        Returns:
            Dict con información de memoria por GPU
        """
        info = {}
        for i in range(self.device_count):
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            total = self.devices[i]['memory_total']
            info[i] = {
                'name': self.devices[i]['name'],
                'allocated_gb': allocated / (1024**3),
                'reserved_gb': reserved / (1024**3),
                'total_gb': total / (1024**3),
                'free_gb': (total - allocated) / (1024**3),
            }
        return info
    
    def print_status(self):
        """Imprime el estado actual de las GPUs."""
        print("\n" + "="*60)
        print("  GPU Status")
        print("="*60)
        
        if self.device_count == 0:
            print("  No GPUs detectadas")
            return
        
        mem_info = self.get_memory_info()
        for idx, info in mem_info.items():
            print(f"\n  GPU {idx}: {info['name']}")
            print(f"    Allocated: {info['allocated_gb']:.2f} GB")
            print(f"    Free:      {info['free_gb']:.2f} GB")
            print(f"    Total:     {info['total_gb']:.2f} GB")
        
        print("\n" + "="*60)


# Singleton global
_gpu_manager: Optional[GPUManager] = None


def get_gpu_manager(config: Optional[Dict] = None) -> GPUManager:
    """
    Obtiene la instancia singleton del GPUManager.
    
    Args:
        config: Configuración (solo se usa en la primera llamada)
        
    Returns:
        Instancia de GPUManager
    """
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager(config)
    return _gpu_manager


def reset_gpu_manager():
    """Reinicia el singleton (útil para testing)."""
    global _gpu_manager
    _gpu_manager = None


if __name__ == '__main__':
    # Test
    manager = GPUManager()
    manager.print_status()
    
    print("\nDevice assignments:")
    print(f"  Whisper:      {manager.get_device_for_whisper()}")
    print(f"  Diarization:  {manager.get_device_for_diarization()}")
    print(f"  Embeddings:   {manager.get_device_for_embeddings()}")
