"""
Sistema de caché persistente para correcciones LLM.
Evita reprocesar textos idénticos entre sesiones.
"""
import json
import hashlib
import os
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
import threading

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Entrada de caché con metadata."""
    text_hash: str
    texto_corregido: str
    cambios: List[str]
    confianza: float
    modelo: str
    created_at: str
    hits: int = 0
    last_accessed: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        return cls(**data)


class CorrectionCache:
    """
    Caché persistente para correcciones LLM.
    
    Características:
    - Persistencia en disco (JSON)
    - Thread-safe
    - Expiración opcional
    - Estadísticas de uso
    - Limpieza automática de entradas antiguas
    """
    
    def __init__(
        self,
        cache_file: str,
        max_entries: int = 10000,
        expire_days: Optional[int] = 30,
        auto_save: bool = True,
        save_interval: int = 100
    ):
        """
        Inicializa el caché.
        
        Args:
            cache_file: Ruta al archivo de caché JSON
            max_entries: Número máximo de entradas
            expire_days: Días antes de expirar (None = no expira)
            auto_save: Guardar automáticamente después de modificaciones
            save_interval: Guardar cada N modificaciones
        """
        self.cache_file = Path(cache_file)
        self.max_entries = max_entries
        self.expire_days = expire_days
        self.auto_save = auto_save
        self.save_interval = save_interval
        
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._modifications = 0
        
        # Estadísticas
        self.stats = {
            'hits': 0,
            'misses': 0,
            'writes': 0,
            'evictions': 0
        }
        
        # Cargar caché existente
        self._load()
    
    def _get_hash(self, text: str) -> str:
        """Genera hash MD5 del texto normalizado."""
        normalized = text.strip().lower()
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def _load(self) -> None:
        """Carga el caché desde disco."""
        if not self.cache_file.exists():
            logger.info(f"Caché no existe, creando nuevo: {self.cache_file}")
            return
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convertir a CacheEntry
            for key, value in data.get('entries', {}).items():
                try:
                    self._cache[key] = CacheEntry.from_dict(value)
                except Exception as e:
                    logger.warning(f"Entrada inválida en caché: {e}")
            
            # Cargar estadísticas
            if 'stats' in data:
                self.stats.update(data['stats'])
            
            logger.info(f"✓ Caché cargado: {len(self._cache)} entradas")
            
            # Limpiar expirados
            if self.expire_days:
                self._cleanup_expired()
                
        except json.JSONDecodeError as e:
            logger.error(f"Error parseando caché: {e}")
            self._cache = {}
        except Exception as e:
            logger.error(f"Error cargando caché: {e}")
            self._cache = {}
    
    def save(self) -> None:
        """Guarda el caché a disco."""
        with self._lock:
            try:
                # Asegurar que el directorio existe
                self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                
                data = {
                    'version': '2.0',
                    'updated_at': datetime.now().isoformat(),
                    'entries_count': len(self._cache),
                    'stats': self.stats,
                    'entries': {
                        k: v.to_dict() for k, v in self._cache.items()
                    }
                }
                
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                logger.debug(f"Caché guardado: {len(self._cache)} entradas")
                self._modifications = 0
                
            except Exception as e:
                logger.error(f"Error guardando caché: {e}")
    
    def get(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Busca una corrección en el caché.
        
        Args:
            text: Texto original
            
        Returns:
            Diccionario con corrección o None si no existe
        """
        text_hash = self._get_hash(text)
        
        with self._lock:
            if text_hash in self._cache:
                entry = self._cache[text_hash]
                
                # Verificar expiración
                if self.expire_days:
                    created = datetime.fromisoformat(entry.created_at)
                    if datetime.now() - created > timedelta(days=self.expire_days):
                        del self._cache[text_hash]
                        self.stats['evictions'] += 1
                        return None
                
                # Actualizar estadísticas
                entry.hits += 1
                entry.last_accessed = datetime.now().isoformat()
                self.stats['hits'] += 1
                
                return {
                    'texto_corregido': entry.texto_corregido,
                    'cambios': entry.cambios,
                    'confianza': entry.confianza,
                    'modelo': entry.modelo,
                    'from_cache': True
                }
            
            self.stats['misses'] += 1
            return None
    
    def set(
        self,
        text: str,
        texto_corregido: str,
        cambios: List[str],
        confianza: float,
        modelo: str = "qwen3:8b"
    ) -> None:
        """
        Agrega una corrección al caché.
        
        Args:
            text: Texto original
            texto_corregido: Texto corregido
            cambios: Lista de cambios aplicados
            confianza: Confianza de la corrección
            modelo: Modelo usado
        """
        text_hash = self._get_hash(text)
        
        with self._lock:
            # Verificar límite de entradas
            if len(self._cache) >= self.max_entries:
                self._evict_oldest()
            
            entry = CacheEntry(
                text_hash=text_hash,
                texto_corregido=texto_corregido,
                cambios=cambios,
                confianza=confianza,
                modelo=modelo,
                created_at=datetime.now().isoformat()
            )
            
            self._cache[text_hash] = entry
            self.stats['writes'] += 1
            self._modifications += 1
            
            # Auto-guardar si corresponde
            if self.auto_save and self._modifications >= self.save_interval:
                self.save()
    
    def _evict_oldest(self) -> None:
        """Elimina la entrada más antigua (LRU)."""
        if not self._cache:
            return
        
        # Encontrar entrada con menor uso / más antigua
        oldest_key = min(
            self._cache.keys(),
            key=lambda k: (
                self._cache[k].hits,
                self._cache[k].created_at
            )
        )
        
        del self._cache[oldest_key]
        self.stats['evictions'] += 1
        logger.debug(f"Evicted cache entry: {oldest_key[:8]}...")
    
    def _cleanup_expired(self) -> None:
        """Limpia entradas expiradas."""
        if not self.expire_days:
            return
        
        now = datetime.now()
        expired = []
        
        for key, entry in self._cache.items():
            created = datetime.fromisoformat(entry.created_at)
            if now - created > timedelta(days=self.expire_days):
                expired.append(key)
        
        for key in expired:
            del self._cache[key]
            self.stats['evictions'] += 1
        
        if expired:
            logger.info(f"Limpiadas {len(expired)} entradas expiradas")
    
    def contains(self, text: str) -> bool:
        """Verifica si un texto está en el caché."""
        return self._get_hash(text) in self._cache
    
    def clear(self) -> None:
        """Limpia todo el caché."""
        with self._lock:
            self._cache.clear()
            self.stats = {
                'hits': 0,
                'misses': 0,
                'writes': 0,
                'evictions': 0
            }
        
        if self.cache_file.exists():
            self.cache_file.unlink()
        
        logger.info("Caché limpiado")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del caché."""
        with self._lock:
            hit_rate = 0.0
            total = self.stats['hits'] + self.stats['misses']
            if total > 0:
                hit_rate = self.stats['hits'] / total * 100
            
            return {
                **self.stats,
                'entries': len(self._cache),
                'hit_rate': round(hit_rate, 2),
                'max_entries': self.max_entries
            }
    
    def __len__(self) -> int:
        return len(self._cache)
    
    def __contains__(self, text: str) -> bool:
        return self.contains(text)
    
    def __del__(self):
        """Guarda al destruir si hay modificaciones pendientes."""
        if self._modifications > 0:
            try:
                self.save()
            except:
                pass


class BatchCorrectionCache(CorrectionCache):
    """
    Extensión del caché optimizada para operaciones batch.
    """
    
    def get_batch(self, texts: List[str]) -> Dict[str, Optional[Dict]]:
        """
        Busca múltiples textos en el caché.
        
        Args:
            texts: Lista de textos a buscar
            
        Returns:
            Diccionario {texto: resultado o None}
        """
        results = {}
        for text in texts:
            results[text] = self.get(text)
        return results
    
    def set_batch(
        self,
        corrections: List[Dict[str, Any]],
        modelo: str = "qwen3:8b"
    ) -> None:
        """
        Agrega múltiples correcciones al caché.
        
        Args:
            corrections: Lista de {text, texto_corregido, cambios, confianza}
            modelo: Modelo usado
        """
        for corr in corrections:
            self.set(
                text=corr['text'],
                texto_corregido=corr['texto_corregido'],
                cambios=corr.get('cambios', []),
                confianza=corr.get('confianza', 0.5),
                modelo=modelo
            )
    
    def filter_uncached(self, texts: List[str]) -> tuple:
        """
        Separa textos cacheados de no cacheados.
        
        Returns:
            Tuple (cached_results: Dict, uncached_texts: List)
        """
        cached = {}
        uncached = []
        
        for text in texts:
            result = self.get(text)
            if result:
                cached[text] = result
            else:
                uncached.append(text)
        
        return cached, uncached


# Singleton global para uso en todo el proyecto
_global_cache: Optional[CorrectionCache] = None


def get_global_cache(
    cache_file: Optional[str] = None,
    **kwargs
) -> CorrectionCache:
    """
    Obtiene o crea el caché global.
    
    Args:
        cache_file: Ruta al archivo (solo necesario la primera vez)
        **kwargs: Argumentos para CorrectionCache
        
    Returns:
        Instancia del caché global
    """
    global _global_cache
    
    if _global_cache is None:
        if cache_file is None:
            # Usar ubicación por defecto
            cache_file = os.path.expanduser(
                "~/.cache/frompodtocast/llm_corrections.json"
            )
        
        _global_cache = BatchCorrectionCache(cache_file, **kwargs)
    
    return _global_cache


def clear_global_cache() -> None:
    """Limpia el caché global."""
    global _global_cache
    if _global_cache:
        _global_cache.clear()
        _global_cache = None

