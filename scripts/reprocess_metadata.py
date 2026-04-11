#!/usr/bin/env python3
"""
Script para re-procesar metadata ya generado.

Re-aplica correcciones LLM y verificación MCP al texto transcrito,
sin necesidad de re-procesar audio.

Uso:
    python reprocess_metadata.py <metadata.json> [--config config.json]
    python reprocess_metadata.py --all --data-dir /path/to/data [--config config.json]
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

# Agregar src al path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

try:
    from text_corrector_llm import TextCorrectorLLM
    LLM_AVAILABLE = True
except ImportError:
    print("⚠️  TextCorrectorLLM no disponible")
    LLM_AVAILABLE = False
    TextCorrectorLLM = None

try:
    from text_verifier_mcp import TextVerifierMCP
    MCP_AVAILABLE = True
except ImportError:
    print("⚠️  TextVerifierMCP no disponible")
    MCP_AVAILABLE = False
    TextVerifierMCP = None


class MetadataReprocessor:
    """
    Re-procesa metadata aplicando correcciones LLM y verificación MCP.
    """
    
    def __init__(self, config: Dict, progress_file: Optional[str] = None):
        """
        Inicializa el re-procesador de metadata.
        
        Args:
            config: Configuración del procesador
            progress_file: Ruta al archivo de progreso para resume
        """
        self.config = config
        llm_config = config.get('llm_correction', {})
        
        # Progress tracking
        self.progress_file = progress_file
        self.progress = self._load_progress()
        
        # Inicializar corrector LLM
        self.llm_corrector = None
        if llm_config.get('enabled', False) and LLM_AVAILABLE:
            try:
                text_config = config.get('text_preprocessing', {})
                self.llm_corrector = TextCorrectorLLM(
                    ollama_host=llm_config.get('ollama_host', 'http://localhost:11434'),
                    model=llm_config.get('model', 'qwen3:14b'),
                    glosario_path=text_config.get('glosario_path'),
                    timeout=llm_config.get('timeout', 120),
                    max_retries=llm_config.get('max_retries', 3),
                    batch_size=llm_config.get('batch_size', 5),
                    enable_cache=llm_config.get('enable_cache', True),
                    cache_file=llm_config.get('cache_file'),
                    max_workers=llm_config.get('max_workers', 2),
                    enable_verification=False  # Deshabilitamos verificación interna
                )
                self.llm_min_confidence = llm_config.get('min_confidence', 0.7)
                print(f"✓ Corrector LLM inicializado (modelo: {llm_config.get('model', 'qwen3:14b')})")
            except Exception as e:
                print(f"✗ Error inicializando corrector LLM: {e}")
                self.llm_corrector = None
        else:
            print("⚠️  Corrector LLM deshabilitado o no disponible")
        
        # Inicializar verificador MCP
        self.mcp_verifier = None
        mcp_config = llm_config.get('mcp_verification', {})
        if llm_config.get('enabled', False) and mcp_config.get('enabled', False) and MCP_AVAILABLE:
            try:
                self.mcp_verifier = TextVerifierMCP(
                    ollama_host=llm_config.get('ollama_host', 'http://localhost:11434'),
                    model=mcp_config.get('model', llm_config.get('model', 'qwen3:14b')),
                    dictionary_path=mcp_config.get('dictionary_path'),
                    timeout=mcp_config.get('timeout', 60),
                    confidence_threshold=mcp_config.get('confidence_threshold', 0.80)
                )
                print(f"✓ Verificador MCP inicializado (modelo: {mcp_config.get('model', 'qwen3:14b')})")
            except Exception as e:
                print(f"✗ Error inicializando verificador MCP: {e}")
                self.mcp_verifier = None
        else:
            print("⚠️  Verificador MCP deshabilitado o no disponible")
    
    def _load_progress(self) -> Dict:
        """Carga el archivo de progreso si existe."""
        if self.progress_file and os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                print(f"✓ Progreso cargado: {len(progress.get('completed', []))} archivos ya procesados")
                return progress
            except Exception as e:
                print(f"⚠️  Error cargando progreso: {e}")
        return {'completed': [], 'failed': [], 'started_at': datetime.now().isoformat()}
    
    def _save_progress(self):
        """Guarda el archivo de progreso."""
        if not self.progress_file:
            return
        try:
            self.progress['updated_at'] = datetime.now().isoformat()
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.progress, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  Error guardando progreso: {e}")
    
    def _mark_completed(self, filepath: str, stats: Dict):
        """Marca un archivo como completado en el progreso."""
        self.progress['completed'].append({
            'file': os.path.basename(filepath),
            'path': filepath,
            'timestamp': datetime.now().isoformat(),
            'entries': stats.get('total_entries', 0),
            'corrected': stats.get('llm_corrected', 0)
        })
        self._save_progress()
    
    def _mark_failed(self, filepath: str, error: str):
        """Marca un archivo como fallido en el progreso."""
        self.progress['failed'].append({
            'file': os.path.basename(filepath),
            'path': filepath,
            'timestamp': datetime.now().isoformat(),
            'error': error
        })
        self._save_progress()
    
    def is_completed(self, filepath: str) -> bool:
        """Verifica si un archivo ya fue procesado exitosamente."""
        basename = os.path.basename(filepath)
        return any(c['file'] == basename for c in self.progress.get('completed', []))
    
    def _save_metadata_intermediate(self, metadata, entries, metadata_path: str):
        """Guarda metadata intermediamente (después de LLM, antes de MCP) para no perder progreso."""
        try:
            if isinstance(metadata, dict) and not isinstance(metadata, list):
                if 'entries' in metadata:
                    metadata['entries'] = entries
                elif 'segments' in metadata:
                    metadata['segments'] = entries
                else:
                    metadata = entries[0] if len(entries) == 1 else entries
                output_data = metadata
            else:
                output_data = entries
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"   💾 Guardado intermedio: {os.path.basename(metadata_path)}")
        except Exception as e:
            print(f"   ⚠️  Error en guardado intermedio: {e}")
    
    def backup_metadata(self, metadata_path: str) -> str:
        """
        Crea backup del archivo de metadata original.
        
        Args:
            metadata_path: Ruta al archivo de metadata
            
        Returns:
            Ruta al archivo de backup
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{metadata_path}.backup_{timestamp}"
        shutil.copy2(metadata_path, backup_path)
        print(f"📦 Backup creado: {backup_path}")
        return backup_path
    
    def reprocess_metadata(self, metadata_path: str, create_backup: bool = True) -> Dict:
        """
        Re-procesa un archivo de metadata.
        
        Args:
            metadata_path: Ruta al archivo JSON de metadata
            create_backup: Si crear backup antes de modificar
            
        Returns:
            Estadísticas del re-procesamiento
        """
        print(f"\n{'='*60}")
        print(f"Re-procesando: {Path(metadata_path).name}")
        print(f"{'='*60}\n")
        
        # Verificar que el archivo existe
        if not os.path.exists(metadata_path):
            print(f"✗ Error: Archivo no encontrado: {metadata_path}")
            return {'error': 'file_not_found'}
        
        # Crear backup
        if create_backup:
            self.backup_metadata(metadata_path)
        
        # Cargar metadata
        print("📖 Cargando metadata...")
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"✗ Error cargando metadata: {e}")
            return {'error': 'load_failed', 'message': str(e)}
        
        # Si metadata es una lista, procesarla directamente
        # Si es un dict, buscar la lista de entries
        if isinstance(metadata, dict):
            # Puede tener diferentes estructuras
            if 'entries' in metadata:
                entries = metadata['entries']
            elif 'segments' in metadata:
                entries = metadata['segments']
            else:
                # Asumir que es un dict con una sola entrada
                entries = [metadata]
        else:
            entries = metadata
        
        print(f"   ✓ Cargadas {len(entries)} entradas\n")
        
        stats = {
            'total_entries': len(entries),
            'llm_corrected': 0,
            'llm_failed': 0,
            'mcp_verified': 0,
            'mcp_reverted': 0,
            'cache_hits': 0,
            'skipped_empty': 0,
            'skipped_already_done': 0
        }
        
        # Extraer textos originales y guardar los que tienen texto original
        # para preservar la corrección anterior
        print("🔍 Identificando textos a re-procesar...")
        texts_to_reprocess = []
        indices_to_reprocess = []
        
        for i, entry in enumerate(entries):
            text = entry.get('text', '').strip()
            if not text:
                stats['skipped_empty'] += 1
                continue
            
            # ENTRY-LEVEL RESUME: Skip entries already reprocessed in this cycle
            llm_corr = entry.get('llm_correction', {})
            if llm_corr.get('reprocessed_at'):
                stats['skipped_already_done'] += 1
                continue
            
            # Usar el texto original si existe (de correcciones previas)
            # De lo contrario, usar el texto actual
            original_text = text
            if 'llm_correction' in entry and 'original' in entry['llm_correction']:
                original_text = entry['llm_correction']['original']
            elif 'text_original' in entry:
                original_text = entry['text_original']
            
            texts_to_reprocess.append(original_text)
            indices_to_reprocess.append(i)
        
        print(f"   ✓ {len(texts_to_reprocess)} textos para re-procesar")
        if stats['skipped_already_done'] > 0:
            print(f"   ⏭️  {stats['skipped_already_done']} entradas ya reprocesadas (saltadas)")
        print(f"   ⊘ {stats['skipped_empty']} entradas sin texto (saltadas)\n")
        
        if not texts_to_reprocess:
            print("⚠️  No hay textos para re-procesar")
            return stats
        
        # Re-procesar con LLM en chunks con guardado intermedio
        if self.llm_corrector:
            print("🤖 Aplicando correcciones LLM (con guardado por batch)...")
            
            # Tamaño de chunk para guardado intermedio (= batch_size del LLM)
            llm_config = self.config.get('llm_correction', {})
            save_chunk_size = llm_config.get('batch_size', 8)
            total_chunks = (len(texts_to_reprocess) + save_chunk_size - 1) // save_chunk_size
            
            print(f"   📦 Procesando {len(texts_to_reprocess)} textos en {total_chunks} chunks (guardado cada {save_chunk_size})")
            
            for chunk_idx in range(total_chunks):
                chunk_start = chunk_idx * save_chunk_size
                chunk_end = min(chunk_start + save_chunk_size, len(texts_to_reprocess))
                chunk_texts = texts_to_reprocess[chunk_start:chunk_end]
                chunk_indices = indices_to_reprocess[chunk_start:chunk_end]
                
                try:
                    # Procesar este chunk con LLM
                    chunk_corrections = self.llm_corrector.correct_batch_optimized(
                        chunk_texts,
                        verify_corrections=False
                    )
                    
                    # Aplicar correcciones de este chunk a los entries
                    for local_idx, (entry_idx, correction) in enumerate(zip(chunk_indices, chunk_corrections)):
                        entry = entries[entry_idx]
                        
                        if correction is None:
                            stats['llm_failed'] += 1
                            continue
                        
                        try:
                            corrected_text, meta = correction
                        except (TypeError, ValueError):
                            stats['llm_failed'] += 1
                            continue
                        
                        # Guardar texto original si no existe
                        if 'text_original' not in entry and 'llm_correction' not in entry:
                            entry['text_original'] = entry['text']
                        
                        if meta.get('from_cache'):
                            stats['cache_hits'] += 1
                        
                        # Aplicar corrección si la confianza es suficiente
                        confianza = meta.get('confianza', 0)
                        original_for_chunk = texts_to_reprocess[chunk_start + local_idx]
                        if 'error' not in meta and confianza >= self.llm_min_confidence:
                            entry['text'] = corrected_text
                            entry['llm_correction'] = {
                                'original': original_for_chunk,
                                'cambios': meta.get('cambios', []),
                                'confianza': confianza,
                                'reprocessed_at': datetime.now().isoformat()
                            }
                            if corrected_text != original_for_chunk:
                                stats['llm_corrected'] += 1
                        elif 'error' in meta:
                            stats['llm_failed'] += 1
                    
                    # GUARDAR DESPUÉS DE CADA CHUNK → a prueba de interrupciones
                    self._save_metadata_intermediate(metadata, entries, metadata_path)
                    
                    done = min(chunk_end, len(texts_to_reprocess))
                    pct = done / len(texts_to_reprocess) * 100
                    print(f"   💾 Chunk {chunk_idx+1}/{total_chunks} guardado ({done}/{len(texts_to_reprocess)}, {pct:.0f}%)")
                    
                except Exception as e:
                    print(f"   ⚠️  Error en chunk {chunk_idx+1}: {e}")
                    # Guardar lo que se tenga hasta ahora
                    self._save_metadata_intermediate(metadata, entries, metadata_path)
                    print(f"   💾 Progreso guardado hasta chunk {chunk_idx}")
                    import traceback
                    traceback.print_exc()
                    break
            
            print(f"\n   ✓ Aplicadas {stats['llm_corrected']} correcciones")
            if stats['cache_hits'] > 0:
                print(f"   ✓ Cache hits: {stats['cache_hits']}")
            if stats['llm_failed'] > 0:
                print(f"   ⚠️  Fallaron {stats['llm_failed']} correcciones")
        print()
        


        # Re-procesar con MCP
        if self.mcp_verifier and self.llm_corrector:
            print("🔐 Verificando con MCP...")
            
            # Preparar datos para verificación
            corrections_to_verify = []
            verification_indices = []
            
            for i, entry in enumerate(entries):
                if entry.get('llm_correction'):
                    original = entry['llm_correction']['original']
                    corrected = entry['text']
                    metadata_llm = entry['llm_correction']
                    corrections_to_verify.append((original, corrected, metadata_llm))
                    verification_indices.append(i)
            
            if corrections_to_verify:
                try:
                    # Verificar en lote
                    verification_results = self.mcp_verifier.verify_batch(corrections_to_verify)
                    
                    # Aplicar resultados
                    for entry_idx, result in zip(verification_indices, verification_results):
                        entry = entries[entry_idx]
                        
                        # Si el texto fue revertido por MCP
                        if result.cambios_revertidos:
                            entry['text'] = result.texto_verificado
                            entry['llm_correction']['mcp_reverted'] = True
                            entry['llm_correction']['mcp_razon'] = result.cambios_revertidos
                            stats['mcp_reverted'] += 1
                        else:
                            # Verificación aceptada
                            entry['text'] = result.texto_verificado
                            entry['llm_correction']['mcp_verified'] = True
                            entry['llm_correction']['mcp_confianza'] = result.confianza
                            stats['mcp_verified'] += 1
                        
                        # Guardar validaciones MCP
                        if result.validaciones_mcp:
                            entry['llm_correction']['mcp_validaciones'] = result.validaciones_mcp
                        
                        # Timestamp de verificación
                        entry['llm_correction']['mcp_verified_at'] = datetime.now().isoformat()
                    
                    mcp_stats = self.mcp_verifier.get_stats()
                    print(f"   ✓ Verificados {stats['mcp_verified']} textos")
                    if stats['mcp_reverted'] > 0:
                        print(f"   ⚠️  Revertidos {stats['mcp_reverted']} textos (regionalismos/términos protegidos)")
                    print(f"   ✓ Consultas al diccionario: {mcp_stats.get('consultas_mcp', 0)}")
                    print(f"   ✓ Confianza promedio: {mcp_stats.get('promedio_confianza', 0):.2f}")
                    print()
                except Exception as e:
                    print(f"✗ Error en verificación MCP: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print("⚠️  Saltando verificación MCP (no disponible)\n")
        
        # Guardar metadata actualizado
        print("💾 Guardando metadata actualizado...")
        try:
            # Determinar la estructura a guardar
            if isinstance(metadata, dict) and not isinstance(metadata, list):
                # Era un dict, actualizar la estructura correspondiente
                if 'entries' in metadata:
                    metadata['entries'] = entries
                elif 'segments' in metadata:
                    metadata['segments'] = entries
                else:
                    # Era un dict con una sola entrada
                    metadata = entries[0] if len(entries) == 1 else entries
                output_data = metadata
            else:
                # Era una lista
                output_data = entries
            
            # Agregar metadata de re-procesamiento
            if isinstance(output_data, dict) and isinstance(entries, list):
                output_data['reprocessing_info'] = {
                    'timestamp': datetime.now().isoformat(),
                    'stats': stats
                }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"   ✓ Metadata guardado: {metadata_path}\n")
        except Exception as e:
            print(f"✗ Error guardando metadata: {e}")
            return {'error': 'save_failed', 'message': str(e)}
        
        return stats
    
    def reprocess_all_in_directory(self, data_dir: str, pattern: str = "*.json") -> Dict:
        """
        Re-procesa todos los archivos de metadata en un directorio.
        
        Args:
            data_dir: Directorio donde buscar archivos de metadata
            pattern: Patrón de archivos a buscar
            
        Returns:
            Estadísticas agregadas
        """
        print(f"\n{'='*60}")
        print(f"Buscando archivos de metadata en: {data_dir}")
        print(f"{'='*60}\n")
        
        # Buscar archivos de metadata
        metadata_files = []
        data_path = Path(data_dir)
        
        # Buscar en subdirectorio metadata/
        metadata_subdir = data_path / 'metadata'
        if metadata_subdir.exists():
            metadata_files.extend(list(metadata_subdir.glob(pattern)))
        
        # Buscar en subdirectorios logs/ (archivos de podcast individuales)
        logs_subdir = data_path / 'logs'
        if logs_subdir.exists():
            # Los logs no son metadata, los saltamos
            pass
        
        # Buscar en directorio raíz
        metadata_files.extend([f for f in data_path.glob(pattern) if f.is_file()])
        
        # Eliminar duplicados y filtrar backups
        metadata_files = list(set(metadata_files))
        metadata_files = [f for f in metadata_files if '.backup_' not in str(f)]
        
        # Filtrar archivos ya completados
        skipped = 0
        pending_files = []
        for f in metadata_files:
            if self.is_completed(str(f)):
                skipped += 1
            else:
                pending_files.append(f)
        
        print(f"📁 Encontrados {len(metadata_files)} archivos de metadata")
        if skipped > 0:
            print(f"⏭️  Saltando {skipped} archivos ya procesados")
        print(f"📋 Pendientes: {len(pending_files)} archivos\n")
        
        if not pending_files:
            print("✅ Todos los archivos ya fueron procesados")
            return {'error': 'all_completed', 'skipped': skipped}
        
        # Procesar cada archivo
        all_stats = {
            'total_files': len(metadata_files),
            'pending_files': len(pending_files),
            'skipped': skipped,
            'successful': 0,
            'failed': 0,
            'total_entries': 0,
            'llm_corrected': 0,
            'mcp_verified': 0,
            'mcp_reverted': 0
        }
        
        for i, metadata_file in enumerate(pending_files):
            print(f"\n[{i+1}/{len(pending_files)}] ", end="")
            try:
                stats = self.reprocess_metadata(str(metadata_file))
                
                if 'error' not in stats:
                    all_stats['successful'] += 1
                    all_stats['total_entries'] += stats.get('total_entries', 0)
                    all_stats['llm_corrected'] += stats.get('llm_corrected', 0)
                    all_stats['mcp_verified'] += stats.get('mcp_verified', 0)
                    all_stats['mcp_reverted'] += stats.get('mcp_reverted', 0)
                    self._mark_completed(str(metadata_file), stats)
                else:
                    all_stats['failed'] += 1
                    self._mark_failed(str(metadata_file), stats.get('error', 'unknown'))
            except Exception as e:
                print(f"✗ Error procesando {metadata_file}: {e}")
                all_stats['failed'] += 1
                self._mark_failed(str(metadata_file), str(e))
        
        # Guardar progreso final
        self._save_progress()
        return all_stats


def load_config(config_path: Optional[str] = None) -> Dict:
    """
    Carga configuración desde archivo o usa defaults.
    
    Args:
        config_path: Ruta al archivo de configuración
        
    Returns:
        Diccionario de configuración
    """
    default_config = {
        'text_preprocessing': {
            'glosario_path': 'config/glosario_terminos.json'
        },
        'llm_correction': {
            'enabled': True,
            'ollama_host': 'http://localhost:11434',
            'model': 'qwen3:14b',
            'timeout': 120,
            'max_retries': 3,
            'batch_size': 5,
            'min_confidence': 0.7,
            'enable_cache': True,
            'max_workers': 2,
            'mcp_verification': {
                'enabled': True,
                'model': 'qwen3:14b',
                'dictionary_path': 'data/diccionario_base.json',
                'timeout': 60,
                'confidence_threshold': 0.80
            }
        }
    }
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            # Merge configs (user config overrides defaults)
            # Deep merge para nested dicts
            for key, value in user_config.items():
                if key in default_config and isinstance(value, dict):
                    default_config[key].update(value)
                else:
                    default_config[key] = value
            print(f"✓ Configuración cargada desde: {config_path}")
        except Exception as e:
            print(f"⚠️  Error cargando configuración: {e}")
            print("   Usando configuración por defecto")
    else:
        print("ℹ️  Usando configuración por defecto")
    
    return default_config


def print_summary(stats: Dict):
    """Imprime resumen de estadísticas."""
    print(f"\n{'='*60}")
    print("RESUMEN DE RE-PROCESAMIENTO")
    print(f"{'='*60}")
    
    if 'error' in stats:
        print(f"❌ Error: {stats['error']}")
        if 'message' in stats:
            print(f"   Mensaje: {stats['message']}")
    else:
        if 'total_files' in stats:
            # Resumen de múltiples archivos
            print(f"Archivos procesados: {stats['successful']}/{stats['total_files']}")
            if stats['failed'] > 0:
                print(f"Archivos fallidos:   {stats['failed']}")
            print(f"Total de entradas:   {stats['total_entries']}")
        else:
            # Resumen de un solo archivo
            print(f"Total de entradas:   {stats['total_entries']}")
            print(f"Entradas vacías:     {stats.get('skipped_empty', 0)}")
        
        print(f"\nCorrección LLM:")
        print(f"  Corregidas:        {stats['llm_corrected']}")
        print(f"  Fallidas:          {stats.get('llm_failed', 0)}")
        print(f"  Cache hits:        {stats.get('cache_hits', 0)}")
        
        print(f"\nVerificación MCP:")
        print(f"  Verificadas:       {stats['mcp_verified']}")
        print(f"  Revertidas:        {stats['mcp_reverted']}")
    
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Re-procesa metadata aplicando correcciones LLM y verificación MCP'
    )
    parser.add_argument(
        'metadata_file',
        nargs='?',
        help='Archivo de metadata JSON a re-procesar'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Re-procesar todos los archivos de metadata en un directorio'
    )
    parser.add_argument(
        '--data-dir',
        default='./data/output',
        help='Directorio donde buscar archivos de metadata (default: ./data/output)'
    )
    parser.add_argument(
        '--config',
        help='Archivo de configuración JSON'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='No crear backup antes de modificar'
    )
    parser.add_argument(
        '--progress-file',
        default='./data/reprocess_progress.json',
        help='Archivo para tracking de progreso (default: ./data/reprocess_progress.json)'
    )
    
    args = parser.parse_args()
    
    # Validar argumentos
    if not args.all and not args.metadata_file:
        parser.error("Debe especificar un archivo de metadata o usar --all")
    
    # Cargar configuración
    config = load_config(args.config)
    
    # Inicializar reprocessor
    print("\n🚀 Inicializando re-procesador...\n")
    reprocessor = MetadataReprocessor(config, progress_file=args.progress_file)
    
    # Verificar que al menos uno de los procesadores está disponible
    if not reprocessor.llm_corrector and not reprocessor.mcp_verifier:
        print("❌ Error: No hay procesadores disponibles (LLM y MCP deshabilitados)")
        sys.exit(1)
    
    # Ejecutar re-procesamiento
    if args.all:
        stats = reprocessor.reprocess_all_in_directory(args.data_dir)
    else:
        stats = reprocessor.reprocess_metadata(
            args.metadata_file,
            create_backup=not args.no_backup
        )
    
    # Mostrar resumen
    print_summary(stats)
    
    # Código de salida
    if 'error' in stats:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
