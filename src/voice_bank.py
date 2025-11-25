"""
Módulo para administrar un banco global de voces con embeddings normalizados.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Callable

import numpy as np


class VoiceBankManager:
    """Gestiona embeddings de hablantes y asignación de IDs globales."""

    def __init__(self, bank_path: str, match_threshold: float = 0.85,
                 id_generator: Optional[Callable[[int], str]] = None):
        self.bank_path = Path(bank_path)
        self.match_threshold = match_threshold
        self.voice_entries: Dict[str, Dict] = {}
        self._id_generator = id_generator
        self._load()

    # ------------------------------------------------------------------ #
    # API pública
    # ------------------------------------------------------------------ #
    def find_best_match(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """Retorna el speaker_id con mayor similitud y la similitud alcanzada."""
        if not self.voice_entries:
            return None, 0.0

        best_id = None
        best_score = -1.0
        for speaker_id, data in self.voice_entries.items():
            ref_emb = data.get("embedding")
            if ref_emb is None:
                continue
            score = self._cosine_similarity(ref_emb, embedding)
            if score > best_score:
                best_score = score
                best_id = speaker_id

        if best_score >= self.match_threshold:
            return best_id, best_score
        return None, best_score

    def add_speaker(self, embedding: np.ndarray) -> Optional[str]:
        """Registra un nuevo hablante en el banco y retorna su ID global."""
        # Validar embedding antes de agregar
        if not self._is_valid_embedding(embedding):
            print(f"   ⚠️  Embedding inválido, no se agregó speaker al banco")
            return None
        
        normalized = self._normalize(embedding)
        if not self._is_valid_embedding(normalized):
            print(f"   ⚠️  Embedding normalizado inválido, no se agregó speaker")
            return None
        
        new_id = self._generate_new_id()
        self.voice_entries[new_id] = {
            "speaker_id": new_id,
            "embedding": normalized,
            "occurrences": 1,
            "last_seen": self._current_timestamp()
        }
        self._save()
        return new_id

    def update_speaker(self, speaker_id: str, embedding: np.ndarray):
        """Actualiza las estadísticas y el embedding promedio de un hablante existente."""
        entry = self.voice_entries.get(speaker_id)
        if entry is None:
            return

        # Validar nuevo embedding
        if not self._is_valid_embedding(embedding):
            print(f"   ⚠️  Embedding inválido, no se actualizó {speaker_id}")
            return

        current_embedding = entry.get("embedding")
        if current_embedding is None or not self._is_valid_embedding(current_embedding):
            entry["embedding"] = self._normalize(embedding)
        else:
            occurrences = entry.get("occurrences", 1)
            updated = (current_embedding * occurrences + embedding) / (occurrences + 1)
            normalized = self._normalize(updated)
            
            # Validar resultado
            if self._is_valid_embedding(normalized):
                entry["embedding"] = normalized
            else:
                print(f"   ⚠️  Embedding actualizado inválido, manteniendo anterior")
                return

        entry["occurrences"] = entry.get("occurrences", 1) + 1
        entry["last_seen"] = self._current_timestamp()
        self._save()

    # ------------------------------------------------------------------ #
    # Utilidades internas
    # ------------------------------------------------------------------ #
    def _load(self):
        if not self.bank_path.exists():
            # Crear directorio si no existe
            os.makedirs(self.bank_path.parent, exist_ok=True)
            self._save()
            return

        try:
            with open(self.bank_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Convertir embeddings a np.array
                for entry in data:
                    speaker_id = entry.get("speaker_id")
                    if not speaker_id:
                        continue
                    embedding_list = entry.get("embedding", [])
                    entry["embedding"] = self._normalize(
                        np.array(embedding_list, dtype=np.float32)
                    ) if embedding_list else None
                    self.voice_entries[speaker_id] = entry
        except Exception as e:
            print(f"⚠️  No se pudo cargar voice_bank ({self.bank_path}): {e}")
            self.voice_entries = {}

    def _save(self):
        serializable = []
        for entry in self.voice_entries.values():
            serializable.append({
                "speaker_id": entry["speaker_id"],
                "embedding": entry["embedding"].tolist() if entry.get("embedding") is not None else [],
                "occurrences": entry.get("occurrences", 1),
                "last_seen": entry.get("last_seen", self._current_timestamp())
            })
        with open(self.bank_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

    def _generate_new_id(self) -> str:
        if self._id_generator:
            return self._id_generator(len(self.voice_entries) + 1)

        existing = sorted(self.voice_entries.keys())
        if not existing:
            return "SPEAKER_GLOBAL_001"

        last = existing[-1]
        try:
            value = int(last.split("_")[-1])
        except ValueError:
            value = len(existing) + 1
        return f"SPEAKER_GLOBAL_{value + 1:03d}"

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        """Normaliza un vector a norma unitaria."""
        if vec is None or len(vec) == 0:
            return vec
        
        # Reemplazar NaN e Inf con 0
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        
        norm = np.linalg.norm(vec)
        if norm == 0 or np.isnan(norm) or np.isinf(norm):
            return vec
        return vec / norm
    
    @staticmethod
    def _is_valid_embedding(embedding: np.ndarray) -> bool:
        """Verifica si un embedding es válido (sin NaN, Inf, norma > 0)."""
        if embedding is None:
            return False
        if not isinstance(embedding, np.ndarray):
            return False
        if len(embedding) == 0:
            return False
        if np.any(np.isnan(embedding)):
            return False
        if np.any(np.isinf(embedding)):
            return False
        
        norm = np.linalg.norm(embedding)
        if norm == 0 or np.isnan(norm) or np.isinf(norm):
            return False
        
        return True

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calcula similitud coseno entre dos embeddings."""
        if a is None or b is None:
            return -1.0
        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            return -1.0
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
        return float(np.dot(a, b) / denom)

    @staticmethod
    def _current_timestamp() -> str:
        return datetime.utcnow().isoformat()

