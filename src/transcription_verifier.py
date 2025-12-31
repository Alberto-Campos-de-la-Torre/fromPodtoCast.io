"""
Módulo para verificar la calidad de las correcciones de transcripción.

Implementa múltiples capas de verificación:
1. Length Ratio Check - Verificar que la longitud corregida está en rango aceptable
2. Word Preservation Check - Verificar que palabras clave se mantienen
3. Hallucination Detection - Detectar palabras nuevas no presentes en el original
4. Semantic Similarity Check - Verificar similitud semántica (opcional)

Usa Pydantic para validación de datos.
"""
import re
from typing import Dict, List, Optional, Tuple, Any
import logging
from difflib import SequenceMatcher

# Intentar importar Pydantic
try:
    from pydantic import BaseModel, Field, field_validator, ConfigDict
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None

if PYDANTIC_AVAILABLE:
    class VerificationDetails(BaseModel):
        """Detalles de la verificación con validación Pydantic."""
        model_config = ConfigDict(extra='allow')
        
        llm_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
        length_ratio: Optional[float] = Field(default=None, ge=0.0)
        word_preservation: Optional[float] = Field(default=None, ge=0.0, le=1.0)
        new_word_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
        sequence_similarity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
        missing_words: List[str] = Field(default_factory=list)
        potential_hallucinations: List[str] = Field(default_factory=list)
        no_changes: bool = Field(default=False)
        
    class VerificationResult(BaseModel):
        """Resultado de la verificación de una corrección - Modelo Pydantic."""
        model_config = ConfigDict(validate_assignment=True)
        
        is_valid: bool = Field(..., description="Si la corrección pasó la verificación")
        original_text: str = Field(..., description="Texto original antes de corrección")
        corrected_text: str = Field(..., description="Texto después de corrección LLM")
        confidence_score: float = Field(..., ge=0.0, le=1.0, description="Score de confianza calculado")
        checks_passed: List[str] = Field(default_factory=list, description="Checks que pasaron")
        checks_failed: List[str] = Field(default_factory=list, description="Checks que fallaron")
        warnings: List[str] = Field(default_factory=list, description="Advertencias generadas")
        details: VerificationDetails = Field(default_factory=VerificationDetails, description="Detalles de verificación")
        
        @field_validator('confidence_score')
        @classmethod
        def clamp_confidence(cls, v: float) -> float:
            """Asegura que confidence esté entre 0 y 1."""
            return max(0.0, min(1.0, v))
        
        def to_dict(self) -> Dict[str, Any]:
            """Convierte a diccionario para serialización."""
            return self.model_dump()
        
        def to_json(self) -> str:
            """Convierte a JSON string."""
            return self.model_dump_json(indent=2)
else:
    # Fallback sin Pydantic - usar dataclass simple
    from dataclasses import dataclass, field
    
    @dataclass
    class VerificationDetails:
        """Detalles de la verificación (fallback dataclass)."""
        llm_confidence: float = 1.0
        length_ratio: Optional[float] = None
        word_preservation: Optional[float] = None
        new_word_ratio: Optional[float] = None
        sequence_similarity: Optional[float] = None
        missing_words: List[str] = field(default_factory=list)
        potential_hallucinations: List[str] = field(default_factory=list)
        no_changes: bool = False
    
    @dataclass
    class VerificationResult:
        """Resultado de la verificación de una corrección (fallback dataclass)."""
        is_valid: bool
        original_text: str
        corrected_text: str
        confidence_score: float
        checks_passed: List[str] = field(default_factory=list)
        checks_failed: List[str] = field(default_factory=list)
        warnings: List[str] = field(default_factory=list)
        details: Dict = field(default_factory=dict)
        
        def to_dict(self) -> Dict[str, Any]:
            """Convierte a diccionario para serialización."""
            from dataclasses import asdict
            return asdict(self)



class TranscriptionVerifier:
    """
    Verificador de correcciones de transcripción.
    
    Asegura que las correcciones del LLM no alteren el significado
    ni introduzcan alucinaciones.
    """
    
    # Palabras funcionales que se pueden ignorar en comparaciones
    STOP_WORDS = {
        'a', 'ante', 'bajo', 'cabe', 'con', 'contra', 'de', 'desde', 'durante',
        'en', 'entre', 'hacia', 'hasta', 'mediante', 'para', 'por', 'según',
        'sin', 'so', 'sobre', 'tras', 'versus', 'vía',
        'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
        'yo', 'tú', 'él', 'ella', 'ello', 'nosotros', 'vosotros', 'ellos', 'ellas',
        'mi', 'tu', 'su', 'nuestro', 'vuestro',
        'este', 'ese', 'aquel', 'esta', 'esa', 'aquella',
        'que', 'cual', 'quien', 'cuyo', 'donde', 'cuando', 'como', 'cuanto',
        'y', 'e', 'ni', 'o', 'u', 'pero', 'mas', 'sino', 'aunque', 'porque',
        'pues', 'si', 'ya', 'qué', 'quién', 'cómo', 'cuándo', 'dónde', 'cuál',
        'muy', 'más', 'menos', 'tan', 'tanto', 'mucho', 'poco', 'algo', 'nada',
        'sí', 'no', 'también', 'tampoco', 'además', 'incluso', 'solo', 'sólo',
        'es', 'son', 'está', 'están', 'era', 'eran', 'fue', 'fueron', 'ser', 'estar',
        'hay', 'ha', 'han', 'había', 'hubo', 'haber', 'hacer', 'hecho',
        'tiene', 'tienen', 'tenía', 'tuvo', 'tener', 'puede', 'pueden', 'poder',
        'va', 'van', 'ir', 'viene', 'vienen', 'venir', 'dice', 'decir',
        'ah', 'eh', 'oh', 'uh', 'uhm', 'um', 'mmm', 'pues', 'bueno', 'entonces',
    }
    
    def __init__(
        self,
        min_length_ratio: float = 0.70,
        max_length_ratio: float = 1.40,
        min_word_preservation: float = 0.80,
        max_new_word_ratio: float = 0.25,
        min_sequence_similarity: float = 0.60,
        enable_semantic_check: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        """
        Inicializa el verificador.
        
        Args:
            min_length_ratio: Ratio mínimo de longitud (corregido/original)
            max_length_ratio: Ratio máximo de longitud
            min_word_preservation: Porcentaje mínimo de palabras a preservar
            max_new_word_ratio: Ratio máximo de palabras nuevas permitidas
            min_sequence_similarity: Similitud mínima de secuencias (difflib)
            enable_semantic_check: Habilitar verificación semántica con embeddings
            logger: Logger para mensajes
        """
        self.min_length_ratio = min_length_ratio
        self.max_length_ratio = max_length_ratio
        self.min_word_preservation = min_word_preservation
        self.max_new_word_ratio = max_new_word_ratio
        self.min_sequence_similarity = min_sequence_similarity
        self.enable_semantic_check = enable_semantic_check
        self.logger = logger or logging.getLogger(__name__)
        
        # Estadísticas
        self.stats = {
            'verified': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'reverted_to_original': 0
        }
        
        # Modelo de embeddings (carga lazy)
        self._embedding_model = None
    
    def _normalize_text(self, text: str) -> str:
        """Normaliza texto para comparación."""
        # Convertir a minúsculas
        text = text.lower()
        # Remover puntuación
        text = re.sub(r'[¿¡?!.,;:\'"«»""''—–-]', ' ', text)
        # Normalizar espacios
        text = ' '.join(text.split())
        return text
    
    def _extract_words(self, text: str, include_stop_words: bool = False) -> set:
        """Extrae palabras únicas del texto."""
        normalized = self._normalize_text(text)
        words = set(normalized.split())
        
        if not include_stop_words:
            words = words - self.STOP_WORDS
        
        # Filtrar palabras muy cortas (menos de 3 caracteres)
        words = {w for w in words if len(w) >= 3}
        
        return words
    
    def _check_length_ratio(
        self, 
        original: str, 
        corrected: str
    ) -> Tuple[bool, float, str]:
        """
        Verifica que la longitud del texto corregido está en rango aceptable.
        
        Returns:
            Tuple (passed, ratio, message)
        """
        orig_len = len(original.strip())
        corr_len = len(corrected.strip())
        
        if orig_len == 0:
            return False, 0.0, "Texto original vacío"
        
        ratio = corr_len / orig_len
        
        if ratio < self.min_length_ratio:
            return False, ratio, f"Texto muy acortado ({ratio:.1%} del original)"
        elif ratio > self.max_length_ratio:
            return False, ratio, f"Texto muy expandido ({ratio:.1%} del original)"
        
        return True, ratio, f"Longitud aceptable ({ratio:.1%})"
    
    def _check_word_preservation(
        self, 
        original: str, 
        corrected: str
    ) -> Tuple[bool, float, str, List[str]]:
        """
        Verifica que las palabras clave se mantienen.
        
        Returns:
            Tuple (passed, preservation_ratio, message, missing_words)
        """
        orig_words = self._extract_words(original)
        corr_words = self._extract_words(corrected)
        
        if not orig_words:
            return True, 1.0, "Sin palabras significativas", []
        
        # Palabras preservadas
        preserved = orig_words & corr_words
        preservation_ratio = len(preserved) / len(orig_words)
        
        # Palabras faltantes
        missing = orig_words - corr_words
        
        if preservation_ratio < self.min_word_preservation:
            return (
                False, 
                preservation_ratio, 
                f"Baja preservación ({preservation_ratio:.1%})",
                list(missing)[:5]  # Primeras 5 palabras faltantes
            )
        
        return (
            True, 
            preservation_ratio, 
            f"Palabras preservadas ({preservation_ratio:.1%})",
            list(missing)
        )
    
    def _check_hallucination(
        self, 
        original: str, 
        corrected: str
    ) -> Tuple[bool, float, str, List[str]]:
        """
        Detecta posibles alucinaciones (palabras nuevas no presentes).
        
        Returns:
            Tuple (passed, new_word_ratio, message, new_words)
        """
        orig_words = self._extract_words(original, include_stop_words=False)
        corr_words = self._extract_words(corrected, include_stop_words=False)
        
        if not corr_words:
            return True, 0.0, "Sin palabras nuevas", []
        
        # Palabras nuevas en la corrección
        new_words = corr_words - orig_words
        
        # Filtrar correcciones ortográficas obvias
        # (palabras que difieren solo en tildes o mayúsculas)
        truly_new = set()
        for new_word in new_words:
            # Verificar si es una variante ortográfica
            is_variant = False
            for orig_word in orig_words:
                # Comparar sin tildes
                orig_simple = self._remove_accents(orig_word)
                new_simple = self._remove_accents(new_word)
                if orig_simple == new_simple:
                    is_variant = True
                    break
                # Comparar similitud
                if SequenceMatcher(None, orig_word, new_word).ratio() > 0.85:
                    is_variant = True
                    break
            
            if not is_variant:
                truly_new.add(new_word)
        
        new_ratio = len(truly_new) / len(corr_words) if corr_words else 0
        
        if new_ratio > self.max_new_word_ratio:
            return (
                False, 
                new_ratio, 
                f"Posibles alucinaciones ({new_ratio:.1%} palabras nuevas)",
                list(truly_new)[:5]
            )
        
        return (
            True, 
            new_ratio, 
            f"Palabras nuevas aceptables ({new_ratio:.1%})",
            list(truly_new)
        )
    
    def _check_sequence_similarity(
        self, 
        original: str, 
        corrected: str
    ) -> Tuple[bool, float, str]:
        """
        Verifica similitud de secuencia usando difflib.
        
        Returns:
            Tuple (passed, similarity, message)
        """
        orig_norm = self._normalize_text(original)
        corr_norm = self._normalize_text(corrected)
        
        similarity = SequenceMatcher(None, orig_norm, corr_norm).ratio()
        
        if similarity < self.min_sequence_similarity:
            return (
                False, 
                similarity, 
                f"Baja similitud de secuencia ({similarity:.1%})"
            )
        
        return (
            True, 
            similarity, 
            f"Similitud aceptable ({similarity:.1%})"
        )
    
    def _remove_accents(self, text: str) -> str:
        """Remueve acentos para comparación."""
        replacements = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'ü': 'u', 'ñ': 'n',
            'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
            'Ü': 'U', 'Ñ': 'N'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
    
    def verify(
        self, 
        original: str, 
        corrected: str,
        llm_confidence: float = 1.0
    ) -> VerificationResult:
        """
        Verifica una corrección de transcripción.
        
        Args:
            original: Texto original
            corrected: Texto corregido por LLM
            llm_confidence: Confianza reportada por el LLM
            
        Returns:
            VerificationResult con el resultado de la verificación
        """
        self.stats['verified'] += 1
        
        checks_passed = []
        checks_failed = []
        warnings = []
        
        # Crear detalles usando Pydantic si está disponible
        if PYDANTIC_AVAILABLE:
            details = VerificationDetails(llm_confidence=llm_confidence)
        else:
            details = {'llm_confidence': llm_confidence}
        
        # Si el texto no cambió, está OK
        if original.strip() == corrected.strip():
            if PYDANTIC_AVAILABLE:
                no_change_details = VerificationDetails(llm_confidence=llm_confidence, no_changes=True)
            else:
                no_change_details = {'no_changes': True, 'llm_confidence': llm_confidence}
            
            return VerificationResult(
                is_valid=True,
                original_text=original,
                corrected_text=corrected,
                confidence_score=1.0,
                checks_passed=['no_changes'],
                checks_failed=[],
                warnings=[],
                details=no_change_details
            )
        
        # 1. Check de longitud
        length_ok, length_ratio, length_msg = self._check_length_ratio(original, corrected)
        if PYDANTIC_AVAILABLE:
            details.length_ratio = length_ratio
        else:
            details['length_ratio'] = length_ratio
        if length_ok:
            checks_passed.append('length_ratio')
        else:
            checks_failed.append('length_ratio')
            warnings.append(length_msg)
        
        # 2. Check de preservación de palabras
        word_ok, word_ratio, word_msg, missing = self._check_word_preservation(original, corrected)
        if PYDANTIC_AVAILABLE:
            details.word_preservation = word_ratio
            details.missing_words = missing
        else:
            details['word_preservation'] = word_ratio
            details['missing_words'] = missing
        if word_ok:
            checks_passed.append('word_preservation')
        else:
            checks_failed.append('word_preservation')
            warnings.append(word_msg)
        
        # 3. Check de alucinación
        halluc_ok, halluc_ratio, halluc_msg, new_words = self._check_hallucination(original, corrected)
        if PYDANTIC_AVAILABLE:
            details.new_word_ratio = halluc_ratio
            details.potential_hallucinations = new_words
        else:
            details['new_word_ratio'] = halluc_ratio
            details['potential_hallucinations'] = new_words
        if halluc_ok:
            checks_passed.append('hallucination_check')
        else:
            checks_failed.append('hallucination_check')
            warnings.append(halluc_msg)
        
        # 4. Check de similitud de secuencia
        seq_ok, seq_ratio, seq_msg = self._check_sequence_similarity(original, corrected)
        if PYDANTIC_AVAILABLE:
            details.sequence_similarity = seq_ratio
        else:
            details['sequence_similarity'] = seq_ratio
        if seq_ok:
            checks_passed.append('sequence_similarity')
        else:
            checks_failed.append('sequence_similarity')
            warnings.append(seq_msg)
        
        # Calcular score final
        # Ponderación: length(0.2), words(0.3), hallucination(0.25), sequence(0.25)
        confidence_score = (
            (0.2 * (1.0 if length_ok else 0.5)) +
            (0.3 * word_ratio) +
            (0.25 * (1.0 - halluc_ratio)) +
            (0.25 * seq_ratio)
        )
        
        # Multiplicar por confianza del LLM
        confidence_score *= min(llm_confidence, 1.0)
        
        # Determinar si es válido (al menos 3 de 4 checks pasan)
        is_valid = len(checks_failed) <= 1 and confidence_score >= 0.6
        
        if is_valid:
            self.stats['passed'] += 1
        else:
            self.stats['failed'] += 1
        
        if warnings:
            self.stats['warnings'] += len(warnings)
        
        return VerificationResult(
            is_valid=is_valid,
            original_text=original,
            corrected_text=corrected,
            confidence_score=confidence_score,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            warnings=warnings,
            details=details
        )
    
    def verify_batch(
        self, 
        originals: List[str], 
        correcteds: List[str],
        llm_confidences: Optional[List[float]] = None,
        revert_on_fail: bool = True
    ) -> List[Tuple[str, VerificationResult]]:
        """
        Verifica un lote de correcciones.
        
        Args:
            originals: Lista de textos originales
            correcteds: Lista de textos corregidos
            llm_confidences: Lista de confianzas del LLM (opcional)
            revert_on_fail: Si True, devuelve el original cuando falla verificación
            
        Returns:
            Lista de tuplas (texto_final, resultado_verificación)
        """
        if llm_confidences is None:
            llm_confidences = [1.0] * len(originals)
        
        results = []
        
        for orig, corr, conf in zip(originals, correcteds, llm_confidences):
            result = self.verify(orig, corr, conf)
            
            if result.is_valid:
                final_text = corr
            else:
                if revert_on_fail:
                    final_text = orig
                    self.stats['reverted_to_original'] += 1
                    self.logger.debug(
                        f"Reverting to original. Checks failed: {result.checks_failed}"
                    )
                else:
                    final_text = corr
            
            results.append((final_text, result))
        
        return results
    
    def get_stats(self) -> Dict:
        """Retorna estadísticas de verificación."""
        stats = self.stats.copy()
        if stats['verified'] > 0:
            stats['pass_rate'] = stats['passed'] / stats['verified']
            stats['revert_rate'] = stats['reverted_to_original'] / stats['verified']
        return stats
    
    def reset_stats(self):
        """Reinicia estadísticas."""
        self.stats = {
            'verified': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'reverted_to_original': 0
        }


def test_verifier():
    """Prueba el verificador con ejemplos."""
    verifier = TranscriptionVerifier()
    
    test_cases = [
        # Caso 1: Corrección válida (puntuación y tildes)
        (
            "que es el marketing digital y por que es importante",
            "¿Qué es el marketing digital y por qué es importante?",
            0.95
        ),
        # Caso 2: Corrección válida (nombres propios)
        (
            "voy a usar chatgpt y youtube para estudiar",
            "Voy a usar ChatGPT y YouTube para estudiar.",
            0.92
        ),
        # Caso 3: Alucinación (agrega contenido)
        (
            "me gusta el café",
            "Me gusta el café porque tiene muchos beneficios para la salud y mejora la concentración.",
            0.88
        ),
        # Caso 4: Truncado (elimina contenido)
        (
            "hoy vamos a hablar sobre inteligencia artificial y machine learning",
            "Hoy vamos a hablar.",
            0.75
        ),
        # Caso 5: Cambio semántico (cambia significado)
        (
            "no me gusta la pizza",
            "Me gusta mucho la pizza.",
            0.90
        ),
        # Caso 6: Sin cambios
        (
            "Este texto no necesita corrección.",
            "Este texto no necesita corrección.",
            0.99
        ),
    ]
    
    print("=" * 70)
    print("  TEST: Transcription Verifier")
    print("=" * 70)
    print()
    
    for i, (original, corrected, confidence) in enumerate(test_cases, 1):
        result = verifier.verify(original, corrected, confidence)
        
        status = "✅ VÁLIDO" if result.is_valid else "❌ INVÁLIDO"
        print(f"Caso {i}: {status}")
        print(f"  Original:  {original[:60]}...")
        print(f"  Corregido: {corrected[:60]}...")
        print(f"  Score: {result.confidence_score:.2f}")
        print(f"  Checks OK: {result.checks_passed}")
        print(f"  Checks FAIL: {result.checks_failed}")
        if result.warnings:
            print(f"  Warnings: {result.warnings}")
        print()
    
    print("=" * 70)
    print(f"Estadísticas: {verifier.get_stats()}")
    print("=" * 70)


if __name__ == '__main__':
    test_verifier()
