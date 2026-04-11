"""
Detector de sobreposición de voces basado en análisis de energía.
Detecta cuando múltiples personas hablan simultáneamente.
"""
import numpy as np
import librosa
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass


@dataclass
class OverlapAnalysis:
    """Resultado del análisis de sobreposición."""
    has_overlap: bool
    overlap_ratio: float  # 0.0 = sin sobreposición, 1.0 = todo sobreposición
    overlap_segments: List[Tuple[float, float]]  # Lista de (start, end) con sobreposición
    confidence: float  # Confianza del análisis
    metrics: Dict[str, float]  # Métricas detalladas


class OverlapDetector:
    """
    Detecta sobreposición de voces usando múltiples métricas de audio:

    1. Varianza de energía: Alta variabilidad indica múltiples fuentes
    2. Spectral flatness: Múltiples voces = espectro más plano/ruidoso
    3. Zero-crossing rate: Variación alta indica múltiples fuentes
    4. Pitch variance: Múltiples F0 simultáneas
    5. Spectral centroid variance: Cambios rápidos en frecuencia dominante
    """

    def __init__(
        self,
        frame_length: int = 2048,
        hop_length: int = 512,
        overlap_threshold: float = 0.3,
        min_overlap_duration: float = 0.5,
        energy_var_threshold: float = 0.15,
        spectral_flatness_threshold: float = 0.3,
        zcr_var_threshold: float = 0.1
    ):
        """
        Inicializa el detector de sobreposición.

        Args:
            frame_length: Tamaño de frame para análisis (samples)
            hop_length: Salto entre frames (samples)
            overlap_threshold: Umbral de ratio de sobreposición para marcar segmento
            min_overlap_duration: Duración mínima de sobreposición para considerar (segundos)
            energy_var_threshold: Umbral de varianza de energía normalizada
            spectral_flatness_threshold: Umbral de spectral flatness
            zcr_var_threshold: Umbral de varianza de ZCR
        """
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.overlap_threshold = overlap_threshold
        self.min_overlap_duration = min_overlap_duration
        self.energy_var_threshold = energy_var_threshold
        self.spectral_flatness_threshold = spectral_flatness_threshold
        self.zcr_var_threshold = zcr_var_threshold

    def analyze(self, audio_path: str) -> OverlapAnalysis:
        """
        Analiza un archivo de audio para detectar sobreposición de voces.

        Args:
            audio_path: Ruta al archivo de audio

        Returns:
            OverlapAnalysis con resultados del análisis
        """
        try:
            # Cargar audio
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            return self.analyze_array(y, sr)
        except Exception as e:
            print(f"   Error analizando {audio_path}: {e}")
            return OverlapAnalysis(
                has_overlap=False,
                overlap_ratio=0.0,
                overlap_segments=[],
                confidence=0.0,
                metrics={}
            )

    def analyze_array(self, y: np.ndarray, sr: int) -> OverlapAnalysis:
        """
        Analiza un array de audio para detectar sobreposición.

        Args:
            y: Array de audio (mono)
            sr: Sample rate

        Returns:
            OverlapAnalysis con resultados
        """
        duration = len(y) / sr

        if duration < 0.5:
            return OverlapAnalysis(
                has_overlap=False,
                overlap_ratio=0.0,
                overlap_segments=[],
                confidence=0.5,
                metrics={'duration': duration, 'reason': 'too_short'}
            )

        # Calcular características por frame
        metrics = {}

        # 1. RMS Energy y su varianza local
        rms = librosa.feature.rms(
            y=y,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )[0]

        # Varianza local de energía (ventana deslizante)
        energy_var = self._local_variance(rms, window_size=10)
        metrics['energy_mean'] = float(np.mean(rms))
        metrics['energy_var'] = float(np.mean(energy_var))
        metrics['energy_var_norm'] = float(np.mean(energy_var) / (np.mean(rms) + 1e-8))

        # 2. Spectral Flatness (Wiener entropy)
        # Valores altos = más ruido/múltiples fuentes
        spectral_flatness = librosa.feature.spectral_flatness(
            y=y,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )[0]
        metrics['spectral_flatness_mean'] = float(np.mean(spectral_flatness))
        metrics['spectral_flatness_std'] = float(np.std(spectral_flatness))

        # 3. Zero-Crossing Rate y su varianza
        zcr = librosa.feature.zero_crossing_rate(
            y=y,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )[0]
        zcr_var = self._local_variance(zcr, window_size=10)
        metrics['zcr_mean'] = float(np.mean(zcr))
        metrics['zcr_var'] = float(np.mean(zcr_var))

        # 4. Spectral Centroid y su varianza
        # Cambios rápidos indican múltiples fuentes
        centroid = librosa.feature.spectral_centroid(
            y=y,
            sr=sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )[0]
        centroid_var = self._local_variance(centroid, window_size=10)
        metrics['centroid_mean'] = float(np.mean(centroid))
        metrics['centroid_var'] = float(np.mean(centroid_var))
        metrics['centroid_var_norm'] = float(np.mean(centroid_var) / (np.mean(centroid) + 1e-8))

        # 5. Spectral Bandwidth (ancho de banda)
        # Múltiples voces = mayor ancho de banda
        bandwidth = librosa.feature.spectral_bandwidth(
            y=y,
            sr=sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )[0]
        metrics['bandwidth_mean'] = float(np.mean(bandwidth))
        metrics['bandwidth_std'] = float(np.std(bandwidth))

        # 6. Análisis de pitch (F0) múltiple usando piptrack
        pitches, magnitudes = librosa.piptrack(
            y=y,
            sr=sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
            threshold=0.1
        )

        # Contar frames con múltiples pitches significativos
        multi_pitch_frames = 0
        total_voiced_frames = 0

        for t in range(pitches.shape[1]):
            # Obtener pitches con magnitud significativa
            valid_pitches = pitches[:, t][magnitudes[:, t] > np.max(magnitudes[:, t]) * 0.3]
            valid_pitches = valid_pitches[valid_pitches > 50]  # Filtrar frecuencias muy bajas

            if len(valid_pitches) > 0:
                total_voiced_frames += 1

                # Verificar si hay múltiples pitches distintos
                if len(valid_pitches) > 1:
                    pitch_range = np.max(valid_pitches) - np.min(valid_pitches)
                    # Si el rango de pitches es > 50 Hz, hay múltiples fuentes
                    if pitch_range > 50:
                        multi_pitch_frames += 1

        multi_pitch_ratio = multi_pitch_frames / (total_voiced_frames + 1) if total_voiced_frames > 0 else 0
        metrics['multi_pitch_ratio'] = float(multi_pitch_ratio)
        metrics['voiced_frames'] = total_voiced_frames

        # Detectar frames con sobreposición usando combinación de métricas
        overlap_frames = self._detect_overlap_frames(
            rms, spectral_flatness, zcr, centroid_var, energy_var, sr
        )

        # Convertir frames a segmentos de tiempo
        overlap_segments = self._frames_to_segments(
            overlap_frames, sr, self.hop_length, self.min_overlap_duration
        )

        # Calcular ratio de sobreposición
        overlap_duration = sum(end - start for start, end in overlap_segments)
        overlap_ratio = overlap_duration / duration if duration > 0 else 0
        metrics['overlap_ratio'] = float(overlap_ratio)

        # Determinar si hay sobreposición significativa
        has_overlap = overlap_ratio >= self.overlap_threshold

        # Calcular confianza basada en consistencia de métricas
        confidence = self._calculate_confidence(metrics)

        return OverlapAnalysis(
            has_overlap=has_overlap,
            overlap_ratio=overlap_ratio,
            overlap_segments=overlap_segments,
            confidence=confidence,
            metrics=metrics
        )

    def _local_variance(self, x: np.ndarray, window_size: int = 10) -> np.ndarray:
        """Calcula varianza local usando ventana deslizante."""
        if len(x) < window_size:
            return np.array([np.var(x)])

        # Usar convolución para calcular media local
        kernel = np.ones(window_size) / window_size
        local_mean = np.convolve(x, kernel, mode='same')

        # Calcular varianza local
        local_var = np.convolve((x - local_mean) ** 2, kernel, mode='same')

        return local_var

    def _detect_overlap_frames(
        self,
        rms: np.ndarray,
        spectral_flatness: np.ndarray,
        zcr: np.ndarray,
        centroid_var: np.ndarray,
        energy_var: np.ndarray,
        sr: int
    ) -> np.ndarray:
        """
        Detecta frames con probable sobreposición.

        Usa combinación de métricas:
        - Alta varianza de energía local
        - Alta spectral flatness
        - Alta varianza de ZCR
        - Alta varianza de centroid

        Returns:
            Array booleano indicando frames con sobreposición
        """
        n_frames = len(rms)

        # Normalizar métricas a [0, 1]
        def normalize(x):
            x_min, x_max = np.min(x), np.max(x)
            if x_max - x_min < 1e-8:
                return np.zeros_like(x)
            return (x - x_min) / (x_max - x_min)

        # Asegurar que todos los arrays tengan la misma longitud
        min_len = min(len(rms), len(spectral_flatness), len(zcr))

        rms_norm = normalize(rms[:min_len])
        sf_norm = normalize(spectral_flatness[:min_len])
        zcr_norm = normalize(zcr[:min_len])

        # Ajustar longitud de varianzas
        if len(centroid_var) > min_len:
            centroid_var = centroid_var[:min_len]
        elif len(centroid_var) < min_len:
            centroid_var = np.pad(centroid_var, (0, min_len - len(centroid_var)), mode='edge')

        if len(energy_var) > min_len:
            energy_var = energy_var[:min_len]
        elif len(energy_var) < min_len:
            energy_var = np.pad(energy_var, (0, min_len - len(energy_var)), mode='edge')

        cv_norm = normalize(centroid_var)
        ev_norm = normalize(energy_var)

        # Score de sobreposición: combinar métricas
        # Pesos basados en importancia empírica
        overlap_score = (
            0.25 * sf_norm +      # Spectral flatness alta = ruido/múltiples fuentes
            0.25 * ev_norm +      # Varianza de energía alta = cambios rápidos
            0.20 * cv_norm +      # Varianza de centroid alta = múltiples frecuencias
            0.15 * zcr_norm +     # ZCR alto = contenido complejo
            0.15 * rms_norm       # RMS alto = múltiples fuentes sumándose
        )

        # Umbral adaptativo basado en percentil
        threshold = np.percentile(overlap_score, 75)  # Top 25% se considera overlap

        # Aplicar umbral mínimo
        threshold = max(threshold, 0.4)

        overlap_frames = overlap_score > threshold

        return overlap_frames

    def _frames_to_segments(
        self,
        overlap_frames: np.ndarray,
        sr: int,
        hop_length: int,
        min_duration: float
    ) -> List[Tuple[float, float]]:
        """
        Convierte frames de sobreposición a segmentos de tiempo.

        Args:
            overlap_frames: Array booleano de frames con sobreposición
            sr: Sample rate
            hop_length: Hop length usado para el análisis
            min_duration: Duración mínima para considerar un segmento

        Returns:
            Lista de tuplas (start_time, end_time)
        """
        segments = []

        if len(overlap_frames) == 0:
            return segments

        # Tiempo por frame
        frame_duration = hop_length / sr

        # Encontrar segmentos contiguos
        in_segment = False
        segment_start = 0

        for i, is_overlap in enumerate(overlap_frames):
            if is_overlap and not in_segment:
                # Inicio de segmento
                segment_start = i * frame_duration
                in_segment = True
            elif not is_overlap and in_segment:
                # Fin de segmento
                segment_end = i * frame_duration
                if segment_end - segment_start >= min_duration:
                    segments.append((segment_start, segment_end))
                in_segment = False

        # Cerrar último segmento si quedó abierto
        if in_segment:
            segment_end = len(overlap_frames) * frame_duration
            if segment_end - segment_start >= min_duration:
                segments.append((segment_start, segment_end))

        return segments

    def _calculate_confidence(self, metrics: Dict[str, float]) -> float:
        """
        Calcula confianza del análisis basada en consistencia de métricas.

        Returns:
            Valor entre 0 y 1
        """
        confidence_factors = []

        # Factor 1: Suficientes frames con voz
        if 'voiced_frames' in metrics:
            voiced_factor = min(1.0, metrics['voiced_frames'] / 50)
            confidence_factors.append(voiced_factor)

        # Factor 2: Consistencia entre métricas de overlap
        if 'spectral_flatness_mean' in metrics and 'energy_var_norm' in metrics:
            # Si ambas métricas indican lo mismo, mayor confianza
            sf_indicates_overlap = metrics['spectral_flatness_mean'] > self.spectral_flatness_threshold
            ev_indicates_overlap = metrics['energy_var_norm'] > self.energy_var_threshold
            consistency = 1.0 if sf_indicates_overlap == ev_indicates_overlap else 0.5
            confidence_factors.append(consistency)

        # Factor 3: Ratio de multi-pitch significativo
        if 'multi_pitch_ratio' in metrics:
            mp_factor = min(1.0, 0.5 + metrics['multi_pitch_ratio'])
            confidence_factors.append(mp_factor)

        if not confidence_factors:
            return 0.5

        return float(np.mean(confidence_factors))

    def get_overlap_severity(self, analysis: OverlapAnalysis) -> str:
        """
        Clasifica la severidad de la sobreposición.

        Returns:
            'none', 'low', 'medium', 'high'
        """
        if not analysis.has_overlap:
            return 'none'

        ratio = analysis.overlap_ratio

        if ratio < 0.15:
            return 'low'
        elif ratio < 0.35:
            return 'medium'
        else:
            return 'high'
