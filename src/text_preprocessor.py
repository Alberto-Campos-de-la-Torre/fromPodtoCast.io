"""
Módulo para pre-procesamiento automático de transcripciones en español.
Aplica correcciones basadas en reglas sin necesidad de LLM.
"""
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class TextPreprocessor:
    """
    Pre-procesador de texto para transcripciones en español.
    
    Aplica correcciones automáticas:
    - Puntuación española (¿? ¡!)
    - Normalización de números a texto
    - Corrección de espaciado
    - Capitalización correcta
    - Correcciones de errores comunes (glosario)
    """
    
    # Mapeo de números a texto en español
    NUMEROS = {
        '0': 'cero', '1': 'uno', '2': 'dos', '3': 'tres', '4': 'cuatro',
        '5': 'cinco', '6': 'seis', '7': 'siete', '8': 'ocho', '9': 'nueve',
        '10': 'diez', '11': 'once', '12': 'doce', '13': 'trece', '14': 'catorce',
        '15': 'quince', '16': 'dieciséis', '17': 'diecisiete', '18': 'dieciocho',
        '19': 'diecinueve', '20': 'veinte', '21': 'veintiuno', '22': 'veintidós',
        '23': 'veintitrés', '24': 'veinticuatro', '25': 'veinticinco',
        '26': 'veintiséis', '27': 'veintisiete', '28': 'veintiocho',
        '29': 'veintinueve', '30': 'treinta', '40': 'cuarenta', '50': 'cincuenta',
        '60': 'sesenta', '70': 'setenta', '80': 'ochenta', '90': 'noventa',
        '100': 'cien', '200': 'doscientos', '300': 'trescientos',
        '400': 'cuatrocientos', '500': 'quinientos', '600': 'seiscientos',
        '700': 'setecientos', '800': 'ochocientos', '900': 'novecientos',
        '1000': 'mil', '1000000': 'un millón'
    }
    
    # Palabras que indican pregunta
    PALABRAS_PREGUNTA = [
        'qué', 'que', 'quién', 'quien', 'cómo', 'como', 'cuándo', 'cuando',
        'dónde', 'donde', 'por qué', 'porqué', 'cuál', 'cual', 'cuánto',
        'cuanto', 'cuántos', 'cuantos', 'acaso', 'verdad', 'cierto',
        'es que', 'será que', 'puede que', 'podría', 'podrías', 'puedes',
        'sabes', 'crees', 'piensas', 'te parece', 'no crees', 'o no'
    ]
    
    # Palabras que indican exclamación
    PALABRAS_EXCLAMACION = [
        'qué', 'cómo', 'cuánto', 'vaya', 'menudo', 'increíble', 'genial',
        'wow', 'guau', 'órale', 'híjole', 'chin', 'chale', 'no manches',
        'no mames', 'qué chido', 'qué padre', 'qué onda', 'sale'
    ]
    
    def __init__(self, glosario_path: Optional[str] = None,
                 fix_punctuation: bool = True,
                 normalize_numbers: bool = True,
                 fix_spacing: bool = True,
                 fix_capitalization: bool = True):
        """
        Inicializa el pre-procesador.
        
        Args:
            glosario_path: Ruta al archivo JSON con correcciones personalizadas
            fix_punctuation: Si True, corrige puntuación española
            normalize_numbers: Si True, convierte números a texto
            fix_spacing: Si True, normaliza espaciado
            fix_capitalization: Si True, corrige mayúsculas
        """
        self.fix_punctuation = fix_punctuation
        self.normalize_numbers = normalize_numbers
        self.fix_spacing = fix_spacing
        self.fix_capitalization = fix_capitalization
        
        # Cargar glosario de correcciones
        self.glosario = self._load_glosario(glosario_path)
        
        # Estadísticas de procesamiento
        self.stats = {
            'processed': 0,
            'punctuation_fixed': 0,
            'numbers_normalized': 0,
            'spacing_fixed': 0,
            'capitalization_fixed': 0,
            'glosario_applied': 0
        }
    
    def preprocess(self, text: str) -> Tuple[str, Dict]:
        """
        Aplica todas las correcciones al texto.
        
        Args:
            text: Texto a procesar
        
        Returns:
            Tuple (texto_corregido, cambios_realizados)
        """
        original = text
        changes = {}
        
        # 1. Normalizar espacios primero
        if self.fix_spacing:
            text, spacing_changes = self._fix_spacing_impl(text)
            if spacing_changes:
                changes['spacing'] = spacing_changes
        
        # 2. Aplicar glosario de correcciones
        text, glosario_changes = self._apply_glosario(text)
        if glosario_changes:
            changes['glosario'] = glosario_changes
        
        # 3. Normalizar números a texto
        if self.normalize_numbers:
            text, number_changes = self._normalize_numbers_impl(text)
            if number_changes:
                changes['numbers'] = number_changes
        
        # 4. Corregir puntuación española
        if self.fix_punctuation:
            text, punct_changes = self._fix_spanish_punctuation(text)
            if punct_changes:
                changes['punctuation'] = punct_changes
        
        # 5. Corregir capitalización
        if self.fix_capitalization:
            text, cap_changes = self._fix_capitalization_impl(text)
            if cap_changes:
                changes['capitalization'] = cap_changes
        
        # Actualizar estadísticas
        self.stats['processed'] += 1
        if changes:
            for key in changes:
                stat_key = f"{key}_fixed" if key != 'glosario' else 'glosario_applied'
                if stat_key in self.stats:
                    self.stats[stat_key] += 1
        
        return text.strip(), changes
    
    def preprocess_batch(self, entries: List[Dict], 
                         text_field: str = 'text') -> List[Dict]:
        """
        Procesa un lote de entradas.
        
        Args:
            entries: Lista de diccionarios con campo de texto
            text_field: Nombre del campo que contiene el texto
        
        Returns:
            Lista de entradas con texto corregido
        """
        processed = []
        
        for entry in entries:
            if text_field not in entry:
                processed.append(entry)
                continue
            
            original_text = entry[text_field]
            corrected_text, changes = self.preprocess(original_text)
            
            # Crear copia del entry con texto corregido
            new_entry = entry.copy()
            new_entry[text_field] = corrected_text
            
            # Guardar texto original y cambios si hubo correcciones
            if corrected_text != original_text:
                new_entry['text_original'] = original_text
                new_entry['text_changes'] = changes
            
            processed.append(new_entry)
        
        return processed
    
    def _load_glosario(self, path: Optional[str]) -> Dict:
        """Carga el glosario de correcciones."""
        default_glosario = {
            'correcciones': {
                # Errores comunes de Whisper
                'que es': 'qué es',
                'como es': 'cómo es',
                'por que': 'por qué',
                'porque ': 'por qué ',  # Al inicio de pregunta
                'cuando ': 'cuándo ',   # En preguntas
                'donde ': 'dónde ',     # En preguntas
                'cual ': 'cuál ',       # En preguntas
                'quien ': 'quién ',     # En preguntas
                
                # Marcas y nombres comunes
                'google': 'Google',
                'youtube': 'YouTube',
                'tiktok': 'TikTok',
                'instagram': 'Instagram',
                'facebook': 'Facebook',
                'whatsapp': 'WhatsApp',
                'twitter': 'Twitter',
                'chatgpt': 'ChatGPT',
                'gpt': 'GPT',
                'openai': 'OpenAI',
                'gemini': 'Gemini',
                'gemina': 'Gemini',  # Error común
                'claude': 'Claude',
                'meta': 'Meta',
                'apple': 'Apple',
                'microsoft': 'Microsoft',
                'amazon': 'Amazon',
                'netflix': 'Netflix',
                'spotify': 'Spotify',
                
                # Acrónimos
                'ia': 'IA',
                'i.a.': 'IA',
                'ai': 'IA',
                'a.i.': 'IA',
                'seo': 'SEO',
                'sem': 'SEM',
                'roi': 'ROI',
                'kpi': 'KPI',
                'b2b': 'B2B',
                'b2c': 'B2C',
                'url': 'URL',
                'api': 'API',
                
                # Correcciones ortográficas comunes
                'atraves': 'a través',
                'atravez': 'a través',
                'enserio': 'en serio',
                'osea': 'o sea',
                'talvez': 'tal vez',
                'almenos': 'al menos',
                'aparte': 'aparte',
                'aveces': 'a veces',
                'deveras': 'de veras',
                'esque': 'es que',
                'nose': 'no sé',
                'haiga': 'haya',
                'nadien': 'nadie',
                'dijistes': 'dijiste',
                'hicistes': 'hiciste',
            },
            'mantener': [
                # Regionalismos mexicanos que no deben corregirse
                'güey', 'wey', 'we',
                'chido', 'chida',
                'neta', 'la neta',
                'chamba', 'chambear',
                'feria', 'lana', 'varo',
                'morro', 'morra',
                'carnal', 'carnala',
                'chale', 'nel', 'simón',
                'órale', 'ándale',
                'no manches', 'no mames',
                'qué onda', 'qué pedo',
                'está cañón', 'está chido',
                'aguas', 'chécalo',
            ]
        }
        
        if path and Path(path).exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    custom = json.load(f)
                    # Merge con glosario por defecto
                    for key in custom:
                        if key in default_glosario:
                            if isinstance(default_glosario[key], dict):
                                default_glosario[key].update(custom[key])
                            elif isinstance(default_glosario[key], list):
                                default_glosario[key].extend(custom[key])
                        else:
                            default_glosario[key] = custom[key]
            except Exception as e:
                print(f"⚠️  Error cargando glosario: {e}")
        
        return default_glosario
    
    def _fix_spacing_impl(self, text: str) -> Tuple[str, List[str]]:
        """Normaliza espaciado en el texto."""
        changes = []
        original = text
        
        # Eliminar espacios múltiples
        text = re.sub(r' +', ' ', text)
        
        # Eliminar espacios antes de puntuación
        text = re.sub(r' +([.,;:!?)])', r'\1', text)
        
        # Añadir espacio después de puntuación si falta
        text = re.sub(r'([.,;:!?])([A-Za-záéíóúñÁÉÍÓÚÑ])', r'\1 \2', text)
        
        # Eliminar espacios después de signos de apertura
        text = re.sub(r'([¿¡(]) +', r'\1', text)
        
        # Eliminar espacios al inicio y final
        text = text.strip()
        
        if text != original:
            changes.append('espaciado_normalizado')
        
        return text, changes
    
    def _apply_glosario(self, text: str) -> Tuple[str, List[str]]:
        """Aplica correcciones del glosario."""
        changes = []
        correcciones = self.glosario.get('correcciones', {})
        
        for error, correccion in correcciones.items():
            # Búsqueda case-insensitive para la mayoría
            pattern = re.compile(re.escape(error), re.IGNORECASE)
            if pattern.search(text):
                # Preservar el caso cuando sea apropiado
                if error.islower() and not correccion.isupper():
                    text = pattern.sub(correccion, text)
                else:
                    text = text.replace(error, correccion)
                changes.append(f'{error} → {correccion}')
        
        return text, changes
    
    def _normalize_numbers_impl(self, text: str) -> Tuple[str, List[str]]:
        """Convierte números a texto en español."""
        changes = []
        
        def number_to_words(num: int) -> str:
            """Convierte un número entero a palabras."""
            if str(num) in self.NUMEROS:
                return self.NUMEROS[str(num)]
            
            if num < 0:
                return 'menos ' + number_to_words(-num)
            
            if num < 100:
                if num < 30:
                    return self.NUMEROS.get(str(num), str(num))
                
                decena = (num // 10) * 10
                unidad = num % 10
                
                if unidad == 0:
                    return self.NUMEROS.get(str(decena), str(decena))
                else:
                    decena_word = self.NUMEROS.get(str(decena), '')
                    unidad_word = self.NUMEROS.get(str(unidad), '')
                    return f"{decena_word} y {unidad_word}"
            
            if num < 1000:
                if num == 100:
                    return 'cien'
                
                centena = (num // 100) * 100
                resto = num % 100
                
                centena_word = self.NUMEROS.get(str(centena), '')
                if centena == 100:
                    centena_word = 'ciento'
                
                if resto == 0:
                    return centena_word
                else:
                    return f"{centena_word} {number_to_words(resto)}"
            
            if num < 1000000:
                miles = num // 1000
                resto = num % 1000
                
                if miles == 1:
                    miles_word = 'mil'
                else:
                    miles_word = f"{number_to_words(miles)} mil"
                
                if resto == 0:
                    return miles_word
                else:
                    return f"{miles_word} {number_to_words(resto)}"
            
            # Para números más grandes, mantener como dígitos
            return str(num)
        
        # Buscar números en el texto (1-4 dígitos para evitar años, IDs, etc.)
        def replace_number(match):
            num_str = match.group(0)
            try:
                num = int(num_str)
                # Solo convertir números menores a 10000 (evitar años, códigos)
                if 0 <= num < 10000:
                    word = number_to_words(num)
                    changes.append(f'{num_str} → {word}')
                    return word
            except ValueError:
                pass
            return num_str
        
        # Patrones para números (evitar números en URLs, emails, etc.)
        # Solo números que están solos o seguidos de palabras comunes
        text = re.sub(
            r'\b(\d{1,4})\b(?=\s|[.,;:!?]|$)',
            replace_number,
            text
        )
        
        # Convertir porcentajes
        def replace_percent(match):
            num = match.group(1)
            try:
                word = number_to_words(int(num))
                changes.append(f'{num}% → {word} por ciento')
                return f'{word} por ciento'
            except:
                return match.group(0)
        
        text = re.sub(r'(\d+)%', replace_percent, text)
        
        return text, changes
    
    def _fix_spanish_punctuation(self, text: str) -> Tuple[str, List[str]]:
        """Añade signos de interrogación y exclamación de apertura."""
        changes = []
        sentences = self._split_sentences(text)
        fixed_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Detectar si es pregunta
            is_question = sentence.endswith('?')
            if is_question and not sentence.startswith('¿'):
                # Verificar si debería tener ¿
                if self._is_likely_question(sentence):
                    sentence = '¿' + sentence
                    changes.append('añadido_¿')
            
            # Detectar si es exclamación
            is_exclamation = sentence.endswith('!')
            if is_exclamation and not sentence.startswith('¡'):
                if self._is_likely_exclamation(sentence):
                    sentence = '¡' + sentence
                    changes.append('añadido_¡')
            
            fixed_sentences.append(sentence)
        
        return ' '.join(fixed_sentences), changes
    
    def _split_sentences(self, text: str) -> List[str]:
        """Divide el texto en oraciones."""
        # Dividir por puntos, signos de interrogación y exclamación
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return sentences
    
    def _is_likely_question(self, sentence: str) -> bool:
        """Determina si una oración es probablemente una pregunta."""
        sentence_lower = sentence.lower()
        
        # Verificar palabras interrogativas al inicio
        for word in self.PALABRAS_PREGUNTA:
            if sentence_lower.startswith(word) or f' {word}' in sentence_lower[:30]:
                return True
        
        # Patrones de pregunta
        question_patterns = [
            r'^(es|son|está|están|fue|fueron|era|eran|será|serán)\s',
            r'^(tiene|tienen|hay|había|habrá)\s',
            r'^(puede|pueden|podría|podrían)\s',
            r'^(cree|crees|piensa|piensas|sabes|sabe)\s',
            r'^(te|le|les|nos)\s+(parece|gusta|interesa)',
        ]
        
        for pattern in question_patterns:
            if re.search(pattern, sentence_lower):
                return True
        
        return False
    
    def _is_likely_exclamation(self, sentence: str) -> bool:
        """Determina si una oración es probablemente una exclamación."""
        sentence_lower = sentence.lower()
        
        for word in self.PALABRAS_EXCLAMACION:
            if sentence_lower.startswith(word):
                return True
        
        # Patrones de exclamación
        exclamation_patterns = [
            r'^(qué|cuánto|cómo)\s+(bueno|malo|bien|mal|grande|increíble)',
            r'^(no\s+)?(manches|mames|way|inventes)',
            r'^(wow|guau|órale|híjole)',
        ]
        
        for pattern in exclamation_patterns:
            if re.search(pattern, sentence_lower):
                return True
        
        return False
    
    def _fix_capitalization_impl(self, text: str) -> Tuple[str, List[str]]:
        """Corrige capitalización después de puntuación."""
        changes = []
        
        # Capitalizar después de . ! ? ¿ ¡
        def capitalize_after_punct(match):
            punct = match.group(1)
            space = match.group(2)
            letter = match.group(3)
            if letter.islower():
                changes.append(f'mayúscula después de {punct}')
                return punct + space + letter.upper()
            return match.group(0)
        
        text = re.sub(
            r'([.!?¿¡])(\s+)([a-záéíóúñ])',
            capitalize_after_punct,
            text
        )
        
        # Capitalizar inicio del texto
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
            changes.append('mayúscula_inicio')
        
        return text, changes
    
    def get_stats(self) -> Dict:
        """Retorna estadísticas de procesamiento."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reinicia las estadísticas."""
        for key in self.stats:
            self.stats[key] = 0


def create_default_glosario(output_path: str):
    """Crea un archivo de glosario por defecto."""
    glosario = {
        "correcciones": {
            "que es": "qué es",
            "como es": "cómo es", 
            "por que": "por qué",
            "gemina": "Gemini",
            "chatgpt": "ChatGPT",
            "youtube": "YouTube",
            "tiktok": "TikTok",
            "instagram": "Instagram",
            "ia": "IA",
            "seo": "SEO"
        },
        "mantener": [
            "güey", "chido", "neta", "chamba",
            "órale", "ándale", "no manches"
        ],
        "eliminar": []
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(glosario, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Glosario creado en: {output_path}")


if __name__ == '__main__':
    # Ejemplo de uso
    preprocessor = TextPreprocessor()
    
    test_texts = [
        "que es el marketing digital y por que es importante",
        "El 50% de las empresas usan ia para mejorar sus ventas",
        "no manches guey , eso esta muy chido !",
        "Gemina es el nuevo modelo de google",
        "tienes 5 minutos para explicar el seo",
    ]
    
    print("=== Test de TextPreprocessor ===\n")
    
    for text in test_texts:
        corrected, changes = preprocessor.preprocess(text)
        print(f"Original:  {text}")
        print(f"Corregido: {corrected}")
        if changes:
            print(f"Cambios:   {changes}")
        print()
    
    print("Estadísticas:", preprocessor.get_stats())


