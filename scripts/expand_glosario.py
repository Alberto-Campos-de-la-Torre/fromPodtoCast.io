#!/usr/bin/env python3
"""
Herramienta para expandir el glosario de términos mediante fuentes verificadas.
Usa APIs médicas oficiales (PubMed/MeSH) para obtener términos reales.
Usa Ollama (qwen3:14b) SOLO para generar variantes incorrectas de Whisper.
"""
import json
import re
import argparse
import time
import requests
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field, asdict
import urllib.parse

@dataclass
class TerminoBase:
    """Estructura para el diccionario_base.json"""
    tipo: str
    categoria: str
    definicion: str
    variantes_incorrectas: List[str]
    forma_correcta: str
    ejemplos: List[str]
    contextos_relacionados: List[str]

    def to_dict(self):
        return asdict(self)

class MedicalTermFetcher:
    """Obtiene términos médicos de fuentes verificadas."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Medical-Glossary-Expander/1.0 (Educational; mailto:dev@example.com)'
        })
    
    def fetch_from_pubmed_mesh(self, query: str, max_results: int = 20) -> List[Dict[str, str]]:
        """Obtiene términos de MeSH (Medical Subject Headings) de PubMed."""
        terms = []
        
        try:
            # API de E-utilities de NCBI para buscar en MeSH
            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
            
            # Paso 1: Buscar términos relacionados
            search_url = f"{base_url}esearch.fcgi"
            params = {
                'db': 'mesh',
                'term': query,
                'retmax': max_results,
                'retmode': 'json'
            }
            
            response = self.session.get(search_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            ids = data.get('esearchresult', {}).get('idlist', [])
            
            if not ids:
                print(f"      ⚠️ No se encontraron términos MeSH para '{query}'")
                return []
            
            # Paso 2: Obtener detalles de cada término
            fetch_url = f"{base_url}esummary.fcgi"
            for mesh_id in ids[:max_results]:
                try:
                    params = {
                        'db': 'mesh',
                        'id': mesh_id,
                        'retmode': 'json'
                    }
                    response = self.session.get(fetch_url, params=params, timeout=10)
                    response.raise_for_status()
                    summary = response.json()
                    
                    result = summary.get('result', {}).get(mesh_id, {})
                    term_name = result.get('ds_meshterms', [''])[0]
                    
                    if term_name:
                        terms.append({
                            'nombre': term_name,
                            'definicion': result.get('scopenote', 'Término médico verificado'),
                            'categoria': query,
                            'fuente': 'PubMed-MeSH'
                        })
                    
                    time.sleep(0.4)  # Rate limiting cortés
                    
                except Exception as e:
                    print(f"      ⚠️ Error obteniendo detalles de MeSH ID {mesh_id}: {e}")
                    continue
            
            print(f"      ✅ Descargados {len(terms)} términos de MeSH")
            
        except Exception as e:
            print(f"      ⚠️ Error accediendo a PubMed API: {e}")
        
        return terms
    
    def fetch_from_wikipedia_medical(self, category: str, max_results: int = 20) -> List[Dict[str, str]]:
        """Obtiene términos de Wikipedia categorías médicas."""
        terms = []
        
        try:
            # API de Wikipedia para obtener páginas de una categoría
            url = "https://es.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'list': 'search',
                'srsearch': f'{category} medicina',
                'srlimit': max_results,
                'format': 'json',
                'srprop': 'snippet'
            }
            
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            for item in data.get('query', {}).get('search', []):
                title = item.get('title', '')
                snippet = item.get('snippet', '')
                
                # Limpiar HTML del snippet
                snippet_clean = re.sub(r'<.*?>', '', snippet)
                
                if title and len(title) > 2 and len(title) < 80:
                    terms.append({
                        'nombre': title,
                        'definicion': snippet_clean[:200] if snippet_clean else 'Término médico de Wikipedia',
                        'categoria': category,
                        'fuente': 'Wikipedia-ES'
                    })
            
            print(f"      ✅ Descargados {len(terms)} términos de Wikipedia")
            
        except Exception as e:
            print(f"      ⚠️ Error accediendo a Wikipedia API: {e}")
        
        return terms

class OllamaGenerator:
    """Manejador de interacción con Ollama."""
    
    def __init__(self, host: str = "http://localhost:11434", model: str = "qwen3:14b"):
        self.host = host
        self.model = model

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Genera respuesta usando Ollama."""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                }
            }
            response = requests.post(f"{self.host}/api/generate", json=payload, timeout=120)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            print(f"⚠️ Error conectando con Ollama: {e}")
            return ""

    def generate_json(self, prompt: str, system_prompt: str = "") -> Optional[Dict]:
        """Genera y parsea respuesta JSON."""
        response_text = self.generate(prompt, system_prompt + "\nResponde ÚNICAMENTE con un JSON válido, sin bloques de código ni markdown.")
        return self._clean_and_parse_json(response_text)

    def _clean_and_parse_json(self, text: str) -> Optional[Any]:
        """Limpia la respuesta para extraer JSON."""
        try:
            text = re.sub(r'```json\s*', '', text)
            text = re.sub(r'```\s*', '', text)
            match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
            if match:
                text = match.group(0)
            return json.loads(text)
        except json.JSONDecodeError:
            print(f"⚠️ Error parseando JSON de LLM. Texto recibido: {text[:100]}...")
            return None
    
    def generate_whisper_errors(self, term: str, context: str = "") -> List[str]:
        """Genera SOLO variantes incorrectas que Whisper produciría."""
        system_prompt = "Eres un experto en errores de transcripción de Whisper en español."
        prompt = f"""Para el término médico REAL "{term}" (contexto: {context}), genera 5-8 variantes incorrectas que Whisper ASR produciría al transcribir audio en español.

Tipos de errores comunes de Whisper:
1. Fragmentación: "hematocrito" → "hema tocrito"
2. Fonéticos: "McBurney" → "Macburney", "Mac Burney"
3. Similitud fonética: "índice tabáquico" → "instabáquico"
4. Homofonía: "vesícula" → "besícula"

Responde con un JSON array de strings:
["variante1", "variante2", "variante3", ...]

NO inventes el término original, solo las variantes incorrectas."""
        
        result = self.generate_json(prompt, system_prompt)
        if result and isinstance(result, list):
            return [v for v in result if isinstance(v, str) and len(v) > 0]
        return []


@dataclass
class GlosarioExpander:
    """Expande el glosario mediante fuentes verificadas + LLM para errores."""
    
    glosario_simple_path: str
    diccionario_base_path: str
    ollama: OllamaGenerator
    fetcher: MedicalTermFetcher
    verbose: bool = True
    
    glosario_simple: Dict = field(default_factory=dict)
    diccionario_base: Dict = field(default_factory=dict)
    
    stats: Dict[str, int] = field(default_factory=lambda: {
        'total_added': 0,
        'simple_added': 0,
        'base_added': 0,
        'terms_fetched': 0
    })
    
    def __post_init__(self):
        self._load_files()
    
    def _load_files(self):
        """Carga los archivos JSON existentes."""
        try:
            with open(self.glosario_simple_path, 'r', encoding='utf-8') as f:
                self.glosario_simple = json.load(f)
        except FileNotFoundError:
            self.glosario_simple = {"correcciones": {}, "mantener": []}

        try:
            with open(self.diccionario_base_path, 'r', encoding='utf-8') as f:
                self.diccionario_base = json.load(f)
        except FileNotFoundError:
            self.diccionario_base = {"version": "2.0.0", "metadata": {}, "terminos": {}}

    def _save_files(self):
        """Guarda los cambios en los archivos JSON."""
        with open(self.glosario_simple_path, 'w', encoding='utf-8') as f:
            self.glosario_simple['correcciones'] = dict(sorted(self.glosario_simple.get('correcciones', {}).items()))
            self.glosario_simple['mantener'] = sorted(list(set(self.glosario_simple.get('mantener', []))))
            json.dump(self.glosario_simple, f, indent=2, ensure_ascii=False)
            
        with open(self.diccionario_base_path, 'w', encoding='utf-8') as f:
            json.dump(self.diccionario_base, f, indent=4, ensure_ascii=False)
            
        if self.verbose:
            print(f"✓ Archivos guardados")

    def _add_to_simple_glosario(self, incorrect: str, correct: str):
        """Añade par incorrecto->correcto al glosario simple."""
        if incorrect.lower() != correct.lower():
            if incorrect not in self.glosario_simple.setdefault('correcciones', {}):
                self.glosario_simple['correcciones'][incorrect] = correct
                self.stats['simple_added'] += 1

    def _add_to_mantener(self, term: str):
        """Añade término a la lista de mantener."""
        if term not in self.glosario_simple.setdefault('mantener', []):
            self.glosario_simple['mantener'].append(term)
    
    def _add_to_diccionario_base(self, term_data: TerminoBase):
        """Añade entrada completa al diccionario base."""
        key = term_data.forma_correcta
        if key not in self.diccionario_base.setdefault('terminos', {}):
            self.diccionario_base['terminos'][key] = term_data.to_dict()
            self.stats['base_added'] += 1
            self.stats['total_added'] += 1
            
            for variante in term_data.variantes_incorrectas:
                self._add_to_simple_glosario(variante, key)
            
            self._add_to_mantener(key)

    def process_expansion(self, categories: List[str], count_per_category: int = 20, source: str = 'pubmed'):
        """Proceso principal: obtener términos verificados + generar variantes con LLM."""
        for cat_desc in categories:
            if self.verbose:
                print(f"\n🎯 Procesando: '{cat_desc}'")
            
            # Paso 1: Obtener términos REALES de fuente verificada
            if source == 'pubmed':
                raw_terms = self.fetcher.fetch_from_pubmed_mesh(cat_desc, max_results=count_per_category)
            elif source == 'wikipedia':
                raw_terms = self.fetcher.fetch_from_wikipedia_medical(cat_desc, max_results=count_per_category)
            else:
                # Híbrido: intentar ambos
                raw_terms = self.fetcher.fetch_from_pubmed_mesh(cat_desc, max_results=count_per_category//2)
                raw_terms += self.fetcher.fetch_from_wikipedia_medical(cat_desc, max_results=count_per_category//2)
            
            self.stats['terms_fetched'] += len(raw_terms)
            
            # Paso 2: Para cada término REAL, generar variantes incorrectas con LLM
            for raw_term in raw_terms:
                term_name = raw_term['nombre']
                
                if term_name in self.diccionario_base.get('terminos', {}):
                    if self.verbose:
                        print(f"      ⏭️  Ya existe: {term_name}")
                    continue
                
                if self.verbose:
                    print(f"      🔍 Procesando: {term_name}")
                
                # Generar variantes incorrectas con LLM
                variantes = self.ollama.generate_whisper_errors(term_name, raw_term.get('categoria', ''))
                
                # Crear término completo
                term_obj = TerminoBase(
                    tipo='termino_medico',
                    categoria=raw_term.get('categoria', 'medicina'),
                    definicion=raw_term.get('definicion', '')[:300],  # Limitar longitud
                    variantes_incorrectas=variantes,
                    forma_correcta=term_name,
                    ejemplos=[f"Término médico: {term_name}"],
                    contextos_relacionados=[cat_desc]
                )
                
                self._add_to_diccionario_base(term_obj)
                
                if self.verbose:
                    print(f"         ✅ Añadido con {len(variantes)} variantes")
                    if variantes:
                        print(f"            Ejemplos: {', '.join(variantes[:3])}")
                
                time.sleep(0.3)  # Rate limiting para Ollama
                        
            self._save_files()

def main():
    parser = argparse.ArgumentParser(description='Expandir glosario con fuentes verificadas (PubMed/Wikipedia) + LLM para errores')
    parser.add_argument('--glosario', default='./config/glosario_terminos.json')
    parser.add_argument('--base', default='./data/diccionario_base.json')
    parser.add_argument('--ollama-host', default='http://localhost:11434')
    parser.add_argument('--source', choices=['pubmed', 'wikipedia', 'hybrid'], default='hybrid',
                       help='Fuente de términos verificados')
    parser.add_argument('--medical', action='store_true', help='Expandir términos médicos')
    parser.add_argument('--regional', action='store_true', help='Expandir regionalismos')
    parser.add_argument('--procedures', action='store_true', help='Expandir procedimientos médicos')
    parser.add_argument('--custom', help='Categoría personalizada')
    parser.add_argument('--count', type=int, default=20, help='Términos por categoría')
    
    args = parser.parse_args()
    
    print(f"🚀 Expansión de glosario (Fuentes verificadas + LLM para variantes)")
    print(f"   Fuente: {args.source.upper()}")
    
    ollama = OllamaGenerator(host=args.ollama_host)
    fetcher = MedicalTermFetcher()
    expander = GlosarioExpander(args.glosario, args.base, ollama, fetcher)
    
    categories = []
    
    if args.medical:
        categories.extend([
            "cardiology",
            "neurology", 
            "gastroenterology",
            "nephrology"
        ])
    
    if args.regional:
        categories.extend([
            "mexican spanish medical terms",
            "latin american medical terminology"
        ])
        
    if args.procedures:
        categories.extend([
            "surgical procedures",
            "medical procedures"
        ])
        
    if args.custom:
        categories.append(args.custom)
    
    if not categories:
        print("⚠️ Selecciona al menos una categoría")
        return

    expander.process_expansion(categories, count_per_category=args.count, source=args.source)
    print("\n✅ Expansión finalizada.")
    print(f"📊 Estadísticas: {expander.stats}")

if __name__ == '__main__':
    main()
