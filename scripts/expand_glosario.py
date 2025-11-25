#!/usr/bin/env python3
"""
Herramienta para expandir el glosario de términos mediante búsquedas web.
Descubre marcas, términos técnicos, acrónimos y errores comunes de transcripción.
"""
import json
import re
import argparse
import time
from pathlib import Path
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field
import urllib.request
import urllib.parse
import ssl

# Crear contexto SSL que no verifica certificados (para simplicidad)
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE


@dataclass
class GlosarioExpander:
    """Expande el glosario mediante búsquedas web y listas predefinidas."""
    
    glosario_path: str
    output_path: Optional[str] = None
    verbose: bool = True
    
    # Datos del glosario
    correcciones: Dict[str, str] = field(default_factory=dict)
    mantener: List[str] = field(default_factory=list)
    
    # Estadísticas
    stats: Dict[str, int] = field(default_factory=lambda: {
        'marcas_added': 0,
        'acronimos_added': 0,
        'errores_added': 0,
        'total_added': 0
    })
    
    def __post_init__(self):
        self.output_path = self.output_path or self.glosario_path
        self._load_glosario()
    
    def _load_glosario(self):
        """Carga el glosario existente."""
        try:
            with open(self.glosario_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.correcciones = data.get('correcciones', {})
                self.mantener = data.get('mantener', [])
        except FileNotFoundError:
            print(f"⚠️  Glosario no encontrado, creando nuevo: {self.glosario_path}")
            self.correcciones = {}
            self.mantener = []
    
    def _save_glosario(self):
        """Guarda el glosario actualizado."""
        data = {
            'correcciones': dict(sorted(self.correcciones.items())),
            'mantener': sorted(list(set(self.mantener))),
            'comentario': 'Glosario expandido automáticamente. Contiene correcciones para errores comunes de Whisper en español.'
        }
        
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            print(f"✓ Glosario guardado en: {self.output_path}")
    
    def _add_correction(self, error: str, correct: str, category: str = 'general'):
        """Añade una corrección al glosario si no existe."""
        if error.lower() != correct.lower() and error not in self.correcciones:
            self.correcciones[error] = correct
            self.stats['total_added'] += 1
            self.stats[f'{category}_added'] = self.stats.get(f'{category}_added', 0) + 1
            return True
        return False
    
    def expand_tech_brands(self):
        """Añade marcas tecnológicas comunes."""
        if self.verbose:
            print("\n📱 Expandiendo marcas tecnológicas...")
        
        tech_brands = {
            # Grandes tecnológicas
            'google': 'Google', 'alphabet': 'Alphabet',
            'apple': 'Apple', 'iphone': 'iPhone', 'ipad': 'iPad', 'imac': 'iMac',
            'macbook': 'MacBook', 'airpods': 'AirPods', 'ios': 'iOS', 'macos': 'macOS',
            'microsoft': 'Microsoft', 'windows': 'Windows', 'xbox': 'Xbox',
            'linkedin': 'LinkedIn', 'github': 'GitHub', 'azure': 'Azure',
            'amazon': 'Amazon', 'aws': 'AWS', 'alexa': 'Alexa', 'kindle': 'Kindle',
            'meta': 'Meta', 'facebook': 'Facebook', 'instagram': 'Instagram',
            'whatsapp': 'WhatsApp', 'messenger': 'Messenger', 'oculus': 'Oculus',
            'twitter': 'Twitter', 'x': 'X',
            'netflix': 'Netflix', 'spotify': 'Spotify', 'disney': 'Disney',
            'tiktok': 'TikTok', 'bytedance': 'ByteDance',
            'snapchat': 'Snapchat', 'pinterest': 'Pinterest',
            'uber': 'Uber', 'airbnb': 'Airbnb', 'lyft': 'Lyft',
            'tesla': 'Tesla', 'spacex': 'SpaceX', 'starlink': 'Starlink',
            'nvidia': 'NVIDIA', 'amd': 'AMD', 'intel': 'Intel',
            'samsung': 'Samsung', 'huawei': 'Huawei', 'xiaomi': 'Xiaomi',
            'sony': 'Sony', 'playstation': 'PlayStation',
            'nintendo': 'Nintendo',
            
            # IA y ML
            'openai': 'OpenAI', 'chatgpt': 'ChatGPT', 'gpt': 'GPT',
            'gpt4': 'GPT-4', 'gpt-4': 'GPT-4', 'gpt3': 'GPT-3',
            'dall-e': 'DALL-E', 'dalle': 'DALL-E',
            'anthropic': 'Anthropic', 'claude': 'Claude',
            'gemini': 'Gemini', 'gemina': 'Gemini', 'bard': 'Bard',
            'copilot': 'Copilot', 'github copilot': 'GitHub Copilot',
            'midjourney': 'Midjourney', 'stable diffusion': 'Stable Diffusion',
            'huggingface': 'Hugging Face', 'hugging face': 'Hugging Face',
            'pytorch': 'PyTorch', 'tensorflow': 'TensorFlow',
            'langchain': 'LangChain', 'llama': 'LLaMA',
            'mistral': 'Mistral', 'perplexity': 'Perplexity',
            'deepmind': 'DeepMind', 'cohere': 'Cohere',
            
            # Herramientas y servicios
            'zoom': 'Zoom', 'slack': 'Slack', 'discord': 'Discord',
            'notion': 'Notion', 'trello': 'Trello', 'asana': 'Asana',
            'figma': 'Figma', 'canva': 'Canva', 'adobe': 'Adobe',
            'photoshop': 'Photoshop', 'illustrator': 'Illustrator',
            'dropbox': 'Dropbox', 'onedrive': 'OneDrive',
            'gmail': 'Gmail', 'outlook': 'Outlook',
            'shopify': 'Shopify', 'woocommerce': 'WooCommerce',
            'wordpress': 'WordPress', 'squarespace': 'Squarespace',
            'stripe': 'Stripe', 'paypal': 'PayPal', 'mercadopago': 'MercadoPago',
            'rappi': 'Rappi', 'didi': 'DiDi',
            
            # Criptomonedas
            'bitcoin': 'Bitcoin', 'ethereum': 'Ethereum', 'solana': 'Solana',
            'binance': 'Binance', 'coinbase': 'Coinbase',
            'nft': 'NFT', 'blockchain': 'blockchain',
            
            # Streaming y contenido
            'youtube': 'YouTube', 'twitch': 'Twitch', 'vimeo': 'Vimeo',
            'hbomax': 'HBO Max', 'hbo max': 'HBO Max',
            'amazon prime': 'Amazon Prime', 'prime video': 'Prime Video',
            'paramount': 'Paramount+', 'hulu': 'Hulu',
            'podcast': 'podcast', 'podcaster': 'podcaster',
        }
        
        added = 0
        for error, correct in tech_brands.items():
            if self._add_correction(error, correct, 'marcas'):
                added += 1
        
        if self.verbose:
            print(f"   ✓ Añadidas {added} marcas tecnológicas")
        
        return added
    
    def expand_acronyms(self):
        """Añade acrónimos técnicos y de negocios."""
        if self.verbose:
            print("\n🔤 Expandiendo acrónimos...")
        
        acronyms = {
            # Tecnología
            'ia': 'IA', 'i.a.': 'IA', 'ai': 'IA', 'a.i.': 'IA',
            'ml': 'ML', 'm.l.': 'ML',
            'api': 'API', 'apis': 'APIs',
            'url': 'URL', 'urls': 'URLs',
            'html': 'HTML', 'css': 'CSS', 'js': 'JavaScript',
            'sql': 'SQL', 'nosql': 'NoSQL',
            'saas': 'SaaS', 'paas': 'PaaS', 'iaas': 'IaaS',
            'cpu': 'CPU', 'gpu': 'GPU', 'ram': 'RAM',
            'ssd': 'SSD', 'hdd': 'HDD',
            'wifi': 'WiFi', 'wi-fi': 'WiFi',
            'usb': 'USB', 'hdmi': 'HDMI',
            'vpn': 'VPN', 'dns': 'DNS', 'ip': 'IP',
            'http': 'HTTP', 'https': 'HTTPS',
            'ssl': 'SSL', 'tls': 'TLS',
            'cdn': 'CDN', 'cms': 'CMS',
            'erp': 'ERP', 'crm': 'CRM',
            'ui': 'UI', 'ux': 'UX', 'ui/ux': 'UI/UX',
            'mvp': 'MVP',
            'qa': 'QA', 'devops': 'DevOps',
            'ci/cd': 'CI/CD', 'cicd': 'CI/CD',
            
            # Marketing y negocios
            'seo': 'SEO', 's.e.o.': 'SEO',
            'sem': 'SEM', 's.e.m.': 'SEM',
            'ppc': 'PPC', 'cpc': 'CPC', 'cpm': 'CPM',
            'ctr': 'CTR', 'cta': 'CTA',
            'roi': 'ROI', 'r.o.i.': 'ROI',
            'kpi': 'KPI', 'kpis': 'KPIs',
            'b2b': 'B2B', 'b2c': 'B2C', 'b2b2c': 'B2B2C',
            'd2c': 'D2C', 'dtc': 'DTC',
            'pr': 'PR', 'rr.pp.': 'RRPP',
            'faq': 'FAQ', 'faqs': 'FAQs',
            'tldr': 'TL;DR', 'tl;dr': 'TL;DR',
            'asap': 'ASAP',
            'ceo': 'CEO', 'cfo': 'CFO', 'cto': 'CTO', 'coo': 'COO',
            'cmo': 'CMO', 'cio': 'CIO',
            'hr': 'HR', 'rrhh': 'RRHH',
            'ebitda': 'EBITDA',
            'ipo': 'IPO',
            
            # Redes sociales
            'dm': 'DM', 'dms': 'DMs',
            'ig': 'IG', 'fb': 'FB', 'tw': 'TW',
            'yt': 'YT', 'tt': 'TT',
            'ugc': 'UGC',
            'reel': 'Reel', 'reels': 'Reels',
            'story': 'Story', 'stories': 'Stories',
            'live': 'live', 'en vivo': 'en vivo',
        }
        
        added = 0
        for error, correct in acronyms.items():
            if self._add_correction(error, correct, 'acronimos'):
                added += 1
        
        if self.verbose:
            print(f"   ✓ Añadidos {added} acrónimos")
        
        return added
    
    def expand_common_errors(self):
        """Añade errores ortográficos comunes en español."""
        if self.verbose:
            print("\n✏️ Expandiendo errores comunes...")
        
        common_errors = {
            # Acentos interrogativos/exclamativos
            'que es': 'qué es', 'que son': 'qué son',
            'como es': 'cómo es', 'como son': 'cómo son',
            'como se': 'cómo se', 'como hacer': 'cómo hacer',
            'por que': 'por qué', 'porque ': 'por qué ',
            'cuando es': 'cuándo es', 'cuando fue': 'cuándo fue',
            'donde esta': 'dónde está', 'donde estan': 'dónde están',
            'cual es': 'cuál es', 'cuales son': 'cuáles son',
            'quien es': 'quién es', 'quienes son': 'quiénes son',
            'cuanto es': 'cuánto es', 'cuantos son': 'cuántos son',
            
            # Errores de acentuación comunes
            'mas': 'más', 'tambien': 'también', 'asi': 'así',
            'despues': 'después', 'ademas': 'además',
            'aqui': 'aquí', 'ahi': 'ahí', 'alla': 'allá',
            'facil': 'fácil', 'dificil': 'difícil',
            'rapido': 'rápido', 'publico': 'público',
            'numero': 'número', 'telefono': 'teléfono',
            'pagina': 'página', 'articulo': 'artículo',
            'informacion': 'información', 'tecnologia': 'tecnología',
            'comunicacion': 'comunicación', 'educacion': 'educación',
            'economia': 'economía', 'politica': 'política',
            
            # Palabras juntas/separadas
            'atraves': 'a través', 'atravez': 'a través',
            'enserio': 'en serio', 'enseguida': 'en seguida',
            'osea': 'o sea', 'talvez': 'tal vez',
            'almenos': 'al menos', 'aparte': 'aparte',
            'aveces': 'a veces', 'acabo': 'a cabo',
            'afuera': 'afuera', 'adentro': 'adentro',
            'deveras': 'de veras', 'deacuerdo': 'de acuerdo',
            'esque': 'es que', 'aver': 'a ver',
            'haber si': 'a ver si', 'haver': 'a ver',
            'nose': 'no sé', 'nomas': 'nomás',
            'porfavor': 'por favor', 'porfa': 'porfa',
            'sobretodo': 'sobre todo', 'entretanto': 'entre tanto',
            'sinembargo': 'sin embargo',
            
            # Conjugaciones incorrectas
            'haiga': 'haya', 'haigan': 'hayan',
            'dijistes': 'dijiste', 'hicistes': 'hiciste',
            'fuistes': 'fuiste', 'vinistes': 'viniste',
            'vistes': 'viste', 'comistes': 'comiste',
            'trajistes': 'trajiste', 'pusistes': 'pusiste',
            'andé': 'anduve', 'andaste': 'anduviste',
            'satisfacido': 'satisfecho',
            'rompido': 'roto', 'imprimido': 'impreso',
            'freido': 'frito',
            
            # Confusiones comunes
            'ahi': 'ahí', 'hay': 'hay', 'ay': 'ay',
            'haya': 'haya', 'halla': 'halla', 'aya': 'aya',
            'echo': 'hecho', 'hecho': 'hecho',
            'asia': 'hacia', 'acia': 'hacia',
            'asta': 'hasta', 'asta luego': 'hasta luego',
            'valla': 'vaya', 'baya': 'vaya',
            'tuvo': 'tuvo', 'tubo': 'tubo',
            'callo': 'callo', 'cayó': 'cayó',
            'basto': 'vasto', 'vasto': 'vasto',
            
            # Anglicismos mal escritos
            'e-mail': 'email', 'e mail': 'email',
            'on-line': 'online', 'on line': 'online',
            'off-line': 'offline', 'off line': 'offline',
            'start-up': 'startup', 'start up': 'startup',
            'feed-back': 'feedback', 'feed back': 'feedback',
            'coach': 'coach', 'coaching': 'coaching',
            'branding': 'branding', 'briefing': 'briefing',
            'marketing': 'marketing', 'merchandising': 'merchandising',
            'target': 'target', 'engagement': 'engagement',
            'influencer': 'influencer', 'influencers': 'influencers',
            'community manager': 'community manager',
            'content creator': 'creador de contenido',
            'streamer': 'streamer', 'streamers': 'streamers',
        }
        
        added = 0
        for error, correct in common_errors.items():
            if self._add_correction(error, correct, 'errores'):
                added += 1
        
        if self.verbose:
            print(f"   ✓ Añadidos {added} errores comunes")
        
        return added
    
    def expand_mexican_slang(self):
        """Añade regionalismos mexicanos a la lista de mantener."""
        if self.verbose:
            print("\n🇲🇽 Expandiendo regionalismos mexicanos...")
        
        mexican_slang = [
            # Expresiones comunes
            'güey', 'wey', 'we', 'guey',
            'chido', 'chida', 'chidos', 'chidas',
            'padre', 'padres', 'padrísimo', 'padrísima',
            'neta', 'la neta', 'neta que sí', 'neta que no',
            'chamba', 'chambear', 'chambeando',
            'feria', 'lana', 'varo', 'varos', 'billete',
            'morro', 'morra', 'morros', 'morras',
            'carnal', 'carnala', 'carnales',
            'compa', 'compas', 'cuate', 'cuates', 'cuatachón',
            'bato', 'vato', 'batos', 'vatos',
            'raza', 'banda', 'la banda',
            
            # Interjecciones
            'órale', 'ándale', 'épale', 'újule', 'híjole',
            'chale', 'nel', 'nel pastel', 'simón', 'simon',
            'va', 'sale', 'sale y vale', 'órale pues',
            'fierro', 'fierro pariente', 'a huevo',
            'no manches', 'no mames', 'no inventes',
            'qué onda', 'qué pedo', 'qué pex', 'qué show',
            'qué rollo', 'qué tranza', 'qué pasó',
            
            # Adjetivos
            'chingón', 'chingona', 'chingones', 'chingonas',
            'gacho', 'gacha', 'bien gacho',
            'cañón', 'está cañón', 'bien cañón',
            'equis', 'equis equis', 'x',
            'naco', 'naca', 'fresa', 'fresas',
            'mamón', 'mamona', 'mamones',
            'cotorro', 'cotorra', 'cotorrear',
            
            # Verbos y acciones
            'chécalo', 'checa', 'checar',
            'aguas', 'aguas con', 'ponte aguas',
            'agarrar la onda', 'ya agarré la onda',
            'echar la hueva', 'huevear', 'hueva',
            'tirar la onda', 'tirando onda',
            'dar el rol', 'echar el rol', 'rolear',
            'ir de reven', 'reven', 'reventón',
            'pistear', 'echarse unas', 'chelas',
            'molestar', 'fregar', 'chingar',
            
            # Sustantivos
            'chela', 'chelas', 'cheve', 'cheves',
            'taco', 'tacos', 'taquear', 'taquería',
            'antro', 'antros', 'antrear',
            'peda', 'pedota', 'cruda',
            'jale', 'jales', 'jalar',
            'paro', 'hacerme un paro', 'me haces el paro',
            'bronca', 'broncas', 'bronco',
            'rollo', 'rollos', 'rollero',
            'cotorreo', 'de cotorreo',
            'desmadre', 'hacer desmadre',
        ]
        
        added = 0
        for term in mexican_slang:
            if term not in self.mantener:
                self.mantener.append(term)
                added += 1
        
        if self.verbose:
            print(f"   ✓ Añadidos {added} regionalismos")
        
        return added
    
    def expand_numbers_and_units(self):
        """Añade correcciones para números y unidades."""
        if self.verbose:
            print("\n🔢 Expandiendo números y unidades...")
        
        numbers_units = {
            # Unidades
            'km': 'kilómetros', 'kms': 'kilómetros',
            'kg': 'kilogramos', 'kgs': 'kilogramos',
            'gr': 'gramos', 'grs': 'gramos',
            'lt': 'litros', 'lts': 'litros',
            'mt': 'metros', 'mts': 'metros',
            'cm': 'centímetros', 'cms': 'centímetros',
            'mm': 'milímetros', 'mms': 'milímetros',
            'hr': 'hora', 'hrs': 'horas',
            'min': 'minutos', 'mins': 'minutos',
            'seg': 'segundos', 'segs': 'segundos',
            
            # Monedas
            'usd': 'dólares', 'USD': 'dólares',
            'mxn': 'pesos', 'MXN': 'pesos',
            'eur': 'euros', 'EUR': 'euros',
            
            # Porcentajes y matemáticas
            '%': ' por ciento',
            'x2': 'por dos', 'x3': 'por tres',
            '2x': 'doble', '3x': 'triple',
            '+': ' más ', '-': ' menos ',
            '=': ' igual a ',
            
            # Ordinaless
            '1ro': 'primero', '2do': 'segundo', '3ro': 'tercero',
            '4to': 'cuarto', '5to': 'quinto',
            '1er': 'primer', '1ra': 'primera',
        }
        
        added = 0
        for error, correct in numbers_units.items():
            if self._add_correction(error, correct, 'numeros'):
                added += 1
        
        if self.verbose:
            print(f"   ✓ Añadidas {added} unidades y números")
        
        return added
    
    def expand_from_web(self, query: str, category: str = 'general', debug: bool = False):
        """
        Busca términos en la web para expandir el glosario.
        
        Args:
            query: Búsqueda a realizar
            category: Categoría de los términos
            debug: Si True, muestra información de depuración
        """
        if self.verbose:
            print(f"\n🌐 Buscando: {query}...")
        
        results = {
            'success': False,
            'error': None,
            'terms_found': 0,
            'source': None
        }
        
        # Intentar múltiples fuentes (Wikipedia primero, es más confiable)
        sources = [
            ('Wikipedia ES', self._search_wikipedia_es),
            ('DuckDuckGo', self._search_duckduckgo),
        ]
        
        all_terms = []
        
        for source_name, search_func in sources:
            try:
                if debug and self.verbose:
                    print(f"   Intentando {source_name}...")
                
                terms = search_func(query, debug)
                
                if terms:
                    added = 0
                    for term in terms:
                        # Crear corrección: versión lowercase -> versión correcta
                        term_lower = term.lower()
                        
                        # Solo añadir si no es exactamente igual (tiene mayúsculas)
                        if term_lower != term:
                            if self._add_correction(term_lower, term, category):
                                added += 1
                                if debug and self.verbose:
                                    print(f"      + {term_lower} → {term}")
                        
                        # También añadir variantes sin acentos comunes
                        term_no_accent = self._remove_accents(term_lower)
                        if term_no_accent != term_lower:
                            if self._add_correction(term_no_accent, term, category):
                                added += 1
                    
                    if added > 0:
                        results['success'] = True
                        results['terms_found'] = added
                        results['source'] = source_name
                        
                        if self.verbose:
                            print(f"   ✓ [{source_name}] Encontrados {added} términos nuevos")
                        
                        return added
                    elif debug and self.verbose:
                        print(f"   ℹ️ [{source_name}] {len(terms)} resultados pero ninguno nuevo")
                    
            except Exception as e:
                results['error'] = str(e)
                if debug and self.verbose:
                    print(f"   ⚠️ [{source_name}] Error: {e}")
                    import traceback
                    traceback.print_exc()
                continue
        
        # Si ninguna fuente funcionó
        if self.verbose:
            if results['error']:
                print(f"   ⚠️ No se encontraron términos. Último error: {results['error']}")
            else:
                print(f"   ⚠️ No se encontraron términos relevantes para: {query}")
        
        return 0
    
    def _search_duckduckgo(self, query: str, debug: bool = False) -> List[str]:
        """Busca en DuckDuckGo HTML."""
        encoded_query = urllib.parse.quote(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'es-MX,es;q=0.9,en;q=0.8',
        }
        
        request = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(request, context=ssl_context, timeout=15) as response:
            html = response.read().decode('utf-8', errors='ignore')
        
        if debug:
            print(f"      Respuesta: {len(html)} bytes")
        
        # Extraer términos de los resultados
        terms = self._extract_terms_from_html(html, debug)
        return terms
    
    def _search_wikipedia_es(self, query: str, debug: bool = False) -> List[str]:
        """Busca en Wikipedia en español."""
        # Construir URL con parámetros correctamente codificados
        base_url = "https://es.wikipedia.org/w/api.php"
        params = {
            'action': 'opensearch',
            'search': query,
            'limit': '15',
            'format': 'json'
        }
        query_string = urllib.parse.urlencode(params)
        url = f"{base_url}?{query_string}"
        
        headers = {
            'User-Agent': 'GlosarioExpander/1.0 (fromPodtoCast project; contact@example.com)',
        }
        
        if debug:
            print(f"      URL: {url[:80]}...")
        
        request = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(request, context=ssl_context, timeout=15) as response:
            data = json.loads(response.read().decode('utf-8'))
        
        if debug:
            print(f"      Resultados Wikipedia: {len(data[1]) if len(data) > 1 else 0}")
            if len(data) > 1 and data[1]:
                print(f"      Muestra: {data[1][:5]}")
        
        # data[1] contiene los títulos de los artículos
        if len(data) > 1 and data[1]:
            valid_terms = []
            for title in data[1]:
                if self._is_valid_wikipedia_term(title):
                    valid_terms.append(title)
            return valid_terms
        
        return []
    
    def _extract_terms_from_html(self, html: str, debug: bool = False) -> List[str]:
        """Extrae términos relevantes del HTML."""
        terms = set()
        
        # Patrones para extraer términos
        patterns = [
            # Marcas y nombres propios (palabras capitalizadas)
            r'\b([A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]+)?)\b',
            # Términos en negrita o enlaces
            r'<b>([^<]+)</b>',
            r'<a[^>]*>([^<]+)</a>',
            # Títulos de resultados
            r'class="result__title"[^>]*>([^<]+)<',
            r'class="result__snippet"[^>]*>([^<]+)<',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, html)
            for match in matches:
                # Limpiar el término
                term = match.strip()
                term = re.sub(r'<[^>]+>', '', term)  # Remover HTML tags
                term = re.sub(r'\s+', ' ', term)  # Normalizar espacios
                
                # Filtrar términos válidos
                if self._is_valid_term(term):
                    terms.add(term)
        
        if debug:
            print(f"      Términos extraídos: {len(terms)}")
            if terms:
                sample = list(terms)[:5]
                print(f"      Muestra: {sample}")
        
        return list(terms)
    
    def _is_valid_term(self, term: str, from_wikipedia: bool = False) -> bool:
        """Verifica si un término es válido para el glosario."""
        if not term or len(term) < 3 or len(term) > 50:
            return False
        
        # Ignorar términos comunes en inglés/español que no son marcas
        ignore_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
            'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
            'los', 'las', 'del', 'que', 'por', 'con', 'una', 'son',
            'para', 'como', 'pero', 'esta', 'este', 'esto', 'puede',
            'Search', 'Results', 'Page', 'Home', 'About', 'Contact',
            'Privacy', 'Terms', 'Help', 'More', 'Next', 'Previous',
            'Code', 'DDG', 'This', 'Transitional', 'UTF', 'DOCTYPE',
            'HTML', 'Lite', 'Get',
        }
        
        if term.lower() in ignore_words or term in ignore_words:
            return False
        
        # Si viene de Wikipedia, es más confiable - aceptar más términos
        if from_wikipedia:
            # Solo ignorar si es un término de navegación
            nav_terms = {'desambiguación', 'anexo', 'categoría', 'wikipedia'}
            if any(nav in term.lower() for nav in nav_terms):
                return False
            return len(term) >= 3 and any(c.isalpha() for c in term)
        
        # Para otras fuentes, debe tener mayúscula
        if not any(c.isupper() for c in term):
            return False
        
        # Ignorar si es solo números o símbolos
        if not any(c.isalpha() for c in term):
            return False
        
        return True
    
    def _is_valid_wikipedia_term(self, term: str) -> bool:
        """Verifica si un término de Wikipedia es válido."""
        if not term or len(term) < 3 or len(term) > 60:
            return False
        
        # Ignorar páginas de desambiguación, anexos, etc.
        skip_patterns = [
            'desambiguación', 'anexo:', 'categoría:', 
            'wikipedia:', 'plantilla:', 'archivo:'
        ]
        
        term_lower = term.lower()
        for pattern in skip_patterns:
            if pattern in term_lower:
                return False
        
        return True
    
    def _remove_accents(self, text: str) -> str:
        """Remueve acentos de un texto."""
        replacements = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'ä': 'a', 'ë': 'e', 'ï': 'i', 'ö': 'o', 'ü': 'u',
            'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
            'ñ': 'n',
        }
        result = text
        for accented, plain in replacements.items():
            result = result.replace(accented, plain)
        return result
    
    def expand_all(self):
        """Ejecuta todas las expansiones disponibles."""
        print("=" * 60)
        print("🔄 Expandiendo glosario de términos")
        print("=" * 60)
        
        total = 0
        total += self.expand_tech_brands()
        total += self.expand_acronyms()
        total += self.expand_common_errors()
        total += self.expand_mexican_slang()
        total += self.expand_numbers_and_units()
        
        self._save_glosario()
        
        print("\n" + "=" * 60)
        print(f"✅ Expansión completada")
        print(f"   Total términos añadidos: {self.stats['total_added']}")
        print(f"   Correcciones totales: {len(self.correcciones)}")
        print(f"   Términos a mantener: {len(self.mantener)}")
        print("=" * 60)
        
        return self.stats
    
    def search_and_expand(self, queries: List[str], debug: bool = False):
        """
        Realiza búsquedas web y expande el glosario.
        
        Args:
            queries: Lista de búsquedas a realizar
            debug: Si True, muestra información de depuración
        """
        print("=" * 60)
        print("🌐 Expandiendo glosario mediante búsquedas web")
        print("=" * 60)
        
        if debug:
            print("   [DEBUG MODE ACTIVADO]")
        
        total = 0
        successful = 0
        failed = 0
        
        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}] ", end="")
            result = self.expand_from_web(query, debug=debug)
            
            if result > 0:
                total += result
                successful += 1
            else:
                failed += 1
            
            # Rate limiting entre búsquedas
            if i < len(queries):
                time.sleep(2)
        
        self._save_glosario()
        
        print("\n" + "=" * 60)
        print(f"✅ Búsquedas completadas")
        print(f"   Exitosas: {successful}/{len(queries)}")
        print(f"   Fallidas: {failed}/{len(queries)}")
        print(f"   Términos añadidos: {total}")
        print("=" * 60)
        
        return total


def main():
    parser = argparse.ArgumentParser(
        description='Expande el glosario de términos para el pre-procesador de texto.'
    )
    parser.add_argument(
        '--glosario', '-g',
        default='./config/glosario_terminos.json',
        help='Ruta al glosario (entrada y salida)'
    )
    parser.add_argument(
        '--output', '-o',
        help='Ruta de salida (opcional, por defecto sobrescribe el original)'
    )
    parser.add_argument(
        '--search', '-s',
        nargs='+',
        help='Búsquedas web a realizar'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Ejecutar todas las expansiones predefinidas'
    )
    parser.add_argument(
        '--brands',
        action='store_true',
        help='Expandir solo marcas tecnológicas'
    )
    parser.add_argument(
        '--acronyms',
        action='store_true',
        help='Expandir solo acrónimos'
    )
    parser.add_argument(
        '--errors',
        action='store_true',
        help='Expandir solo errores comunes'
    )
    parser.add_argument(
        '--slang',
        action='store_true',
        help='Expandir solo regionalismos mexicanos'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Modo silencioso'
    )
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Modo debug - muestra información detallada de búsquedas'
    )
    
    args = parser.parse_args()
    
    expander = GlosarioExpander(
        glosario_path=args.glosario,
        output_path=args.output,
        verbose=not args.quiet
    )
    
    if args.all or not any([args.brands, args.acronyms, args.errors, args.slang, args.search]):
        expander.expand_all()
    else:
        if args.brands:
            expander.expand_tech_brands()
        if args.acronyms:
            expander.expand_acronyms()
        if args.errors:
            expander.expand_common_errors()
        if args.slang:
            expander.expand_mexican_slang()
        if args.search:
            expander.search_and_expand(args.search, debug=args.debug)
        
        expander._save_glosario()
    
    return 0


if __name__ == '__main__':
    exit(main())

