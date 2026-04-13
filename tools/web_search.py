"""
Herramienta de búsqueda web para el sistema multi-agente.
Busca información en StackOverflow y GitHub para casos complejos.
"""
import re
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Resultado de una búsqueda web."""
    title: str
    url: str
    snippet: str
    source: str  # 'stackoverflow', 'github', 'other'
    relevance_score: float


class WebSearchTool:
    """
    Herramienta de búsqueda web con cache.
    Busca información técnica para ayudar en la generación de código.
    """

    # Callback para notificar visualización externa (ej: chainlit)
    _status_callback = None

    # Keywords que activan búsqueda automática
    ERROR_KEYWORDS = [
        'error', 'exception', 'traceback', 'no funciona', 'falla',
        'import error', 'module not found', 'attribute error',
        'key error', 'index error', 'value error', 'type error',
        'syntax error', 'runtime error', 'bug', 'fix',
        'not working', 'does not work', 'fails', 'broken'
    ]

    # Librerías comunes para las que buscar documentación
    POPULAR_LIBS = [
        'numpy', 'pandas', 'tensorflow', 'pytorch', 'sklearn',
        'django', 'flask', 'fastapi', 'requests', 'sqlalchemy',
        'matplotlib', 'seaborn', 'plotly', 'opencv', 'pillow',
        'asyncio', 'threading', 'multiprocessing', 'socket',
        'json', 'csv', 'xml', 'yaml', 'toml',
        'os', 'sys', 'pathlib', 'subprocess', 'logging'
    ]

    def __init__(self, cache_ttl: int = 3600):
        """
        Inicializa la herramienta de búsqueda.

        Args:
            cache_ttl: Tiempo de vida del cache en segundos (default: 1 hora)
        """
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._last_search_time: Dict[str, float] = {}
        self._search_available = False

        # Intentar importar la herramienta de búsqueda si está disponible
        try:
            # La herramienta search_web está disponible como tool del sistema
            # pero no como módulo importable directamente
            self._search_available = True
        except Exception:
            self._search_available = False

    @classmethod
    def set_status_callback(cls, callback):
        """Establece callback para notificar estado de búsqueda."""
        cls._status_callback = callback

    def _notify_status(self, stage: str, message: str, data: Dict = None):
        """Notifica estado de búsqueda para visualización."""
        # Terminal
        prefix = "🔍" if stage == "start" else "✅" if stage == "success" else "❌" if stage == "error" else "📊"
        print(f"[WebSearch] {prefix} {message}")

        # Chainlit (via callback)
        if self._status_callback:
            try:
                self._status_callback(stage, message, data or {})
            except Exception as e:
                print(f"[WebSearch] Error en callback: {e}")

    def should_search(self, prompt: str, code_analysis: str = None) -> bool:
        """
        Determina si se debe activar la búsqueda web.

        Args:
            prompt: Solicitud del usuario
            code_analysis: Análisis previo del código (opcional)

        Returns:
            True si se debe buscar, False en caso contrario
        """
        prompt_lower = prompt.lower()

        # Detectar keywords de error
        has_error_keywords = any(kw in prompt_lower for kw in self.ERROR_KEYWORDS)

        # Detectar librerías populares
        has_library = any(lib in prompt_lower for lib in self.POPULAR_LIBS)

        should = has_error_keywords or has_library

        if should:
            self._notify_status("detected", f"Búsqueda web activada para: {prompt[:50]}...")

        return should

    def search(
        self,
        query: str,
        domain: Optional[str] = None,
        max_results: int = 5
    ) -> List[SearchResult]:
        """
        Realiza una búsqueda web con cache.
        NOTA: Requiere implementación con herramienta de búsqueda real.

        Args:
            query: Query de búsqueda
            domain: Dominio preferido (stackoverflow, github)
            max_results: Máximo de resultados

        Returns:
            Lista de SearchResult (vacía por ahora - placeholder)
        """
        domain_str = f" [{domain}]" if domain else ""
        self._notify_status("start", f"Iniciando búsqueda{domain_str}: {query[:60]}...")

        # Verificar cache
        cache_key = f"{query}:{domain}:{max_results}"
        cached = self._get_from_cache(cache_key)
        if cached:
            self._notify_status("cache", f"Usando cache ({len(cached)} resultados)", {"count": len(cached)})
            return cached

        # Placeholder - la búsqueda real requiere integración con herramienta externa
        self._notify_status("searching", f"Buscando... (placeholder - implementación pendiente)")

        # Simular resultados de ejemplo para demostración
        # En implementación real, aquí se haría la búsqueda
        results = []

        self._notify_status("success", f"Búsqueda completada: {len(results)} resultados", {"count": len(results)})

        # Guardar en cache
        if results:
            self._save_to_cache(cache_key, results)

        return results

    def search_for_error(
        self,
        error_message: str,
        context: str = None,
        max_results: int = 3
    ) -> List[SearchResult]:
        """
        Busca soluciones para un error específico.

        Args:
            error_message: Mensaje de error
            context: Contexto adicional (ej: librería usada)
            max_results: Máximo de resultados

        Returns:
            Lista de SearchResult
        """
        self._notify_status("start", f"Buscando solución para error: {error_message[:50]}...")

        query_parts = [error_message]
        if context:
            query_parts.append(context)
        query_parts.append('solution fix python')

        query = ' '.join(query_parts)
        return self.search(query, domain='stackoverflow', max_results=max_results)

    def search_for_library_usage(
        self,
        library: str,
        task: str,
        max_results: int = 3
    ) -> List[SearchResult]:
        """
        Busca ejemplos de uso de una librería.

        Args:
            library: Nombre de la librería
            task: Tarea a realizar
            max_results: Máximo de resultados

        Returns:
            Lista de SearchResult
        """
        self._notify_status("start", f"Buscando ejemplos de {library} para: {task[:40]}...")

        query = f"{library} {task} python example"
        return self.search(query, max_results=max_results)

    def format_for_prompt(self, results: List[SearchResult], max_chars: int = 2000) -> str:
        """
        Formatea los resultados para incluir en un prompt.

        Args:
            results: Lista de resultados
            max_chars: Máximo de caracteres

        Returns:
            String formateado
        """
        if not results:
            return ""

        parts = ["\n--- INFORMACIÓN DE BÚSQUEDA WEB ---\n"]

        total_chars = len(parts[0])
        for result in results:
            entry = f"[{result.source.upper()}] {result.title}\n"
            entry += f"URL: {result.url}\n"
            entry += f"{result.snippet[:300]}...\n\n"

            if total_chars + len(entry) > max_chars:
                break

            parts.append(entry)
            total_chars += len(entry)

        parts.append("--- FIN DE BÚSQUEDA ---\n")

        return ''.join(parts)

    def _get_from_cache(self, key: str) -> Optional[List[SearchResult]]:
        """Obtiene resultado del cache si es válido."""
        if key not in self._cache:
            return None

        cached_time = self._last_search_time.get(key, 0)
        if time.time() - cached_time > self.cache_ttl:
            del self._cache[key]
            del self._last_search_time[key]
            return None

        return self._cache[key]

    def _save_to_cache(self, key: str, results: List[SearchResult]):
        """Guarda resultado en cache."""
        self._cache[key] = results
        self._last_search_time[key] = time.time()


# Instancia global para uso del sistema
web_search = WebSearchTool()
