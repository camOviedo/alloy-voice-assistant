"""
Sistema de memoria JSON persistente para agentes.
"""
import json
import hashlib
import os
import time
from pathlib import Path
from typing import Any, Optional

from config import SUGGESTIONS_PATH


class JSONMemory:
    """Memoria persistente basada en archivos JSON para agentes."""

    def __init__(self, agent_name: str, memory_dir: str = None):
        """
        Inicializa la memoria para un agente específico.

        Args:
            agent_name: Nombre del agente (ej: 'vision', 'editor')
            memory_dir: Directorio donde guardar los archivos de memoria
        """
        self.agent_name = agent_name
        self.memory_dir = Path(memory_dir) if memory_dir else Path(SUGGESTIONS_PATH)
        self.memory_file = self.memory_dir / f"{agent_name}_memory.json"
        self._data = {}
        self._load()

    def _load(self) -> None:
        """Carga los datos de memoria desde archivo JSON."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    self._data = json.load(f)
            except Exception as e:
                print(f"[Memory] Error cargando memoria de {self.agent_name}: {e}")
                self._data = {}
        else:
            self._data = {}

    def _save(self) -> None:
        """Guarda los datos de memoria a archivo JSON."""
        try:
            self.memory_dir.mkdir(parents=True, exist_ok=True)
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Memory] Error guardando memoria de {self.agent_name}: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Obtiene un valor de memoria."""
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Guarda un valor en memoria."""
        self._data[key] = value
        self._save()

    def delete(self, key: str) -> bool:
        """Elimina una clave de memoria."""
        if key in self._data:
            del self._data[key]
            self._save()
            return True
        return False

    def get_all(self) -> dict:
        """Obtiene todo el contenido de la memoria."""
        return self._data.copy()

    def clear(self) -> None:
        """Limpia toda la memoria del agente."""
        self._data = {}
        self._save()

    @staticmethod
    def compute_hash(data: str) -> str:
        """Computa un hash simple para identificar datos."""
        return hashlib.md5(data.encode('utf-8')).hexdigest()[:16]


class VisionMemory(JSONMemory):
    """Memoria especializada para el agente de visión.
    Almacena caché de análisis de imágenes."""

    def __init__(self, memory_dir: str = None):
        super().__init__("vision", memory_dir)

    def get_cached_analysis(self, image_hash: str) -> Optional[dict]:
        """
        Busca si existe un análisis previo para una imagen similar.

        Args:
            image_hash: Hash de la imagen (MD5 del base64)

        Returns:
            Dict con el análisis previo o None si no existe
        """
        cache = self.get("image_cache", {})
        if image_hash in cache:
            entry = cache[image_hash]
            # Verificar si el caché es reciente (< 24 horas)
            age = time.time() - entry.get("timestamp", 0)
            if age < 86400:  # 24 horas
                return entry
            else:
                # Eliminar entrada expirada
                del cache[image_hash]
                self.set("image_cache", cache)
        return None

    def cache_analysis(self, image_hash: str, analysis: str, prompt: str) -> None:
        """
        Guarda el análisis de una imagen en caché.

        Args:
            image_hash: Hash de la imagen
            analysis: Texto del análisis
            prompt: Prompt que se usó para analizar
        """
        cache = self.get("image_cache", {})
        cache[image_hash] = {
            "analysis": analysis,
            "prompt": prompt,
            "timestamp": time.time()
        }
        # Limitar caché a 50 entradas más recientes
        if len(cache) > 50:
            sorted_items = sorted(cache.items(), key=lambda x: x[1].get("timestamp", 0), reverse=True)
            cache = dict(sorted_items[:50])
        self.set("image_cache", cache)


class EditorMemory(JSONMemory):
    """Memoria especializada para el agente editor.
    Mantiene historial de modificaciones recientes."""

    def __init__(self, memory_dir: str = None):
        super().__init__("editor", memory_dir)

    def add_modification(self, filename: str, original_code: str, new_code: str, description: str) -> str:
        """
        Registra una modificación en el historial.

        Args:
            filename: Nombre del archivo modificado
            original_code: Código original
            new_code: Código nuevo
            description: Descripción de los cambios

        Returns:
            ID de la modificación
        """
        history = self.get("modification_history", [])
        mod_id = f"{filename}_{int(time.time())}"

        mod_entry = {
            "id": mod_id,
            "file": filename,
            "timestamp": time.time(),
            "description": description,
            "original_hash": self.compute_hash(original_code),
            "new_hash": self.compute_hash(new_code),
            "lines_changed": len(new_code.splitlines()) - len(original_code.splitlines())
        }

        history.append(mod_entry)
        # Mantener solo últimas 20 modificaciones
        if len(history) > 20:
            history = history[-20:]

        self.set("modification_history", history)
        return mod_id

    def get_recent_modifications(self, filename: str = None, count: int = 5) -> list:
        """
        Obtiene modificaciones recientes.

        Args:
            filename: Filtrar por archivo específico (opcional)
            count: Número de modificaciones a retornar

        Returns:
            Lista de modificaciones recientes
        """
        history = self.get("modification_history", [])
        if filename:
            history = [h for h in history if h.get("file") == filename]
        return history[-count:]

    def get_context_for_file(self, filename: str) -> str:
        """
        Genera contexto de modificaciones previas para un archivo.

        Args:
            filename: Nombre del archivo

        Returns:
            String con contexto de modificaciones previas
        """
        recent = self.get_recent_modifications(filename, count=3)
        if not recent:
            return ""

        context_parts = ["\n--- Modificaciones recientes en este archivo ---"]
        for mod in recent:
            lines_info = f"({mod['lines_changed']:+.0f} líneas)" if mod.get('lines_changed') else ""
            context_parts.append(
                f"- [{mod['id']}] {mod['description'][:60]}... {lines_info}"
            )
        context_parts.append("--- Fin del historial ---\n")

        return "\n".join(context_parts)
