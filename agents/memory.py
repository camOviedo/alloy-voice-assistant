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


class ProjectMemory(JSONMemory):
    """Memoria de análisis de proyecto para el CodeAgent.
    Almacena análisis de archivos individuales y resumen del proyecto.
    Se invalida automáticamente si el archivo cambia (checksum)."""

    def __init__(self, memory_dir: str = None):
        super().__init__("project", memory_dir)

    def get_file_analysis(self, filepath: str, content: str = None) -> dict:
        """
        Obtiene análisis de archivo si existe y es válido.

        Args:
            filepath: Ruta del archivo
            content: Contenido actual del archivo (para validar checksum)

        Returns:
            Dict con el análisis o None si no existe o está desactualizado
        """
        analyses = self.get("file_analyses", {})
        if filepath not in analyses:
            return None

        entry = analyses[filepath]

        # Si se proporciona contenido, validar checksum
        if content is not None:
            current_hash = self.compute_hash(content)
            if entry.get("checksum") != current_hash:
                print(f"[ProjectMemory] Análisis desactualizado para {filepath}")
                return None

        return entry

    def save_file_analysis(self, filepath: str, content: str, analysis: dict) -> None:
        """
        Guarda análisis de un archivo con su checksum.

        Args:
            filepath: Ruta del archivo
            content: Contenido actual del archivo
            analysis: Dict con el análisis del CodeAgent
        """
        analyses = self.get("file_analyses", {})

        analyses[filepath] = {
            "analysis": analysis.get("analysis", ""),
            "summary": analysis.get("summary", ""),
            "files_affected": analysis.get("files_affected", [filepath]),
            "lines_to_modify": analysis.get("lines_to_modify", []),
            "approach": analysis.get("approach", ""),
            "considerations": analysis.get("considerations", []),
            "checksum": self.compute_hash(content),
            "timestamp": time.time(),
            "line_count": len(content.splitlines())
        }

        self.set("file_analyses", analyses)
        print(f"[ProjectMemory] Análisis guardado para {filepath}")

    def get_project_summary(self) -> str:
        """Obtiene resumen general del proyecto si existe."""
        return self.get("project_summary", "")

    def save_project_summary(self, summary: str, files_analyzed: list) -> None:
        """
        Guarda resumen general del proyecto.

        Args:
            summary: Texto del resumen
            files_analyzed: Lista de archivos incluidos en el resumen
        """
        self.set("project_summary", summary)
        self.set("summary_files", files_analyzed)
        self.set("summary_timestamp", time.time())

    def get_cached_files_count(self) -> int:
        """Retorna cuántos archivos tienen análisis en caché."""
        return len(self.get("file_analyses", {}))

    def clear_outdated_analyses(self, project_files: list) -> int:
        """
        Limpia análisis de archivos que ya no existen en el proyecto.

        Args:
            project_files: Lista de archivos actualmente en el proyecto

        Returns:
            Número de análisis eliminados
        """
        analyses = self.get("file_analyses", {})
        to_remove = [f for f in analyses if f not in project_files]

        for f in to_remove:
            del analyses[f]

        if to_remove:
            self.set("file_analyses", analyses)
            print(f"[ProjectMemory] Limpiados {len(to_remove)} análisis obsoletos")

        return len(to_remove)

    def get_all_cached_analyses(self) -> dict:
        """
        Obtiene todos los análisis cacheados.

        Returns:
            Dict {filepath: analysis_data}
        """
        return self.get("file_analyses", {})

    def build_context_from_cache(self, target_file: str = None) -> str:
        """
        Construye contexto de proyecto desde caché.

        Args:
            target_file: Archivo específico a modificar (opcional)

        Returns:
            String con contexto del proyecto
        """
        analyses = self.get("file_analyses", {})
        if not analyses:
            return ""

        context_parts = ["\n=== CONTEXTO DEL PROYECTO (desde memoria) ==="]

        # Incluir resumen si existe
        summary = self.get_project_summary()
        if summary:
            context_parts.append(f"\n**Resumen del proyecto:**\n{summary[:1000]}...")

        # Incluir análisis del archivo objetivo primero
        if target_file and target_file in analyses:
            entry = analyses[target_file]
            context_parts.append(f"\n**Análisis previo de {target_file}:**")
            context_parts.append(f"- Resumen: {entry.get('summary', 'N/A')}")
            context_parts.append(f"- Enfoque: {entry.get('approach', 'N/A')}")
            if entry.get('lines_to_modify'):
                context_parts.append(f"- Líneas relevantes: {entry['lines_to_modify']}")

        # Incluir conteo de archivos analizados
        context_parts.append(f"\n*Total de archivos analizados en caché: {len(analyses)}*")
        context_parts.append("=== FIN DEL CONTEXTO ===\n")

        return "\n".join(context_parts)
