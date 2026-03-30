"""
Agente Vision - analiza imágenes y extrae información relevante.
Usa modelo de visión cuantizado (Q3) y memoria caché.
"""
import base64
import os
from typing import Dict, Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from agents.memory import VisionMemory


class VisionAgent:
    """
    Agente especializado en análisis de imágenes/screenshots.
    Usa memoria caché para evitar re-procesar imágenes similares.
    """

    def __init__(self, model_name: str = "qwen3-vl:8b", memory_dir: str = None):
        """
        Inicializa el agente de visión.

        Args:
            model_name: Modelo Ollama vision-capable (preferiblemente Q3 cuantizado)
            memory_dir: Directorio para la memoria caché
        """
        self.model_name = model_name
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.2,  # Consistente con Modelfile (precisión visual)
            num_ctx=8192  # Contexto moderado para visión
        )
        self.memory = VisionMemory(memory_dir)

        self.system_prompt = """Eres un agente de visión especializado en analizar pantallas, interfaces y código visible en imágenes.

TU MISIÓN:
1. Extraer TEXTO relevante visible en la imagen (errores, logs, código, UI)
2. Identificar ELEMENTOS visuales importantes (ventanas, botones, gráficos)
3. Describir el CONTEXTO (¿qué aplicación? ¿qué está pasando?)
4. Detectar ERRORES o problemas visibles

FORMATO DE SALIDA:
Devuelve tu análisis en este formato estructurado:

**Texto extraído:**
- Lista de textos importantes encontrados

**Elementos visuales:**
- Descripción de UI, ventanas, elementos clave

**Análisis de código (si aplica):**
- Si hay código visible, resume qué muestra
- Identifica errores de sintaxis o lógica visibles

**Contexto:**
- ¿Qué aplicación/software se ve?
- ¿Qué acción está realizando el usuario?
- ¿Hay algún error o problema evidente?

REGLAS:
- Sé CONCISO pero completo
- Prioriza información útil para debugging o desarrollo
- Si hay errores visibles, descríbelos detalladamente"""

    def _compute_image_hash(self, image_b64: str) -> str:
        """Computa un hash simple de la imagen para caché."""
        return self.memory.compute_hash(image_b64[:1000])  # Usar primeros 1000 chars para velocidad

    def _load_image_from_file(self, image_path: str) -> str:
        """Carga una imagen desde archivo y la codifica en base64."""
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        return base64.b64encode(image_bytes).decode('utf-8')

    def analyze_image(self, image_b64: str = None, image_path: str = None, context_prompt: str = None) -> Dict[str, Any]:
        """
        Analiza una imagen y extrae información relevante.

        Args:
            image_b64: Imagen en base64 (opcional)
            image_path: Ruta a imagen en disco (opcional)
            context_prompt: Contexto adicional sobre qué buscar

        Returns:
            Dict con el análisis y metadatos
        """
        # Si se proporciona image_path, cargar la imagen
        if image_path and not image_b64:
            if os.path.exists(image_path):
                print(f"[VisionAgent] Cargando imagen desde: {image_path}")
                image_b64 = self._load_image_from_file(image_path)
            else:
                print(f"[VisionAgent] Archivo no encontrado: {image_path}")
                return {
                    "analysis": f"Error: Archivo de imagen no encontrado: {image_path}",
                    "from_cache": False,
                    "image_hash": "",
                    "tokens_used": 0,
                    "error": "File not found"
                }

        if not image_b64:
            return {
                "analysis": "Error: No se proporcionó imagen para analizar",
                "from_cache": False,
                "image_hash": "",
                "tokens_used": 0,
                "error": "No image provided"
            }

        # Verificar caché
        image_hash = self._compute_image_hash(image_b64)
        cached = self.memory.get_cached_analysis(image_hash)

        if cached:
            print(f"[VisionAgent] Usando análisis en caché (hash: {image_hash[:8]}...)")
            return {
                "analysis": cached["analysis"],
                "from_cache": True,
                "image_hash": image_hash,
                "tokens_used": 0  # No usamos tokens si viene de caché
            }

        # Preparar prompt específico si hay contexto
        user_content = context_prompt if context_prompt else "Analiza esta imagen y extrae toda la información relevante."

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(
                content=[
                    {"type": "text", "text": user_content},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                    }
                ]
            )
        ]

        try:
            print(f"[VisionAgent] Analizando imagen con {self.model_name}...")
            response = self.llm.invoke(messages)
            analysis = response.content

            # Guardar en caché
            self.memory.cache_analysis(image_hash, analysis, user_content)

            # Estimar tokens (imagen ~750 tokens por 224x224)
            est_image_tokens = int(len(image_b64) * 0.75)
            est_output_tokens = len(analysis.split())

            return {
                "analysis": analysis,
                "from_cache": False,
                "image_hash": image_hash,
                "tokens_used": {
                    "image": est_image_tokens,
                    "output": est_output_tokens
                }
            }
        except Exception as e:
            print(f"[VisionAgent] Error analizando imagen: {e}")
            return {
                "analysis": f"Error al analizar imagen: {e}",
                "from_cache": False,
                "image_hash": image_hash,
                "tokens_used": 0,
                "error": str(e)
            }

    def analyze_code_screenshot(self, image_b64: str, filename_hint: str = None) -> Dict[str, Any]:
        """
        Analiza específicamente un screenshot de código.

        Args:
            image_b64: Imagen en base64
            filename_hint: Nombre de archivo sospechado (opcional)

        Returns:
            Dict con análisis enfocado en código
        """
        context = "Esta imagen muestra código en una pantalla. "
        if filename_hint:
            context += f"Parece ser del archivo {filename_hint}. "
        context += "Extrae TODO el código visible, identifica errores, y describe la estructura."

        return self.analyze_image(image_b64, context)

    def analyze_error_screen(self, image_b64: str) -> Dict[str, Any]:
        """
        Analiza específicamente una pantalla de error.

        Args:
            image_b64: Imagen en base64

        Returns:
            Dict con análisis enfocado en errores
        """
        context = "Esta imagen muestra un error en pantalla. Extrae el mensaje de error completo, el stack trace si es visible, y describe el contexto de la aplicación."

        return self.analyze_image(image_b64, context)

    def clear_cache(self) -> None:
        """Limpia el caché de imágenes."""
        self.memory.clear()
        print("[VisionAgent] Caché de imágenes limpiado")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del caché."""
        cache = self.memory.get("image_cache", {})
        return {
            "cached_images": len(cache),
            "memory_file": str(self.memory.memory_file)
        }
