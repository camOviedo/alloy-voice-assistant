"""
Agente Coordinador - enrutador de flujo de trabajo.
Usa un modelo pequeño y rápido para decidir qué agentes ejecutar.
"""
import json
from typing import Dict, Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama


class CoordinatorAgent:
    """
    Agente que analiza el prompt del usuario y decide el flujo de ejecución.
    No requiere memoria - es stateless.
    """

    def __init__(self, model_name: str = "qwen2.5:3b"):
        """
        Inicializa el agente coordinador.

        Args:
            model_name: Modelo Ollama a usar (pequeño y rápido)
        """
        self.model_name = model_name
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.1,  # Baja temperatura para decisiones consistentes
            num_ctx=4096,  # Contexto suficiente para coordinación
            format="json"
        )

        self.system_prompt = """Eres un coordinador de flujo de trabajo que analiza solicitudes de usuarios y decide qué agentes especializados deben ejecutarse.

TU TAREA: Analiza la solicitud del usuario y devuelve un JSON con la decisión de enrutamiento.

DECISIONES POSIBLES:
1. "vision": true/false - ¿Necesita analizar imágenes/screenshots?
2. "code_analysis": true/false - ¿Necesita analizar código existente?
3. "code_generation": true/false - ¿Necesita generar/modificar código?
4. "direct_response": true/false - ¿Es una pregunta simple que no requiere agentes?

REGLAS DETALLADAS:
- "vision": true SOLO si el prompt menciona explícitamente imágenes, pantalla, screenshot, "qué ves", "muestra", etc.
- "code_analysis": true si menciona archivos, código, modificar, cambiar, actualizar, fix, bug, error, importaciones
- "code_generation": true SIEMPRE que el usuario pida: corregir, modificar, cambiar, actualizar, implementar, crear, añadir, eliminar, arreglar, reparar, solucionar CÓDIGO
- "direct_response": true para preguntas simples, saludos, o consultas generales SIN código

REGLA CRÍTICA:
Si el usuario dice "corrigelo", "modificalo", "cambialo", "arreglalo", "solucionalo" o similares en referencia a código:
- "code_analysis": true
- "code_generation": true (OBLIGATORIO - el editor debe ejecutarse para hacer los cambios)

Ejemplo 1 - Solicitud de corrección (DEBE generar código):
Usuario: "hay un error en tracker.py, corrige las importaciones"
Salida: {"vision": false, "code_analysis": true, "code_generation": true, "direct_response": false, "reasoning": "El usuario reporta un error y pide corrección, requiere análisis y generación de código modificado"}

Ejemplo 2 - Solo análisis (NO generar código):
Usuario: "explica qué hace este archivo"
Salida: {"vision": false, "code_analysis": true, "code_generation": false, "direct_response": false, "reasoning": "El usuario solo pide explicación, no modificación"}

IMPORTANTE:
- Devuelve SIEMPRE un JSON válido
- NO agregues texto explicativo fuera del JSON
- Usa true/false (minúsculas, sin comillas)
- Cuando el usuario pida CORREGIR/ARREGLAR/MODIFICAR código, code_generation DEBE ser true"""

    def analyze_request(self, prompt: str, has_image: bool = False) -> Dict[str, Any]:
        """
        Analiza la solicitud del usuario y decide el flujo.

        Args:
            prompt: Texto del usuario
            has_image: Si hay imagen disponible

        Returns:
            Dict con la decisión de enrutamiento
        """
        # Construir el mensaje del usuario
        user_content = f"Solicitud del usuario: {prompt}\n\n"
        if has_image:
            user_content += "Hay una imagen/screenshot adjunta disponible para análisis.\n"
        else:
            user_content += "NO hay imagen disponible - el usuario solo proporcionó texto.\n"

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_content)
        ]

        try:
            response = self.llm.invoke(messages)
            # Parsear la respuesta JSON
            try:
                result = json.loads(response.content)
                # Validar campos requeridos
                return {
                    "vision": result.get("vision", False) and has_image,
                    "code_analysis": result.get("code_analysis", False),
                    "code_generation": result.get("code_generation", False),
                    "direct_response": result.get("direct_response", False),
                    "reasoning": result.get("reasoning", "")
                }
            except json.JSONDecodeError:
                print(f"[Coordinator] Error parseando JSON: {response.content}")
                # Fallback a heurística simple
                return self._fallback_analysis(prompt, has_image)
        except Exception as e:
            print(f"[Coordinator] Error en LLM: {e}")
            return self._fallback_analysis(prompt, has_image)

    def _fallback_analysis(self, prompt: str, has_image: bool) -> Dict[str, Any]:
        """
        Análisis de fallback basado en heurísticas simples cuando el LLM falla.
        """
        prompt_lower = prompt.lower()

        # Keywords para visión - EXPANDIDOS para capturar más casos
        vision_keywords = [
            "pantalla", "imagen", "imágenes", "imagenes", "screenshot", "captura", 
            "ventana", "foto", "fotos", "video", "ver", "mirar", "mira",
            "qué ves", "que ves", "muestra", "muéstrame", "muestrame", "muéstrame",
            "error en", "analyze image", "what do you see", "screen", "look at",
            "detectar", "números", "numeros", "puerta", "salida"
        ]
        needs_vision = has_image and any(kw in prompt_lower for kw in vision_keywords)

        # Si hay imágenes disponibles y se menciona cualquier cosa relacionada con 
        # detectar, ver, analizar visualmente - forzar visión
        if has_image and any(kw in prompt_lower for kw in ["detectar", "números", "numeros", "puerta", "salida"]):
            needs_vision = True

        # Keywords para código
        code_keywords = [
            "modifica", "modificar", "cambia", "cambiar", "actualiza", "actualizar",
            "fix", "bug", "error", "archivo", "file", "código", "code",
            "implementa", "implementar", "crea", "crear"
        ]
        needs_code = any(kw in prompt_lower for kw in code_keywords)

        # Keywords específicas para generación/corrección de código
        generation_keywords = [
            "corrige", "corrigelo", "corregir", "soluciona", "solucionalo",
            "arregla", "arreglalo", "repara", "reparar", "haz", "genera",
            "escribe", "escribir", "añade", "agrega", "quita", "elimina",
            "borra", "modifica esto", "cambia esto"
        ]
        needs_generation = any(kw in prompt_lower for kw in generation_keywords)

        # Si pide corregir errores, forzar ambos análisis y generación
        if needs_generation:
            needs_code = True

        # Detectar si es una pregunta simple/directa
        simple_keywords = [
            "hola", "hello", "buenos días", "buenas", "qué tal", "cómo estás",
            "gracias", "adiós", "bye", "qué es", "cómo", "dime"
        ]
        is_simple = any(kw in prompt_lower for kw in simple_keywords) and not needs_code and not needs_generation

        return {
            "vision": needs_vision,
            "code_analysis": needs_code,
            "code_generation": needs_generation or needs_code,  # Si necesita código, probablemente también generación
            "direct_response": is_simple,
            "reasoning": "Análisis heurístico de fallback (LLM no disponible)"
        }

    def get_execution_plan(self, prompt: str, has_image: bool = False) -> Dict[str, Any]:
        """
        Genera un plan de ejecución completo con la secuencia de agentes.

        Args:
            prompt: Texto del usuario
            has_image: Si hay imagen disponible

        Returns:
            Dict con la decisión y plan de ejecución
        """
        analysis = self.analyze_request(prompt, has_image)

        # Construir secuencia de ejecución
        execution_sequence = []

        if analysis["vision"]:
            execution_sequence.append("vision")

        if analysis["code_analysis"]:
            execution_sequence.append("code")

        if analysis["code_generation"]:
            execution_sequence.append("editor")

        if analysis["direct_response"] or not execution_sequence:
            execution_sequence.append("direct")

        return {
            **analysis,
            "execution_sequence": execution_sequence,
            "requires_image": analysis["vision"],
            "requires_file_context": analysis["code_analysis"] or analysis["code_generation"]
        }
