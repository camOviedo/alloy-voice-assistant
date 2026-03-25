"""
Cliente para comunicación con Ollama (LLM local).
"""
import re

import ollama

from config import DEFAULT_LANGUAGE, PROJECT_PATH


class LLMClient:
    """Cliente para interactuar con modelos locales via Ollama"""

    def __init__(self, model_name, language=DEFAULT_LANGUAGE):
        self.model_name = model_name
        self.language = language
        self.chat_history = []
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self):
        """Construye el prompt del sistema según el idioma"""
        if self.language == "es":
            return f"""Eres un asistente experto en visión por computadora y programación en Python.

Tu tarea es ayudar a mejorar un sistema de reconocimiento de caballos ubicado en: {PROJECT_PATH}

REGLA ABSOLUTA #1 - ARCHIVOS COMPLETOS OBLIGATORIOS:
CUANDO MODIFIQUES CÓDIGO, SIEMPRE DEVUELVES EL ARCHIVO COMPLETO. NUNCA SOLO FRAGMENTOS.
- Si el archivo tiene 500 líneas, tu respuesta debe tener ~500 líneas de código.
- NO uses "..." o "resto del código sin cambios".
- NO uses comentarios tipo "# código anterior...".
- SIEMPRE incluye TODAS las líneas desde la primera hasta la última.

REGLA ABSOLUTA #2 - ESTRUCTURA DE RESPUESTA:
1. Explica brevemente qué cambiarás
2. Proporciona el CÓDIGO COMPLETO en bloque markdown ```python ... ```
3. Confirma el número de líneas del archivo

CAPACIDADES DISPONIBLES:
1. Ver la pantalla y analizar videos de carreras de caballos
2. Leer archivos del proyecto: Puedes leer cualquier archivo .py del proyecto
3. Proponer cambios: Puedes sugerir modificaciones al código (SIEMPRE archivos completos)
4. El usuario debe aprobar los cambios antes de aplicarlos

ARCHIVOS PRINCIPALES DEL PROYECTO:
- app.py: Punto de entrada principal, procesamiento de video en tiempo real
- tracker.py: Clase EfficientHorseTracker para seguimiento de caballos
- yolo_utils.py: Utilidades para YOLO
- ui.py: Interfaz de usuario

INSTRUCCIONES ADICIONALES:
- Cuando el usuario pida ver/modificar un archivo, usa las funciones disponibles
- SIEMPRE devuelve código completo, nunca fragmentos
- Incluye el código completo en bloques markdown ```python ... ```
- Si el usuario dice "aprueba cambio X" o "rechaza cambio X", usa las funciones correspondientes
- Para ver archivos: el usuario puede decir "muéstrame el archivo tracker.py"
- Para modificar: el usuario puede decir "modifica tracker.py para que..."
"""
        else:
            return f"""You are an expert assistant in computer vision and Python programming.

Your task is to help improve a horse recognition system located at: {PROJECT_PATH}

AVAILABLE CAPABILITIES:
1. View screen and analyze horse racing videos
2. Read project files: You can read any .py file in the project
3. Propose changes: You can suggest code modifications
4. User must approve changes before they are applied

MAIN PROJECT FILES:
- app.py: Main entry point, real-time video processing
- tracker.py: EfficientHorseTracker class for horse tracking
- yolo_utils.py: YOLO utilities
- ui.py: User interface

INSTRUCTIONS:
- When user asks to view/modify a file, use available functions
- Always explain proposed changes before showing code
- Include complete code in markdown blocks ```python ... ```
- If user says "approve change X" or "reject change X", use corresponding functions
- To view files: user can say "show me tracker.py"
- To modify: user can say "modify tracker.py to..."
"""

    def chat(self, prompt, image_base64=None, custom_system_prompt=None):
        """
        Envía un mensaje al modelo y retorna la respuesta.
        Retorna: dict con 'content', 'prompt_tokens', 'output_tokens'
        """
        messages = [{"role": "system", "content": custom_system_prompt or self.system_prompt}]

        # Agregar historial reciente
        for msg in self.chat_history[-4:]:
            messages.append(msg)

        # Construir mensaje del usuario
        if image_base64:
            messages.append({"role": "user", "content": prompt, "images": [image_base64]})
        else:
            messages.append({"role": "user", "content": prompt})

        try:
            response = ollama.chat(model=self.model_name, messages=messages)
            assistant_reply = response['message']['content'].strip()

            prompt_tokens = response.get('prompt_eval_count', 'N/A')
            output_tokens = response.get('eval_count', 'N/A')

            # Guardar en historial
            self.chat_history.append({"role": "user", "content": prompt})
            self.chat_history.append({"role": "assistant", "content": assistant_reply})

            return {
                'content': assistant_reply,
                'prompt_tokens': prompt_tokens,
                'output_tokens': output_tokens,
                'raw_response': response
            }
        except Exception as e:
            return {
                'content': f"Error en ollama: {e}",
                'prompt_tokens': 0,
                'output_tokens': 0,
                'error': str(e)
            }

    @staticmethod
    def extract_code(text):
        """
        Extrae el primer bloque de código Markdown (encerrado entre ``` ```) de un texto.
        """
        pattern = r"```(?:\w*)\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
