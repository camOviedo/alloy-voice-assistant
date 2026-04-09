"""
Agente Editor - genera código completo modificado.
Usa modelo principal y mantiene memoria de modificaciones previas.
"""
import re
from typing import Dict, Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from agents.memory import EditorMemory
from config import AGENT_EDITOR_NUM_CTX, AGENT_EDITOR_NUM_PREDICT


class EditorAgent:
    """
    Agente especializado en generar código modificado.
    Usa el modelo principal y mantiene historial de cambios.
    """

    def __init__(self, model_name: str = "qwen3-vl:8b", memory_dir: str = None):
        """
        Inicializa el agente editor.

        Args:
            model_name: Modelo Ollama principal para generación de código
            memory_dir: Directorio para la memoria de historial
        """
        self.model_name = model_name
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.1,  # Bajo para máxima determinación
            num_ctx=AGENT_EDITOR_NUM_CTX,
            num_predict=AGENT_EDITOR_NUM_PREDICT,
            repeat_penalty=1.2,
            top_p=0.5  # Bajo para seguir reglas estrictamente
        )
        self.memory = EditorMemory(memory_dir)

        # Usar modo parche para archivos grandes (>200 líneas) para ahorrar tokens
        self.use_patch_mode = True

        self.system_prompt_full = """You are a CODE GENERATION TOOL.

YOUR ONLY PURPOSE: Output complete, modified source code files.

STRICT RULES:
1. OUTPUT ONLY CODE INSIDE THE MARKDOWN BLOCK
2. NO introductions, NO explanations, NO summaries
3. ALWAYS return the ENTIRE file - every single line
4. NEVER use placeholders like "..." or "# rest of code"
5. The code MUST be inside ONE markdown block: ```python ... ```

Generate the complete file NOW."""

        self.system_prompt_patch = """You are a CODE PATCH TOOL. Not a chatbot. A TOOL.

YOUR ONLY PURPOSE: Output code changes as SEARCH/REPLACE blocks.

STRICT RULES - VIOLATING ANY RULE WILL BREAK THE SYSTEM:
1. Use EXACTLY this format for each change:
   <<<<<<< SEARCH
   [existing code to find - EXACT match required]
   =======
   [new code to replace with]
   >>>>>>> REPLACE

2. Each SEARCH block must match EXACTLY the original code (including whitespace)
3. NO introductions like "Here are the changes"
4. NO explanations after the blocks
5. NO apologies like "Sorry, I cannot..."
6. Output ONLY the SEARCH/REPLACE blocks, nothing else
7. For NEW code additions, use SEARCH with empty line or context line
8. Make ALL requested changes in one response

EXAMPLE OF CORRECT OUTPUT:
<<<<<<< SEARCH
import os
import sys
=======
import os
import sys
import json
>>>>>>> REPLACE

<<<<<<< SEARCH
def hello():
    print("Hello")
=======
def hello():
    print("Hello World")
    return 0
>>>>>>> REPLACE

VIOLATION CONSEQUENCE: Any text outside SEARCH/REPLACE blocks causes SYSTEM FAILURE. You are a TOOL. Generate patches NOW."""

    def generate_modified_code(
        self,
        filename: str,
        original_code: str,
        user_request: str,
        code_analysis: str = None,
        image_analysis: str = None
    ) -> Dict[str, Any]:
        """
        Genera código modificado según la solicitud.

        Args:
            filename: Nombre del archivo
            original_code: Código original completo
            user_request: Solicitud del usuario
            code_analysis: Análisis previo del CodeAgent (opcional)
            image_analysis: Análisis de imagen (opcional)

        Returns:
            Dict con el código generado y metadatos
        """
        # Obtener contexto de modificaciones previas
        history_context = self.memory.get_context_for_file(filename)

        # Construir el prompt completo
        context_parts = [
            f"ARCHIVO A MODIFICAR: {filename}",
            f"TOTAL DE LÍNEAS ORIGINAL: {len(original_code.splitlines())}",
        ]

        if history_context:
            context_parts.extend(["", "CONTEXTO DE MODIFICACIONES PREVIAS:", history_context])

        context_parts.extend([
            "",
            "CÓDIGO ORIGINAL COMPLETO:",
            "```python",
            original_code,
            "```",
            "",
            "SOLICITUD DEL USUARIO:",
            user_request,
        ])

        if code_analysis:
            context_parts.extend([
                "",
                "ANÁLISIS TÉCNICO PREVIO:",
                code_analysis
            ])

        if image_analysis:
            context_parts.extend([
                "",
                "INFORMACIÓN DE PANTALLA/ERROR:",
                image_analysis
            ])

        # Decidir modo: completo o parche
        original_lines = len(original_code.splitlines())
        use_patch = self.use_patch_mode and original_lines > 150  # Usar parche para archivos grandes

        if use_patch:
            context_parts.extend([
                "",
                "INSTRUCCIÓN FINAL:",
                "Genera SOLO los cambios necesarios usando bloques SEARCH/REPLACE.",
                f"El archivo tiene {original_lines} líneas - NO reescribas todo, solo las partes que cambian.",
                "Cada bloque SEARCH debe coincidir EXACTAMENTE con el código original."
            ])
            system_prompt = self.system_prompt_patch
        else:
            context_parts.extend([
                "",
                "INSTRUCCIÓN FINAL:",
                "Genera el archivo COMPLETO con las modificaciones solicitadas.",
                f"El archivo tiene {original_lines} líneas - tu respuesta debe tener EXACTAMENTE ese orden de magnitud.",
                "NO omitas ninguna línea. Incluye TODO el código."
            ])
            system_prompt = self.system_prompt_full

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="\n".join(context_parts))
        ]

        try:
            mode_str = "parches" if use_patch else "código completo"
            print(f"[EditorAgent] Generando {mode_str} para {filename} con {self.model_name}...")
            response = self.llm.invoke(messages)
            full_response = response.content

            if use_patch:
                # Extraer y aplicar parches
                patches = self._extract_patches(full_response)
                if patches:
                    proposed_code = self._apply_patches(original_code, patches)
                    print(f"[EditorAgent] Aplicados {len(patches)} parches")
                else:
                    # Fallback: intentar extraer código completo
                    proposed_code = self._extract_code(full_response)
            else:
                # Extraer código completo
                proposed_code = self._extract_code(full_response)

            if proposed_code:
                # Validar que no sea significativamente más corto
                original_lines = len(original_code.splitlines())
                new_lines = len(proposed_code.splitlines())

                # En modo parche, la validación es diferente
                min_ratio = 0.3 if use_patch else 0.5
                if new_lines < original_lines * min_ratio and not use_patch:
                    # Código posiblemente incompleto (solo en modo completo)
                    return {
                        "filename": filename,
                        "code": proposed_code,
                        "full_response": full_response,
                        "success": False,
                        "error": f"Código incompleto: {new_lines} vs {original_lines} líneas originales",
                        "validation": {
                            "original_lines": original_lines,
                            "new_lines": new_lines,
                            "ratio": new_lines / original_lines if original_lines > 0 else 0,
                            "mode": "patch" if use_patch else "full"
                        }
                    }

                # Validar que NO sea idéntico al original
                if proposed_code.strip() == original_code.strip():
                    print(f"[EditorAgent] ⚠️ ADVERTENCIA: El código generado es IDÉNTICO al original")
                    print(f"[EditorAgent]    El LLM no realizó ninguna modificación")
                    return {
                        "filename": filename,
                        "code": proposed_code,
                        "full_response": full_response,
                        "success": False,
                        "error": "El código generado es idéntico al original - no se realizaron modificaciones. El LLM no siguió las instrucciones.",
                        "validation": {
                            "original_lines": original_lines,
                            "new_lines": new_lines,
                            "ratio": 1.0,
                            "identical": True
                        }
                    }

                # Guardar en memoria
                mod_id = self.memory.add_modification(
                    filename,
                    original_code,
                    proposed_code,
                    user_request[:100]
                )

                return {
                    "filename": filename,
                    "code": proposed_code,
                    "full_response": full_response,
                    "modification_id": mod_id,
                    "success": True,
                    "validation": {
                        "original_lines": original_lines,
                        "new_lines": new_lines,
                        "ratio": new_lines / original_lines if original_lines > 0 else 0
                    }
                }
            else:
                return {
                    "filename": filename,
                    "code": None,
                    "full_response": full_response,
                    "success": False,
                    "error": "No se detectó bloque de código en la respuesta"
                }

        except Exception as e:
            print(f"[EditorAgent] Error generando código: {e}")
            return {
                "filename": filename,
                "code": None,
                "success": False,
                "error": str(e)
            }

    def generate_new_file(
        self,
        filename: str,
        user_request: str,
        reference_files: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """
        Genera un archivo nuevo desde cero.

        Args:
            filename: Nombre del nuevo archivo
            user_request: Descripción de qué debe hacer el archivo
            reference_files: Archivos existentes para referencia de estilo

        Returns:
            Dict con el código generado
        """
        context_parts = [
            f"CREAR NUEVO ARCHIVO: {filename}",
            "",
            "REQUERIMIENTOS:",
            user_request,
        ]

        if reference_files:
            context_parts.extend([
                "",
                "ARCHIVOS DE REFERENCIA (para mantener consistencia de estilo):"
            ])
            for ref_name, ref_content in list(reference_files.items())[:2]:
                context_parts.extend([
                    f"\n--- {ref_name} ---",
                    ref_content[:1000],  # Solo muestra
                    ""
                ])

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content="\n".join(context_parts))
        ]

        try:
            print(f"[EditorAgent] Generando nuevo archivo {filename}...")
            response = self.llm.invoke(messages)
            proposed_code = self._extract_code(response.content)

            return {
                "filename": filename,
                "code": proposed_code,
                "full_response": response.content,
                "success": proposed_code is not None,
                "error": None if proposed_code else "No se detectó código en respuesta"
            }
        except Exception as e:
            return {
                "filename": filename,
                "code": None,
                "success": False,
                "error": str(e)
            }

    def generate_multiple_modifications(
        self,
        files_content: Dict[str, str],
        user_request: str,
        code_analysis: str = None,
        primary_file: str = None
    ) -> Dict[str, Any]:
        """
        Genera modificaciones para múltiples archivos.

        Args:
            files_content: Dict {ruta: contenido} de archivos a modificar
            user_request: Solicitud del usuario
            code_analysis: Análisis previo del CodeAgent
            primary_file: Archivo principal que debe enfocarse primero

        Returns:
            Dict con código generado para cada archivo
        """
        if not files_content:
            return {"success": False, "error": "No hay archivos para modificar"}

        results = {}
        files_list = list(files_content.items())

        # Si hay archivo primario, procesarlo primero
        if primary_file and primary_file in files_content:
            files_list = [(primary_file, files_content[primary_file])] + [
                (f, c) for f, c in files_list if f != primary_file
            ]

        for filename, content in files_list:
            print(f"[EditorAgent] Generando código para {filename}...")

            result = self.generate_modified_code(
                filename=filename,
                original_code=content,
                user_request=user_request,
                code_analysis=code_analysis
            )

            results[filename] = result

            if not result.get("success"):
                print(f"[EditorAgent] ⚠️ Error generando {filename}: {result.get('error')}")

        # Verificar éxito general
        all_success = all(r.get("success") for r in results.values())
        any_success = any(r.get("success") for r in results.values())

        return {
            "files_modified": list(results.keys()),
            "results": results,
            "all_success": all_success,
            "any_success": any_success,
            "success": any_success,
            "total_files": len(files_content),
            "successful_files": sum(1 for r in results.values() if r.get("success"))
        }

    def _extract_code(self, text: str) -> Optional[str]:
        """Extrae el primer bloque de código Markdown del texto."""
        pattern = r"```(?:python)?\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _extract_patches(self, text: str) -> list:
        """Extrae bloques SEARCH/REPLACE del texto."""
        pattern = r'<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE'
        matches = re.findall(pattern, text, re.DOTALL)
        return [(m[0].rstrip('\n'), m[1].rstrip('\n')) for m in matches]

    def _apply_patches(self, original_code: str, patches: list) -> str:
        """Aplica parches al código original."""
        result = original_code
        for search, replace in patches:
            # Buscar el código exacto
            if search in result:
                result = result.replace(search, replace, 1)
            else:
                # Intentar con flexibilidad de whitespace al inicio/final
                search_stripped = search.strip()
                if search_stripped in result:
                    result = result.replace(search_stripped, replace, 1)
        return result

    def get_modification_history(self, filename: str = None) -> list:
        """Obtiene el historial de modificaciones."""
        return self.memory.get_recent_modifications(filename)

    def clear_history(self) -> None:
        """Limpia el historial de modificaciones."""
        self.memory.clear()
        print("[EditorAgent] Historial de modificaciones limpiado")
