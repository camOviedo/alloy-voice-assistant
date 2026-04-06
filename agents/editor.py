"""
Agente Editor - genera código completo modificado.
Usa modelo principal y mantiene memoria de modificaciones previas.
"""
import re
from typing import Dict, Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from agents.memory import EditorMemory


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
            temperature=0.6,
            num_ctx=65536,  # Mayor contexto para archivos completos
            num_predict=32768,
            repeat_penalty=1.1,
            top_p=0.9
        )
        self.memory = EditorMemory(memory_dir)

        self.system_prompt = """Eres un agente editor de código experto.

TU MISIÓN:
Generar el CÓDIGO COMPLETO modificado según los requerimientos.

REGLAS ABSOLUTAS - OBLIGATORIAS:
1. DEVUELVE SIEMPRE EL ARCHIVO COMPLETO - NUNCA solo fragmentos
2. Incluye TODAS las líneas: imports, clases, funciones, métodos
3. NO uses "..." o "# resto del código" o "# código sin cambios"
4. NO hagas resúmenes del código faltante
5. El código debe estar en UN SOLO bloque markdown ```python ... ```
6. Explica los cambios DESPUÉS del bloque de código, no antes
7. DEBES hacer cambios REALES al código - no devuelvas el código sin modificar
8. Si la solicitud implica añadir nueva funcionalidad, implementa métodos/clases nuevas
9. Si la solicitud implica modificar comportamiento, cambia la lógica existente

FORMATO DE RESPUESTA CORRECTO:
```python
# [Todo el código del archivo, línea por línea]
# Incluyendo imports, clases, funciones, TODO
# NO omitas NADA
```

**Resumen de cambios:**
- Lista breve de modificaciones realizadas

ADVERTENCIA CRÍTICA: Si devuelves el código sin cambios (idéntico al original), el sistema RECHAZARÁ automáticamente tu respuesta y se perderá el trabajo. DEBES implementar las modificaciones solicitadas."""

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

        context_parts.extend([
            "",
            "INSTRUCCIÓN FINAL:",
            "Genera el archivo COMPLETO con las modificaciones solicitadas.",
            f"El archivo tiene {len(original_code.splitlines())} líneas - tu respuesta debe tener EXACTAMENTE ese orden de magnitud.",
            "NO omitas ninguna línea. Incluye TODO el código."
        ])

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content="\n".join(context_parts))
        ]

        try:
            print(f"[EditorAgent] Generando código para {filename} con {self.model_name}...")
            response = self.llm.invoke(messages)
            full_response = response.content

            # Extraer el código
            proposed_code = self._extract_code(full_response)

            if proposed_code:
                # Validar que no sea significativamente más corto
                original_lines = len(original_code.splitlines())
                new_lines = len(proposed_code.splitlines())

                if new_lines < original_lines * 0.5:
                    # Código posiblemente incompleto
                    return {
                        "filename": filename,
                        "code": proposed_code,
                        "full_response": full_response,
                        "success": False,
                        "error": f"Código incompleto: {new_lines} vs {original_lines} líneas originales",
                        "validation": {
                            "original_lines": original_lines,
                            "new_lines": new_lines,
                            "ratio": new_lines / original_lines if original_lines > 0 else 0
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
        """
        Extrae el primer bloque de código Markdown del texto.
        """
        pattern = r"```(?:python)?\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def get_modification_history(self, filename: str = None) -> list:
        """Obtiene el historial de modificaciones."""
        return self.memory.get_recent_modifications(filename)

    def clear_history(self) -> None:
        """Limpia el historial de modificaciones."""
        self.memory.clear()
        print("[EditorAgent] Historial de modificaciones limpiado")
