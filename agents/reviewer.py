"""
Agente Revisor - verifica la calidad del código generado.
Revisa errores de sintaxis, imports faltantes, inconsistencias y cumplimiento de requisitos.
"""
import ast
import re
from typing import Dict, Any, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama


class ReviewerAgent:
    """
    Agente especializado en revisar código generado.
    Detecta errores antes de entregar al usuario y sugiere correcciones.
    """

    def __init__(self, model_name: str = "qwen2.5-coder:14b"):
        """
        Inicializa el agente revisor.

        Args:
            model_name: Modelo Ollama para revisión
        """
        self.model_name = model_name
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.2,
            num_ctx=16384,
            num_predict=8192,
            repeat_penalty=1.05
        )

        self.system_prompt = """Eres un revisor de código experto. Tu misión es verificar que el código generado sea correcto y completo.

TU TAREA:
Revisar el código generado comparándolo con el original y la solicitud del usuario. Detecta:

1. ERRORES DE SINTAXIS:
   - Código Python inválido
   - Indentación incorrecta
   - Paréntesis/corchetes sin cerrar

2. IMPORTS FALTANTES:
   - Módulos importados en el original pero no en el generado
   - Nuevos imports necesarios para el código agregado

3. CÓDIGO OMITIDO:
   - Funciones/clases del original que faltan en el generado
   - Líneas importantes que fueron eliminadas accidentalmente

4. INCONSISTENCIAS:
   - Variables no definidas
   - Funciones llamadas pero no existentes
   - Cambios de nombre sin actualizar referencias

5. CUMPLIMIENTO:
   - ¿El código cumple con la solicitud del usuario?
   - ¿Faltan requisitos mencionados?

FORMATO DE RESPUESTA (JSON estricto):
{
  "approved": true/false,
  "issues": [
    {
      "severity": "error|warning|info",
      "type": "syntax|import|omission|inconsistency|requirement",
      "description": "Descripción del problema",
      "line": número_de_línea_o_null,
      "suggestion": "Cómo corregirlo"
    }
  ],
  "corrected_code": "Código completo corregido (solo si hay issues)",
  "summary": "Resumen breve de la revisión"
}

REGLAS:
- approved = true solo si no hay issues de severity "error"
- Si hay errores, proporciona corrected_code completo
- Sé específico en las descripciones de issues
- No inventes problemas que no existen"""

    def review_code(
        self,
        filename: str,
        original_code: str,
        generated_code: str,
        user_request: str,
        code_analysis: str = None
    ) -> Dict[str, Any]:
        """
        Revisa el código generado.

        Args:
            filename: Nombre del archivo
            original_code: Código original completo
            generated_code: Código generado por el editor
            user_request: Solicitud del usuario
            code_analysis: Análisis previo del CodeAgent (opcional)

        Returns:
            Dict con resultado de la revisión
        """
        # Primero hacer análisis estático rápido
        static_issues = self._static_analysis(original_code, generated_code)

        # Si hay errores de sintaxis graves, no necesitamos consultar al LLM
        syntax_errors = [i for i in static_issues if i["severity"] == "error" and i["type"] == "syntax"]
        if syntax_errors:
            return {
                "approved": False,
                "issues": static_issues,
                "corrected_code": None,
                "summary": f"Se encontraron {len(syntax_errors)} errores de sintaxis que deben corregirse",
                "from_static_analysis": True
            }

        # Construir prompt para el LLM
        context_parts = [
            f"ARCHIVO: {filename}",
            "",
            "SOLICITUD DEL USUARIO:",
            user_request,
            ""
        ]

        if code_analysis:
            context_parts.extend([
                "ANÁLISIS TÉCNICO PREVIO:",
                code_analysis,
                ""
            ])

        context_parts.extend([
            "CÓDIGO ORIGINAL:",
            "```python",
            original_code[:2000],  # Limitar para no saturar
            "```",
            "",
            "CÓDIGO GENERADO A REVISAR:",
            "```python",
            generated_code,
            "```",
            "",
            "ISSUES DETECTADOS EN ANÁLISIS ESTÁTICO:",
            self._format_issues(static_issues) if static_issues else "Ninguno"
        ])

        context_parts.append("\nRealiza la revisión completa y devuelve el JSON.")

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content="\n".join(context_parts))
        ]

        try:
            print(f"[ReviewerAgent] Revisando código generado...")
            response = self.llm.invoke(messages)

            # Intentar parsear JSON
            result = self._parse_review_response(response.content, generated_code)

            # Combinar con issues estáticos
            all_issues = static_issues + result.get("issues", [])

            # Re-evaluar aprobación
            has_errors = any(i["severity"] == "error" for i in all_issues)

            return {
                "approved": not has_errors,
                "issues": all_issues,
                "corrected_code": result.get("corrected_code") if has_errors else None,
                "summary": result.get("summary", "Revisión completada"),
                "from_static_analysis": False
            }

        except Exception as e:
            print(f"[ReviewerAgent] Error en revisión LLM: {e}")
            # Fallback: devolver resultado basado solo en análisis estático
            return {
                "approved": len(static_issues) == 0,
                "issues": static_issues,
                "corrected_code": None,
                "summary": f"Revisión por LLM falló. Issues estáticos: {len(static_issues)}",
                "error": str(e),
                "from_static_analysis": True
            }

    def _static_analysis(self, original_code: str, generated_code: str) -> List[Dict[str, Any]]:
        """
        Realiza análisis estático rápido del código generado.
        """
        issues = []

        # 1. Verificar sintaxis Python
        try:
            ast.parse(generated_code)
        except SyntaxError as e:
            issues.append({
                "severity": "error",
                "type": "syntax",
                "description": f"Error de sintaxis: {e.msg}",
                "line": e.lineno,
                "suggestion": f"Revisar la línea {e.lineno}: {e.text}"
            })
        except Exception as e:
            issues.append({
                "severity": "error",
                "type": "syntax",
                "description": f"Error parsing código: {str(e)}",
                "line": None,
                "suggestion": "El código generado no es Python válido"
            })

        # 2. Detectar imports faltantes del original
        original_imports = self._extract_imports(original_code)
        generated_imports = self._extract_imports(generated_code)

        for imp in original_imports:
            if imp not in generated_imports:
                issues.append({
                    "severity": "error",
                    "type": "import",
                    "description": f"Import faltante: {imp}",
                    "line": None,
                    "suggestion": f"Agregar 'import {imp}' o 'from {imp} import ...'"
                })

        # 3. Verificar que no haya "..." o comentarios de código omitido
        omit_patterns = [
            r'#\s*\.\.\.\s*',
            r'#\s*resto del código',
            r'#\s*código sin cambios',
            r'#\s*implementación anterior',
            r'\.\.\.\s*$'  # Ellipsis sola en línea
        ]
        for pattern in omit_patterns:
            if re.search(pattern, generated_code, re.IGNORECASE):
                issues.append({
                    "severity": "error",
                    "type": "omission",
                    "description": "Se detectó código omitido con '...' o comentarios",
                    "line": None,
                    "suggestion": "Reemplazar con el código completo del archivo original"
                })
                break  # Solo reportar una vez

        # 4. Verificar longitud (posible código truncado)
        original_lines = len(original_code.splitlines())
        generated_lines = len(generated_code.splitlines())

        if generated_lines < original_lines * 0.5:
            issues.append({
                "severity": "error",
                "type": "omission",
                "description": f"Código significativamente más corto: {generated_lines} vs {original_lines} líneas",
                "line": None,
                "suggestion": "El código generado parece incompleto"
            })

        return issues

    def _extract_imports(self, code: str) -> set:
        """
        Extrae los módulos importados del código.
        """
        imports = set()
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])
        except:
            pass  # Si no parsea, devolver vacío
        return imports

    def _format_issues(self, issues: List[Dict[str, Any]]) -> str:
        """
        Formatea issues para el prompt.
        """
        if not issues:
            return "Ninguno"

        lines = []
        for i, issue in enumerate(issues, 1):
            line_info = f" (línea {issue['line']})" if issue.get('line') else ""
            lines.append(f"{i}. [{issue['severity'].upper()}] {issue['type']}{line_info}: {issue['description']}")
        return "\n".join(lines)

    def _parse_review_response(self, content: str, fallback_code: str) -> Dict[str, Any]:
        """
        Parsea la respuesta del LLM buscando JSON.
        """
        # Intentar extraer JSON
        try:
            # Buscar bloque JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                import json
                result = json.loads(json_match.group())
                return {
                    "approved": result.get("approved", False),
                    "issues": result.get("issues", []),
                    "corrected_code": result.get("corrected_code"),
                    "summary": result.get("summary", "Revisión completada")
                }
        except json.JSONDecodeError as e:
            print(f"[ReviewerAgent] Error parseando JSON: {e}")
            # JSON malformado - devolver como no aprobado para revisión manual
            return {
                "approved": False,
                "issues": [{
                    "severity": "error",
                    "type": "parse_error",
                    "description": f"Error parseando respuesta JSON del revisor: {e}",
                    "line": None,
                    "suggestion": "Revisar manualmente el código generado"
                }],
                "corrected_code": None,
                "summary": f"Error parseando JSON: {e}"
            }
        except Exception as e:
            print(f"[ReviewerAgent] Error inesperado: {e}")

        # Fallback: interpretar texto
        content_lower = content.lower()
        approved = "aprobado" in content_lower or "correcto" in content_lower

        # Intentar extraer código corregido
        corrected = self._extract_code(content) if not approved else None

        return {
            "approved": approved,
            "issues": [],
            "corrected_code": corrected,
            "summary": "Revisión parseada de texto (JSON no válido)"
        }

    def _extract_code(self, text: str) -> Optional[str]:
        """
        Extrae el primer bloque de código Markdown.
        """
        pattern = r"```(?:python)?\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
