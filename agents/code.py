"""
Agente Código - analiza código existente y determina qué cambios son necesarios.
Usa modelo de visión (Q4) y ProjectMemory para cachear análisis entre sesiones.
"""
from typing import Dict, Any, Optional, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from agents.memory import ProjectMemory


class CodeAgent:
    """
    Agente especializado en análisis de código.
    Examina el código existente y planifica modificaciones necesarias.
    Usa ProjectMemory para cachear análisis entre sesiones.
    """

    def __init__(self, model_name: str = "qwen3-vl:8b", memory_dir: str = None):
        """
        Inicializa el agente de código.

        Args:
            model_name: Modelo Ollama para análisis (preferiblemente Q4 cuantizado)
            memory_dir: Directorio para la memoria persistente
        """
        self.model_name = model_name
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.2,
            top_p=0.85,
            num_ctx=12288,  # Mayor contexto para código
            num_predict=8192,
            repeat_penalty=1.05
        )
        self.memory = ProjectMemory(memory_dir)

        self.system_prompt = """Eres un agente de análisis de código experto.

TU MISIÓN:
Analizar el código existente y determinar QUÉ cambios son necesarios según la solicitud del usuario.

REGLAS IMPORTANTES:
1. NO generes código nuevo - solo ANALIZA y DESCRIBE los cambios necesarios
2. Identifica qué líneas/archivos necesitan modificación
3. Describe el enfoque técnico recomendado
4. Señala posibles riesgos o consideraciones

FORMATO DE SALIDA:
Devuelve tu análisis en secciones claras:

**Análisis de la solicitud:**
- Qué quiere lograr el usuario

**Archivos afectados:**
- Lista de archivos que necesitan cambios

**Cambios identificados:**
- Descripción detallada de qué modificar en cada archivo
- Líneas o funciones específicas a cambiar

**Enfoque recomendado:**
- Estrategia técnica sugerida

**Consideraciones:**
- Riesgos potenciales
- Dependencias a tener en cuenta"""

    def analyze_modification_request(
        self,
        filename: str,
        code_content: str,
        user_request: str,
        image_analysis: str = None
    ) -> Dict[str, Any]:
        """
        Analiza una solicitud de modificación de código.

        Args:
            filename: Nombre del archivo a modificar
            code_content: Contenido actual del archivo
            user_request: Solicitud del usuario
            image_analysis: Análisis previo de imagen (si aplica)

        Returns:
            Dict con el análisis y plan de cambios
        """
        # Verificar si existe análisis en caché válido
        cached = self.memory.get_file_analysis(filename, code_content)
        if cached:
            print(f"[CodeAgent] Usando análisis en caché para {filename}")
            return {
                "filename": filename,
                "analysis": cached["analysis"],
                "summary": cached["summary"],
                "files_affected": cached["files_affected"],
                "lines_to_modify": cached["lines_to_modify"],
                "approach": cached["approach"],
                "considerations": cached["considerations"],
                "from_cache": True,
                "success": True
            }

        # Construir prompt de análisis
        context_parts = [
            f"ARCHIVO: {filename}",
            f"TOTAL DE LÍNEAS: {len(code_content.splitlines())}",
            "",
            "CONTENIDO ACTUAL DEL ARCHIVO:",
            "```python",
            code_content[:4000],  # Limitar para no saturar
            "```",
            "",
            f"SOLICITUD DEL USUARIO:\n{user_request}",
        ]

        if image_analysis:
            context_parts.extend([
                "",
                "INFORMACIÓN DE PANTALLA (posible error/contexto visual):",
                image_analysis
            ])

        user_prompt = "\n".join(context_parts)

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt)
        ]

        try:
            print(f"[CodeAgent] Analizando {filename} con {self.model_name}...")
            response = self.llm.invoke(messages)
            analysis = response.content

            # Extraer información estructurada
            extracted_info = self._extract_structured_info(analysis)

            result = {
                "filename": filename,
                "analysis": analysis,
                "summary": extracted_info.get("summary", ""),
                "files_affected": extracted_info.get("files", [filename]),
                "lines_to_modify": extracted_info.get("lines", []),
                "approach": extracted_info.get("approach", ""),
                "considerations": extracted_info.get("considerations", []),
                "from_cache": False,
                "success": True
            }

            # Guardar en caché
            self.memory.save_file_analysis(filename, code_content, result)

            return result
        except Exception as e:
            print(f"[CodeAgent] Error analizando código: {e}")
            return {
                "filename": filename,
                "analysis": f"Error en análisis: {e}",
                "summary": "",
                "files_affected": [filename],
                "success": False,
                "error": str(e)
            }

    def analyze_multiple_files(
        self,
        files_content: Dict[str, str],
        user_request: str
    ) -> Dict[str, Any]:
        """
        Analiza múltiples archivos para determinar qué modificar.

        Args:
            files_content: Dict {filename: content}
            user_request: Solicitud del usuario

        Returns:
            Dict con análisis de todos los archivos
        """
        file_summaries = []
        for fname, content in files_content.items():
            lines = len(content.splitlines())
            file_summaries.append(f"- {fname}: {lines} líneas")

        context_parts = [
            "ARCHIVOS EN EL PROYECTO:",
            "\n".join(file_summaries),
            "",
            "SOLICITUD DEL USUARIO:",
            user_request,
            "",
            "INSTRUCCIÓN:",
            "Determina QUÉ archivo(s) debe(n) modificarse y describe los cambios necesarios.",
            "Solo analiza - NO generes código todavía."
        ]

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content="\n".join(context_parts))
        ]

        try:
            print(f"[CodeAgent] Analizando {len(files_content)} archivos...")
            response = self.llm.invoke(messages)
            analysis = response.content

            return {
                "files_analyzed": list(files_content.keys()),
                "analysis": analysis,
                "success": True
            }
        except Exception as e:
            return {
                "files_analyzed": list(files_content.keys()),
                "analysis": f"Error: {e}",
                "success": False,
                "error": str(e)
            }

    def _extract_structured_info(self, analysis: str) -> Dict[str, Any]:
        """
        Extrae información estructurada del análisis de texto.
        """
        info = {
            "summary": "",
            "files": [],
            "lines": [],
            "approach": "",
            "considerations": []
        }

        lines = analysis.split('\n')
        current_section = None

        for line in lines:
            line_lower = line.lower()

            if 'resumen' in line_lower or 'summary' in line_lower:
                current_section = 'summary'
                continue
            elif 'archivo' in line_lower or 'files' in line_lower:
                current_section = 'files'
                continue
            elif 'línea' in line_lower or 'line' in line_lower:
                current_section = 'lines'
                continue
            elif 'enfoque' in line_lower or 'approach' in line_lower:
                current_section = 'approach'
                continue
            elif 'consideraci' in line_lower or 'consideration' in line_lower:
                current_section = 'considerations'
                continue

            # Extraer información según la sección
            if current_section == 'summary' and line.strip() and not line.startswith('-'):
                info['summary'] += line + " "
            elif current_section == 'files' and line.strip().startswith('-'):
                # Extraer nombre de archivo
                import re
                match = re.search(r'(\w+\.py)', line)
                if match:
                    info['files'].append(match.group(1))
            elif current_section == 'considerations' and line.strip().startswith('-'):
                info['considerations'].append(line.strip()[1:].strip())

        return info
