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
            num_ctx=21504,  # Mayor contexto para código
            num_predict=16384,
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
        Usa memoria del proyecto si está disponible para priorizar archivos relevantes.

        Args:
            files_content: Dict {filename: content}
            user_request: Solicitud del usuario

        Returns:
            Dict con el análisis y archivos identificados para modificar
        """
        # Verificar si hay un resumen del proyecto en memoria
        project_summary = self.memory.get_project_summary()
        cached_analyses = self.memory.get_all_cached_analyses()

        # Construir lista de archivos con info de caché
        file_summaries = []
        prioritized_files = []
        other_files = []

        for fname, content in files_content.items():
            lines = len(content.splitlines())
            summary_line = f"- {fname}: {lines} líneas"

            # Si hay análisis en caché, añadir info
            if fname in cached_analyses:
                cached = cached_analyses[fname]
                summary_cached = cached.get('summary', '')
                if summary_cached:
                    summary_line += f" (cache: {summary_cached[:60]}...)"
                # Archivos con caché tienen prioridad
                prioritized_files.append(summary_line)
            else:
                other_files.append(summary_line)

        # Ordenar: primero los que tienen caché, luego el resto
        file_summaries = prioritized_files + other_files

        context_parts = [
            "ARCHIVOS EN EL PROYECTO:",
            "\n".join(file_summaries[:15]),  # Limitar a 15 archivos para no saturar
            "",
        ]

        # Añadir contexto del proyecto si existe
        if project_summary:
            context_parts.extend([
                "RESUMEN DEL PROYECTO (de análisis previo):",
                project_summary[:1000],
                "",
            ])

        context_parts.extend([
            "SOLICITUD DEL USUARIO:",
            user_request,
            "",
            "INSTRUCCIÓN:",
            "1. Usa el RESUMEN DEL PROYECTO si está disponible para entender la estructura",
            "2. Identifica QUÉ archivo(s) debe(n) modificarse según la solicitud",
            "3. Prioriza los archivos principales del proyecto (app.py, tracker.py, etc.)",
            "4. Ignora archivos de datos, scripts auxiliares o de prueba a menos que sean relevantes",
            "5. Describe los cambios necesarios en cada archivo identificado",
            "Solo analiza - NO generes código todavía."
        ])

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content="\n".join(context_parts))
        ]

        try:
            print(f"[CodeAgent] Analizando {len(files_content)} archivos...")
            if project_summary:
                print(f"[CodeAgent] Usando resumen del proyecto en memoria ({len(project_summary)} chars)")
            if cached_analyses:
                print(f"[CodeAgent] {len(cached_analyses)} archivos con análisis en caché")

            response = self.llm.invoke(messages)
            analysis = response.content

            # Extraer información estructurada
            extracted_info = self._extract_structured_info(analysis)

            # Determinar archivos afectados
            files_affected = extracted_info.get("files", [])

            # Si no se detectaron archivos, usar los mencionados en el análisis
            if not files_affected:
                import re
                for fname in files_content.keys():
                    if re.search(rf'\b{re.escape(fname)}\b', analysis, re.IGNORECASE):
                        files_affected.append(fname)

            # Guardar análisis de cada archivo afectado en caché
            for fname in files_affected:
                if fname in files_content:
                    file_result = {
                        "filename": fname,
                        "analysis": analysis,
                        "summary": extracted_info.get("summary", ""),
                        "files_affected": files_affected,
                        "approach": extracted_info.get("approach", ""),
                        "considerations": extracted_info.get("considerations", []),
                        "success": True
                    }
                    self.memory.save_file_analysis(fname, files_content[fname], file_result)

            # Guardar resumen del proyecto
            if extracted_info.get("summary"):
                self.memory.save_project_summary(
                    extracted_info.get("summary", ""),
                    list(files_content.keys())
                )

            return {
                "files_analyzed": list(files_content.keys()),
                "files_affected": files_affected,
                "analysis": analysis,
                "summary": extracted_info.get("summary", ""),
                "approach": extracted_info.get("approach", ""),
                "success": True
            }
        except Exception as e:
            print(f"[CodeAgent] Error en análisis multi-archivo: {e}")
            import traceback
            traceback.print_exc()
            return {
                "files_analyzed": list(files_content.keys()),
                "files_affected": [],
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

    def discover_related_files(self, target_file: str, project_path: str) -> list:
        """
        Descubre archivos relacionados con el archivo objetivo mediante análisis de imports.

        Args:
            target_file: Ruta del archivo objetivo
            project_path: Ruta raíz del proyecto

        Returns:
            Lista de rutas de archivos relacionados
        """
        import ast
        import os

        related_files = []
        target_dir = os.path.dirname(target_file) or project_path

        try:
            with open(target_file, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)
            imports = set()

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])

            # Buscar archivos que correspondan a los imports
            for root, _, files in os.walk(project_path):
                for file in files:
                    if not file.endswith('.py'):
                        continue

                    file_path = os.path.join(root, file)
                    module_name = file[:-3]  # quitar .py

                    if module_name in imports:
                        related_files.append(file_path)

        except Exception as e:
            print(f"[CodeAgent] Error descubriendo archivos relacionados: {e}")

        return related_files

    def analyze_with_context(
        self,
        target_file: str,
        target_content: str,
        related_files: Dict[str, str],
        user_request: str,
        image_analysis: str = None
    ) -> Dict[str, Any]:
        """
        Analiza el archivo objetivo con contexto de archivos relacionados.

        Args:
            target_file: Archivo principal a modificar
            target_content: Contenido del archivo principal
            related_files: Dict {ruta: contenido} de archivos relacionados
            user_request: Solicitud del usuario
            image_analysis: Análisis de imagen (opcional)

        Returns:
            Dict con análisis incluyendo archivos afectados
        """
        # Construir contexto de archivos relacionados
        context_parts = [
            f"ARCHIVO PRINCIPAL: {target_file}",
            f"TOTAL DE LÍNEAS: {len(target_content.splitlines())}",
            "",
            "CONTENIDO DEL ARCHIVO PRINCIPAL:",
            "```python",
            target_content[:3000],
            "```",
        ]

        if related_files:
            context_parts.extend([
                "",
                "ARCHIVOS RELACIONADOS (imports/contexto):",
                ""
            ])
            for filepath, content in list(related_files.items())[:3]:  # Máximo 3
                context_parts.extend([
                    f"--- {filepath} ---",
                    "```python",
                    content[:1500],
                    "```",
                    ""
                ])

        context_parts.extend([
            "",
            f"SOLICITUD DEL USUARIO:\n{user_request}",
            "",
            "INSTRUCCIÓN:",
            "1. Analiza qué cambios son necesarios en el ARCHIVO PRINCIPAL",
            "2. Identifica si alguno de los ARCHIVOS RELACIONADOS también necesita cambios",
            "3. Especifica qué archivos deben modificarse y por qué",
            "4. Describe los cambios necesarios en cada archivo"
        ])

        if image_analysis:
            context_parts.extend([
                "",
                "INFORMACIÓN DE PANTALLA:",
                image_analysis
            ])

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content="\n".join(context_parts))
        ]

        try:
            print(f"[CodeAgent] Analizando con contexto de {len(related_files)} archivos relacionados...")
            response = self.llm.invoke(messages)
            analysis = response.content

            # Extraer información estructurada
            extracted_info = self._extract_structured_info(analysis)

            # Determinar archivos a modificar
            files_to_modify = extracted_info.get("files", [target_file])
            if target_file not in files_to_modify:
                files_to_modify.insert(0, target_file)

            result = {
                "filename": target_file,
                "analysis": analysis,
                "summary": extracted_info.get("summary", ""),
                "files_affected": files_to_modify,
                "related_context": list(related_files.keys()),
                "lines_to_modify": extracted_info.get("lines", []),
                "approach": extracted_info.get("approach", ""),
                "considerations": extracted_info.get("considerations", []),
                "success": True
            }

            # Guardar en caché solo el análisis principal
            self.memory.save_file_analysis(target_file, target_content, result)

            return result

        except Exception as e:
            print(f"[CodeAgent] Error en análisis con contexto: {e}")
            return {
                "filename": target_file,
                "analysis": f"Error: {e}",
                "files_affected": [target_file],
                "success": False,
                "error": str(e)
            }
