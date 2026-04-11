"""
Workflow de LangGraph - grafo de flujo de trabajo multi-agente.
"""
from typing import Dict, Any, Optional, TypedDict

from langgraph.graph import StateGraph, END

from agents.coordinator import CoordinatorAgent
from agents.vision import VisionAgent
from agents.code import CodeAgent
from agents.editor import EditorAgent
from agents.reviewer import ReviewerAgent
from tools.web_search import web_search


class AgentState(TypedDict):
    """Estado compartido entre los agentes en el grafo."""
    # Input del usuario
    user_prompt: str
    image_b64: Optional[str]
    image_path: Optional[str]
    image_paths: Optional[list]  # Lista de rutas a imágenes capturadas
    has_image: bool

    # Decisiones del coordinador
    needs_vision: bool
    needs_code_analysis: bool
    needs_code_generation: bool
    direct_response: bool
    coordinator_reasoning: Optional[str]  # Razonamiento del coordinador

    # Resultados de agentes
    vision_analysis: Optional[str]
    code_analysis: Optional[Dict[str, Any]]
    editor_result: Optional[Dict[str, Any]]
    review_result: Optional[Dict[str, Any]]

    # Control de flujo
    review_iterations: int
    web_search_results: Optional[str]

    # Contexto de archivos
    target_file: Optional[str]
    file_content: Optional[str]
    
    # Soporte multi-archivo (cascada)
    cascade_analysis: Optional[Dict[str, Any]]  # Análisis de dependencias
    all_files_content: Optional[Dict[str, str]]  # Todos los archivos a modificar
    project_path: Optional[str]  # Ruta del proyecto
    
    # Visión
    images_already_cropped: bool  # True si las imágenes ya están recortadas (ej: de captures/)

    # Output final
    final_response: str
    success: bool
    error: Optional[str]

    # Métricas
    tokens_used: Dict[str, int]
    execution_path: list


class AgentWorkflow:
    """
    Grafo de flujo de trabajo que orquesta los agentes.
    """

    def __init__(
        self,
        coordinator_model: str = "qwen2.5:3b",
        vision_model: str = "qwen3-vl:8b-vision",
        code_model: str = "qwen3-coder-30b",
        editor_model: str = "qwen3-coder-30b",
        reviewer_model: str = "qwen2.5-coder:14b",
        memory_dir: str = None
    ):
        """
        Inicializa el workflow con todos los agentes.

        Args:
            coordinator_model: Modelo para el coordinador
            vision_model: Modelo para el agente de visión
            code_model: Modelo para el agente de código
            editor_model: Modelo para el agente editor
            reviewer_model: Modelo para el agente revisor
            memory_dir: Directorio para memoria persistente
        """
        # Inicializar agentes
        self.coordinator = CoordinatorAgent(coordinator_model)
        self.vision = VisionAgent(vision_model, memory_dir)
        self.code = CodeAgent(code_model, memory_dir)
        self.editor = EditorAgent(editor_model, memory_dir)
        self.reviewer = ReviewerAgent(reviewer_model)

        # Construir el grafo
        self.workflow = self._build_graph()
        self.app = self.workflow.compile()

    def _build_graph(self) -> StateGraph:
        """Construye el grafo de flujo de trabajo."""

        # Definir el grafo
        workflow = StateGraph(AgentState)

        # Agregar nodos
        workflow.add_node("coordinator", self._run_coordinator)
        workflow.add_node("vision", self._run_vision)
        workflow.add_node("code", self._run_code)
        workflow.add_node("editor", self._run_editor)
        workflow.add_node("reviewer", self._run_reviewer)
        workflow.add_node("direct_response", self._run_direct_response)
        workflow.add_node("finalize", self._finalize)

        # Definir el flujo condicional desde el coordinador
        workflow.set_entry_point("coordinator")

        workflow.add_conditional_edges(
            "coordinator",
            self._route_from_coordinator,
            {
                "vision": "vision",
                "code": "code",
                "editor": "editor",
                "direct": "direct_response",
            }
        )

        # Después de visión, puede ir a code (si necesita análisis), editor (si solo generación), o finalizar
        workflow.add_conditional_edges(
            "vision",
            self._route_from_vision,
            {
                "code": "code",
                "editor": "editor",
                "finalize": "finalize",
            }
        )

        # Después de code, va a editor si necesita generación
        workflow.add_conditional_edges(
            "code",
            self._route_from_code,
            {
                "editor": "editor",
                "finalize": "finalize",
            }
        )

        # Después de editor, va a reviewer para verificación
        workflow.add_conditional_edges(
            "editor",
            self._route_from_editor,
            {
                "reviewer": "reviewer",
                "finalize": "finalize",
            }
        )

        # Después de reviewer, puede volver a editor (si hay correcciones) o finalizar
        workflow.add_conditional_edges(
            "reviewer",
            self._route_from_reviewer,
            {
                "editor": "editor",
                "finalize": "finalize",
            }
        )

        # Direct response va a finalizar
        workflow.add_edge("direct_response", "finalize")

        # Finalizar termina
        workflow.add_edge("finalize", END)

        return workflow

    def _run_coordinator(self, state: AgentState) -> AgentState:
        """Ejecuta el agente coordinador."""
        print("[Workflow] Ejecutando coordinador...")

        plan = self.coordinator.get_execution_plan(
            state["user_prompt"],
            state["has_image"]
        )

        state["needs_vision"] = plan["vision"]
        state["needs_code_analysis"] = plan["code_analysis"]
        state["needs_code_generation"] = plan["code_generation"]
        state["direct_response"] = plan["direct_response"]
        state["execution_path"] = plan.get("execution_sequence", [])
        state["coordinator_reasoning"] = plan.get("reasoning", "")  # Guardar razonamiento

        print(f"[Workflow] Plan: {state['execution_path']}")
        print(f"[Workflow] Razonamiento: {plan.get('reasoning', 'N/A')}")

        return state

    def _route_from_coordinator(self, state: AgentState) -> str:
        """Decide la siguiente ruta desde el coordinador."""
        if state["direct_response"]:
            return "direct"
        elif state["needs_vision"]:
            return "vision"
        elif state["needs_code_analysis"]:
            # El análisis de código debe ir primero; si también necesita generación,
            # el flujo irá a editor después del análisis vía _route_from_code
            return "code"
        elif state["needs_code_generation"]:
            # Solo generación sin análisis previo (raro pero posible)
            return "editor"
        else:
            return "direct"

    def _run_vision(self, state: AgentState) -> AgentState:
        """Ejecuta el agente de visión."""
        image_b64 = state.get("image_b64")
        image_path = state.get("image_path")
        image_paths = state.get("image_paths")
        
        # Si hay múltiples imágenes, usar análisis en batch (más eficiente)
        if image_paths and len(image_paths) > 1:
            print(f"[Workflow] Ejecutando agente de visión en BATCH sobre {len(image_paths)} imágenes...")
            print(f"[Workflow] Estrategia: recortar todas primero, luego analizar")
            
            # Usar método batch
            # Si las imágenes ya están recortadas (de captures/), no recortar de nuevo
            should_crop = not state.get("images_already_cropped", False)
            
            result = self.vision.analyze_images_batch(
                image_paths=image_paths,
                context_prompt=state["user_prompt"],
                auto_crop=should_crop  # Solo recortar si son imágenes nuevas
            )
            
            if result.get("success"):
                batch_info = result.get("batch_info", {})
                state["vision_analysis"] = result.get("analysis", "")
                state["tokens_used"]["vision"] = batch_info.get("total_image_tokens", 0)
                
                # Log de estadísticas
                print(f"[Workflow] Visión: BATCH completado")
                print(f"  - Imágenes procesadas: {batch_info.get('processed_images', 0)}")
                print(f"  - Imágenes recortadas: {batch_info.get('cropped_images', 0)}")
                print(f"  - Ahorro de tokens: ~{batch_info.get('total_token_savings', 0)}")
                print(f"  - Tokens usados: {batch_info.get('total_image_tokens', 0)}")
            else:
                state["vision_analysis"] = f"Error en análisis batch: {result.get('error', 'desconocido')}"
                state["tokens_used"]["vision"] = 0
                print(f"[Workflow] Visión: ERROR en batch - {result.get('error')}")
            
        elif image_b64 or image_path:
            # Análisis de imagen única (retrocompatibilidad)
            print("[Workflow] Ejecutando agente de visión (imagen única)...")
            
            result = self.vision.analyze_image(
                image_b64=image_b64,
                image_path=image_path,
                context_prompt=state["user_prompt"]
            )
            
            state["vision_analysis"] = result.get("analysis", "")
            
            # Trackear tokens
            tokens = result.get("tokens_used", 0)
            if isinstance(tokens, dict):
                state["tokens_used"]["vision"] = tokens.get("image", 0) + tokens.get("output", 0)
            else:
                state["tokens_used"]["vision"] = 0 if result.get("from_cache") else 1000
            
            if result.get("from_cache"):
                print("[Workflow] Visión: usando resultado en caché")
            else:
                print(f"[Workflow] Visión: análisis completado ({state['tokens_used']['vision']} tokens)")
        else:
            state["vision_analysis"] = None
            
        return state

    def _route_from_vision(self, state: AgentState) -> str:
        """Decide la ruta después del análisis de visión."""
        if state["needs_code_analysis"]:
            return "code"
        if state["needs_code_generation"]:
            return "editor"
        return "finalize"

    def _run_code(self, state: AgentState) -> AgentState:
        """Ejecuta el agente de código (análisis)."""
        if not state["target_file"] or not state["file_content"]:
            state["code_analysis"] = {"error": "No hay archivo para analizar"}
            return state

        print("[Workflow] Ejecutando agente de código...")

        # Verificar si necesita búsqueda web
        web_search_results = None
        if web_search.should_search(state["user_prompt"]):
            print("[Workflow] Detectada posible necesidad de búsqueda web...")
            # Buscar información sobre el error/tema
            search_query = f"python {state['user_prompt'][:100]}"
            results = web_search.search(search_query, domain='stackoverflow', max_results=3)
            if results:
                web_search_results = web_search.format_for_prompt(results)
                print(f"[Workflow] Web search: {len(results)} resultados encontrados")
                state["tokens_used"]["web_search"] = len(web_search_results) // 4

        # Análisis con detección de cascada si hay múltiples archivos disponibles
        all_files = state.get("all_files_content")
        project_path = state.get("project_path")
        
        if all_files and len(all_files) > 1:
            print("[Workflow] Detectados múltiples archivos - análisis con detección de cascada...")
            cascade_result = self.code.analyze_with_cascade_detection(
                target_file=state["target_file"],
                target_content=state["file_content"],
                all_files=all_files,
                user_request=state["user_prompt"],
                project_path=project_path or "."
            )
            
            state["cascade_analysis"] = cascade_result
            
            if cascade_result.get("cascade_required"):
                dep_files = cascade_result.get("dependent_files", [])
                print(f"[Workflow] ⚠️ Cambios en cascada detectados: {len(dep_files)} archivos dependientes")
                for dep in dep_files:
                    print(f"  - {dep['file']}")
            else:
                print("[Workflow] No se detectaron dependencias que requieran cambios en cascada")
            
            # Usar el análisis base para compatibilidad
            result = cascade_result.get("base_analysis", {})
        else:
            # Análisis simple (un solo archivo)
            result = self.code.analyze_modification_request(
                filename=state["target_file"],
                code_content=state["file_content"],
                user_request=state["user_prompt"],
                image_analysis=state.get("vision_analysis")
            )

        state["code_analysis"] = result
        state["web_search_results"] = web_search_results
        state["tokens_used"]["code"] = len(str(result).split()) // 4 if not result.get("from_cache") else 0

        if result.get("from_cache"):
            print("[Workflow] Código: usando análisis en caché")
        else:
            print(f"[Workflow] Código: análisis completado")

        return state

    def _route_from_code(self, state: AgentState) -> str:
        """Decide la ruta después del análisis de código."""
        code_analysis = state.get("code_analysis") or {}
        if state["needs_code_generation"] and code_analysis.get("success", False):
            return "editor"
        return "finalize"

    def _run_editor(self, state: AgentState) -> AgentState:
        """Ejecuta el agente editor (generación de código)."""
        target_file = state.get("target_file")
        file_content = state.get("file_content")
        cascade_analysis = state.get("cascade_analysis")
        
        print(f"[Workflow] Editor input - target_file: {target_file}")
        print(f"[Workflow] Editor input - file_content length: {len(file_content) if file_content else 0}")
        
        if not target_file or not file_content:
            state["editor_result"] = {"error": "No hay archivo para editar", "success": False}
            print(f"[Workflow] Editor: error - falta archivo o contenido")
            return state

        # Verificar si hay cambios en cascada requeridos
        if cascade_analysis and cascade_analysis.get("cascade_required"):
            print("[Workflow] Ejecutando editor en modo multi-archivo (cascada)...")
            
            # Preparar contenido de todos los archivos a modificar
            all_files = state.get("all_files_content", {})
            files_to_modify = cascade_analysis.get("all_files_to_modify", [target_file])
            
            # Filtrar solo archivos que tenemos contenido
            files_content = {}
            for fname in files_to_modify:
                if fname in all_files:
                    files_content[fname] = all_files[fname]
                elif fname == target_file:
                    files_content[fname] = file_content
            
            # Preparar análisis combinado
            analysis_parts = []
            if state.get("code_analysis"):
                analysis_parts.append(state["code_analysis"].get("analysis", ""))
            if state.get("web_search_results"):
                analysis_parts.append(state["web_search_results"])
            code_analysis_combined = "\n\n".join(analysis_parts)
            
            # Generar cambios multi-archivo
            result = self.editor.generate_multi_file_changes(
                files_content=files_content,
                user_request=state["user_prompt"],
                cascade_analysis=cascade_analysis,
                code_analysis=code_analysis_combined
            )
            
            state["editor_result"] = result
            
            if result.get("success"):
                successful = result.get("successful_files", 0)
                total = result.get("total_files", 0)
                print(f"[Workflow] Editor: {successful}/{total} archivos modificados en cascada")
            else:
                print(f"[Workflow] Editor: error en cambios en cascada - {result.get('error', 'desconocido')}")
            
            state["tokens_used"]["editor"] = len(str(result.get("full_response", "")).split()) // 4
            return state
        
        # Modo simple: un solo archivo
        print("[Workflow] Ejecutando agente editor (modo simple)...")

        # Preparar análisis combinado (código + web search)
        analysis_parts = []
        if state.get("code_analysis"):
            analysis_parts.append(state["code_analysis"].get("analysis", ""))
        if state.get("web_search_results"):
            analysis_parts.append(state["web_search_results"])

        code_analysis_combined = "\n\n".join(analysis_parts)

        result = self.editor.generate_modified_code(
            filename=state["target_file"],
            original_code=state["file_content"],
            user_request=state["user_prompt"],
            code_analysis=code_analysis_combined,
            image_analysis=state.get("vision_analysis")
        )

        state["editor_result"] = result
        
        # Defensive check for None result
        if result is None:
            state["editor_result"] = {"error": "Editor returned None", "success": False}
            state["tokens_used"]["editor"] = 0
            print(f"[Workflow] Editor: error - retornó None")
            return state
        
        state["tokens_used"]["editor"] = len(result.get("full_response", "").split()) // 4

        if result.get("success"):
            print(f"[Workflow] Editor: código generado ({result['validation']['new_lines']} líneas)")
        else:
            print(f"[Workflow] Editor: error - {result.get('error', 'desconocido')}")

        return state

    def _route_from_editor(self, state: AgentState) -> str:
        """Decide si el código generado necesita revisión."""
        editor_result = state.get("editor_result")

        # Si el editor falló, no hay nada que revisar
        if not editor_result or not editor_result.get("success"):
            return "finalize"

        # Si no hay código generado, finalizar
        if not editor_result.get("code"):
            return "finalize"

        # Siempre revisar el código generado
        return "reviewer"

    def _run_reviewer(self, state: AgentState) -> AgentState:
        """Ejecuta el agente revisor para verificar el código generado."""
        editor_result = state.get("editor_result")

        if not editor_result or not editor_result.get("success"):
            state["review_result"] = {"approved": False, "error": "No hay código para revisar"}
            return state

        print("[Workflow] Ejecutando agente revisor...")

        # Incrementar contador de iteraciones
        state["review_iterations"] = state.get("review_iterations", 0) + 1

        result = self.reviewer.review_code(
            filename=state["target_file"],
            original_code=state["file_content"],
            generated_code=editor_result["code"],
            user_request=state["user_prompt"],
            code_analysis=(state.get("code_analysis") or {}).get("analysis", "")
        )

        state["review_result"] = result
        state["tokens_used"]["reviewer"] = len(str(result).split()) // 4

        if result.get("approved"):
            print(f"[Workflow] Revisor: código aprobado ({len(result.get('issues', []))} observaciones menores)")
        else:
            issue_count = len([i for i in result.get("issues", []) if i.get("severity") == "error"])
            print(f"[Workflow] Revisor: código rechazado - {issue_count} errores encontrados")

        return state

    def _route_from_reviewer(self, state: AgentState) -> str:
        """Decide la ruta después de la revisión."""
        review_result = state.get("review_result")
        iterations = state.get("review_iterations", 0)

        # Si está aprobado, finalizar
        if review_result and review_result.get("approved"):
            return "finalize"

        # Si hay código corregido por el revisor y no excedimos iteraciones, volver a editor
        if review_result and review_result.get("corrected_code") and iterations < 2:
            # Actualizar el editor_result con el código corregido
            state["editor_result"]["code"] = review_result["corrected_code"]
            print(f"[Workflow] Aplicando correcciones del revisor (iteración {iterations})")
            return "editor"

        # Si el error es solo de parseo (no de código), permitir que el usuario revise manualmente
        if review_result and review_result.get("issues"):
            issues = review_result.get("issues", [])
            # Filtrar solo errores de parseo vs errores reales de código
            parse_errors = [i for i in issues if i.get("type") == "parse_error"]
            real_errors = [i for i in issues if i.get("type") not in ["parse_error", "info"]]

            # Si solo hay errores de parseo, aprobar para revisión manual
            if parse_errors and not real_errors:
                print("[Workflow] Revisor: error de parseo, código aprobado para revisión manual")
                # Marcar como aprobado para que se muestre al usuario
                review_result["approved"] = True
                review_result["needs_manual_review"] = True
                return "finalize"

        # Si hay errores reales pero no tenemos corrección, o excedimos iteraciones, finalizar igual
        if iterations >= 2:
            print("[Workflow] Máximo de revisiones alcanzado, finalizando...")

        return "finalize"

    def _run_direct_response(self, state: AgentState) -> AgentState:
        """Genera una respuesta directa sin agentes especializados."""
        # Usar el propio LLM del coordinador para respuestas simples
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content="Eres un asistente útil. Responde de manera concisa."),
            HumanMessage(content=state["user_prompt"])
        ]

        try:
            response = self.coordinator.llm.invoke(messages)
            state["final_response"] = response.content
            state["tokens_used"]["direct"] = len(response.content.split()) // 4
        except Exception as e:
            state["final_response"] = f"Error generando respuesta: {e}"

        return state

    def _finalize(self, state: AgentState) -> AgentState:
        """Finaliza el workflow y prepara la respuesta."""
        print("[Workflow] Finalizando...")

        editor_result = state.get("editor_result")
        
        # Caso 1: Editor generó código multi-archivo (cascada)
        if editor_result and editor_result.get("success") and "results" in editor_result:
            results = editor_result["results"]
            successful_files = [f for f, r in results.items() if r.get("success")]
            total_files = len(results)
            
            state["final_response"] = f"Cambios generados para {len(successful_files)}/{total_files} archivos"
            state["success"] = len(successful_files) > 0
            
            # Log detallado
            print(f"[Workflow] Multi-file: {len(successful_files)} archivos exitosos")
            for fname, result in results.items():
                status = "✅" if result.get("success") else "❌"
                print(f"  {status} {fname}: {result.get('error', 'OK')}")

        # Caso 2: Editor generó código simple
        elif editor_result and editor_result.get("success"):
            code = editor_result["code"]
            mod_id = editor_result.get("modification_id", "")
            state["final_response"] = f"Código modificado generado (ID: {mod_id[:8]}...)"
            state["success"] = True

        # Caso 3: Hay análisis de visión pero no código
        elif state.get("vision_analysis") and not editor_result:
            state["final_response"] = f"Análisis de imagen:\n{state['vision_analysis'][:500]}..."
            state["success"] = True

        # Caso 4: Hay análisis de código pero no editor
        elif state.get("code_analysis") and not editor_result:
            analysis = state["code_analysis"].get("analysis", "")
            state["final_response"] = f"Análisis:\n{analysis[:500]}..."
            state["success"] = True

        # Caso 5: Hay error
        elif state.get("error"):
            state["success"] = False

        # Total de tokens
        total = sum(state["tokens_used"].values())
        print(f"[Workflow] Total tokens estimados: {total}")
        print(f"[Workflow] Camino de ejecución: {' -> '.join(state['execution_path'])}")

        return state

    def run(
        self,
        prompt: str,
        image_b64: Optional[str] = None,
        image_path: Optional[str] = None,
        image_paths: Optional[list] = None,
        images_already_cropped: bool = False,
        target_file: Optional[str] = None,
        file_content: Optional[str] = None,
        all_files_content: Optional[Dict[str, str]] = None,
        project_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ejecuta el workflow completo.

        Args:
            prompt: Solicitud del usuario
            image_b64: Imagen en base64 (opcional)
            image_path: Ruta a imagen guardada (opcional)
            image_paths: Lista de rutas a imágenes (opcional)
            images_already_cropped: Si True, las imágenes ya están recortadas (no recortar de nuevo)
            target_file: Archivo objetivo para modificaciones (opcional)
            file_content: Contenido del archivo objetivo (opcional)
            all_files_content: Dict de todos los archivos del proyecto para análisis de cascada (opcional)
            project_path: Ruta del proyecto (opcional)

        Returns:
            Dict con el resultado final
        """
        # Estado inicial
        has_image = (
            (image_b64 is not None and len(image_b64) > 100) or 
            (image_path is not None) or
            (image_paths is not None and len(image_paths) > 0)
        )
        
        has_multi_file = all_files_content is not None and len(all_files_content) > 1
        
        print(f"[Workflow] run() called with target_file={target_file}, file_content_length={len(file_content) if file_content else 0}, has_image={has_image}, multi_file={has_multi_file}")
        
        initial_state: AgentState = {
            "user_prompt": prompt,
            "image_b64": image_b64,
            "image_path": image_path,
            "image_paths": image_paths,
            "has_image": has_image,
            "needs_vision": False,
            "needs_code_analysis": False,
            "needs_code_generation": False,
            "direct_response": False,
            "coordinator_reasoning": None,
            "vision_analysis": None,
            "code_analysis": None,
            "editor_result": None,
            "review_result": None,
            "review_iterations": 0,
            "web_search_results": None,
            "cascade_analysis": None,
            "all_files_content": all_files_content,
            "project_path": project_path,
            "images_already_cropped": images_already_cropped,
            "target_file": target_file,
            "file_content": file_content,
            "final_response": "",
            "success": False,
            "error": None,
            "tokens_used": {},
            "execution_path": []
        }

        # Ejecutar el grafo
        print("\n" + "="*60)
        print("🔄 INICIANDO WORKFLOW MULTI-AGENTE")
        print("="*60)

        final_state = self.app.invoke(initial_state)

        print("="*60)
        print("✅ WORKFLOW COMPLETADO")
        print("="*60 + "\n")

        return {
            "response": final_state["final_response"],
            "success": final_state["success"],
            "vision_analysis": final_state.get("vision_analysis"),
            "code_analysis": final_state.get("code_analysis"),
            "editor_result": final_state.get("editor_result"),
            "execution_path": final_state["execution_path"],
            "tokens_used": final_state["tokens_used"],
            "error": final_state.get("error")
        }
