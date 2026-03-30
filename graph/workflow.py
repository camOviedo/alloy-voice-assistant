"""
Workflow de LangGraph - grafo de flujo de trabajo multi-agente.
"""
from typing import Dict, Any, Optional, TypedDict

from langgraph.graph import StateGraph, END

from agents.coordinator import CoordinatorAgent
from agents.vision import VisionAgent
from agents.code import CodeAgent
from agents.editor import EditorAgent


class AgentState(TypedDict):
    """Estado compartido entre los agentes en el grafo."""
    # Input del usuario
    user_prompt: str
    image_b64: Optional[str]
    image_path: Optional[str]
    has_image: bool

    # Decisiones del coordinador
    needs_vision: bool
    needs_code_analysis: bool
    needs_code_generation: bool
    direct_response: bool

    # Resultados de agentes
    vision_analysis: Optional[str]
    code_analysis: Optional[Dict[str, Any]]
    editor_result: Optional[Dict[str, Any]]

    # Contexto de archivos
    target_file: Optional[str]
    file_content: Optional[str]

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
        code_model: str = "qwen3-vl:8b-rapido",
        editor_model: str = "qwen3-vl:8b-editor",
        memory_dir: str = None
    ):
        """
        Inicializa el workflow con todos los agentes.

        Args:
            coordinator_model: Modelo para el coordinador
            vision_model: Modelo para el agente de visión
            code_model: Modelo para el agente de código
            editor_model: Modelo para el agente editor
            memory_dir: Directorio para memoria persistente
        """
        # Inicializar agentes
        self.coordinator = CoordinatorAgent(coordinator_model)
        self.vision = VisionAgent(vision_model, memory_dir)
        self.code = CodeAgent(code_model, memory_dir)
        self.editor = EditorAgent(editor_model, memory_dir)

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

        # Después de visión, siempre va a code si necesita código, o directo a finalizar
        workflow.add_conditional_edges(
            "vision",
            self._route_from_vision,
            {
                "code": "code",
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

        # Editor siempre va a finalizar
        workflow.add_edge("editor", "finalize")

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

        print(f"[Workflow] Plan: {state['execution_path']}")
        print(f"[Workflow] Razonamiento: {plan.get('reasoning', 'N/A')}")

        return state

    def _route_from_coordinator(self, state: AgentState) -> str:
        """Decide la siguiente ruta desde el coordinador."""
        if state["direct_response"]:
            return "direct"
        elif state["needs_vision"]:
            return "vision"
        elif state["needs_code_generation"]:
            return "editor"
        elif state["needs_code_analysis"]:
            return "code"
        else:
            return "direct"

    def _run_vision(self, state: AgentState) -> AgentState:
        """Ejecuta el agente de visión."""
        image_b64 = state.get("image_b64")
        image_path = state.get("image_path")
        
        if not image_b64 and not image_path:
            state["vision_analysis"] = None
            return state

        print("[Workflow] Ejecutando agente de visión...")

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

        return state

    def _route_from_vision(self, state: AgentState) -> str:
        """Decide la ruta después del análisis de visión."""
        if state["needs_code_analysis"]:
            return "code"
        return "finalize"

    def _run_code(self, state: AgentState) -> AgentState:
        """Ejecuta el agente de código (análisis)."""
        if not state["target_file"] or not state["file_content"]:
            state["code_analysis"] = {"error": "No hay archivo para analizar"}
            return state

        print("[Workflow] Ejecutando agente de código...")

        result = self.code.analyze_modification_request(
            filename=state["target_file"],
            code_content=state["file_content"],
            user_request=state["user_prompt"],
            image_analysis=state.get("vision_analysis")
        )

        state["code_analysis"] = result
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
        if not state["target_file"] or not state["file_content"]:
            state["editor_result"] = {"error": "No hay archivo para editar", "success": False}
            return state

        print("[Workflow] Ejecutando agente editor...")

        result = self.editor.generate_modified_code(
            filename=state["target_file"],
            original_code=state["file_content"],
            user_request=state["user_prompt"],
            code_analysis=(state.get("code_analysis") or {}).get("analysis", ""),
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

        # Si hay resultado del editor, usarlo
        if state.get("editor_result") and state["editor_result"].get("success"):
            code = state["editor_result"]["code"]
            mod_id = state["editor_result"].get("modification_id", "")
            state["final_response"] = f"Código modificado generado (ID: {mod_id[:8]}...)"
            state["success"] = True

        # Si hay análisis de visión pero no código
        elif state.get("vision_analysis") and not state.get("editor_result"):
            state["final_response"] = f"Análisis de imagen:\n{state['vision_analysis'][:500]}..."
            state["success"] = True

        # Si hay análisis de código pero no editor
        elif state.get("code_analysis") and not state.get("editor_result"):
            analysis = state["code_analysis"].get("analysis", "")
            state["final_response"] = f"Análisis:\n{analysis[:500]}..."
            state["success"] = True

        # Si hay error
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
        target_file: Optional[str] = None,
        file_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ejecuta el workflow completo.

        Args:
            prompt: Solicitud del usuario
            image_b64: Imagen en base64 (opcional)
            image_path: Ruta a imagen guardada (opcional)
            target_file: Archivo objetivo para modificaciones (opcional)
            file_content: Contenido del archivo objetivo (opcional)

        Returns:
            Dict con el resultado final
        """
        # Estado inicial
        initial_state: AgentState = {
            "user_prompt": prompt,
            "image_b64": image_b64,
            "image_path": image_path,
            "has_image": (image_b64 is not None and len(image_b64) > 100) or (image_path is not None),
            "needs_vision": False,
            "needs_code_analysis": False,
            "needs_code_generation": False,
            "direct_response": False,
            "vision_analysis": None,
            "code_analysis": None,
            "editor_result": None,
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
