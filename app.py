"""
Interfaz Chainlit para el Asistente Multi-Agente
Reemplaza la terminal con una UI web moderna
"""
import asyncio
import os
import time
from typing import Dict, Any, Optional, List

import chainlit as cl
from chainlit.types import ThreadDict

from config import (
    DEFAULT_MODEL,
    PROJECT_PATH,
    SUGGESTIONS_PATH,
    CAPTURES_PATH,
    AGENT_COORDINATOR_MODEL,
    AGENT_VISION_MODEL,
    AGENT_CODE_MODEL,
    AGENT_EDITOR_MODEL,
    DEFAULT_MONITOR,
    DEFAULT_SCALE_DISPLAY,
    DEFAULT_MAX_WIDTH,
    DEFAULT_JPEG_QUALITY,
    VISION_CAPTURE_DURATION,
    VISION_CAPTURE_FPS,
)
from assistant_core import Assistant
from file_manager import ProjectFileManager
from screen_capture import ScreenStream
from graph.workflow import AgentWorkflow


# Variables globales para la sesión
assistant: Optional[Assistant] = None
file_manager: Optional[ProjectFileManager] = None
screen_stream: Optional[ScreenStream] = None
vision_active = False


@cl.on_chat_start
async def on_chat_start():
    """Inicializa la sesión del asistente"""
    global assistant, file_manager

    # Inicializar asistente
    assistant = Assistant(
        model_name=DEFAULT_MODEL,
        language="es",
        project_path=PROJECT_PATH,
        vision_timeout=5
    )

    # Inicializar file manager
    file_manager = ProjectFileManager(PROJECT_PATH)

    # Guardar en user session
    cl.user_session.set("assistant", assistant)
    cl.user_session.set("file_manager", file_manager)
    cl.user_session.set("vision_active", False)
    cl.user_session.set("captures", [])

    # Mensaje de bienvenida
    welcome_msg = f"""🤖 **Asistente de Código con Visión - Modo Web**

📁 Proyecto: `{PROJECT_PATH}`

**Comandos disponibles:**
• Escribe cualquier prompt para interactuar
• "muestra archivo.py" - Ver contenido
• "lista archivos" - Listar archivos Python
• "modifica archivo.py para..." - Proponer cambio
• "cambios pendientes" - Ver cambios propuestos

**Modo Visión:**
Escribe prompts con palabras como "pantalla", "imagen", "mira" para activar el análisis visual.

**Nota:** Las capturas de pantalla se realizan localmente en tu máquina (privado).
"""

    await cl.Message(content=welcome_msg).send()

    # Mostrar acciones rápidas
    actions = [
        cl.Action(name="list_files", label="📁 Archivos", payload={"value": "list_files"}),
        cl.Action(name="pending_changes", label="⏳ Pendientes", payload={"value": "pending_changes"}),
    ]
    await cl.Message(
        content="**Acciones rápidas:**",
        actions=actions
    ).send()


@cl.action_callback("list_files")
async def on_list_files(action):
    """Lista archivos del proyecto"""
    fm = cl.user_session.get("file_manager")
    files, error = fm.list_python_files()

    if error:
        await cl.Message(content=f"❌ {error}").send()
        return

    if not files:
        await cl.Message(content="📂 No hay archivos Python en el proyecto.").send()
        return

    # Crear botones para cada archivo
    actions = [
        cl.Action(name=f"view_file", label=f"📄 {f}", payload={"value": f})
        for f in files[:10]
    ]

    file_list = "\n".join([f"• `{f}`" for f in files])
    await cl.Message(
        content=f"📁 **Archivos Python ({len(files)} total):**\n\n{file_list}",
        actions=actions
    ).send()


@cl.action_callback("view_file")
async def on_view_file(action):
    """Muestra contenido de un archivo"""
    filename = action.payload.get("value")
    fm = cl.user_session.get("file_manager")
    content, error = fm.read_file(filename)

    if error:
        await cl.Message(content=f"❌ {error}").send()
        return

    # Mostrar en bloque de código
    await cl.Message(
        content=f"📄 **{filename}** ({len(content)} caracteres)\n\n```python\n{content[:2000]}\n```"
    ).send()

    if len(content) > 2000:
        await cl.Message(content=f"... *(archivo truncado, mostrando 2000/{len(content)} caracteres)*").send()


@cl.action_callback("pending_changes")
async def on_pending_changes(action):
    """Muestra cambios pendientes"""
    fm = cl.user_session.get("file_manager")
    pending = fm.get_pending_changes()

    if not pending:
        await cl.Message(content="✅ **No hay cambios pendientes.**").send()
        return

    for change_id, change in pending.items():
        diff = fm.generate_diff(change_id)
        msg_content = f"""⏳ **Cambio: `{change_id}`**

📄 Archivo: `{change['file']}`
📝 {change.get('description', 'Sin descripción')[:100]}

```diff
{diff[:800] if diff else 'No hay diff disponible'}
```
"""
        actions = [
            cl.Action(name="approve_change", label="✅ Aprobar", payload={"value": change_id}),
            cl.Action(name="reject_change", label="❌ Rechazar", payload={"value": change_id}),
        ]
        await cl.Message(content=msg_content, actions=actions).send()


@cl.action_callback("approve_change")
async def on_approve_change(action):
    """Aprueba un cambio"""
    change_id = action.payload.get("value")
    fm = cl.user_session.get("file_manager")
    success, result = fm.approve_change(change_id)

    if success:
        await cl.Message(content=f"✅ **{result}**").send()
    else:
        await cl.Message(content=f"❌ **{result}**").send()


@cl.action_callback("reject_change")
async def on_reject_change(action):
    """Rechaza un cambio"""
    change_id = action.payload.get("value")
    fm = cl.user_session.get("file_manager")
    success, result = fm.reject_change(change_id)

    if success:
        await cl.Message(content=f"🗑️ **{result}**").send()
    else:
        await cl.Message(content=f"❌ **{result}**").send()


def get_existing_capture_images():
    """Obtiene lista de imágenes existentes en la carpeta captures/"""
    try:
        if not os.path.exists(CAPTURES_PATH):
            return []

        # Extensiones de imagen soportadas
        valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')

        images = []
        for file in sorted(os.listdir(CAPTURES_PATH)):
            if file.lower().endswith(valid_extensions):
                full_path = os.path.join(CAPTURES_PATH, file)
                images.append(full_path)

        return images
    except Exception as e:
        print(f"⚠️ Error buscando imágenes existentes: {e}")
        return []


@cl.action_callback("use_existing_images")
async def on_use_existing_images(action):
    """Carga imágenes existentes de la carpeta captures/"""
    pending = cl.user_session.get("pending_prompt", "")

    # Obtener imágenes existentes
    existing_images = get_existing_capture_images()

    if not existing_images:
        await cl.Message(content="❌ No hay imágenes en la carpeta captures/").send()
        return

    # Guardar en sesión
    cl.user_session.set("captures", existing_images)

    # Mostrar imágenes cargadas
    await cl.Message(content=f"✅ **{len(existing_images)} imágenes cargadas desde captures/**").send()

    elements = []
    for path in existing_images[:4]:  # Mostrar máximo 4
        if os.path.exists(path):
            elements.append(cl.Image(name=os.path.basename(path), path=path, display="inline"))

    if elements:
        await cl.Message(
            content="📸 **Imágenes a analizar:**",
            elements=elements
        ).send()

    # Procesar el prompt pendiente
    if pending:
        await cl.Message(content="▶️ **Continuando con el análisis...**").send()
        await process_prompt(pending)
        cl.user_session.set("pending_prompt", "")


@cl.action_callback("capture_screen")
async def on_capture_screen(action):
    """Activa captura de pantalla local"""
    await capture_screens()


async def capture_screens():
    """Realiza captura de pantalla local usando ScreenStream"""
    global screen_stream

    actions_msg = await cl.Message(content="📷 **Iniciando captura de pantalla local...**").send()

    try:
        # Calcular frames
        num_frames = int(VISION_CAPTURE_DURATION * VISION_CAPTURE_FPS)
        delay_between_frames = VISION_CAPTURE_DURATION / max(num_frames - 1, 1) if num_frames > 1 else 0

        # Iniciar stream
        screen_stream = ScreenStream(
            monitor=DEFAULT_MONITOR,
            scale_display=DEFAULT_SCALE_DISPLAY,
            max_width=DEFAULT_MAX_WIDTH,
            jpeg_quality=DEFAULT_JPEG_QUALITY
        ).start()

        if not screen_stream.running:
            await cl.Message(content="❌ **No se pudo iniciar la captura de pantalla**").send()
            return []

        # Esperar a que tenga frames
        wait_time = 0
        while screen_stream.frame is None and screen_stream.running and wait_time < 3:
            await asyncio.sleep(0.1)
            wait_time += 0.1

        if screen_stream.frame is None:
            await cl.Message(content="❌ **No se pudo obtener frames**").send()
            screen_stream.stop()
            return []

        # Capturar frames
        image_paths = []
        for i in range(num_frames):
            b64, path = screen_stream.capture_and_save_frame()
            if path:
                image_paths.append(path)
            if i < num_frames - 1:
                await asyncio.sleep(delay_between_frames)

        # Detener stream
        screen_stream.stop()

        # Actualizar sesión
        cl.user_session.set("captures", image_paths)

        # Mostrar imágenes capturadas
        await actions_msg.update(content=f"✅ **{len(image_paths)} imágenes capturadas**")

        elements = []
        for path in image_paths[:4]:  # Mostrar máximo 4
            if os.path.exists(path):
                elements.append(cl.Image(name=os.path.basename(path), path=path, display="inline"))

        if elements:
            await cl.Message(
                content="📸 **Imágenes capturadas:**",
                elements=elements
            ).send()

        return image_paths

    except Exception as e:
        await cl.Message(content=f"❌ **Error en captura:** {e}").send()
        if screen_stream:
            screen_stream.stop()
        return []


def needs_vision_analysis(prompt: str) -> bool:
    """Detecta si el prompt requiere análisis de imagen"""
    prompt_lower = prompt.lower()
    vision_keywords = [
        "pantalla", "imagen", "foto", "captura", "ventana", "interfaz",
        "video", "stream", "cámara", "webcam", "monitor", "escritorio",
        "muestra", "muéstrame", "ver pantalla", "que ves", "qué ves",
        "analiza la imagen", "describe la imagen", "en la pantalla",
        "error en pantalla", "lo que ves", "screenshot", "screen",
        "image", "picture", "window", "interface", "desktop",
        "what do you see", "analyze image", "show me", "look at"
    ]
    return any(kw in prompt_lower for kw in vision_keywords)


def extract_filename(prompt: str) -> Optional[str]:
    """Extrae nombre de archivo del prompt"""
    import re
    patterns = [
        r'(\w+\.py)',
        r'archivo\s+(\S+\.py)',
        r'file\s+(\S+\.py)'
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


@cl.on_message
async def on_message(message: cl.Message):
    """Procesa mensajes del usuario"""
    prompt = message.content
    assistant = cl.user_session.get("assistant")
    fm = cl.user_session.get("file_manager")

    # Detectar si necesita visión
    needs_vision = needs_vision_analysis(prompt)

    # Si necesita visión y no hay capturas, preguntar por fuente
    if needs_vision:
        captures = cl.user_session.get("captures", [])
        if not captures:
            # Verificar si hay imágenes existentes en captures/
            existing_images = get_existing_capture_images()

            if existing_images:
                # Hay imágenes existentes - ofrecer opciones
                actions = [
                    cl.Action(name="use_existing_images", label=f"📂 Usar {len(existing_images)} imágenes existentes", payload={"value": "existing"}),
                    cl.Action(name="capture_screen", label="📷 Capturar pantalla nueva", payload={"value": "capture"}),
                    cl.Action(name="continue_without_capture", label="📝 Continuar sin imagen", payload={"value": "skip"})
                ]
                await cl.Message(
                    content=f"👁️ **Detecté que tu prompt podría necesitar análisis visual.**\n\n📁 Encontré **{len(existing_images)}** imágenes en la carpeta `captures/`\n\n¿Qué deseas hacer?",
                    actions=actions
                ).send()
            else:
                # No hay imágenes existentes
                actions = [
                    cl.Action(name="capture_screen", label="📷 Capturar Pantalla", payload={"value": "capture"}),
                    cl.Action(name="continue_without_capture", label="📝 Continuar sin imagen", payload={"value": "skip"})
                ]
                await cl.Message(
                    content="👁️ **Detecté que tu prompt podría necesitar análisis visual.**\n\n¿Deseas capturar la pantalla primero?",
                    actions=actions
                ).send()
            # Guardar el prompt para procesarlo después
            cl.user_session.set("pending_prompt", prompt)
            return

    # Procesar el prompt
    await process_prompt(prompt)


@cl.action_callback("select_file_for_edit")
async def on_select_file_for_edit(action):
    """Selecciona archivo para editar"""
    filename = action.payload.get("filename")
    prompt = action.payload.get("prompt")
    if filename and prompt:
        await cl.Message(content=f"✏️ **Archivo seleccionado:** `{filename}`").send()
        await process_with_workflow(prompt, force_filename=filename)


@cl.action_callback("continue_without_capture")
async def on_continue_without_capture(action):
    """Continúa sin captura de pantalla"""
    pending = cl.user_session.get("pending_prompt", "")
    if pending:
        await process_prompt(pending)
        cl.user_session.set("pending_prompt", "")


async def process_prompt(prompt: str):
    """Procesa el prompt del usuario"""
    assistant = cl.user_session.get("assistant")
    fm = cl.user_session.get("file_manager")

    # Verificar comandos específicos
    prompt_lower = prompt.lower()

    # Comando: lista archivos
    if any(cmd in prompt_lower for cmd in ["lista archivos", "list files", "archivos python"]):
        files, error = fm.list_python_files()
        if error:
            await cl.Message(content=f"❌ {error}").send()
        else:
            file_list = "\n".join([f"• `{f}`" for f in files])
            await cl.Message(content=f"📁 **Archivos Python ({len(files)}):**\n\n{file_list}").send()
        return

    # Comando: muestra archivo
    if any(cmd in prompt_lower for cmd in ["muestra", "muéstrame", "ver archivo"]):
        filename = extract_filename(prompt)
        if filename:
            content, error = fm.read_file(filename)
            if error:
                await cl.Message(content=f"❌ {error}").send()
            else:
                await cl.Message(
                    content=f"📄 **{filename}**\n\n```python\n{content[:2000]}\n```"
                ).send()
            return

    # Comando: cambios pendientes
    if any(cmd in prompt_lower for cmd in ["pendientes", "cambios pendientes", "pending"]):
        pending = fm.get_pending_changes()
        if not pending:
            await cl.Message(content="✅ No hay cambios pendientes.").send()
        else:
            for change_id, change in pending.items():
                await cl.Message(
                    content=f"⏳ `{change_id}`: `{change['file']}` - {change.get('description', 'Sin descripción')[:50]}..."
                ).send()
        return

    # Comando: aprobar cambio
    if any(cmd in prompt_lower for cmd in ["aprueba", "aprobar", "approve"]):
        import re
        match = re.search(r'cambio[_\s]*(\w+)|change[_\s]*(\w+)|#?(\d+)', prompt_lower)
        if match:
            change_id = match.group(1) or match.group(2) or match.group(3)
            pending = fm.get_pending_changes()
            full_id = None
            for pid in pending.keys():
                if change_id in pid:
                    full_id = pid
                    break
            if full_id:
                success, result = fm.approve_change(full_id)
                if success:
                    await cl.Message(content=f"✅ {result}").send()
                else:
                    await cl.Message(content=f"❌ {result}").send()
            else:
                await cl.Message(content=f"❌ No encontré cambio con ID '{change_id}'").send()
            return

    # Comando: rechazar cambio
    if any(cmd in prompt_lower for cmd in ["rechaza", "rechazar", "reject"]):
        import re
        match = re.search(r'cambio[_\s]*(\w+)|change[_\s]*(\w+)|#?(\d+)', prompt_lower)
        if match:
            change_id = match.group(1) or match.group(2) or match.group(3)
            pending = fm.get_pending_changes()
            full_id = None
            for pid in pending.keys():
                if change_id in pid:
                    full_id = pid
                    break
            if full_id:
                success, result = fm.reject_change(full_id)
                if success:
                    await cl.Message(content=f"🗑️ {result}").send()
                else:
                    await cl.Message(content=f"❌ {result}").send()
            else:
                await cl.Message(content=f"❌ No encontré cambio con ID '{change_id}'").send()
            return

    # Procesar con workflow multi-agente
    await process_with_workflow(prompt)


async def process_with_workflow(prompt: str, force_filename: str = None):
    """Procesa el prompt usando el workflow multi-agente con visualización de pasos"""
    captures = cl.user_session.get("captures", [])
    fm = cl.user_session.get("file_manager")

    # Extraer archivo si se menciona
    filename = force_filename or extract_filename(prompt)
    content = None
    if filename:
        content, error = fm.read_file(filename)
        if error:
            await cl.Message(content=f"❌ No se pudo leer `{filename}`: {error}").send()
            return

    # Si no hay archivo pero el prompt parece querer modificar código -> MODO SMART
    if not filename and any(kw in prompt.lower() for kw in ["corrige", "modifica", "cambia", "elimina", "error", "bug", "fix", "arregla"]):
        # Modo SMART: Analizar todos los archivos para detectar cuál modificar
        files, _ = fm.list_python_files()
        if files:
            await cl.Message(content="🔍 **Modo SMART**: Analizando archivos para detectar cuál modificar...").send()

            # Crear workflow temporal solo para usar el CodeAgent
            temp_workflow = AgentWorkflow(
                coordinator_model=AGENT_COORDINATOR_MODEL,
                vision_model=AGENT_VISION_MODEL,
                code_model=AGENT_CODE_MODEL,
                editor_model=AGENT_EDITOR_MODEL,
            )

            # Leer contenido de todos los archivos
            files_content = {}
            for fname in files:
                content, _ = fm.read_file(fname)
                if content:
                    files_content[fname] = content

            # Analizar múltiples archivos
            multi_analysis = temp_workflow.code.analyze_multiple_files(
                files_content=files_content,
                user_request=prompt
            )

            if multi_analysis.get("success"):
                files_affected = multi_analysis.get("files_affected", [])
                analysis_text = multi_analysis.get("analysis", "")

                # Mostrar análisis en step
                async with cl.Step(name="🔍 Smart Analysis", type="llm") as step:
                    step_output = f"**Análisis multi-archivo:**\n\n{analysis_text[:800]}..."
                    if files_affected:
                        step_output += f"\n\n**Archivos identificados:** {', '.join(files_affected)}"
                    step.output = step_output

                if files_affected:
                    # Usar el primer archivo identificado
                    filename = files_affected[0]
                    content = files_content.get(filename, "")
                    await cl.Message(content=f"✅ **Archivo detectado automáticamente:** `{filename}`").send()
                else:
                    # Fallback: usar el primer archivo mencionado en el análisis
                    import re
                    for fname in files:
                        if re.search(rf'\b{re.escape(fname)}\b', analysis_text, re.IGNORECASE):
                            filename = fname
                            content = files_content.get(fname, "")
                            await cl.Message(content=f"✅ **Archivo detectado:** `{filename}` (por mención en análisis)").send()
                            break

            if not filename:
                # Si el análisis smart falló, fallback a preguntar
                actions = [
                    cl.Action(name="select_file_for_edit", label=f"📄 {f}", payload={"filename": f, "prompt": prompt})
                    for f in files[:8]
                ]
                await cl.Message(
                    content="⚠️ **No se pudo detectar automáticamente qué archivo modificar.**\n\nTu prompt sugiere que quieres modificar código. ¿Cuál archivo deseas editar?",
                    actions=actions
                ).send()
                return

    if not filename:
        await cl.Message(content="❌ No se detectó archivo para modificar. Por favor menciona el archivo (ej: `corrige app.py`).").send()
        return

    # Crear workflow
    workflow = AgentWorkflow(
        coordinator_model=AGENT_COORDINATOR_MODEL,
        vision_model=AGENT_VISION_MODEL,
        code_model=AGENT_CODE_MODEL,
        editor_model=AGENT_EDITOR_MODEL,
    )

    # Ejecutar con steps visuales
    async with cl.Step(name="🤖 Workflow Multi-Agente", type="run") as main_step:

        # Step 1: Coordinator
        async with cl.Step(name="📋 Coordinator", type="llm") as step:
            plan = workflow.coordinator.get_execution_plan(prompt, has_image=bool(captures))
            reasoning = plan.get('reasoning', 'N/A')
            execution_sequence = plan.get('execution_sequence', [])

            step_output = f"**Decisión del Coordinador:**\n• Visión: {'✅' if plan.get('vision') else '❌'}\n• Análisis: {'✅' if plan.get('code_analysis') else '❌'}\n• Generación: {'✅' if plan.get('code_generation') else '❌'}\n\n**Razonamiento:** {reasoning[:300]}...\n\n**Secuencia:** {' → '.join(execution_sequence)}"

            step.output = step_output

        # Ejecutar workflow completo
        result = workflow.run(
            prompt=prompt,
            image_paths=captures if captures else None,
            target_file=filename,
            file_content=content
        )

        # Mostrar pasos ejecutados basado en el resultado
        execution_path = result.get("execution_path", [])

        # Step 2: Vision (si se ejecutó)
        if "vision" in execution_path and result.get("vision_analysis"):
            async with cl.Step(name="👁️ Vision Agent", type="tool") as step:
                vision_text = result["vision_analysis"]
                step_output = f"**Análisis visual:**\n{vision_text[:500]}..."
                if len(vision_text) > 500:
                    step_output += f"\n\n*(Total: {len(vision_text)} caracteres)*"
                step.output = step_output

        # Step 3: Code (si se ejecutó)
        if "code" in execution_path and result.get("code_analysis"):
            async with cl.Step(name="💻 Code Agent", type="llm") as step:
                code_analysis = result["code_analysis"]
                analysis_text = code_analysis.get("analysis", "")
                summary = code_analysis.get("summary", "")

                step_output = f"**Resumen:** {summary}\n\n**Análisis:**\n{analysis_text[:600]}..."
                step.output = step_output

        # Step 4: Editor (si se ejecutó)
        if "editor" in execution_path and result.get("editor_result"):
            async with cl.Step(name="✏️ Editor Agent", type="llm") as step:
                editor_result = result["editor_result"]
                if editor_result.get("success"):
                    code_preview = editor_result.get("code", "")[:400]
                    lines = editor_result.get("validation", {}).get("new_lines", 0)
                    step_output = f"**Código generado:** {lines} líneas\n\n```python\n{code_preview}...\n```"
                else:
                    step_output = f"**Error:** {editor_result.get('error', 'Error desconocido')}"
                step.output = step_output

        # Step 5: Reviewer (si se ejecutó)
        if result.get("review_result"):
            async with cl.Step(name="🔍 Reviewer Agent", type="llm") as step:
                review_result = result["review_result"]
                approved = review_result.get("approved", False)
                issues = review_result.get("issues", [])

                status = "✅ **APROBADO**" if approved else "⚠️ **CON OBSERVACIONES**"
                issues_text = "\n".join([f"• {i.get('message', 'N/A')}" for i in issues[:5]])

                step_output = f"{status}\n\n**Issues encontrados:** {len(issues)}\n{issues_text}"
                step.output = step_output

        # Mostrar métricas finales
        tokens_used = result.get("tokens_used", {})
        total_tokens = sum(tokens_used.values())

        async with cl.Step(name="📊 Métricas", type="tool") as step:
            metrics = []
            for agent, tokens in sorted(tokens_used.items()):
                if tokens > 0:
                    metrics.append(f"• {agent.capitalize()}: {tokens:,} tokens")
            metrics_text = "\n".join(metrics) if metrics else "No hay métricas disponibles"

            step_output = f"**Tokens consumidos por agente:**\n{metrics_text}\n\n**Total:** {total_tokens:,} tokens"
            step.output = step_output

        # DEBUG: Ver qué tenemos en el resultado
        print(f"[DEBUG] Result keys: {result.keys()}")
        print(f"[DEBUG] editor_result: {result.get('editor_result')}")
        print(f"[DEBUG] filename: {filename}")

        # Extraer datos del resultado ANTES de cerrar el paso
        editor_result = result.get("editor_result")
        code_analysis = result.get("code_analysis")
        vision_analysis = result.get("vision_analysis")
        has_editor_result = editor_result and editor_result.get("success")
        has_code_only = code_analysis and not editor_result
        has_vision_only = vision_analysis and not code_analysis

        # Guardar datos en variables de sesión para usar fuera del paso
        if has_editor_result:
            cl.user_session.set("pending_proposal", {
                "filename": filename,
                "proposed_code": editor_result.get("code"),
                "proposal_result": None,  # Se calculará después
                "prompt": prompt
            })

    # ============================================================
    # FUERA DEL PASO PRINCIPAL - Enviar mensajes de resultado
    # ============================================================

    # Caso 1: Editor generó código
    if has_editor_result:
        pending = cl.user_session.get("pending_proposal")
        proposed_code = pending["proposed_code"]
        filename = pending["filename"]

        print(f"[DEBUG] Entrando a bloque de editor. proposed_code length: {len(proposed_code) if proposed_code else 0}")

        # Verificar que tenemos filename válido
        if not filename:
            await cl.Message(content="❌ Error interno: no se detectó archivo para guardar el cambio.").send()
            return

        if not proposed_code:
            await cl.Message(content="❌ Error: el editor no generó código.").send()
            return

        # Proponer cambio
        try:
            print(f"[DEBUG] Llamando fm.propose_change con filename={filename}")
            success, proposal_result = fm.propose_change(
                filename,
                proposed_code,
                description=prompt[:100]
            )
            print(f"[DEBUG] propose_change result: success={success}, proposal_result={proposal_result}")
        except Exception as e:
            await cl.Message(content=f"❌ Error al guardar propuesta: {e}").send()
            import traceback
            traceback.print_exc()
            return

        if success:
            print(f"[DEBUG] Enviando mensajes con botones de aprobación...")

            # Preparar código truncado para mostrar
            code_preview = proposed_code[:1500]
            code_truncated = len(proposed_code) > 1500

            # Mensaje principal
            response_content = f"""✅ **Código modificado generado y guardado**

⏳ **ID del cambio:** `{proposal_result}`
📁 **Archivo:** `{filename}`

**Vista previa del código:**
```python
{code_preview}
```
{"*(Código truncado, total: " + str(len(proposed_code)) + " caracteres)*" if code_truncated else ""}
"""
            print(f"[DEBUG] Enviando mensaje principal...")
            await cl.Message(content=response_content).send()

            # Enviar acciones de forma independiente
            print(f"[DEBUG] Enviando acciones...")

            async def send_actions():
                actions = [
                    cl.Action(name="approve_change", label="✅ Aprobar", payload={"value": proposal_result}),
                    cl.Action(name="reject_change", label="❌ Rechazar", payload={"value": proposal_result}),
                ]
                await cl.Message(content="**¿Deseas aplicar este cambio?**", actions=actions).send()
                print(f"[DEBUG] Acciones enviadas desde tarea")

            # Crear tarea independiente
            import asyncio
            task = asyncio.create_task(send_actions())
            print(f"[DEBUG] Tarea creada: {task}")
        else:
            await cl.Message(content=f"❌ Error guardando propuesta: {proposal_result}").send()

    # Caso 2: Solo análisis de código
    elif has_code_only:
        code_analysis = result["code_analysis"]
        analysis_text = code_analysis.get("analysis", "")
        summary = code_analysis.get("summary", "")

        response = f"""📋 **Análisis de código completado**

**Resumen:** {summary}

**Análisis detallado:**
{analysis_text[:1200]}
"""
        if len(analysis_text) > 1200:
            response += f"\n\n*(Análisis truncado, total: {len(analysis_text)} caracteres)*"

        response += f"\n\n📈 **Tokens consumidos:** {total_tokens:,}"

        await cl.Message(content=response).send()

    # Caso 3: Solo análisis visual
    elif has_vision_only:
        vision_text = result["vision_analysis"]

        response = f"""👁️ **Análisis visual completado**

{vision_text[:1200]}
"""
        if len(vision_text) > 1200:
            response += f"\n\n*(Análisis truncado, total: {len(vision_text)} caracteres)*"

        response += f"\n\n📈 **Tokens consumidos:** {total_tokens:,}"

        await cl.Message(content=response).send()

    # Caso 4: Respuesta directa o fallback
    else:
        final_response = result.get('response', 'No hay respuesta')
        await cl.Message(content=f"📝 **Respuesta:**\n\n{final_response}").send()

    # Limpiar capturas después de usarlas
    cl.user_session.set("captures", [])
