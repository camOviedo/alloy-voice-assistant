"""
Núcleo del asistente - orquesta LLM, voz, visión y gestión de archivos.
"""
import re
import time

from config import (
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_VISION_TIMEOUT,
    PROJECT_PATH,
    AGENT_COORDINATOR_MODEL,
    AGENT_VISION_MODEL,
    AGENT_CODE_MODEL,
    AGENT_EDITOR_MODEL,
)
from file_manager import ProjectFileManager
from llm_client import LLMClient
from voice import VoiceManager
from graph.workflow import AgentWorkflow
from screen_capture import capture_single_screenshot


class Assistant:
    def __init__(self, model_name=DEFAULT_MODEL, language=DEFAULT_LANGUAGE,
                 project_path=None, vision_timeout=DEFAULT_VISION_TIMEOUT,
                 vision_start_callback=None, vision_stop_callback=None):
        """
        model_name: modelo en Ollama (ej: "qwen2.5-vl:7b")
        language: idioma para Whisper y para el prompt del sistema (es, en, etc.)
        project_path: ruta al proyecto a modificar
        vision_timeout: segundos que dura el modo visión activado
        vision_start_callback: función a llamar para iniciar captura de pantalla
        vision_stop_callback: función a llamar para detener captura de pantalla
        """
        self.model_name = model_name
        self.language = language
        self.vision_timeout = vision_timeout
        self.vision_active_until = 0  # Timestamp cuando expira el modo visión
        self.vision_start_callback = vision_start_callback
        self.vision_stop_callback = vision_stop_callback
        self.vision_get_image_callback = None  # Se establecerá desde main.py

        # Inicializar gestor de archivos del proyecto
        self.project_path = project_path or PROJECT_PATH
        self.file_manager = ProjectFileManager(self.project_path)

        # Inicializar gestor de voz
        self.voice = VoiceManager(language=self.language)

        # Inicializar cliente LLM
        self.llm = LLMClient(model_name=self.model_name, language=self.language)

        # Inicializar workflow multi-agente
        self.agent_workflow = AgentWorkflow(
            coordinator_model=AGENT_COORDINATOR_MODEL,
            vision_model=AGENT_VISION_MODEL,
            code_model=AGENT_CODE_MODEL,
            editor_model=AGENT_EDITOR_MODEL,
        )

    def answer(self, prompt, image_base64=None, image_path=None):
        if not prompt or not prompt.strip():
            print("Prompt vacío, ignorando.")
            return

        current_time = time.time()

        # Detectar si el prompt necesita análisis de imagen/visión
        needs_vision = self._needs_vision_analysis(prompt)

        # Si detecta necesidad de visión, capturar imagen única
        if needs_vision and not image_path:
            print(f"👁️ Modo visión detectado en prompt - capturando imagen única...")
            try:
                image_base64, image_path = capture_single_screenshot()
                if image_path:
                    print(f"✅ Imagen capturada y guardada: {image_path}")
            except Exception as e:
                print(f"⚠️ Error capturando imagen: {e}")
                image_base64 = None
                image_path = None

        # Si no hay imagen, continuar en modo texto
        if not image_base64 and not image_path:
            print("📝 Modo texto (sin imagen) - ahorrando tokens")

        print("Prompt:", prompt)
        prompt_lower = prompt.lower()

        # Manejar comandos específicos
        if any(cmd in prompt_lower for cmd in ["muestra", "muéstrame", "ver archivo", "show", "ver el archivo"]):
            filename = self._extract_filename(prompt)
            if filename:
                self._handle_file_view(filename)
                return

        if any(cmd in prompt_lower for cmd in ["lista archivos", "list files", "que archivos hay", "archivos python"]):
            self._handle_list_files()
            return

        if any(cmd in prompt_lower for cmd in ["aprueba", "aprobar", "approve", "aceptar", "acepta"]):
            change_id = self._extract_change_id(prompt)
            if change_id:
                self._handle_approve_change(change_id)
                return

        if any(cmd in prompt_lower for cmd in ["rechaza", "rechazar", "reject", "cancelar", "cancela"]):
            change_id = self._extract_change_id(prompt)
            if change_id:
                self._handle_reject_change(change_id)
                return

        if any(cmd in prompt_lower for cmd in ["pendientes", "pending", "cambios pendientes", "ver cambios"]):
            self._handle_show_pending()
            return

        if any(cmd in prompt_lower for cmd in ["diff", "diferencias", "ver diferencias"]):
            change_id = self._extract_change_id(prompt)
            if change_id:
                self._handle_show_diff(change_id)
                return

        # Comando para activar visión manualmente por N segundos
        if any(cmd in prompt_lower for cmd in ["activa vision", "activar vision", "modo vision", "enable vision", "vision on"]):
            seconds = self._extract_seconds(prompt)
            duration = seconds if seconds else self.vision_timeout
            self.vision_active_until = time.time() + duration
            print(f"👁️ Modo visión ACTIVADO MANUALMENTE por {duration}s")
            self.voice.speak(f"Modo visión activado por {duration} segundos")
            return

        # Comando para desactivar visión
        if any(cmd in prompt_lower for cmd in ["desactiva vision", "desactivar vision", "modo texto", "disable vision", "vision off"]):
            self.vision_active_until = 0
            print("📝 Modo visión DESACTIVADO - volviendo a modo texto")
            self.voice.speak("Modo visión desactivado")
            return

        if any(cmd in prompt_lower for cmd in ["modifica", "modificar", "cambia", "cambiar", "update", "modify"]):
            filename = self._extract_filename(prompt)
            if filename:
                self._handle_file_modification(prompt, filename, image_base64, image_path)

    def _needs_vision_analysis(self, prompt):
        """Detecta si el prompt requiere análisis de imagen/visión"""
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

        for keyword in vision_keywords:
            if keyword in prompt_lower:
                return True
        return False

    def set_vision_get_image_callback(self, callback):
        """Establece el callback para obtener imagen de la captura."""
        self.vision_get_image_callback = callback

    def _get_vision_image(self):
        """Obtiene imagen del sistema de captura si está disponible."""
        if self.vision_get_image_callback:
            try:
                return self.vision_get_image_callback()
            except Exception as e:
                print(f"⚠️ Error obteniendo imagen: {e}")
        return None

    def _normal_response(self, prompt, image_base64):
        """Respuesta normal del asistente"""
        result = self.llm.chat(prompt, image_base64=image_base64)

        assistant_reply = result['content']
        prompt_tokens = result['prompt_tokens']
        output_tokens = result['output_tokens']

        # Mostrar información de tokens
        if image_base64:
            image_chars = len(image_base64)
            est_image_tokens = int(image_chars * 0.75)
            print(f"📊 Tokens - Prompt texto: {prompt_tokens} | Imagen (~): {est_image_tokens:,} | Generados: {output_tokens}")
            total = prompt_tokens + est_image_tokens if prompt_tokens != 'N/A' else est_image_tokens
            print(f"📊 Contexto usado: {total:,} / 262144 disponibles")
        else:
            print(f"📊 Tokens - Prompt: {prompt_tokens} | Generados: {output_tokens} (sin imagen)")

        print("Response:", assistant_reply)

        if assistant_reply:
            self.voice.speak(assistant_reply[:300])

    def _extract_filename(self, prompt):
        """Extrae nombre de archivo del prompt"""
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

    def _extract_change_id(self, prompt):
        """Extrae ID de cambio del prompt"""
        patterns = [
            r'cambio[_\s]*(\w+)',
            r'change[_\s]*(\w+)',
            r'#?(\d+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _extract_seconds(self, prompt):
        """Extrae número de segundos del prompt"""
        match = re.search(r'(\d+)\s*(?:s|seg|segundos|seconds)', prompt.lower())
        if match:
            return int(match.group(1))
        match = re.search(r'\b(\d{1,3})\b', prompt)
        if match:
            num = int(match.group(1))
            if 1 <= num <= 300:
                return num
        return None

    def _handle_file_view(self, filename):
        """Maneja solicitud de ver archivo"""
        content, error = self.file_manager.read_file(filename)
        if error:
            reply = f"❌ {error}"
        else:
            reply = f"📄 Contenido de `{filename}`:\n\n```python\n{content[:2000]}\n```"
            if len(content) > 2000:
                reply += f"\n\n... (archivo truncado, total: {len(content)} caracteres)"

        print("Response:", reply)
        self.llm.chat_history.append({"role": "user", "content": f"Ver {filename}"})
        self.llm.chat_history.append({"role": "assistant", "content": reply})
        self.voice.speak(f"Aquí está el contenido de {filename}")

    def _handle_list_files(self):
        """Lista archivos Python del proyecto"""
        files, error = self.file_manager.list_python_files()
        if error:
            reply = f"❌ {error}"
        else:
            reply = "📁 Archivos Python en el proyecto:\n" + "\n".join([f"  - {f}" for f in files])

        print("Response:", reply)
        self.llm.chat_history.append({"role": "user", "content": "Listar archivos"})
        self.llm.chat_history.append({"role": "assistant", "content": reply})
        self.voice.speak(f"Hay {len(files)} archivos Python en el proyecto")

    def _handle_show_pending(self):
        """Muestra cambios pendientes"""
        pending = self.file_manager.get_pending_changes()
        if not pending:
            reply = "✅ No hay cambios pendientes de aprobación."
        else:
            reply = "⏳ Cambios pendientes:\n\n"
            for change_id, change in pending.items():
                reply += f"• **{change_id}**: `{change['file']}` - {change.get('description', 'Sin descripción')[:50]}...\n"
            reply += "\nDi 'aprueba cambio [ID]' o 'rechaza cambio [ID]' para gestionarlos."

        print("Response:", reply)
        self.voice.speak(reply[:200])

    def _handle_show_diff(self, change_id):
        """Muestra diferencias de un cambio"""
        diff = self.file_manager.generate_diff(change_id)
        if diff is None:
            reply = f"❌ Cambio '{change_id}' no encontrado"
        else:
            reply = f"📊 Diferencias para `{change_id}`:\n\n```diff\n{diff[:1500]}\n```"
            if len(diff) > 1500:
                reply += "\n\n... (diff truncado)"

        print("Response:", reply)
        self.voice.speak(f"Mostrando diferencias del cambio {change_id}")

    def _handle_approve_change(self, change_id):
        """Aprueba un cambio pendiente"""
        pending = self.file_manager.get_pending_changes()
        full_id = None
        for pid in pending.keys():
            if change_id in pid:
                full_id = pid
                break

        if not full_id:
            reply = f"❌ No encontré cambio con ID '{change_id}'. Verifica con 'cambios pendientes'."
        else:
            success, result = self.file_manager.approve_change(full_id)
            if success:
                reply = f"✅ {result}"
            else:
                reply = f"❌ {result}"

        print("Response:", reply)
        self.llm.chat_history.append({"role": "user", "content": f"Aprobar {change_id}"})
        self.llm.chat_history.append({"role": "assistant", "content": reply})
        self.voice.speak(reply[:150])

    def _handle_reject_change(self, change_id):
        """Rechaza un cambio pendiente"""
        pending = self.file_manager.get_pending_changes()
        full_id = None
        for pid in pending.keys():
            if change_id in pid:
                full_id = pid
                break

        if not full_id:
            reply = f"❌ No encontré cambio con ID '{change_id}'. Verifica con 'cambios pendientes'."
        else:
            success, result = self.file_manager.reject_change(full_id)
            if success:
                reply = f"🗑️ {result}"
            else:
                reply = f"❌ {result}"

        print("Response:", reply)
        self.llm.chat_history.append({"role": "user", "content": f"Rechazar {change_id}"})
        self.llm.chat_history.append({"role": "assistant", "content": reply})
        self.voice.speak(reply[:150])

    def _handle_file_modification(self, prompt, filename, image_base64=None, image_path=None):
        """Maneja solicitud de modificación de archivo usando workflow multi-agente"""
        content, error = self.file_manager.read_file(filename)
        if error:
            reply = f"❌ {error}"
            print("Response:", reply)
            self.voice.speak(reply)
            return

        # Usar el workflow multi-agente para procesar la modificación
        print(f"\n🤖 Iniciando workflow multi-agente para modificar {filename}...")

        result = self.agent_workflow.run(
            prompt=prompt,
            image_b64=image_base64,
            image_path=image_path,
            target_file=filename,
            file_content=content
        )

        # Defensive check for None result
        if result is None:
            print("⚠️ Workflow retornó None, usando método legacy...")
            self._handle_file_modification_legacy(prompt, filename, image_base64, content)
            return

        # Mostrar métricas del workflow
        total_tokens = sum((result.get("tokens_used") or {}).values())
        execution_path = result.get("execution_path") or []

        print(f"\n📊 Workflow completado:")
        print(f"   - Camino: {' -> '.join(execution_path)}")
        print(f"   - Tokens usados: ~{total_tokens}")

        # Procesar resultado del editor
        editor_result = result.get("editor_result")

        if editor_result and editor_result.get("success"):
            proposed_code = editor_result.get("code")
            mod_id = editor_result.get("modification_id", "unknown")

            # Proponer el cambio
            success, proposal_result = self.file_manager.propose_change(
                filename,
                proposed_code,
                description=prompt[:100]
            )

            if success:
                change_id = proposal_result
                assistant_reply = (
                    f"✅ He analizado y modificado `{filename}` usando el workflow multi-agente.\n\n"
                    f"📊 Camino de ejecución: {' -> '.join(execution_path)}\n"
                    f"💾 Tokens optimizados: ~{total_tokens} (usando modelos especializados)\n\n"
                    f"⏳ **Cambio propuesto guardado como: `{change_id}`**"
                )

                print("Response:", assistant_reply)
                self.llm.chat_history.append({"role": "user", "content": prompt})
                self.llm.chat_history.append({"role": "assistant", "content": assistant_reply})
                self.voice.speak(f"He preparado una modificación usando {len(execution_path)} agentes. Revisa en pantalla y usa el teclado para decidir.")
                self._show_pending_change_menu(change_id)
                return
            else:
                assistant_reply = f"❌ Error guardando propuesta: {proposal_result}"
        elif editor_result:
            assistant_reply = f"❌ El agente editor no pudo generar código: {editor_result.get('error', 'Error desconocido')}"
        elif result.get("code_analysis"):
            # Mostrar análisis del CodeAgent cuando no hay código generado
            code_analysis = result["code_analysis"]
            analysis_text = code_analysis.get("analysis", "")
            summary = code_analysis.get("summary", "")
            
            assistant_reply = f"📋 **Análisis de código completado**\n\n"
            if summary:
                assistant_reply += f"**Resumen:** {summary}\n\n"
            assistant_reply += f"**Análisis detallado:**\n{analysis_text[:1500]}"
            if len(analysis_text) > 1500:
                assistant_reply += f"\n\n... (análisis truncado, total: {len(analysis_text)} caracteres)"
            
            assistant_reply += f"\n\n📊 Camino de ejecución: {' -> '.join(execution_path)}"
            assistant_reply += f"\n💾 Tokens usados: ~{total_tokens}"
            
            print("Response:", assistant_reply)
            self.llm.chat_history.append({"role": "user", "content": prompt})
            self.llm.chat_history.append({"role": "assistant", "content": assistant_reply})
            self.voice.speak("He completado el análisis del código. Revisa los detalles en pantalla.")
            return
        else:
            # Fallback al método original si el workflow no generó código
            print("⚠️ Workflow no generó código, usando método legacy...")
            self._handle_file_modification_legacy(prompt, filename, image_base64, content)
            return

        print("Response:", assistant_reply)
        self.llm.chat_history.append({"role": "user", "content": prompt})
        self.llm.chat_history.append({"role": "assistant", "content": assistant_reply})
        self.voice.speak(f"Hubo un problema preparando la modificación.")

    def _handle_file_modification_legacy(self, prompt, filename, image_base64, content):
        """Método legacy de modificación (fallback)"""
        context_prompt = f"""El usuario quiere modificar el archivo `{filename}`.

CONTENIDO ACTUAL DEL ARCHIVO ({len(content.splitlines())} líneas total):
```python
{content}
```

SOLICITUD DEL USUARIO:
{prompt}

⚠️⚠️⚠️ REGLAS ABSOLUTAS - OBLIGATORIAS:
1. DEVUELVE EL ARCHIVO COMPLETO. NO solo los cambios. NO fragmentos.
2. El archivo tiene {len(content.splitlines())} líneas - tu respuesta debe tener EXACTAMENTE ~{len(content.splitlines())} líneas de código.
3. INCLUYE TODO: imports, clases, funciones, métodos, todo el código.
4. NO uses "..." o "# resto del código" o "# código anterior sin cambios".
5. NO hagas resúmenes del código que falta.
6. Devuelve el código en UN SOLO bloque markdown ```python ... ```
7. Explica DESPUÉS del código, no antes.

FORMATO CORRECTO:
```python
[TODO el código del archivo, línea por línea]
```

Si devuelves solo un fragmento, el sistema RECHAZARÁ automáticamente tu respuesta."""

        result = self.llm.chat(context_prompt, image_base64=image_base64)
        assistant_reply = result['content']

        # Mostrar info de tokens
        if image_base64:
            image_chars = len(image_base64)
            est_image_tokens = int(image_chars * 0.75)
            total_context = (result['prompt_tokens'] + est_image_tokens) if result['prompt_tokens'] != 'N/A' else est_image_tokens
            print(f"📊 Tokens - Prompt texto: {result['prompt_tokens']} | Imagen (~): {est_image_tokens:,} | Generados: {result['output_tokens']}")
            print(f"📊 Contexto usado: ~{total_context:,} / 262144 disponibles ({100*total_context/262144:.1f}%)")
        else:
            print(f"📊 Tokens - Prompt: {result['prompt_tokens']} | Generados: {result['output_tokens']} (sin imagen)")

        # Extraer código propuesto
        proposed_code = self.llm.extract_code(assistant_reply)

        if proposed_code:
            success, result = self.file_manager.propose_change(
                filename,
                proposed_code,
                description=prompt[:100]
            )
            if success:
                change_id = result
                assistant_reply += f"\n\n⏳ **Cambio propuesto guardado como: `{change_id}`**\n"
                print("Response:", assistant_reply)
                self.llm.chat_history.append({"role": "user", "content": prompt})
                self.llm.chat_history.append({"role": "assistant", "content": assistant_reply})
                self.voice.speak(f"He preparado una modificación. Revisa en pantalla y usa el teclado para decidir.")
                self._show_pending_change_menu(change_id)
                return

        print("Response:", assistant_reply)
        self.llm.chat_history.append({"role": "user", "content": prompt})
        self.llm.chat_history.append({"role": "assistant", "content": assistant_reply})
        self.voice.speak(f"He preparado una modificación para {filename}. Revisa la propuesta y aprueba o rechaza el cambio.")

    def _handle_file_modification_smart(self, prompt, image_base64):
        """Maneja solicitud de modificación sin archivo especificado - usa LLM para elegir archivo"""
        files, error = self.file_manager.list_python_files()
        if error:
            reply = f"❌ Error listando archivos: {error}"
            print("Response:", reply)
            self.voice.speak(reply)
            return

        smart_prompt = f"""El usuario quiere hacer una modificación en el código pero no especificó qué archivo.

ARCHIVOS DISPONIBLES EN EL PROYECTO:
{chr(10).join([f"  - {f}" for f in files])}

SOLICITUD DEL USUARIO:
{prompt}

⚠️⚠️⚠️ REGLAS ABSOLUTAS - OBLIGATORIAS:
1. Analiza la solicitud y determina CUÁL archivo debería modificarse
2. Responde indicando PRIMERO el nombre del archivo a modificar (ej: "Archivo: app.py")
3. LEE TODO el archivo y proporciona el CÓDIGO COMPLETO MODIFICADO
4. DEVUELVE EL ARCHIVO COMPLETO, NO solo los cambios, NO fragmentos
5. El código debe estar en UN SOLO bloque markdown ```python ... ```
6. Explica brevemente los cambios DESPUÉS del bloque de código
7. NO uses "..." o "# resto del código" - incluye TODO el código

ADVERTENCIA: Si devuelves solo un fragmento, el cambio será RECHAZADO automáticamente."""

        result = self.llm.chat(smart_prompt, image_base64=image_base64)
        assistant_reply = result['content']

        if image_base64:
            print(f"📊 Tokens - Prompt: {result['prompt_tokens']} | Generados: {result['output_tokens']} (con imagen)")
        else:
            print(f"📊 Tokens - Prompt: {result['prompt_tokens']} | Generados: {result['output_tokens']} (sin imagen)")

        # Extraer nombre de archivo de la respuesta
        filename_match = re.search(r'[Aa]rchivo:\s*(\w+\.py)', assistant_reply)
        if not filename_match:
            filename_match = re.search(r'[Ff]ile:\s*(\w+\.py)', assistant_reply)
        if not filename_match:
            filename_match = re.search(r'(\w+\.py)', assistant_reply[:200])

        if filename_match:
            detected_filename = filename_match.group(1)
            if detected_filename in files:
                proposed_code = self.llm.extract_code(assistant_reply)

                if proposed_code:
                    success, result = self.file_manager.propose_change(
                        detected_filename,
                        proposed_code,
                        description=prompt[:100]
                    )
                    if success:
                        change_id = result
                        assistant_reply += f"\n\n⏳ **Cambio propuesto guardado como: `{change_id}`**\n"
                        print("Response:", assistant_reply)
                        self.llm.chat_history.append({"role": "user", "content": prompt})
                        self.llm.chat_history.append({"role": "assistant", "content": assistant_reply})
                        self.voice.speak(f"He preparado una modificación. Revisa en pantalla y usa el teclado para decidir.")
                        self._show_pending_change_menu(change_id)
                        return
                else:
                    assistant_reply += "\n\n⚠️ No se detectó código propuesto en la respuesta."
            else:
                assistant_reply += f"\n\n⚠️ El archivo '{detected_filename}' no existe en el proyecto."
        else:
            assistant_reply += "\n\n⚠️ No pude determinar qué archivo modificar. Por favor especifica el archivo."

        print("Response:", assistant_reply)
        self.llm.chat_history.append({"role": "user", "content": prompt})
        self.llm.chat_history.append({"role": "assistant", "content": assistant_reply})
        self.voice.speak(f"He analizado tu solicitud. Revisa la propuesta en pantalla.")

    def _show_pending_change_menu(self, change_id):
        """Muestra menú interactivo para gestionar un cambio propuesto"""
        print("\n" + "="*60)
        print("📝 CAMBIO PENDIENTE - SELECCIONA UNA OPCIÓN:")
        print("="*60)
        print(f"ID: {change_id}")
        print("\n  [1] ✅ Aplicar cambio (sin commit)")
        print("  [2] ✅ Aplicar cambio y hacer commit")
        print("  [3] ❌ Rechazar y descartar")
        print("  [4] 📊 Ver diferencias (diff)")
        print("  [5] ⏭️  Dejar pendiente (decidir luego)")
        print("="*60)
        print("Ingresa número (1-5): ", end="", flush=True)

        try:
            import select
            import sys

            ready, _, _ = select.select([sys.stdin], [], [], 60)

            if ready:
                choice = sys.stdin.readline().strip()

                if choice == "1":
                    success, result = self.file_manager.approve_change(change_id, auto_commit=False)
                    if success:
                        print(f"\n✅ {result}")
                        print("💡 Usa 'git status' para ver los cambios y luego 'git commit' cuando estés listo.")
                        self.voice.speak("Cambio aplicado sin commit. Tú tienes el control del git.")
                    else:
                        print(f"\n❌ Error: {result}")
                        self.voice.speak("Hubo un error al aplicar el cambio")

                elif choice == "2":
                    success, result = self.file_manager.approve_change(change_id, auto_commit=True)
                    if success:
                        print(f"\n✅ {result}")
                        self.voice.speak("Cambio aplicado y commiteado correctamente")
                    else:
                        print(f"\n❌ Error: {result}")
                        self.voice.speak("Hubo un error al aplicar el cambio")

                elif choice == "3":
                    success, result = self.file_manager.reject_change(change_id)
                    if success:
                        print(f"\n🗑️ {result}")
                        self.voice.speak("Cambio rechazado y eliminado")
                    else:
                        print(f"\n❌ Error: {result}")

                elif choice == "4":
                    diff = self.file_manager.generate_diff(change_id)
                    if diff:
                        print(f"\n📊 DIFERENCIAS:\n{'='*60}")
                        print(diff[:2000])
                        if len(diff) > 2000:
                            print("... (diff truncado)")
                        print(f"{'='*60}")
                        self._show_pending_change_menu(change_id)
                        return
                    else:
                        print("\n❌ No se pudo generar el diff")

                elif choice == "5":
                    print(f"\n⏳ Cambio {change_id} dejado pendiente.")
                    print("Puedes gestionarlo luego diciendo 'cambios pendientes'")
                    self.voice.speak("Cambio guardado para revisar más tarde")

                else:
                    print(f"\n⚠️ Opción '{choice}' no válida. Cambio dejado pendiente.")
                    self.voice.speak("Opción no válida, cambio guardado para revisar luego")

            else:
                print("\n⏱️ Tiempo expirado. Cambio dejado pendiente.")
                print(f"Di 'aprueba cambio {change_id}' para aplicarlo más tarde.")
                self.voice.speak("Tiempo expirado, cambio guardado para revisar luego")

        except Exception as e:
            print(f"\n⚠️ Error en menú: {e}. Cambio dejado pendiente.")
            print(f"ID para gestionar luego: {change_id}")
