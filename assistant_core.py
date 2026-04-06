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
    SUGGESTIONS_PATH,
    AGENT_COORDINATOR_MODEL,
    AGENT_VISION_MODEL,
    AGENT_CODE_MODEL,
    AGENT_EDITOR_MODEL,
)
from file_manager import ProjectFileManager
from llm_client import LLMClient
from voice import VoiceManager
from graph.workflow import AgentWorkflow
from screen_capture import capture_single_screenshot, ScreenStream


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
            memory_dir=SUGGESTIONS_PATH,
        )

    def answer(self, prompt, image_base64=None, image_path=None, image_paths=None):
        if not prompt or not prompt.strip():
            print("Prompt vacío, ignorando.")
            return

        current_time = time.time()

        # Detectar si el prompt necesita análisis de imagen/visión
        needs_vision = self._needs_vision_analysis(prompt)

        image_paths = []
        
        # Si detecta necesidad de visión, preguntar por fuente de imágenes
        if needs_vision:
            print(f"👁️ Modo visión detectado")
            
            # Verificar si hay imágenes existentes en captures/
            existing_images = self._get_existing_capture_images()
            
            if existing_images:
                # Mostrar menú de opciones
                print(f"\n📁 Se encontraron {len(existing_images)} imágenes en captures/")
                print("\n" + "="*50)
                print("   ¿Qué deseas hacer?")
                print("="*50)
                print("   [1] 📂 Usar imágenes existentes en carpeta")
                print("   [2] 📷 Hacer nuevas capturas de pantalla")
                print("="*50)
                print("   Ingresa número (1-2): ", end="", flush=True)
                
                try:
                    choice = input().strip()
                    
                    if choice == "1":
                        image_paths = existing_images
                        print(f"\n✅ Usando {len(image_paths)} imágenes existentes")
                        
                        # Pausa para revisión antes de análisis
                        print("\n⏸️  PAUSA DE REVISIÓN")
                        print("   Puedes eliminar las imágenes irrelevantes de captures/")
                        print("   Presiona ENTER cuando estés listo para continuar...")
                        try:
                            input()
                            print("▶️  Continuando con el análisis...\n")
                        except (EOFError, KeyboardInterrupt):
                            print("\n⚠️  Continuando sin esperar...")
                            
                    elif choice == "2":
                        print("\n📷 Iniciando nuevas capturas...")
                        image_paths = self._capture_new_images()
                    else:
                        print("\n⚠️ Opción no válida. Usando imágenes existentes.")
                        image_paths = existing_images
                        
                except (EOFError, KeyboardInterrupt):
                    print("\n⚠️ Entrada cancelada. Usando imágenes existentes.")
                    image_paths = existing_images
            else:
                # No hay imágenes existentes, hacer capturas nuevas
                print(f"\n📷 No hay imágenes en captures/. Iniciando captura...")
                image_paths = self._capture_new_images()

        # Si no hay imágenes, continuar en modo texto
        if not image_paths:
            print("📝 Modo texto (sin imágenes) - ahorrando tokens")

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

        if any(cmd in prompt_lower for cmd in ["modifica", "modificar", "cambia", "cambiar", "update", "modify", "aplica", "aplicar", "ajusta", "ajustar", "corrige", "corregir", "actualiza", "actualizar"]):
            filename = self._extract_filename(prompt)
            if filename:
                self._handle_file_modification(prompt, filename, image_base64, image_path, image_paths)
                return
            else:
                # No se especificó archivo, usar modo smart
                self._handle_file_modification_smart(prompt, image_base64)
                return

        # Si no es ningún comando específico, usar respuesta normal o workflow según corresponda
        self._normal_response(prompt, image_base64)

    def _get_existing_capture_images(self):
        """Obtiene lista de imágenes existentes en la carpeta captures/"""
        try:
            from config import CAPTURES_PATH
            import os
            
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

    def _capture_new_images(self):
        """Realiza nueva captura de imágenes y retorna lista de paths"""
        image_paths = []
        
        try:
            from config import (
                DEFAULT_MONITOR, DEFAULT_SCALE_DISPLAY, DEFAULT_MAX_WIDTH, DEFAULT_JPEG_QUALITY,
                VISION_CAPTURE_DURATION, VISION_CAPTURE_FPS
            )
            
            # Calcular frames dinámicamente basado en tiempo × FPS
            num_frames = int(VISION_CAPTURE_DURATION * VISION_CAPTURE_FPS)
            delay_between_frames = VISION_CAPTURE_DURATION / max(num_frames - 1, 1) if num_frames > 1 else 0
            
            # Iniciar stream de captura continua
            try:
                screen_stream = ScreenStream(
                    monitor=DEFAULT_MONITOR,
                    scale_display=DEFAULT_SCALE_DISPLAY,
                    max_width=DEFAULT_MAX_WIDTH,
                    jpeg_quality=DEFAULT_JPEG_QUALITY
                ).start()
            except Exception as e:
                print(f"❌ Error iniciando captura de pantalla: {e}")
                print("💡 Posibles causas: No hay display X11 disponible, o estás usando Wayland")
                print("💡 Intenta usar imágenes existentes en captures/ o especifica un archivo de imagen")
                return []
            
            # Verificar si el stream se inició correctamente
            if not screen_stream.running:
                print("❌ No se pudo iniciar la captura de pantalla")
                return []
            
            print("📹 Stream de captura iniciado - capturando frames...")
            
            # Esperar a que el stream tenga frames (con timeout)
            wait_time = 0
            while screen_stream.frame is None and screen_stream.running and wait_time < 3:
                time.sleep(0.1)
                wait_time += 0.1
            
            if screen_stream.frame is None:
                print("❌ No se pudo obtener frames del stream")
                screen_stream.stop()
                return []
            
            # Capturar frames calculados dinámicamente (tiempo × FPS)
            print(f"📹 Capturando {num_frames} frames en {VISION_CAPTURE_DURATION}s ({VISION_CAPTURE_FPS} fps)...")
            for i in range(num_frames):
                b64, path = screen_stream.capture_and_save_frame()
                if path:
                    image_paths.append(path)
                    print(f"  📸 Frame {i+1}/{num_frames} capturado")
                if i < num_frames - 1:
                    time.sleep(delay_between_frames)
            
            # DETENER el stream
            screen_stream.stop()
            print(f"✅ Captura completada - {len(image_paths)} frames guardados")
            print("🛑 Stream de video detenido")
            
            # PAUSA para permitir eliminar imágenes irrelevantes
            if image_paths:
                print(f"\n⏸️  PAUSA DE REVISIÓN - {len(image_paths)} imágenes capturadas en: captures/")
                print("   Puedes eliminar las imágenes irrelevantes ahora.")
                print("   Presiona ENTER cuando estés listo para continuar...")
                try:
                    input()
                    print("▶️  Continuando con el análisis...\n")
                except (EOFError, KeyboardInterrupt):
                    print("\n⚠️  Continuando sin esperar...")
                    
        except Exception as e:
            print(f"⚠️ Error en captura: {e}")
            import traceback
            traceback.print_exc()
        
        return image_paths

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
        """Respuesta normal del asistente - con guardado en memoria para análisis de proyecto"""
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

        # Detectar si es un análisis de proyecto y guardar en memoria
        self._save_project_analysis_to_memory(prompt, assistant_reply)

        if assistant_reply:
            self.voice.speak(assistant_reply[:300])

    def _save_project_analysis_to_memory(self, prompt, analysis):
        """Detecta si el prompt/respuesta contiene análisis de proyecto y lo guarda en ProjectMemory"""
        import re

        # Detectar si es un prompt de análisis de proyecto
        project_analysis_keywords = [
            'analiza el proyecto', 'analiza el código', 'conocimiento del proyecto',
            'entiende el proyecto', 'estructura del proyecto', 'archivos del proyecto',
            'analyze the project', 'understand the project', 'project structure'
        ]

        prompt_lower = prompt.lower()
        is_project_analysis = any(kw in prompt_lower for kw in project_analysis_keywords)

        if not is_project_analysis:
            return

        try:
            # Extraer archivos mencionados en el análisis
            files_mentioned = re.findall(r'(\w+\.py)', analysis)
            unique_files = list(set(files_mentioned))

            # Extraer un resumen del análisis (primeros 1000 chars o hasta el primer punto seguido de espacio)
            summary = analysis[:1000].strip()
            if len(analysis) > 1000:
                summary += "..."

            # Guardar en ProjectMemory
            self.agent_workflow.code.memory.save_project_summary(summary, unique_files)
            print(f"💾 Análisis del proyecto guardado en memoria ({len(unique_files)} archivos identificados)")

            # También guardar análisis individual para archivos principales
            for fname in unique_files[:5]:  # Solo los 5 principales
                content, _ = self.file_manager.read_file(fname)
                if content:
                    file_analysis = {
                        "filename": fname,
                        "analysis": f"Archivo identificado en análisis de proyecto: {fname}",
                        "summary": f"Archivo principal del proyecto",
                        "files_affected": unique_files,
                        "success": True
                    }
                    self.agent_workflow.code.memory.save_file_analysis(fname, content, file_analysis)

        except Exception as e:
            print(f"⚠️ No se pudo guardar análisis en memoria: {e}")

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

    def _handle_file_modification(self, prompt, filename, image_base64=None, image_path=None, image_paths=None):
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
            image_paths=image_paths,
            target_file=filename,
            file_content=content
        )

        # Defensive check for None result
        if result is None:
            print("⚠️ Workflow retornó None, usando método legacy...")
            self._handle_file_modification_legacy(prompt, filename, image_base64, content)
            return

        # Mostrar métricas del workflow con desglose por agente
        tokens_used = result.get("tokens_used") or {}
        execution_path = result.get("execution_path") or []
        total_tokens = sum(tokens_used.values())
        
        # Contexto disponible (basado en los modelos utilizados)
        CONTEXT_WINDOW = 262144  # qwen3 models: 262144 tokens
        available_tokens = CONTEXT_WINDOW - total_tokens
        
        print(f"\n📊 Workflow completado:")
        print(f"   - Camino: {' -> '.join(execution_path)}")
        print(f"\n   📈 Tokens consumidos por agente:")
        for agent, tokens in sorted(tokens_used.items()):
            percentage = (tokens / CONTEXT_WINDOW) * 100
            print(f"      • {agent:12}: {tokens:>8,} tokens ({percentage:>5.2f}%)")
        print(f"   ─────────────────────────────────")
        print(f"      • {'TOTAL':12}: {total_tokens:>8,} tokens ({(total_tokens/CONTEXT_WINDOW)*100:.2f}%)")
        print(f"\n   💾 Disponible: {available_tokens:,} tokens / {CONTEXT_WINDOW:,} ({(available_tokens/CONTEXT_WINDOW)*100:.1f}% libre)")

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
            
                # Formatear desglose de tokens para la respuesta
                tokens_breakdown = "\n".join([f"   • {agent}: {tokens:,} tokens" for agent, tokens in sorted(tokens_used.items())])
                
                assistant_reply = (
                    f"✅ He analizado y modificado `{filename}` usando el workflow multi-agente.\n\n"
                    f"📊 Camino de ejecución: {' -> '.join(execution_path)}\n"
                    f"📈 Tokens consumidos:\n{tokens_breakdown}\n"
                    f"   ──────────────────\n"
                    f"   • Total: {total_tokens:,} / 262,144 disponibles ({(total_tokens/262144)*100:.1f}%)\n\n"
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
            assistant_reply = f"❌ El agente editor no pudo generar código: {editor_result.get('error', 'Error desconocido')}\n\n"
            # Añadir info de tokens aunque falle
            tokens_breakdown = "\n".join([f"   • {agent}: {tokens:,} tokens" for agent, tokens in sorted(tokens_used.items())])
            assistant_reply += f"📈 Tokens consumidos antes del error:\n{tokens_breakdown}\n   • Total: {total_tokens:,} tokens"
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
            
            # Añadir desglose detallado de tokens
            tokens_breakdown = "\n".join([f"   • {agent}: {tokens:,} tokens" for agent, tokens in sorted(tokens_used.items())])
            assistant_reply += f"\n\n📊 Camino de ejecución: {' -> '.join(execution_path)}"
            assistant_reply += f"\n📈 Tokens consumidos:\n{tokens_breakdown}"
            assistant_reply += f"\n   ──────────────────"
            assistant_reply += f"\n   • Total: {total_tokens:,} / 262,144 disponibles ({(total_tokens/262144)*100:.1f}%)"
            
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

    def _handle_file_modification_smart(self, prompt, image_base64=None, image_path=None, image_paths=None):
        """Maneja solicitud de modificación sin archivo especificado - usa workflow multi-agente con análisis multi-archivo"""
        files, error = self.file_manager.list_python_files()
        if error:
            reply = f"❌ Error listando archivos: {error}"
            print("Response:", reply)
            self.voice.speak(reply)
            return

        if not files:
            reply = "❌ No hay archivos Python en el proyecto para analizar."
            print("Response:", reply)
            self.voice.speak(reply)
            return

        print(f"\n🔍 Modo SMART: Analizando {len(files)} archivos para determinar cuáles modificar...")
        print(f"   Archivos disponibles: {', '.join(files[:5])}{'...' if len(files) > 5 else ''}")

        # Usar el CodeAgent para analizar múltiples archivos y determinar cuáles modificar
        try:
            # Primero analizar todos los archivos juntos para identificar cuáles necesitan cambios
            files_content = {}
            for fname in files:
                content, _ = self.file_manager.read_file(fname)
                if content:
                    files_content[fname] = content

            if not files_content:
                reply = "❌ No se pudieron leer los archivos del proyecto."
                print("Response:", reply)
                self.voice.speak(reply)
                return

            # Usar CodeAgent para análisis multi-archivo
            print(f"\n🤖 Iniciando análisis multi-archivo con CodeAgent...")
            multi_analysis = self.agent_workflow.code.analyze_multiple_files(
                files_content=files_content,
                user_request=prompt
            )

            if not multi_analysis.get("success"):
                print(f"⚠️ Análisis multi-archivo falló, usando método legacy...")
                self._handle_file_modification_smart_legacy(prompt, image_base64)
                return

            analysis_text = multi_analysis.get("analysis", "")
            print(f"\n� Análisis del CodeAgent:")
            print(f"   {analysis_text[:500]}...")

            # Extraer archivos afectados del análisis
            import re
            files_to_modify = []
            for fname in files:
                # Buscar menciones de archivos en el análisis
                if re.search(rf'\b{re.escape(fname)}\b', analysis_text, re.IGNORECASE):
                    files_to_modify.append(fname)

            # Si no se detectaron archivos específicos, usar el primer archivo mencionado o el principal
            if not files_to_modify:
                # Intentar extraer cualquier nombre de archivo .py del análisis
                matches = re.findall(r'(\w+\.py)', analysis_text)
                for match in matches:
                    if match in files:
                        files_to_modify.append(match)

            # Si aún no hay archivos, usar el primero como fallback
            if not files_to_modify and files:
                files_to_modify = [files[0]]
                print(f"   ⚠️ No se detectaron archivos específicos, usando: {files[0]}")

            print(f"\n📁 Archivos identificados para modificación: {', '.join(files_to_modify)}")

            # Si hay múltiples archivos, procesar el primero y dejar pendientes los demás
            # (o podríamos extender el workflow para manejar múltiples archivos)
            primary_file = files_to_modify[0]
            primary_content = files_content.get(primary_file, "")

            if not primary_content:
                reply = f"❌ No se pudo leer el contenido de {primary_file}"
                print("Response:", reply)
                self.voice.speak(reply)
                return

            print(f"\n🤖 Iniciando workflow multi-agente para modificar {primary_file}...")
            if len(files_to_modify) > 1:
                print(f"   (Otros archivos identificados: {', '.join(files_to_modify[1:])})")

            result = self.agent_workflow.run(
                prompt=prompt,
                image_b64=image_base64,
                image_path=image_path,
                image_paths=image_paths,
                target_file=primary_file,
                file_content=primary_content
            )

            # Procesar resultado del workflow
            self._process_workflow_result(result, prompt, primary_file)

        except Exception as e:
            print(f"⚠️ Error en análisis multi-archivo: {e}")
            import traceback
            traceback.print_exc()
            print(f"   Fallback al método legacy...")
            self._handle_file_modification_smart_legacy(prompt, image_base64)

    def _process_workflow_result(self, result, prompt, filename):
        """Procesa el resultado del workflow multi-agente y genera la respuesta final"""
        # Defensive check for None result
        if result is None:
            print("⚠️ Workflow retornó None")
            assistant_reply = "❌ Error: El workflow no retornó resultado."
            print("Response:", assistant_reply)
            self.voice.speak("Hubo un error en el análisis.")
            return

        # Mostrar métricas del workflow con desglose por agente
        tokens_used = result.get("tokens_used") or {}
        execution_path = result.get("execution_path") or []
        total_tokens = sum(tokens_used.values())

        # Contexto disponible (basado en los modelos utilizados)
        CONTEXT_WINDOW = 262144
        available_tokens = CONTEXT_WINDOW - total_tokens

        print(f"\n📊 Workflow completado:")
        print(f"   - Camino: {' -> '.join(execution_path) if execution_path else 'N/A'}")
        print(f"\n   📈 Tokens consumidos por agente:")
        for agent, tokens in sorted(tokens_used.items()):
            percentage = (tokens / CONTEXT_WINDOW) * 100
            print(f"      • {agent:12}: {tokens:>8,} tokens ({percentage:>5.2f}%)")
        print(f"   ─────────────────────────────────")
        print(f"      • {'TOTAL':12}: {total_tokens:>8,} tokens ({(total_tokens/CONTEXT_WINDOW)*100:.2f}%)")
        print(f"\n   💾 Disponible: {available_tokens:,} tokens / {CONTEXT_WINDOW:,} ({(available_tokens/CONTEXT_WINDOW)*100:.1f}% libre)")

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

                # Formatear desglose de tokens para la respuesta
                tokens_breakdown = "\n".join([f"   • {agent}: {tokens:,} tokens" for agent, tokens in sorted(tokens_used.items())])

                assistant_reply = (
                    f"✅ He analizado y modificado `{filename}` usando el workflow multi-agente.\n\n"
                    f"📊 Camino de ejecución: {' -> '.join(execution_path) if execution_path else 'N/A'}\n"
                    f"📈 Tokens consumidos:\n{tokens_breakdown}\n"
                    f"   ──────────────────\n"
                    f"   • Total: {total_tokens:,} / 262,144 disponibles ({(total_tokens/262144)*100:.1f}%)\n\n"
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
            assistant_reply = f"❌ El agente editor no pudo generar código: {editor_result.get('error', 'Error desconocido')}\n\n"
            # Añadir info de tokens aunque falle
            tokens_breakdown = "\n".join([f"   • {agent}: {tokens:,} tokens" for agent, tokens in sorted(tokens_used.items())])
            assistant_reply += f"📈 Tokens consumidos antes del error:\n{tokens_breakdown}\n   • Total: {total_tokens:,} tokens"
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

            # Añadir desglose detallado de tokens
            tokens_breakdown = "\n".join([f"   • {agent}: {tokens:,} tokens" for agent, tokens in sorted(tokens_used.items())])
            assistant_reply += f"\n\n📊 Camino de ejecución: {' -> '.join(execution_path) if execution_path else 'N/A'}"
            assistant_reply += f"\n📈 Tokens consumidos:\n{tokens_breakdown}"
            assistant_reply += f"\n   ──────────────────"
            assistant_reply += f"\n   • Total: {total_tokens:,} / 262,144 disponibles ({(total_tokens/262144)*100:.1f}%)"

            print("Response:", assistant_reply)
            self.llm.chat_history.append({"role": "user", "content": prompt})
            self.llm.chat_history.append({"role": "assistant", "content": assistant_reply})
            self.voice.speak("He completado el análisis del código. Revisa los detalles en pantalla.")
            return
        else:
            assistant_reply = f"⚠️ Workflow completado pero no se generó código ni análisis."

        print("Response:", assistant_reply)
        self.llm.chat_history.append({"role": "user", "content": prompt})
        self.llm.chat_history.append({"role": "assistant", "content": assistant_reply})
        self.voice.speak(f"He completado el análisis. Revisa los resultados en pantalla.")

    def _handle_file_modification_smart_legacy(self, prompt, image_base64=None):
        """Método legacy para modificación smart (fallback cuando el análisis multi-archivo falla)"""
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
        import re
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
