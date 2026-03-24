import base64
import cv2
import numpy as np
import ollama
import pyttsx3
import mss
import threading
import time
from threading import Lock, Thread
from faster_whisper import WhisperModel
import speech_recognition as sr
import io
import os
import re
import subprocess
import difflib
import json
from pathlib import Path

# Opcional: suprimir warnings de ALSA y Qt (puedes descomentar si quieres)
# os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
# os.environ['ALSA_CARD'] = "Generic"

# -------------------------------------------------------------------
# Configuración del proyecto objetivo
# -------------------------------------------------------------------
PROJECT_PATH = os.path.expanduser("~/Proyectos/reconocimiento_caballos")
SUGGESTIONS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sugerencias_pendientes")

# Crear directorio para sugerencias pendientes
os.makedirs(SUGGESTIONS_PATH, exist_ok=True)

# -------------------------------------------------------------------
# Gestor de archivos del proyecto
# -------------------------------------------------------------------
class ProjectFileManager:
    """Gestiona la lectura y modificación de archivos del proyecto objetivo"""
    
    def __init__(self, project_path):
        self.project_path = Path(project_path)
        self.pending_changes = {}  # {file_path: {original, proposed, description}}
        self._load_pending_changes()
    
    def _load_pending_changes(self):
        """Carga cambios pendientes desde archivo JSON"""
        pending_file = Path(SUGGESTIONS_PATH) / "pending_changes.json"
        if pending_file.exists():
            try:
                with open(pending_file, 'r', encoding='utf-8') as f:
                    self.pending_changes = json.load(f)
            except Exception as e:
                print(f"Error cargando cambios pendientes: {e}")
                self.pending_changes = {}
    
    def _save_pending_changes(self):
        """Guarda cambios pendientes a archivo JSON"""
        pending_file = Path(SUGGESTIONS_PATH) / "pending_changes.json"
        try:
            with open(pending_file, 'w', encoding='utf-8') as f:
                json.dump(self.pending_changes, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error guardando cambios pendientes: {e}")
    
    def read_file(self, relative_path):
        """Lee un archivo del proyecto"""
        file_path = self.project_path / relative_path
        if not file_path.exists():
            return None, f"Archivo no encontrado: {relative_path}"
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content, None
        except Exception as e:
            return None, f"Error leyendo archivo: {e}"
    
    def list_python_files(self):
        """Lista todos los archivos Python del proyecto"""
        try:
            py_files = list(self.project_path.glob("*.py"))
            return [f.name for f in py_files], None
        except Exception as e:
            return None, f"Error listando archivos: {e}"
    
    def propose_change(self, relative_path, new_content, description=""):
        """Propone un cambio a un archivo"""
        original_content, error = self.read_file(relative_path)
        if error:
            return False, error
        
        # VALIDACIÓN DE SEGURIDAD: detectar si el código es significativamente más corto
        original_lines = len(original_content.splitlines())
        new_lines = len(new_content.splitlines())
        
        if new_lines < original_lines * 0.5:
            # El código nuevo tiene menos del 50% de líneas del original
            warning = f"⚠️ ADVERTENCIA: El código propuesto ({new_lines} líneas) es mucho más corto que el original ({original_lines} líneas)."
            print(f"\n{'='*60}")
            print(warning)
            print("Esto puede indicar que el LLM devolvió solo un fragmento en lugar del archivo completo.")
            print("El cambio NO se ha guardado. Intenta nuevamente especificando 'archivo completo'.")
            print(f"{'='*60}\n")
            return False, f"Código propuesto incompleto: {new_lines} vs {original_lines} líneas. El cambio fue rechazado automáticamente."
        
        change_id = f"{relative_path}_{int(time.time())}"
        self.pending_changes[change_id] = {
            "file": relative_path,
            "original": original_content,
            "proposed": new_content,
            "description": description,
            "timestamp": time.time()
        }
        self._save_pending_changes()
        return True, change_id
    
    def get_pending_changes(self):
        """Obtiene todos los cambios pendientes"""
        return self.pending_changes
    
    def approve_change(self, change_id, auto_commit=False):
        """Aprueba y aplica un cambio pendiente (sin commit automático por defecto)"""
        if change_id not in self.pending_changes:
            return False, "Cambio no encontrado"
        
        change = self.pending_changes[change_id]
        file_path = self.project_path / change["file"]
        
        try:
            # Guardar backup
            backup_path = str(file_path) + ".backup"
            with open(file_path, 'r', encoding='utf-8') as f:
                original = f.read()
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original)
            
            # Aplicar cambio
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(change["proposed"])
            
            result_msg = f"✅ Cambio aplicado a {change['file']}"
            
            # Solo hacer commit si se solicita explícitamente
            if auto_commit:
                result = self._git_commit(change["file"], change.get("description", "Actualización via asistente"))
                result_msg += f" | {result}"
            else:
                result_msg += " (SIN COMMIT - cambios en working directory)"
            
            # Eliminar de pendientes
            del self.pending_changes[change_id]
            self._save_pending_changes()
            
            return True, result_msg
        except Exception as e:
            return False, f"Error aplicando cambio: {e}"
    
    def reject_change(self, change_id):
        """Rechaza y elimina un cambio pendiente"""
        if change_id not in self.pending_changes:
            return False, "Cambio no encontrado"
        
        del self.pending_changes[change_id]
        self._save_pending_changes()
        return True, "Cambio rechazado y eliminado"
    
    def _git_commit(self, file_path, message):
        """Hace commit del cambio en git"""
        try:
            # Cambiar al directorio del proyecto
            original_dir = os.getcwd()
            os.chdir(self.project_path)
            
            # Añadir archivo
            subprocess.run(["git", "add", file_path], check=True, capture_output=True)
            
            # Commit
            result = subprocess.run(
                ["git", "commit", "-m", f"[Asistente] {message}"],
                capture_output=True,
                text=True
            )
            
            os.chdir(original_dir)
            
            if result.returncode == 0:
                return f"✅ Cambio aplicado y commiteado: {message}"
            else:
                return f"⚠️ Archivo actualizado pero commit falló: {result.stderr}"
        except Exception as e:
            return f"⚠️ Archivo actualizado pero error en git: {e}"
    
    def generate_diff(self, change_id):
        """Genera un diff del cambio propuesto"""
        if change_id not in self.pending_changes:
            return None
        
        change = self.pending_changes[change_id]
        original_lines = change["original"].splitlines(keepends=True)
        proposed_lines = change["proposed"].splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            original_lines,
            proposed_lines,
            fromfile=f"a/{change['file']}",
            tofile=f"b/{change['file']}"
        )
        return "".join(diff)
class ScreenStream:
    def __init__(self, monitor=1, scale_display=0.5, max_width=800, jpeg_quality=50):
        """
        monitor: índice del monitor a capturar (1 = principal, 2 = secundario, etc.)
        scale_display: factor de escala para la ventana de previsualización
        max_width: ancho máximo en píxeles para la imagen enviada al LLM (reduce tokens)
        jpeg_quality: calidad JPEG (menor = menos tokens pero más compresión)
        """
        self.monitor_index = monitor
        self.scale_display = scale_display
        self.max_width = max_width
        self.jpeg_quality = jpeg_quality
        self.frame = None
        self.running = False
        self.lock = Lock()
        # No inicializamos mss aquí, lo haremos en el hilo

    def start(self):
        if self.running:
            return self
        self.running = True
        self.thread = Thread(target=self.update)
        self.thread.start()
        # Esperar a que haya un primer frame
        while self.frame is None and self.running:
            time.sleep(0.1)
        return self

    def update(self):
        # Crear una instancia de mss dentro del hilo
        with mss.mss() as sct:
            # Obtener la región del monitor seleccionado
            # Nota: monitors[0] es todos los monitores combinados, [1] el principal, etc.
            if self.monitor_index >= len(sct.monitors):
                self.monitor_index = 1  # fallback al principal
            monitor_region = sct.monitors[self.monitor_index]
            width = monitor_region['width']
            height = monitor_region['height']
            
            while self.running:
                # Capturar la pantalla
                screenshot = sct.grab(monitor_region)
                # Convertir a numpy array (BGRA) y luego a BGR (sin canal alpha)
                img = np.array(screenshot)
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                with self.lock:
                    self.frame = frame

    def read(self, encode=False):
        with self.lock:
            if self.frame is None:
                return None
            frame = self.frame.copy()
        if encode:
            # Redimensionar si es necesario para reducir tokens
            h, w = frame.shape[:2]
            if w > self.max_width:
                scale = self.max_width / w
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                print(f"📸 Imagen redimensionada: {w}x{h} → {new_w}x{new_h}")
            # Comprimir a JPEG con calidad reducida para menor tamaño
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            _, buffer = cv2.imencode(".jpeg", frame, encode_params)
            b64 = base64.b64encode(buffer).decode('utf-8')
            est_tokens = int(len(b64) * 0.75)
            print(f"📸 Imagen: {len(b64):,} chars (~{est_tokens:,} tokens) | Calidad: {self.jpeg_quality}%")
            return b64
        return frame

    def read_display(self):
        """Devuelve una versión redimensionada para mostrar en ventana"""
        with self.lock:
            if self.frame is None:
                # Retornar un frame negro de tamaño pequeño
                return np.zeros((100, 100, 3), dtype=np.uint8)
            frame = self.frame.copy()
        if self.scale_display != 1.0:
            h, w = frame.shape[:2]
            new_w = int(w * self.scale_display)
            new_h = int(h * self.scale_display)
            frame = cv2.resize(frame, (new_w, new_h))
        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

# -------------------------------------------------------------------
# Asistente con modelo local (Ollama + Whisper GPU + TTS local)
# -------------------------------------------------------------------
class Assistant:
    def __init__(self, model_name="qwen3-vl:8b-extreme", language="es", project_path=None, vision_timeout=5):
        """
        model_name: modelo en Ollama (ej: "qwen2.5-vl:7b")
        language: idioma para Whisper y para el prompt del sistema (es, en, etc.)
        project_path: ruta al proyecto a modificar
        vision_timeout: segundos que dura el modo visión activado (default: 30s)
        """
        self.model_name = model_name
        self.language = language
        self.vision_timeout = vision_timeout
        self.vision_active_until = 0  # Timestamp cuando expira el modo visión
        
        # Inicializar gestor de archivos del proyecto
        self.project_path = project_path or PROJECT_PATH
        self.file_manager = ProjectFileManager(self.project_path)
        
        # Inicializar Whisper local en GPU con FP16
        print(f"Cargando modelo Whisper en GPU para idioma: {self.language}...")
        self.whisper_model = WhisperModel(
            "base",           # puedes cambiar a "small", "medium", "large-v3"
            device="cuda",
            compute_type="float16"
        )
        print("Modelo Whisper listo.")
        
        # Inicializar TTS local
        self.tts_engine = pyttsx3.init()
        # --- AJUSTES PARA VOZ MÁS NATURAL ---
        # 1. Reducir la velocidad (rate) para que sea más pausado
        # El valor por defecto suele ser 200 palabras por minuto.
        # Probaremos con 140, pero puedes ajustarlo según prefieras.
        current_rate = self.tts_engine.getProperty('rate')
        new_rate = 140  # o current_rate - 50
        self.tts_engine.setProperty('rate', new_rate)
        print(f"Velocidad de voz ajustada a {new_rate} (anterior: {current_rate})")
        
        # 2. Aumentar ligeramente el volumen (opcional)
        self.tts_engine.setProperty('volume', 0.9)  # rango 0.0-1.0
        
        # 3. Seleccionar la mejor voz en español disponible
        voices = self.tts_engine.getProperty('voices')
        spanish_voice = None
        # Algunas voces comunes en Windows: 'Microsoft Sabina', 'Microsoft Helena'
        # En Linux pueden llamarse 'mbrola_es1', 'es_ES', etc.
        for voice in voices:
            voice_name = voice.name.lower()
            voice_id = voice.id.lower()
            # Criterios de búsqueda para español
            if any(x in voice_name for x in ['spanish', 'español', 'es_', 'mb-es']):
                spanish_voice = voice.id
                print(f"✓ Voz encontrada: {voice.name}")
                break
            # Alternativa: buscar por ID (ej. 'mb-es1' para Mbrola Spanish)
            if 'mb-es' in voice_id or 'es_' in voice_id:
                spanish_voice = voice.id
                print(f"✓ Voz encontrada por ID: {voice.name}")
                break
        
        if spanish_voice:
            self.tts_engine.setProperty('voice', spanish_voice)
            print("Voz en español seleccionada.")
        else:
            print("No se encontró voz en español, se usará la predeterminada.")
            
        # Historial de conversación simple (lista de dicts)
        self.chat_history = []
        
        # Prompt del sistema en el idioma seleccionado
        if language == "es":
            self.system_prompt = f"""Eres un asistente experto en visión por computadora y programación en Python.

Tu tarea es ayudar a mejorar un sistema de reconocimiento de caballos ubicado en: {self.project_path}

REGLA ABSOLUTA #1 - ARCHIVOS COMPLETOS OBLIGATORIOS:
CUANDO MODIFIQUES CÓDIGO, SIEMPRE DEVUELVES EL ARCHIVO COMPLETO. NUNCA SOLO FRAGMENTOS.
- Si el archivo tiene 500 líneas, tu respuesta debe tener ~500 líneas de código.
- NO uses "..." o "resto del código sin cambios".
- NO uses comentarios tipo "# código anterior...".
- SIEMPRE incluye TODAS las líneas desde la primera hasta la última.

REGLA ABSOLUTA #2 - ESTRUCTURA DE RESPUESTA:
1. Explica brevemente qué cambiarás
2. Proporciona el CÓDIGO COMPLETO en bloque markdown ```python ... ```
3. Confirma el número de líneas del archivo

CAPACIDADES DISPONIBLES:
1. Ver la pantalla y analizar videos de carreras de caballos
2. Leer archivos del proyecto: Puedes leer cualquier archivo .py del proyecto
3. Proponer cambios: Puedes sugerir modificaciones al código (SIEMPRE archivos completos)
4. El usuario debe aprobar los cambios antes de aplicarlos

ARCHIVOS PRINCIPALES DEL PROYECTO:
- app.py: Punto de entrada principal, procesamiento de video en tiempo real
- tracker.py: Clase EfficientHorseTracker para seguimiento de caballos
- yolo_utils.py: Utilidades para YOLO
- ui.py: Interfaz de usuario

INSTRUCCIONES ADICIONALES:
- Cuando el usuario pida ver/modificar un archivo, usa las funciones disponibles
- SIEMPRE devuelve código completo, nunca fragmentos
- Incluye el código completo en bloques markdown ```python ... ```
- Si el usuario dice "aprueba cambio X" o "rechaza cambio X", usa las funciones correspondientes
- Para ver archivos: el usuario puede decir "muéstrame el archivo tracker.py"
- Para modificar: el usuario puede decir "modifica tracker.py para que..."
"""
        else:
            self.system_prompt = f"""You are an expert assistant in computer vision and Python programming.

Your task is to help improve a horse recognition system located at: {self.project_path}

AVAILABLE CAPABILITIES:
1. View screen and analyze horse racing videos
2. Read project files: You can read any .py file in the project
3. Propose changes: You can suggest code modifications
4. User must approve changes before they are applied

MAIN PROJECT FILES:
- app.py: Main entry point, real-time video processing
- tracker.py: EfficientHorseTracker class for horse tracking
- yolo_utils.py: YOLO utilities
- ui.py: User interface

INSTRUCTIONS:
- When user asks to view/modify a file, use available functions
- Always explain proposed changes before showing code
- Include complete code in markdown blocks ```python ... ```
- If user says "approve change X" or "reject change X", use corresponding functions
- To view files: user can say "show me tracker.py"
- To modify: user can say "modify tracker.py to..."
"""

    def answer(self, prompt, image_base64=None):
        if not prompt or not prompt.strip():
            print("Prompt vacío, ignorando.")
            return
        
        # Verificar si el modo visión ha expirado
        current_time = time.time()
        vision_expired = current_time > self.vision_active_until
        
        # Detectar si el prompt necesita análisis de imagen/visión
        needs_vision = self._needs_vision_analysis(prompt)
        
        # Si detecta necesidad de visión, activar temporalmente
        if needs_vision:
            self.vision_active_until = current_time + self.vision_timeout
            remaining = self.vision_timeout
            print(f"👁️ Modo visión ACTIVADO por {remaining}s (detectado en prompt)")
        
        # Verificar si estamos en modo visión activo
        vision_active = current_time <= self.vision_active_until
        
        if vision_active and image_base64:
            remaining = int(self.vision_active_until - current_time)
            print(f"�️ Modo visión activo - {remaining}s restantes")
        elif vision_active and not image_base64:
            print("⚠️ Modo visión activo pero imagen no disponible")
            vision_active = False
        
        # Si no está activo el modo visión, no usar imagen
        if not vision_active:
            image_base64 = None
            print("📝 Modo texto (sin imagen) - ahorrando tokens")

        print("Prompt:", prompt)
        prompt_lower = prompt.lower()
        
        # Manejar comandos específicos
        if any(cmd in prompt_lower for cmd in ["muestra", "muéstrame", "ver archivo", "show", "ver el archivo"]):
            # Extraer nombre de archivo
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
            self._tts_local(f"Modo visión activado por {duration} segundos")
            return
        
        # Comando para desactivar visión
        if any(cmd in prompt_lower for cmd in ["desactiva vision", "desactivar vision", "modo texto", "disable vision", "vision off"]):
            self.vision_active_until = 0
            print("📝 Modo visión DESACTIVADO - volviendo a modo texto")
            self._tts_local("Modo visión desactivado")
            return
        if any(cmd in prompt_lower for cmd in ["modifica", "modificar", "cambia", "cambiar", "update", "modify"]):
            filename = self._extract_filename(prompt)
            if filename:
                self._handle_file_modification(prompt, filename, image_base64)
                return
            else:
                # No se especificó archivo - preguntar al LLM cuál archivo modificar
                self._handle_file_modification_smart(prompt, image_base64)
                return
        
        # Si no coincide con ningún comando específico, usar respuesta normal del asistente
        self._normal_response(prompt, image_base64)

    def _needs_vision_analysis(self, prompt):
        """Detecta si el prompt requiere análisis de imagen/visión"""
        prompt_lower = prompt.lower()
        
        # Palabras clave que indican necesidad de visión
        vision_keywords = [
            # Español
            "pantalla", "imagen", "foto", "captura", "ventana", "interfaz",
            "video", "stream", "cámara", "webcam", "monitor", "escritorio",
            "muestra", "muéstrame", "ver pantalla", "que ves", "qué ves",
            "analiza la imagen", "describe la imagen", "en la pantalla",
            "error en pantalla", "lo que ves", "screenshot", "screen",
            # Inglés
            "screen", "image", "picture", "window", "interface", "desktop",
            "what do you see", "analyze image", "show me", "look at"
        ]
        
        for keyword in vision_keywords:
            if keyword in prompt_lower:
                return True
        
        return False

    # Respuesta normal del asistente
    def _normal_response(self, prompt, image_base64):
        """Respuesta normal del asistente"""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Agregar historial
        for msg in self.chat_history[-4:]:
            messages.append(msg)
        
        # Solo incluir imagen si está disponible
        if image_base64:
            messages.append({"role": "user", "content": prompt, "images": [image_base64]})
        else:
            messages.append({"role": "user", "content": prompt})
        
        try:
            response = ollama.chat(model=self.model_name, messages=messages)
            assistant_reply = response['message']['content'].strip()
            # Mostrar información detallada de tokens
            prompt_tokens = response.get('prompt_eval_count', 'N/A')
            output_tokens = response.get('eval_count', 'N/A')
            total_duration = response.get('total_duration', 'N/A')
            # Estimar tokens de imagen si está presente
            if image_base64:
                image_chars = len(image_base64)
                est_image_tokens = int(image_chars * 0.75)
                print(f"📊 Tokens - Prompt texto: {prompt_tokens} | Imagen (~): {est_image_tokens:,} | Generados: {output_tokens}")
                print(f"📊 Contexto usado: {prompt_tokens + est_image_tokens if prompt_tokens != 'N/A' else est_image_tokens:,} / 262144 disponibles")
            else:
                print(f"📊 Tokens - Prompt: {prompt_tokens} | Generados: {output_tokens} (sin imagen - ahorrando ~45k tokens)")
        except Exception as e:
            assistant_reply = f"Error en ollama: {e}"
        
        print("Response:", assistant_reply)
        self.chat_history.append({"role": "user", "content": prompt})
        self.chat_history.append({"role": "assistant", "content": assistant_reply})
        
        if assistant_reply:
            self._tts_local(assistant_reply[:300])  # Limitar longitud para TTS

    def _extract_filename(self, prompt):
        """Extrae nombre de archivo del prompt"""
        # Patrones comunes: "archivo.py", "el archivo tracker.py", "modifica app.py"
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
        # Buscar patrón: "cambio_XXX" o "change_XXX" o número
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
        # Buscar solo número razonable (1-300)
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
        self.chat_history.append({"role": "user", "content": f"Ver {filename}"})
        self.chat_history.append({"role": "assistant", "content": reply})
        self._tts_local(f"Aquí está el contenido de {filename}")
    
    def _handle_list_files(self):
        """Lista archivos Python del proyecto"""
        files, error = self.file_manager.list_python_files()
        if error:
            reply = f"❌ {error}"
        else:
            reply = "📁 Archivos Python en el proyecto:\n" + "\n".join([f"  - {f}" for f in files])
        
        print("Response:", reply)
        self.chat_history.append({"role": "user", "content": "Listar archivos"})
        self.chat_history.append({"role": "assistant", "content": reply})
        self._tts_local(f"Hay {len(files)} archivos Python en el proyecto")
    
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
        self._tts_local(reply[:200])
    
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
        self._tts_local(f"Mostrando diferencias del cambio {change_id}")
    
    def _handle_approve_change(self, change_id):
        """Aprueba un cambio pendiente"""
        # Buscar cambio por ID parcial o completo
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
        self.chat_history.append({"role": "user", "content": f"Aprobar {change_id}"})
        self.chat_history.append({"role": "assistant", "content": reply})
        self._tts_local(reply[:150])
    
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
        self.chat_history.append({"role": "user", "content": f"Rechazar {change_id}"})
        self.chat_history.append({"role": "assistant", "content": reply})
        self._tts_local(reply[:150])
    
    def _handle_file_modification(self, prompt, filename, image_base64):
        """Maneja solicitud de modificación de archivo"""
        # Leer archivo actual
        content, error = self.file_manager.read_file(filename)
        if error:
            reply = f"❌ {error}"
            print("Response:", reply)
            self._tts_local(reply)
            return
        
        # Construir mensaje para el modelo con contexto
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

        # Llamar al modelo
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Solo incluir imagen si está disponible (para no gastar tokens innecesariamente)
        if image_base64:
            messages.append({"role": "user", "content": context_prompt, "images": [image_base64]})
        else:
            messages.append({"role": "user", "content": context_prompt})
        
        try:
            response = ollama.chat(model=self.model_name, messages=messages)
            assistant_reply = response['message']['content'].strip()
            # Mostrar información detallada de tokens
            prompt_tokens = response.get('prompt_eval_count', 'N/A')
            output_tokens = response.get('eval_count', 'N/A')
            # Estimar tokens de imagen si está presente
            if image_base64:
                image_chars = len(image_base64)
                est_image_tokens = int(image_chars * 0.75)
                total_context = (prompt_tokens + est_image_tokens) if prompt_tokens != 'N/A' else est_image_tokens
                print(f"📊 Tokens - Prompt texto: {prompt_tokens} | Imagen (~): {est_image_tokens:,} | Generados: {output_tokens}")
                print(f"📊 Contexto usado: ~{total_context:,} / 262144 disponibles ({100*total_context/262144:.1f}%)")
            else:
                print(f"📊 Tokens - Prompt: {prompt_tokens} | Generados: {output_tokens} (sin imagen - ahorrando tokens)")
        except Exception as e:
            assistant_reply = f"Error generando respuesta: {e}"
        
        # Extraer código propuesto
        proposed_code = self._extract_code(assistant_reply)
        
        if proposed_code:
            # Crear propuesta de cambio
            success, result = self.file_manager.propose_change(
                filename, 
                proposed_code, 
                description=prompt[:100]
            )
            if success:
                change_id = result
                assistant_reply += f"\n\n⏳ **Cambio propuesto guardado como: `{change_id}`**\n"
                print("Response:", assistant_reply)
                self.chat_history.append({"role": "user", "content": prompt})
                self.chat_history.append({"role": "assistant", "content": assistant_reply})
                self._tts_local(f"He preparado una modificación. Revisa en pantalla y usa el teclado para decidir.")
                # Mostrar menú interactivo
                self._show_pending_change_menu(change_id)
                return
        
        print("Response:", assistant_reply)
        self.chat_history.append({"role": "user", "content": prompt})
        self.chat_history.append({"role": "assistant", "content": assistant_reply})
        self._tts_local(f"He preparado una modificación para {filename}. Revisa la propuesta y aprueba o rechaza el cambio.")

    def _handle_file_modification_smart(self, prompt, image_base64):
        """Maneja solicitud de modificación sin archivo especificado - usa LLM para elegir archivo"""
        # Obtener lista de archivos
        files, error = self.file_manager.list_python_files()
        if error:
            reply = f"❌ Error listando archivos: {error}"
            print("Response:", reply)
            self._tts_local(reply)
            return
        
        # Construir prompt para el LLM para que elija archivo y genere código
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

        # Llamar al modelo
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Solo incluir imagen si está disponible
        if image_base64:
            messages.append({"role": "user", "content": smart_prompt, "images": [image_base64]})
        else:
            messages.append({"role": "user", "content": smart_prompt})
        
        try:
            response = ollama.chat(model=self.model_name, messages=messages)
            assistant_reply = response['message']['content'].strip()
            # Mostrar información de tokens
            prompt_tokens = response.get('prompt_eval_count', 'N/A')
            output_tokens = response.get('eval_count', 'N/A')
            if image_base64:
                print(f"📊 Tokens - Prompt: {prompt_tokens} | Generados: {output_tokens} (con imagen)")
            else:
                print(f"📊 Tokens - Prompt: {prompt_tokens} | Generados: {output_tokens} (sin imagen)")
        except Exception as e:
            assistant_reply = f"Error generando respuesta: {e}"
            print("Response:", assistant_reply)
            self._tts_local(assistant_reply)
            return
        
        # Extraer nombre de archivo de la respuesta
        filename_match = re.search(r'[Aa]rchivo:\s*(\w+\.py)', assistant_reply)
        if not filename_match:
            filename_match = re.search(r'[Ff]ile:\s*(\w+\.py)', assistant_reply)
        if not filename_match:
            # Buscar cualquier .py mencionado al inicio
            filename_match = re.search(r'(\w+\.py)', assistant_reply[:200])
        
        if filename_match:
            detected_filename = filename_match.group(1)
            # Verificar que el archivo existe
            if detected_filename in files:
                # Extraer código propuesto
                proposed_code = self._extract_code(assistant_reply)
                
                if proposed_code:
                    # Crear propuesta de cambio
                    success, result = self.file_manager.propose_change(
                        detected_filename, 
                        proposed_code, 
                        description=prompt[:100]
                    )
                    if success:
                        change_id = result
                        assistant_reply += f"\n\n⏳ **Cambio propuesto guardado como: `{change_id}`**\n"
                        print("Response:", assistant_reply)
                        self.chat_history.append({"role": "user", "content": prompt})
                        self.chat_history.append({"role": "assistant", "content": assistant_reply})
                        self._tts_local(f"He preparado una modificación. Revisa en pantalla y usa el teclado para decidir.")
                        # Mostrar menú interactivo
                        self._show_pending_change_menu(change_id)
                        return  # Importante: retornar para no duplicar el print de Response
                else:
                    assistant_reply += "\n\n⚠️ No se detectó código propuesto en la respuesta."
            else:
                assistant_reply += f"\n\n⚠️ El archivo '{detected_filename}' no existe en el proyecto."
        else:
            assistant_reply += "\n\n⚠️ No pude determinar qué archivo modificar. Por favor especifica el archivo (ej: 'modifica app.py para...')."
        
        print("Response:", assistant_reply)
        self.chat_history.append({"role": "user", "content": prompt})
        self.chat_history.append({"role": "assistant", "content": assistant_reply})
        self._tts_local(f"He analizado tu solicitud. Revisa la propuesta en pantalla.")

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
            # Leer input del usuario con timeout
            import select
            import sys
            
            # Esperar input por 60 segundos
            ready, _, _ = select.select([sys.stdin], [], [], 60)
            
            if ready:
                choice = sys.stdin.readline().strip()
                
                if choice == "1":
                    # Aplicar sin commit
                    success, result = self.file_manager.approve_change(change_id, auto_commit=False)
                    if success:
                        print(f"\n✅ {result}")
                        print("💡 Usa 'git status' para ver los cambios y luego 'git commit' cuando estés listo.")
                        self._tts_local("Cambio aplicado sin commit. Tú tienes el control del git.")
                    else:
                        print(f"\n❌ Error: {result}")
                        self._tts_local("Hubo un error al aplicar el cambio")
                        
                elif choice == "2":
                    # Aplicar con commit
                    success, result = self.file_manager.approve_change(change_id, auto_commit=True)
                    if success:
                        print(f"\n✅ {result}")
                        self._tts_local("Cambio aplicado y commiteado correctamente")
                    else:
                        print(f"\n❌ Error: {result}")
                        self._tts_local("Hubo un error al aplicar el cambio")
                        
                elif choice == "3":
                    success, result = self.file_manager.reject_change(change_id)
                    if success:
                        print(f"\n🗑️ {result}")
                        self._tts_local("Cambio rechazado y eliminado")
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
                        # Volver a mostrar menú después de ver diff
                        self._show_pending_change_menu(change_id)
                        return
                    else:
                        print("\n❌ No se pudo generar el diff")
                        
                elif choice == "5":
                    print(f"\n⏳ Cambio {change_id} dejado pendiente.")
                    print("Puedes gestionarlo luego diciendo 'cambios pendientes'")
                    self._tts_local("Cambio guardado para revisar más tarde")
                    
                else:
                    print(f"\n⚠️ Opción '{choice}' no válida. Cambio dejado pendiente.")
                    self._tts_local("Opción no válida, cambio guardado para revisar luego")
                    
            else:
                print("\n⏱️ Tiempo expirado. Cambio dejado pendiente.")
                print(f"Di 'aprueba cambio {change_id}' para aplicarlo más tarde.")
                self._tts_local("Tiempo expirado, cambio guardado para revisar luego")
                
        except Exception as e:
            print(f"\n⚠️ Error en menú: {e}. Cambio dejado pendiente.")
            print(f"ID para gestionar luego: {change_id}")

    def _tts_local(self, text):
        """Texto a voz con pyttsx3 (local)"""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def transcribe_audio(self, audio_data):
        """
        Transcribe audio usando faster-whisper.
        audio_data: instancia de speech_recognition.AudioData
        """
        # Convertir AudioData a bytes WAV
        wav_bytes = audio_data.get_wav_data()
        # Usar faster-whisper
        segments, info = self.whisper_model.transcribe(
            io.BytesIO(wav_bytes),
            language=self.language,  # idioma especificado
            task="transcribe",
            beam_size=5,
            vad_filter=True,          # filtro de actividad de voz para mejor precisión
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        text = " ".join([seg.text for seg in segments])
        return text.strip()
    


    @staticmethod
    def _extract_code(text):
        """
        Extrae el primer bloque de código Markdown (encerrado entre ``` ```) de un texto.
        """
        pattern = r"```(?:\w*)\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
# -------------------------------------------------------------------
# Captura de audio en segundo plano con speech_recognition
# -------------------------------------------------------------------
def audio_callback(recognizer, audio):
    """Se ejecuta cuando se detecta voz en el micrófono"""
    try:
        # Usar nuestro asistente para transcribir localmente
        prompt = assistant.transcribe_audio(audio)
        # Obtener imagen actual de la pantalla
        image_b64 = screen_stream.read(encode=True)
        if image_b64 is None:
            print("Imagen no disponible aún, esperando...")
            return
        # Obtener respuesta del modelo
        assistant.answer(prompt, image_b64)
    except sr.UnknownValueError:
        print("No se entendió el audio")
    except Exception as e:
        print(f"Error procesando audio: {e}")

# -------------------------------------------------------------------
# Inicialización
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Iniciar captura de pantalla
    print("Iniciando captura de pantalla...")
    # Para ver los monitores disponibles, podemos imprimirlos (opcional)
    with mss.mss() as sct:
        print("Monitores detectados:", sct.monitors)
    # Elegir monitor: 1 para el principal, 2 para el segundo, etc.
    # max_width=896 para mantener imágenes ~30-40k tokens (vs 258k)
    screen_stream = ScreenStream(monitor=1, scale_display=0.5, max_width=896, jpeg_quality=45).start()
    print("Captura de pantalla iniciada.")

    # Crear asistente con idioma español y timeout de visión de 10 segundos
    assistant = Assistant(model_name="qwen3-vl:8b-extreme", language="es", project_path=PROJECT_PATH, vision_timeout=5)
    
    # Mostrar información inicial
    print("\n" + "="*60)
    print("🤖 ASISTENTE DE CÓDIGO CON VISIÓN")
    print("="*60)
    print(f"📁 Proyecto objetivo: {PROJECT_PATH}")
    print(f"📂 Sugerencias pendientes: {SUGGESTIONS_PATH}")
    print("\n📋 COMANDOS DISPONIBLES:")
    print("   • 'muestrame [archivo.py]' - Ver contenido de archivo")
    print("   • 'lista archivos' - Listar archivos Python")
    print("   • 'modifica [archivo.py] para...' - Proponer cambio")
    print("   • 'cambios pendientes' - Ver cambios propuestos")
    print("   • 'aprueba cambio [ID]' - Aplicar cambio")
    print("   • 'rechaza cambio [ID]' - Descartar cambio")
    print("   • 'ver diferencias [ID]' - Ver diff del cambio")
    print("   • 'activa vision [segundos]' - Activar modo visión temporalmente")
    print("   • 'desactiva vision' - Desactivar modo visión")
    print("   • 'q' o ESC - Salir")
    print("="*60 + "\n")

    # Configurar reconocimiento de voz
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 4.0
    recognizer.phrase_threshold = 0.3
    
    # Obtener micrófono
    microphone = sr.Microphone()
    with microphone as source:
        print("Calibrando micrófono para ruido ambiente...")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print("Micrófono calibrado.")
    
    # Variable para controlar si estamos escuchando
    is_listening = False
    stop_listening = None
    
    def toggle_voice_listening(enable):
        """Activa o desactiva la escucha de voz en segundo plano"""
        global is_listening, stop_listening
        if enable and not is_listening:
            print("🎤 Escucha por voz ACTIVADA")
            stop_listening = recognizer.listen_in_background(microphone, audio_callback)
            is_listening = True
        elif not enable and is_listening:
            print("🛑 Escucha por voz DESACTIVADA")
            if stop_listening:
                stop_listening(wait_for_stop=False)
            is_listening = False
    
    def show_input_menu():
        """Muestra menú para elegir método de entrada"""
        print("\n" + "="*50)
        print("🎯 SELECCIONA MÉTODO DE ENTRADA:")
        print("="*50)
        print("  [1] ⌨️  Escribir prompt (teclado)")
        print("  [2] 🎤 Hablar prompt (voz)")
        print("  [q] 🚪 Salir")
        print("="*50)
        print("Ingresa opción (1/2/q): ", end="", flush=True)
    
    def process_text_input():
        """Procesa entrada por teclado"""
        try:
            prompt = input().strip()
            if prompt.lower() in ['q', 'salir', 'exit']:
                return False
            if prompt:
                image_b64 = screen_stream.read(encode=True)
                if image_b64:
                    assistant.answer(prompt, image_b64)
                else:
                    print("⚠️ Imagen no disponible, esperando...")
            return True
        except EOFError:
            return False
        except Exception as e:
            print(f"❌ Error procesando entrada: {e}")
            return True
    
    print("\n✅ Asistente listo. Elige cómo quieres interactuar.")
    
    # Flag para controlar si el menú ya fue mostrado
    menu_shown = False
    
    # Bucle principal: mostrar pantalla y gestionar entrada
    try:
        while True:
            frame_display = screen_stream.read_display()
            cv2.imshow("Screen Capture", frame_display)
            
            # Mostrar menú solo una vez (si no está escuchando por voz)
            if not is_listening:
                if not menu_shown:
                    show_input_menu()
                    menu_shown = True
                
                # Esperar input con timeout para mantener la ventana responsive
                import select
                import sys
                
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if ready:
                    choice = sys.stdin.readline().strip()
                    menu_shown = False  # Resetear para mostrar menú nuevamente
                    
                    if choice.lower() == 'q':
                        break
                    elif choice == '1':
                        # Entrada por teclado
                        toggle_voice_listening(False)
                        print("\n✏️ Escribe tu prompt y presiona ENTER:")
                        print("> ", end="", flush=True)
                        if not process_text_input():
                            break
                    elif choice == '2':
                        # Entrada por voz
                        print("\n🎤 Habla ahora... (la escucha está activa)")
                        print("   Presiona Ctrl+C o espera para volver al menú")
                        toggle_voice_listening(True)
                        try:
                            # Esperar mientras escucha
                            time.sleep(2)
                        except KeyboardInterrupt:
                            toggle_voice_listening(False)
                    else:
                        print(f"\n⚠️ Opción '{choice}' no válida")
            else:
                # Está escuchando por voz, solo actualizar ventana
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    toggle_voice_listening(False)
                    break
                time.sleep(0.1)
                
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
                
    except KeyboardInterrupt:
        print("\n\nInterrupción por teclado.")
    finally:
        print("\nDeteniendo...")
        if is_listening and stop_listening:
            stop_listening(wait_for_stop=False)
        screen_stream.stop()
        cv2.destroyAllWindows()
        print("Programa terminado.")