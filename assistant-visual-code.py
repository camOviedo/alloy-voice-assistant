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

# Opcional: suprimir warnings de ALSA y Qt (puedes descomentar si quieres)
# os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
# os.environ['ALSA_CARD'] = "Generic"

# -------------------------------------------------------------------
# Clase ScreenStream (corregida para evitar error de display en hilos)
# -------------------------------------------------------------------
class ScreenStream:
    def __init__(self, monitor=1, scale_display=0.5):
        """
        monitor: índice del monitor a capturar (1 = principal, 2 = secundario, etc.)
        scale_display: factor de escala para la ventana de previsualización
        """
        self.monitor_index = monitor
        self.scale_display = scale_display
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
            # Comprimir a JPEG con calidad 70 para menor tamaño
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            _, buffer = cv2.imencode(".jpeg", frame, encode_params)
            return base64.b64encode(buffer).decode('utf-8')
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
    def __init__(self, model_name="qwen3-vl:8b", language="es"):
        """
        model_name: modelo en Ollama (ej: "qwen2.5-vl:7b")
        language: idioma para Whisper y para el prompt del sistema (es, en, etc.)
        """
        self.model_name = model_name
        self.language = language
        
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
            self.system_prompt = """
            Eres un asistente experto en visión por computadora y programación en Python. 
            Tu tarea es ayudar a mejorar un sistema de reconocimiento de caballos. 
            Cuando el usuario te pida modificar el código, debes generar el nuevo código 
            explicando los cambios. Siempre que sea posible, incluye el código completo 
            dentro de bloques markdown ```python ... ```.
            """
        else:
            self.system_prompt = """
            You are an expert assistant in computer vision and Python programming.
            Your task is to help improve a horse recognition system.
            When the user asks you to modify the code, you must generate the new code explaining the changes.
            Whenever possible, include the complete code within markdown blocks ```python ...  ```.
            """

    def answer(self, prompt, image_base64):
        if not prompt or not prompt.strip():
            print("Prompt vacío, ignorando.")
            return
        
        if image_base64 is None:
            print("Error: imagen no disponible.")
            return

        print("Prompt:", prompt)

        # Construir mensajes para Ollama
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Agregar historial (últimos 6 intercambios para no saturar)
        for msg in self.chat_history[-6:]:
            messages.append(msg)
        
        # Mensaje actual del usuario con imagen
        user_msg = {
            "role": "user",
            "content": prompt,
            "images": [image_base64]
        }
        messages.append(user_msg)

        # Llamada a Ollama
        try:
            response = ollama.chat(model=self.model_name, messages=messages)
            assistant_reply = response['message']['content'].strip()
        except Exception as e:
            print(f"Error en ollama: {e}")
            assistant_reply = "Lo siento, tuve un problema al procesar la imagen."

        print("Response:", assistant_reply)

        # Guardar en historial
        self.chat_history.append({"role": "user", "content": prompt})
        self.chat_history.append({"role": "assistant", "content": assistant_reply})

        # Detectar si el prompt pide modificar código
        if "modifica" in prompt.lower() or "cambia el código" in prompt.lower():
            # Extraer el bloque de código de la respuesta (p.ej. entre ```python ... ```)
            codigo_extraido = self._extract_code(assistant_reply)
            if codigo_extraido:
                # Guardar en un archivo (con timestamp para no sobrescribir)
                filename = f"sugerencia_{int(time.time())}.py"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(codigo_extraido)
                print(f"Código guardado en {filename}")
                assistant_reply += f" He guardado el código en {filename}. Revísalo antes de usarlo."

        # TTS local
        if assistant_reply:
            self._tts_local(assistant_reply)

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
    screen_stream = ScreenStream(monitor=1, scale_display=0.5).start()
    print("Captura de pantalla iniciada.")

    # Crear asistente con idioma español
    assistant = Assistant(model_name="qwen3-vl:8b", language="es")

    # Configurar reconocimiento de voz
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300  # ajuste fino (puede necesitar calibración)
    recognizer.dynamic_energy_threshold = True
    
    # Obtener micrófono
    microphone = sr.Microphone()
    with microphone as source:
        print("Calibrando micrófono para ruido ambiente...")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print("Micrófono calibrado.")

    # Escuchar en segundo plano
    print("Escuchando en segundo plano. Presiona 'q' en la ventana para salir.")
    stop_listening = recognizer.listen_in_background(microphone, audio_callback)

    # Bucle principal: mostrar la pantalla capturada
    try:
        while True:
            frame_display = screen_stream.read_display()
            cv2.imshow("Screen Capture", frame_display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' o ESC
                break
    except KeyboardInterrupt:
        print("\nInterrupción por teclado.")
    finally:
        print("Deteniendo...")
        screen_stream.stop()
        cv2.destroyAllWindows()
        stop_listening(wait_for_stop=False)
        print("Programa terminado.")