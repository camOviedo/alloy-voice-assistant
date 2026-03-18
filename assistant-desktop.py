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
        # Configurar voz en español si está disponible
        voices = self.tts_engine.getProperty('voices')
        spanish_voice = None
        for voice in voices:
            if 'spanish' in voice.name.lower() or 'español' in voice.name.lower() or 'es_' in voice.id:
                spanish_voice = voice.id
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
Eres un asistente ingenioso que usará el historial de la conversación y la imagen 
proporcionada por el usuario para responder a sus preguntas. Tu trabajo es responder 
preguntas.

Usa pocas palabras en tus respuestas. Ve directo al grano. No uses emoticonos ni emojis.

Sé amigable y útil. Muestra algo de personalidad.
"""
        else:
            self.system_prompt = """
You are a witty assistant that will use the chat history and the image 
provided by the user to answer its questions. Your job is to answer 
questions.

Use few words on your answers. Go straight to the point. Do not use any
emoticons or emojis. 

Be friendly and helpful. Show some personality.
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