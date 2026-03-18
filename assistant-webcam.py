import base64
import cv2
import numpy as np
import ollama
from threading import Lock, Thread
from cv2 import VideoCapture, imencode
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

# Para Whisper local
from faster_whisper import WhisperModel
import speech_recognition as sr
import io
import wave

# TTS local
import pyttsx3

# -------------------------------------------------------------------
# Clase WebcamStream (sin cambios)
# -------------------------------------------------------------------
class WebcamStream:
    def __init__(self):
        self.stream = VideoCapture(index=0)
        _, self.frame = self.stream.read()
        self.running = False
        self.lock = Lock()

    def start(self):
        if self.running:
            return self
        self.running = True
        self.thread = Thread(target=self.update)
        self.thread.start()
        return self

    def update(self):
        while self.running:
            _, frame = self.stream.read()
            with self.lock:
                self.frame = frame

    def read(self, encode=False):
        with self.lock:
            frame = self.frame.copy()
        if encode:
            _, buffer = imencode(".jpeg", frame)
            return base64.b64encode(buffer).decode('utf-8')
        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.stream.release()

# -------------------------------------------------------------------
# Asistente con modelo local y soporte para español
# -------------------------------------------------------------------
class Assistant:
    def __init__(self, model_name="qwen3-vl:8b", language="es"):
        """
        model_name: nombre del modelo en Ollama (ej: "qwen2.5-vl:7b")
        language: idioma para Whisper y para las interacciones ('es' para español)
        """
        self.model_name = model_name
        self.language = language

        # Inicializar Whisper local con GPU (si está disponible)
        print(f"Cargando modelo Whisper en GPU para idioma: {self.language}...")
        self.whisper_model = WhisperModel(
            "base",                # puedes cambiarlo a "small", "medium", etc.
            device="cuda",
            compute_type="float16"
        )
        print("Modelo Whisper listo.")

        # Inicializar TTS local
        self.tts_engine = pyttsx3.init(driverName='espeak')
        # Configurar voz en español si está disponible
        self._set_spanish_voice()

        # Historial de conversación
        self.chat_history = ChatMessageHistory()

        # System prompt en español (puedes modificarlo a tu gusto)
        self.system_prompt = """
        Eres un asistente ingenioso que usará el historial de la conversación y la imagen 
        proporcionada por el usuario para responder sus preguntas. Tu trabajo es responder 
        preguntas.

        Usa pocas palabras en tus respuestas. Ve directo al grano. No uses emoticonos ni emojis.

        Sé amable y útil. Muestra algo de personalidad.
        """

    def _set_spanish_voice(self):
        """Configura una voz en español para pyttsx3 si está disponible."""
        voices = self.tts_engine.getProperty('voices')
        # Buscar una voz que contenga 'spanish' o 'español' en el id o nombre
        spanish_voice = None
        for voice in voices:
            # Algunas voces tienen el idioma en el id, otras en el nombre
            if 'spanish' in voice.id.lower() or 'español' in voice.id.lower() or \
               'es' in voice.id.lower() or 'spanish' in voice.name.lower():
                spanish_voice = voice.id
                break
        if spanish_voice:
            self.tts_engine.setProperty('voice', spanish_voice)
            print("Voz en español seleccionada.")
        else:
            print("No se encontró voz en español, se usará la voz por defecto.")

    def answer(self, prompt, image_base64):
        if not prompt:
            return
        print("Prompt:", prompt)

        # Preparar mensajes para Ollama
        messages = [
            {"role": "system", "content": self.system_prompt},
            # Agregar historial previo (últimos 6 mensajes)
            *[{"role": msg.type, "content": msg.content} for msg in self.chat_history.messages[-6:]],
            {
                "role": "user",
                "content": prompt,
                "images": [image_base64]
            }
        ]

        # Llamada a Ollama
        response = ollama.chat(model=self.model_name, messages=messages)
        assistant_reply = response['message']['content'].strip()

        print("Response:", assistant_reply)

        # Guardar en historial
        self.chat_history.add_user_message(prompt)
        self.chat_history.add_ai_message(assistant_reply)

        # TTS local (en español, si se configuró la voz)
        if assistant_reply:
            self._tts_local(assistant_reply)

    def _tts_local(self, text):
        """Texto a voz con pyttsx3 (local)"""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def transcribe_audio(self, audio_data):
        """
        Transcribe audio usando faster-whisper con el idioma especificado.
        """
        wav_bytes = audio_data.get_wav_data()
        segments, info = self.whisper_model.transcribe(
            io.BytesIO(wav_bytes),
            language=self.language,   # Usamos el idioma configurado
            task="transcribe",
            beam_size=5
        )
        text = " ".join([seg.text for seg in segments])
        return text

# -------------------------------------------------------------------
# Captura de audio en segundo plano
# -------------------------------------------------------------------
def audio_callback(recognizer, audio):
    try:
        prompt = assistant.transcribe_audio(audio)
        image_b64 = webcam_stream.read(encode=True)
        assistant.answer(prompt, image_b64)
    except Exception as e:
        print(f"Error procesando audio: {e}")

# -------------------------------------------------------------------
# Inicialización
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Iniciar webcam
    webcam_stream = WebcamStream().start()

    # Crear asistente con modelo local y español
    assistant = Assistant(model_name="qwen3-vl:8b", language="es")

    # Configurar reconocimiento de voz con speech_recognition
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)

    # Escuchar en segundo plano
    stop_listening = recognizer.listen_in_background(microphone, audio_callback)

    # Bucle principal: mostrar la webcam
    try:
        while True:
            cv2.imshow("Webcam", webcam_stream.read())
            if cv2.waitKey(1) in [27, ord("q")]:
                break
    finally:
        webcam_stream.stop()
        cv2.destroyAllWindows()
        stop_listening(wait_for_stop=False)