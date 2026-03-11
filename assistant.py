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
# Opción: usar ChatOllama de langchain (requiere langchain-ollama)
# from langchain_ollama import ChatOllama

# Para Whisper local
from faster_whisper import WhisperModel
import speech_recognition as sr
import io
import wave

# Opcional: TTS local
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
# Asistente con modelo local (Ollama + Whisper + TTS opcional)
# -------------------------------------------------------------------
class Assistant:
    def __init__(self, model_name="qwen2.5-vl:7b"):
        """
        model_name: nombre del modelo en Ollama (ej: "qwen2.5-vl:7b")
        """
        self.model_name = model_name
        # Inicializar Whisper local en GPU con FP16 para máximo rendimiento
        print("Cargando modelo Whisper en GPU...")
        self.whisper_model = WhisperModel(
            "base",                     # o "tiny", "small", "medium", "large-v3"
            device="cuda",               # <--- CAMBIO CLAVE: Usar GPU
            compute_type="float16"       # <--- CAMBIO CLAVE: Usar FP16 para aceleración
        )
        print("Modelo Whisper listo en GPU.")
        # Inicializar TTS local (opcional)
        self.tts_engine = pyttsx3.init()  # si no quieres TTS, comenta esta línea
        # Historial de conversación
        self.chat_history = ChatMessageHistory()
        # System prompt
        self.system_prompt = """
        You are a witty assistant that will use the chat history and the image 
        provided by the user to answer its questions. Your job is to answer 
        questions.

        Use few words on your answers. Go straight to the point. Do not use any
        emoticons or emojis. 

        Be friendly and helpful. Show some personality.
        """

    def answer(self, prompt, image_base64):
        if not prompt:
            return
        print("Prompt:", prompt)

        # Preparar mensajes para Ollama
        messages = [
            {"role": "system", "content": self.system_prompt},
            # Agregar historial previo (últimos 6 mensajes para no saturar)
            *[{"role": msg.type, "content": msg.content} for msg in self.chat_history.messages[-6:]],
            {
                "role": "user",
                "content": prompt,
                "images": [image_base64]   # Imagen en base64
            }
        ]

        # Llamada a Ollama
        response = ollama.chat(model=self.model_name, messages=messages)
        assistant_reply = response['message']['content'].strip()

        print("Response:", assistant_reply)

        # Guardar en historial (como objetos LangChain)
        self.chat_history.add_user_message(prompt)
        self.chat_history.add_ai_message(assistant_reply)

        # Opcional: TTS local
        if assistant_reply:
            self._tts_local(assistant_reply)

    def _tts_local(self, text):
        """Texto a voz con pyttsx3 (local)"""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    # También podemos mantener un método TTS con OpenAI si aún lo quieres
    # pero aquí todo es local.

    def transcribe_audio(self, audio_data):
        """
        Transcribe audio proveniente de speech_recognition usando faster-whisper.
        audio_data: instancia de speech_recognition.AudioData
        """
        # Convertir AudioData a bytes WAV
        wav_bytes = audio_data.get_wav_data()
        # faster-whisper lee directamente de un buffer de memoria
        segments, info = self.whisper_model.transcribe(
            io.BytesIO(wav_bytes),
            language="en",          # puedes cambiarlo o dejarlo None para autodetección
            task="transcribe",
            beam_size=5
        )
        text = " ".join([seg.text for seg in segments])
        return text

# -------------------------------------------------------------------
# Captura de audio en segundo plano con speech_recognition
# -------------------------------------------------------------------
def audio_callback(recognizer, audio):
    """Se ejecuta cuando se detecta voz en el micrófono"""
    try:
        # Usar nuestro asistente para transcribir localmente
        prompt = assistant.transcribe_audio(audio)
        # Obtener imagen actual de la webcam
        image_b64 = webcam_stream.read(encode=True)
        # Obtener respuesta del modelo
        assistant.answer(prompt, image_b64)
    except Exception as e:
        print(f"Error procesando audio: {e}")

# -------------------------------------------------------------------
# Inicialización
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Iniciar webcam
    webcam_stream = WebcamStream().start()

    # Crear asistente con modelo local (ajusta el nombre si es otro)
    assistant = Assistant(model_name="qwen3-vl:8b")

    # Configurar reconocimiento de voz con speech_recognition
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)

    # Escuchar en segundo plano (usa un hilo)
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