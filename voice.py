"""
Módulo de voz: Text-to-Speech (TTS) y Speech-to-Text (STT).
"""
import io

import pyttsx3
import speech_recognition as sr
from faster_whisper import WhisperModel

from config import (
    DEFAULT_LANGUAGE,
    DEFAULT_TTS_RATE,
    DEFAULT_TTS_VOLUME,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL_SIZE,
)


class VoiceManager:
    """Gestiona TTS (texto a voz) y STT (voz a texto)"""

    def __init__(self, language=DEFAULT_LANGUAGE):
        self.language = language
        self.tts_engine = None
        self.whisper_model = None
        self._init_tts()
        self._init_whisper()

    def _init_tts(self):
        """Inicializa el motor TTS local"""
        self.tts_engine = pyttsx3.init()
        # Ajustar velocidad
        current_rate = self.tts_engine.getProperty('rate')
        self.tts_engine.setProperty('rate', DEFAULT_TTS_RATE)
        print(f"Velocidad de voz ajustada a {DEFAULT_TTS_RATE} (anterior: {current_rate})")

        # Ajustar volumen
        self.tts_engine.setProperty('volume', DEFAULT_TTS_VOLUME)

        # Seleccionar voz en español si está disponible
        voices = self.tts_engine.getProperty('voices')
        spanish_voice = None
        for voice in voices:
            voice_name = voice.name.lower()
            voice_id = voice.id.lower()
            if any(x in voice_name for x in ['spanish', 'español', 'es_', 'mb-es']):
                spanish_voice = voice.id
                print(f"✓ Voz encontrada: {voice.name}")
                break
            if 'mb-es' in voice_id or 'es_' in voice_id:
                spanish_voice = voice.id
                print(f"✓ Voz encontrada por ID: {voice.name}")
                break

        if spanish_voice:
            self.tts_engine.setProperty('voice', spanish_voice)
            print("Voz en español seleccionada.")
        else:
            print("No se encontró voz en español, se usará la predeterminada.")

    def _init_whisper(self):
        """Inicializa el modelo Whisper para STT"""
        print(f"Cargando modelo Whisper en GPU para idioma: {self.language}...")
        self.whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE
        )
        print("Modelo Whisper listo.")

    def speak(self, text):
        """Convierte texto a voz"""
        if self.tts_engine:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()

    def transcribe(self, audio_data):
        """
        Transcribe audio usando faster-whisper.
        audio_data: instancia de speech_recognition.AudioData
        """
        if not self.whisper_model:
            return ""

        # Convertir AudioData a bytes WAV
        wav_bytes = audio_data.get_wav_data()
        # Usar faster-whisper
        segments, info = self.whisper_model.transcribe(
            io.BytesIO(wav_bytes),
            language=self.language,
            task="transcribe",
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        text = " ".join([seg.text for seg in segments])
        return text.strip()
