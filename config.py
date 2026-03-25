"""
Configuración central del asistente de voz y visión.
"""
import os

# -------------------------------------------------------------------
# Configuración del proyecto objetivo
# -------------------------------------------------------------------
PROJECT_PATH = os.path.expanduser("~/Proyectos/reconocimiento_caballos")
SUGGESTIONS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sugerencias_pendientes")

# Crear directorio para sugerencias pendientes
os.makedirs(SUGGESTIONS_PATH, exist_ok=True)

# -------------------------------------------------------------------
# Configuración del modelo LLM
# -------------------------------------------------------------------
DEFAULT_MODEL = "qwen3-vl:8b-extreme"
DEFAULT_LANGUAGE = "es"
DEFAULT_VISION_TIMEOUT = 5

# -------------------------------------------------------------------
# Configuración de captura de pantalla
# -------------------------------------------------------------------
DEFAULT_MONITOR = 1
DEFAULT_SCALE_DISPLAY = 0.5
DEFAULT_MAX_WIDTH = 896
DEFAULT_JPEG_QUALITY = 45

# -------------------------------------------------------------------
# Configuración de voz
# -------------------------------------------------------------------
DEFAULT_TTS_RATE = 140
DEFAULT_TTS_VOLUME = 0.9

# -------------------------------------------------------------------
# Configuración de Whisper
# -------------------------------------------------------------------
WHISPER_MODEL_SIZE = "base"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"
