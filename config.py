"""
Configuración central del asistente de voz y visión.
"""
import os

# -------------------------------------------------------------------
# Configuración del proyecto objetivo
# -------------------------------------------------------------------
PROJECT_PATH = os.path.expanduser("~/Proyectos/reconocimiento_caballos")
SUGGESTIONS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sugerencias_pendientes")
CAPTURES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")

# Crear directorios necesarios
os.makedirs(SUGGESTIONS_PATH, exist_ok=True)
os.makedirs(CAPTURES_PATH, exist_ok=True)

# -------------------------------------------------------------------
# Configuración de agentes LangChain (multi-agente)
# -------------------------------------------------------------------
AGENT_COORDINATOR_MODEL = "qwen2.5:3b"  # Pequeño, rápido, texto-only
AGENT_VISION_MODEL = "qwen3-vl:8b-vision"     # Vision-capable
AGENT_CODE_MODEL = "qwen2.5-coder:14b"       # Analisis de codigo
AGENT_EDITOR_MODEL = "qwen2.5-coder:14b"  # Generacion de codigo

# Cuantización recomendada:
# - Vision: Q3 (más rápido)
# - Code: Q4 (mejor calidad)
# - Editor: sin cuantizar o Q4 (máxima calidad)

# -------------------------------------------------------------------
# Configuración del modelo LLM (legacy - para compatibilidad)
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

# Configuración de captura de video para modo visión
# Duración total de la captura en segundos
VISION_CAPTURE_DURATION = 20  # segundos
# Frames por segundo a capturar (cantidad de screenshots = DURATION × FPS)
VISION_CAPTURE_FPS = 1  # 1 fps = 20 screenshots en 20 segundos

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
