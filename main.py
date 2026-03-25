"""
Punto de entrada principal del asistente de voz y visión.
"""
import sys
import time

import cv2
import mss
import speech_recognition as sr

from assistant_core import Assistant
from config import (
    DEFAULT_JPEG_QUALITY,
    DEFAULT_MAX_WIDTH,
    DEFAULT_MODEL,
    DEFAULT_MONITOR,
    DEFAULT_SCALE_DISPLAY,
    PROJECT_PATH,
    SUGGESTIONS_PATH,
)
from screen_capture import ScreenStream


# Variable global para el callback de audio
assistant = None
screen_stream = None


def audio_callback(recognizer, audio):
    """Se ejecuta cuando se detecta voz en el micrófono"""
    try:
        prompt = assistant.voice.transcribe(audio)
        image_b64 = screen_stream.read(encode=True)
        if image_b64 is None:
            print("Imagen no disponible aún, esperando...")
            return
        assistant.answer(prompt, image_b64)
    except sr.UnknownValueError:
        print("No se entendió el audio")
    except Exception as e:
        print(f"Error procesando audio: {e}")


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


def main():
    global assistant, screen_stream

    # Iniciar captura de pantalla
    print("Iniciando captura de pantalla...")
    with mss.mss() as sct:
        print("Monitores detectados:", sct.monitors)

    screen_stream = ScreenStream(
        monitor=DEFAULT_MONITOR,
        scale_display=DEFAULT_SCALE_DISPLAY,
        max_width=DEFAULT_MAX_WIDTH,
        jpeg_quality=DEFAULT_JPEG_QUALITY
    ).start()
    print("Captura de pantalla iniciada.")

    # Crear asistente
    assistant = Assistant(
        model_name=DEFAULT_MODEL,
        language="es",
        project_path=PROJECT_PATH,
        vision_timeout=5
    )

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

    microphone = sr.Microphone()
    with microphone as source:
        print("Calibrando micrófono para ruido ambiente...")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print("Micrófono calibrado.")

    # Variables de control
    is_listening = False
    stop_listening = None

    def toggle_voice_listening(enable):
        """Activa o desactiva la escucha de voz en segundo plano"""
        nonlocal is_listening, stop_listening
        if enable and not is_listening:
            print("🎤 Escucha por voz ACTIVADA")
            stop_listening = recognizer.listen_in_background(microphone, audio_callback)
            is_listening = True
        elif not enable and is_listening:
            print("🛑 Escucha por voz DESACTIVADA")
            if stop_listening:
                stop_listening(wait_for_stop=False)
            is_listening = False

    print("\n✅ Asistente listo. Elige cómo quieres interactuar.")

    menu_shown = False

    # Bucle principal
    try:
        while True:
            frame_display = screen_stream.read_display()
            cv2.imshow("Screen Capture", frame_display)

            if not is_listening:
                if not menu_shown:
                    show_input_menu()
                    menu_shown = True

                import select

                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if ready:
                    choice = sys.stdin.readline().strip()
                    menu_shown = False

                    if choice.lower() == 'q':
                        break
                    elif choice == '1':
                        toggle_voice_listening(False)
                        print("\n✏️ Escribe tu prompt y presiona ENTER:")
                        print("> ", end="", flush=True)
                        if not process_text_input():
                            break
                    elif choice == '2':
                        print("\n🎤 Habla ahora... (la escucha está activa)")
                        print("   Presiona Ctrl+C o espera para volver al menú")
                        toggle_voice_listening(True)
                        try:
                            time.sleep(2)
                        except KeyboardInterrupt:
                            toggle_voice_listening(False)
                    else:
                        print(f"\n⚠️ Opción '{choice}' no válida")
            else:
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


if __name__ == "__main__":
    main()
