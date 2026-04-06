"""
Punto de entrada principal del asistente de voz y visión.
"""
import sys
import threading
import time
from queue import Queue

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
vision_active = False

# Variables para procesamiento asíncrono de prompts
pending_result_queue = Queue()
current_prompt_thread = None


def start_vision():
    """Inicia la captura de pantalla y el modo visión"""
    global screen_stream, vision_active
    if screen_stream is None or not vision_active:
        print("\n👁️ Iniciando modo visión...")
        with mss.mss() as sct:
            print("Monitores detectados:", sct.monitors)

        screen_stream = ScreenStream(
            monitor=DEFAULT_MONITOR,
            scale_display=DEFAULT_SCALE_DISPLAY,
            max_width=DEFAULT_MAX_WIDTH,
            jpeg_quality=DEFAULT_JPEG_QUALITY
        ).start()
        vision_active = True
        print("✅ Captura de pantalla iniciada.")
        assistant.vision_active_until = time.time() + 3600  # 1 hora por defecto
        return True
    return False


def stop_vision():
    """Detiene la captura de pantalla y el modo visión"""
    global screen_stream, vision_active
    if screen_stream and vision_active:
        print("\n🛑 Deteniendo modo visión...")
        screen_stream.stop()
        screen_stream = None
        vision_active = False
        assistant.vision_active_until = 0
        cv2.destroyAllWindows()
        print("✅ Captura de pantalla detenida.")
        return True
    return False


def toggle_vision(enable):
    """Activa o desactiva el modo visión"""
    if enable:
        return start_vision()
    else:
        return stop_vision()


def audio_callback(recognizer, audio):
    """Se ejecuta cuando se detecta voz en el micrófono"""
    global current_prompt_thread
    try:
        prompt = assistant.voice.transcribe(audio)
        
        # La captura de imagen se maneja internamente en assistant_core
        # cuando se detectan palabras clave de visión
        
        # Ejecutar assistant.answer() en un hilo separado
        def run_assistant():
            try:
                assistant.answer(prompt)
                pending_result_queue.put(('done', None))
            except Exception as e:
                pending_result_queue.put(('error', str(e)))

        current_prompt_thread = threading.Thread(target=run_assistant, daemon=True)
        current_prompt_thread.start()
        print("⏳ Procesando entrada de voz...")
    except sr.UnknownValueError:
        print("No se entendió el audio")
    except Exception as e:
        print(f"Error procesando audio: {e}")


def show_input_menu():
    """Muestra menú para elegir método de entrada"""
    status_icon = "🟢" if vision_active else "🔴"
    print("\n" + "="*50)
    print("🎯 SELECCIONA MÉTODO DE ENTRADA:")
    print("="*50)
    print(f"  [1] ⌨️  Escribir prompt (teclado)")
    print(f"  [2] 🎤 Hablar prompt (voz)")
    print(f"  [v] {status_icon} {'Desactivar' if vision_active else 'Activar'} visión")
    print("  [q] 🚪 Salir")
    print("="*50)
    print(f"Estado visión: {'ACTIVA' if vision_active else 'INACTIVA'}")
    print("Ingresa opción (1/2/v/q): ", end="", flush=True)


def drain_stdin(timeout=1.0):
    """Drena todo el contenido residual de stdin usando múltiples métodos"""
    import select
    import time
    import sys
    import termios
    
    # Método 1: Intentar limpiar buffer del terminal con TCIFLUSH
    try:
        # Obtener atributos del terminal
        fd = sys.stdin.fileno()
        old_attr = termios.tcgetattr(fd)
        # Limpiar buffer de entrada del terminal
        termios.tcflush(fd, termios.TCIFLUSH)
    except:
        pass
    
    # Método 2: Leer todo lo disponible con select
    start_time = time.time()
    last_data_time = start_time
    while (time.time() - start_time) < timeout:
        ready, _, _ = select.select([sys.stdin], [], [], 0.05)
        if ready:
            try:
                # Leer en chunks más grandes
                chunk = sys.stdin.read(4096)
                if chunk:
                    last_data_time = time.time()
                else:
                    if (time.time() - last_data_time) > 0.2:
                        break
            except:
                break
        else:
            if (time.time() - last_data_time) > 0.2:
                break
        time.sleep(0.01)


def process_text_input_async():
    """Procesa entrada por teclado de forma asíncrona (no bloqueante)"""
    global current_prompt_thread
    import select
    try:
        prompt = input().strip()
        # Leer líneas adicionales si están disponibles (para pegar textos multilinea)
        # Usar timeout más largo para asegurar que todo el pegado se complete
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.2)
            if ready:
                extra_line = sys.stdin.readline()
                if extra_line:
                    prompt += "\n" + extra_line.rstrip('\n')
                else:
                    break
            else:
                break

        # Drenar cualquier contenido residual
        drain_stdin(timeout=1.0)

        if prompt.lower() in ['q', 'salir', 'exit']:
            return False, None
        if prompt:
            # La captura de imagen se maneja internamente en assistant_core
            # cuando se detectan palabras clave de visión

            # Ejecutar assistant.answer() en un hilo separado
            def run_assistant():
                try:
                    assistant.answer(prompt)
                    pending_result_queue.put(('done', None))
                except Exception as e:
                    pending_result_queue.put(('error', str(e)))

            current_prompt_thread = threading.Thread(target=run_assistant, daemon=True)
            current_prompt_thread.start()
            return True, 'processing'
        return True, None
    except EOFError:
        return False, None
    except Exception as e:
        print(f"❌ Error procesando entrada: {e}")
        return True, None


def process_pending_result():
    """Procesa el resultado pendiente si está listo"""
    try:
        status, data = pending_result_queue.get_nowait()
        return status, data
    except:
        return None, None


def main():
    global assistant, screen_stream, vision_active

    # Crear asistente (la captura de pantalla se maneja internamente)
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
    print("\n👁️ MODO VISIÓN:")
    print("   La captura se activa AUTOMÁTICAMENTE cuando detectas")
    print("   palabras como: 'pantalla', 'imagen', 'mira', 'video'...")
    print("   • 'activa vision' - Activar captura continua manual")
    print("   • 'desactiva vision' - Desactivar modo visión")
    print("   • 'v' en menú - Toggle visión continua")
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
            # Mostrar ventana de captura solo si está activa
            if vision_active and screen_stream:
                frame_display = screen_stream.read_display()
                if frame_display is not None:
                    cv2.imshow("Screen Capture", frame_display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break

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
                        should_continue, status = process_text_input_async()
                        if not should_continue:
                            break
                        # Si se está procesando un prompt, mantener bucle activo mostrando visión
                        if status == 'processing':
                            print("⏳ Procesando prompt... (presiona 'q' para salir)")
                            processing = True
                            while processing:
                                # Mantener ventana de captura activa mientras procesa
                                if vision_active and screen_stream:
                                    frame_display = screen_stream.read_display()
                                    if frame_display is not None:
                                        cv2.imshow("Screen Capture", frame_display)
                                key = cv2.waitKey(50) & 0xFF
                                if key == ord('q') or key == 27:
                                    break
                                # Verificar si el procesamiento terminó
                                result_status, _ = process_pending_result()
                                if result_status is not None:
                                    processing = False
                                    if result_status == 'error':
                                        print("❌ Error procesando prompt")
                                    # Esperar a que el hilo termine completamente antes de continuar
                                    if current_prompt_thread and current_prompt_thread.is_alive():
                                        current_prompt_thread.join(timeout=1.0)
                                    # Drenar cualquier contenido residual del buffer antes de volver al menú
                                    drain_stdin(timeout=1.0)
                                time.sleep(0.05)
                    elif choice == '2':
                        print("\n🎤 Habla ahora... (la escucha está activa)")
                        print("   Presiona Ctrl+C o espera para volver al menú")
                        toggle_voice_listening(True)
                        try:
                            time.sleep(2)
                        except KeyboardInterrupt:
                            toggle_voice_listening(False)
                    elif choice.lower() == 'v':
                        # Toggle visión
                        if vision_active:
                            stop_vision()
                        else:
                            start_vision()
                    else:
                        print(f"\n⚠️ Opción '{choice}' no válida")
            else:
                # En modo escucha, solo checkear teclas si visión está activa
                if vision_active and screen_stream:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        toggle_voice_listening(False)
                        break
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nInterrupción por teclado.")
    finally:
        print("\nDeteniendo...")
        if is_listening and stop_listening:
            stop_listening(wait_for_stop=False)
        if vision_active:
            stop_vision()
        cv2.destroyAllWindows()
        print("Programa terminado.")


if __name__ == "__main__":
    main()
