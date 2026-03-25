"""
Captura de pantalla en tiempo real usando mss.
"""
import base64
import time
from threading import Lock, Thread

import cv2
import mss
import numpy as np

from config import (
    DEFAULT_JPEG_QUALITY,
    DEFAULT_MAX_WIDTH,
    DEFAULT_MONITOR,
    DEFAULT_SCALE_DISPLAY,
)


class ScreenStream:
    def __init__(self, monitor=DEFAULT_MONITOR, scale_display=DEFAULT_SCALE_DISPLAY,
                 max_width=DEFAULT_MAX_WIDTH, jpeg_quality=DEFAULT_JPEG_QUALITY):
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
