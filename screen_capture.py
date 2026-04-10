"""
Captura de pantalla en tiempo real usando mss.
"""
import base64
import os
import time
from datetime import datetime
from threading import Lock, Thread

import cv2
import mss
import numpy as np

from config import (
    DEFAULT_JPEG_QUALITY,
    DEFAULT_MAX_WIDTH,
    DEFAULT_MONITOR,
    DEFAULT_SCALE_DISPLAY,
    CAPTURES_PATH,
    AUTO_CROP_VIDEO,
)
from image_cropper import crop_to_video, detect_video_region


def capture_single_screenshot(
    monitor=DEFAULT_MONITOR,
    max_width=DEFAULT_MAX_WIDTH,
    jpeg_quality=DEFAULT_JPEG_QUALITY,
    save_to_disk=True,
    captures_path=CAPTURES_PATH,
    auto_crop=AUTO_CROP_VIDEO
):
    """
    Captura una única screenshot y opcionalmente la guarda en disco.
    
    Args:
        monitor: índice del monitor a capturar (1 = principal)
        max_width: ancho máximo para redimensionar
        jpeg_quality: calidad JPEG
        save_to_disk: si True, guarda la imagen en disco
        captures_path: ruta donde guardar la imagen
        auto_crop: si True, detecta y recorta región de video automáticamente
        
    Returns:
        tuple: (image_base64, file_path, crop_metadata) - imagen en base64, ruta y metadata del recorte
    """
    with mss.mss() as sct:
        # Verificar monitor válido
        if monitor >= len(sct.monitors):
            monitor = 1
        
        monitor_region = sct.monitors[monitor]
        
        # Capturar pantalla
        screenshot = sct.grab(monitor_region)
        img = np.array(screenshot)
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        h, w = frame.shape[:2]
        
        # Auto-crop: detectar y recortar región de video
        crop_metadata = {'cropped': False}
        if auto_crop:
            cropped_frame, region, crop_metadata = crop_to_video(frame, padding=10)
            if crop_metadata['cropped']:
                frame = cropped_frame
                print(f"✂️  Auto-crop aplicado: {crop_metadata['reduction_percent']}% reducción, ~{crop_metadata['estimated_token_savings']} tokens ahorrados")
        
        h, w = frame.shape[:2]
        
        # Redimensionar si es necesario
        if w > max_width:
            scale = max_width / w
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"📸 Imagen redimensionada: {w}x{h} → {new_w}x{new_h}")
        
        # Comprimir a JPEG
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        _, buffer = cv2.imencode(".jpeg", frame, encode_params)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        est_tokens = int(len(image_base64) * 0.75)
        print(f"📸 Imagen capturada: {len(image_base64):,} chars (~{est_tokens:,} tokens)")
        
        file_path = None
        if save_to_disk:
            # Generar nombre de archivo con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            file_path = os.path.join(captures_path, filename)
            
            # Guardar imagen
            with open(file_path, 'wb') as f:
                f.write(buffer)
            print(f"💾 Imagen guardada: {file_path}")
        
        return image_base64, file_path, crop_metadata


class ScreenStream:
    def __init__(self, monitor=DEFAULT_MONITOR, scale_display=DEFAULT_SCALE_DISPLAY,
                 max_width=DEFAULT_MAX_WIDTH, jpeg_quality=DEFAULT_JPEG_QUALITY,
                 auto_crop=AUTO_CROP_VIDEO):
        """
        monitor: índice del monitor a capturar (1 = principal, 2 = secundario, etc.)
        scale_display: factor de escala para la ventana de previsualización
        max_width: ancho máximo en píxeles para la imagen enviada al LLM (reduce tokens)
        jpeg_quality: calidad JPEG (menor = menos tokens pero más compresión)
        auto_crop: si True, detecta y recorta región de video automáticamente
        """
        self.monitor_index = monitor
        self.scale_display = scale_display
        self.max_width = max_width
        self.jpeg_quality = jpeg_quality
        self.auto_crop = auto_crop
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

    def capture_and_save_frame(self, save_to_disk=True, captures_path=CAPTURES_PATH):
        """
        Captura el frame actual del stream y lo guarda en disco.
        
        Returns:
            tuple: (image_base64, file_path, crop_metadata) - imagen en base64, ruta y metadata del recorte
        """
        with self.lock:
            if self.frame is None:
                return None, None, None
            frame = self.frame.copy()
        
        # Auto-crop: detectar y recortar región de video
        crop_metadata = {'cropped': False}
        if self.auto_crop:
            cropped_frame, region, crop_metadata = crop_to_video(frame, padding=10)
            if crop_metadata['cropped']:
                frame = cropped_frame
                print(f"✂️  Frame auto-crop: {crop_metadata['reduction_percent']}% reducción, ~{crop_metadata['estimated_token_savings']} tokens ahorrados")
        
        h, w = frame.shape[:2]
        
        # Redimensionar si es necesario
        if w > self.max_width:
            scale = self.max_width / w
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"📸 Frame redimensionado: {w}x{h} → {new_w}x{new_h}")
        
        # Comprimir a JPEG
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        _, buffer = cv2.imencode(".jpeg", frame, encode_params)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        est_tokens = int(len(image_base64) * 0.75)
        print(f"📸 Frame capturado: {len(image_base64):,} chars (~{est_tokens:,} tokens)")
        
        file_path = None
        if save_to_disk:
            # Generar nombre de archivo con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"frame_{timestamp}.jpg"
            file_path = os.path.join(captures_path, filename)
            
            # Guardar imagen
            with open(file_path, 'wb') as f:
                f.write(buffer)
            print(f"💾 Frame guardado: {file_path}")
        
        return image_base64, file_path, crop_metadata

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
