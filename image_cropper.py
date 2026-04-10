"""
Recorte inteligente de imágenes para detectar y extraer regiones de video.
Reduce tokens al enviar solo el área relevante al LLM de visión.
"""
import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from pathlib import Path


def detect_video_region(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Detecta la región de reproducción de video en una imagen.
    
    Usa múltiples técnicas:
    1. Análisis de bordes para encontrar rectángulos grandes
    2. Análisis de color para detectar áreas dinámicas típicas de video
    3. Filtrado por tamaño mínimo (evitar falsos positivos pequeños)
    
    Args:
        image: Imagen numpy array (BGR)
        
    Returns:
        Tuple (x, y, w, h) de la región de video, o None si no se detecta
    """
    if image is None or image.size == 0:
        return None
    
    height, width = image.shape[:2]
    min_area = (width * height) * 0.05  # Mínimo 5% de la pantalla
    max_area = (width * height) * 0.95  # Máximo 95% de la pantalla
    
    # Estrategia 1: Detección de bordes y contornos
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilatar bordes para conectar líneas cercanas
    kernel = np.ones((5, 5), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = w / h if h > 0 else 0
        
        # Filtrar por tamaño y aspect ratio típico de video (16:9, 4:3, etc.)
        if min_area < area < max_area:
            # Aspect ratios comunes de video: 16:9 (1.78), 4:3 (1.33), 21:9 (2.33)
            if 1.2 <= aspect_ratio <= 2.5:
                candidates.append({
                    'rect': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'score': area * (1 - abs(aspect_ratio - 1.78) / 1.78)  # Preferir 16:9
                })
    
    # Estrategia 2: Análisis de varianza de color (áreas con contenido rico)
    # Las regiones de video típicamente tienen más varianza de color que UI estática
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Dividir imagen en grid y analizar varianza
    grid_size = 50
    variance_regions = []
    
    for y in range(0, height - grid_size, grid_size):
        for x in range(0, width - grid_size, grid_size):
            region = hsv[y:y+grid_size, x:x+grid_size]
            variance = np.var(region[:, :, 1])  # Varianza del canal de saturación
            if variance > 500:  # Umbral de varianza
                variance_regions.append((x, y, variance))
    
    # Si encontramos regiones de alta varianza, buscar clusters
    if len(variance_regions) > 10:
        # Agrupar puntos cercanos
        points = np.array([(x + grid_size//2, y + grid_size//2) for x, y, _ in variance_regions])
        if len(points) > 5:
            # Calcular bounding box de puntos de alta varianza
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            
            w_dyn = x_max - x_min + grid_size * 2
            h_dyn = y_max - y_min + grid_size * 2
            
            area_dyn = w_dyn * h_dyn
            if min_area < area_dyn < max_area:
                aspect_dyn = w_dyn / h_dyn if h_dyn > 0 else 0
                if 1.2 <= aspect_dyn <= 2.5:
                    candidates.append({
                        'rect': (max(0, x_min - grid_size), max(0, y_min - grid_size), 
                                min(w_dyn, width - x_min), min(h_dyn, height - y_min)),
                        'area': area_dyn,
                        'aspect_ratio': aspect_dyn,
                        'score': area_dyn * 0.8  # Ligeramente menor prioridad que bordes
                    })
    
    # Seleccionar el mejor candidato
    if candidates:
        best = max(candidates, key=lambda c: c['score'])
        return best['rect']
    
    return None


def detect_video_region_from_path(image_path: str) -> Optional[Tuple[int, int, int, int]]:
    """
    Carga una imagen desde archivo y detecta la región de video.
    
    Args:
        image_path: Ruta al archivo de imagen
        
    Returns:
        Tuple (x, y, w, h) o None
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
    return detect_video_region(image)


def crop_to_video(image: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None,
                  padding: int = 10) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]], Dict]:
    """
    Recorta la imagen a la región de video detectada.
    
    Args:
        image: Imagen numpy array
        region: Región (x, y, w, h) - si None, se detecta automáticamente
        padding: Píxeles de padding alrededor de la región
        
    Returns:
        Tuple de (imagen_recortada, region_usada, metadatos)
    """
    if image is None:
        return None, None, {'error': 'No image provided'}
    
    # Detectar automáticamente si no se proporciona región
    if region is None:
        region = detect_video_region(image)
    
    if region is None:
        return image, None, {
            'cropped': False,
            'reason': 'No video region detected',
            'original_shape': image.shape[:2]
        }
    
    x, y, w, h = region
    height, width = image.shape[:2]
    
    # Aplicar padding con límites de la imagen
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + w + padding)
    y2 = min(height, y + h + padding)
    
    cropped = image[y1:y2, x1:x2]
    
    # Calcular métricas de reducción
    original_pixels = width * height
    cropped_pixels = cropped.shape[1] * cropped.shape[0]
    reduction_pct = (1 - cropped_pixels / original_pixels) * 100 if original_pixels > 0 else 0
    
    # Estimar ahorro de tokens (~0.75 tokens por char base64)
    # Una imagen JPEG típica genera ~0.75 * bytes_base64 tokens
    estimated_token_reduction = int(original_pixels * 0.001 * reduction_pct / 100)
    
    metadata = {
        'cropped': True,
        'original_shape': (height, width),
        'cropped_shape': cropped.shape[:2],
        'region': (x1, y1, x2 - x1, y2 - y1),
        'reduction_percent': round(reduction_pct, 1),
        'estimated_token_savings': estimated_token_reduction
    }
    
    return cropped, (x1, y1, x2 - x1, y2 - y1), metadata


def crop_image_file(input_path: str, output_path: Optional[str] = None,
                    auto_detect: bool = True) -> Dict:
    """
    Procesa un archivo de imagen, detecta y recorta la región de video.
    
    Args:
        input_path: Ruta de la imagen de entrada
        output_path: Ruta para guardar la imagen recortada (opcional)
        auto_detect: Si True, detecta automáticamente la región
        
    Returns:
        Dict con resultado y metadatos
    """
    image = cv2.imread(input_path)
    if image is None:
        return {
            'success': False,
            'error': f'Could not load image: {input_path}'
        }
    
    cropped, region, metadata = crop_to_video(image, padding=10)
    
    result = {
        'success': True,
        'input_path': input_path,
        'metadata': metadata
    }
    
    if output_path and metadata.get('cropped'):
        cv2.imwrite(output_path, cropped)
        result['output_path'] = output_path
    
    return result


def batch_crop_images(input_dir: str, output_dir: Optional[str] = None,
                      prefix: str = "cropped_") -> Dict:
    """
    Procesa múltiples imágenes en un directorio.
    
    Args:
        input_dir: Directorio con imágenes
        output_dir: Directorio de salida (si None, sobrescribe)
        prefix: Prefijo para archivos recortados
        
    Returns:
        Dict con estadísticas del procesamiento
    """
    from pathlib import Path
    
    input_path = Path(input_dir)
    if not input_path.exists():
        return {'success': False, 'error': f'Directory not found: {input_dir}'}
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = input_path
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images = [f for f in input_path.iterdir() 
              if f.is_file() and f.suffix.lower() in valid_extensions]
    
    processed = 0
    cropped_count = 0
    total_savings = 0
    
    for img_file in images:
        output_file = output_path / f"{prefix}{img_file.name}"
        result = crop_image_file(str(img_file), str(output_file))
        
        if result['success']:
            processed += 1
            if result['metadata'].get('cropped'):
                cropped_count += 1
                total_savings += result['metadata'].get('estimated_token_savings', 0)
    
    return {
        'success': True,
        'total_images': len(images),
        'processed': processed,
        'cropped': cropped_count,
        'total_token_savings': total_savings,
        'avg_reduction_percent': round((cropped_count / max(processed, 1)) * 100, 1) if processed > 0 else 0
    }


if __name__ == "__main__":
    # Test con imágenes existentes
    import sys
    
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        print(f"Testing detection on: {test_path}")
        
        region = detect_video_region_from_path(test_path)
        if region:
            x, y, w, h = region
            print(f"✅ Video region detected: x={x}, y={y}, w={w}, h={h}")
            print(f"   Aspect ratio: {w/h:.2f}")
            
            # Mostrar resultado
            image = cv2.imread(test_path)
            cropped, _, meta = crop_to_video(image, region)
            print(f"   Original: {meta['original_shape']}")
            print(f"   Cropped: {meta['cropped_shape']}")
            print(f"   Reduction: {meta['reduction_percent']}%")
            print(f"   Est. token savings: ~{meta['estimated_token_savings']}")
        else:
            print("❌ No video region detected")
    else:
        # Test batch en captures/
        captures_dir = Path(__file__).parent / "captures"
        if captures_dir.exists():
            print(f"Batch processing {captures_dir}...")
            stats = batch_crop_images(str(captures_dir), prefix="test_cropped_")
            print(f"\nResults:")
            print(f"  Images: {stats['processed']}/{stats['total_images']}")
            print(f"  Cropped: {stats['cropped']}")
            print(f"  Token savings: ~{stats['total_token_savings']}")
