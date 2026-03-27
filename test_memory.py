#!/usr/bin/env python3
"""
Script de prueba para verificar que ProjectMemory funciona correctamente.
Prueba el cacheo de análisis entre sesiones.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import directo para evitar cargar dependencias de langchain
import importlib.util
spec = importlib.util.spec_from_file_location("memory", "agents/memory.py")
memory_module = importlib.util.module_from_spec(spec)

# Necesitamos cargar config primero
spec_config = importlib.util.spec_from_file_location("config", "config.py")
config_module = importlib.util.module_from_spec(spec_config)
spec_config.loader.exec_module(config_module)
sys.modules['config'] = config_module

spec.loader.exec_module(memory_module)
ProjectMemory = memory_module.ProjectMemory

def test_project_memory():
    """Prueba el sistema de memoria de proyecto."""
    print("="*60)
    print("🧪 PRUEBA DE ProjectMemory")
    print("="*60)

    # Crear instancia de memoria
    memory = ProjectMemory()
    print(f"✓ Memoria inicializada: {memory.memory_file}")

    # Simular contenido de archivo
    test_file = "test_module.py"
    test_content = "def hello():\n    return 'world'\n"

    # Simular resultado de análisis
    test_analysis = {
        "analysis": "El archivo contiene una función simple",
        "summary": "Función hello que retorna 'world'",
        "files_affected": [test_file],
        "lines_to_modify": [1],
        "approach": "Modificar el retorno de la función",
        "considerations": ["No hay dependencias"]
    }

    # Guardar análisis
    print(f"\n💾 Guardando análisis para {test_file}...")
    memory.save_file_analysis(test_file, test_content, test_analysis)

    # Verificar que se guardó
    cached_count = memory.get_cached_files_count()
    print(f"✓ Archivos en caché: {cached_count}")

    # Recuperar análisis
    print(f"\n📂 Recuperando análisis para {test_file}...")
    cached = memory.get_file_analysis(test_file, test_content)

    if cached:
        print("✓ Análisis recuperado exitosamente:")
        print(f"  - Resumen: {cached.get('summary', 'N/A')}")
        print(f"  - Enfoque: {cached.get('approach', 'N/A')}")
        print(f"  - Líneas a modificar: {cached.get('lines_to_modify', [])}")
    else:
        print("❌ No se pudo recuperar el análisis")
        return False

    # Probar invalidación por cambio de contenido
    print(f"\n🔄 Probando invalidación por cambio de contenido...")
    modified_content = "def hello():\n    return 'world modified'\n"
    cached_invalid = memory.get_file_analysis(test_file, modified_content)

    if cached_invalid is None:
        print("✓ Caché correctamente invalidado (checksum diferente)")
    else:
        print("❌ El caché no se invalidó correctamente")
        return False

    # Probar build_context_from_cache
    print(f"\n📋 Construyendo contexto desde caché...")
    context = memory.build_context_from_cache(test_file)
    if context and "CONTEXTO DEL PROYECTO" in context:
        print("✓ Contexto generado correctamente")
        print(f"  (Longitud: {len(context)} caracteres)")
    else:
        print("❌ Error generando contexto")
        return False

    # Limpiar archivo de prueba
    print(f"\n🧹 Limpiando archivo de prueba...")
    if memory.memory_file.exists():
        memory.memory_file.unlink()
        print("✓ Archivo de memoria eliminado")

    print("\n" + "="*60)
    print("✅ TODAS LAS PRUEBAS PASARON")
    print("="*60)
    return True

if __name__ == "__main__":
    try:
        success = test_project_memory()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error en pruebas: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
