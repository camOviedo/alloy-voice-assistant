"""
Gestor de archivos del proyecto - maneja lectura, modificaciones propuestas y git.
"""
import json
import os
import subprocess
import time
import difflib
from pathlib import Path

from config import SUGGESTIONS_PATH


class ProjectFileManager:
    """Gestiona la lectura y modificación de archivos del proyecto objetivo"""

    def __init__(self, project_path):
        self.project_path = Path(project_path)
        self.pending_changes = {}  # {file_path: {original, proposed, description}}
        self._load_pending_changes()

    def _load_pending_changes(self):
        """Carga cambios pendientes desde archivo JSON"""
        pending_file = Path(SUGGESTIONS_PATH) / "pending_changes.json"
        if pending_file.exists():
            try:
                with open(pending_file, 'r', encoding='utf-8') as f:
                    self.pending_changes = json.load(f)
            except Exception as e:
                print(f"Error cargando cambios pendientes: {e}")
                self.pending_changes = {}

    def _save_pending_changes(self):
        """Guarda cambios pendientes a archivo JSON"""
        pending_file = Path(SUGGESTIONS_PATH) / "pending_changes.json"
        try:
            with open(pending_file, 'w', encoding='utf-8') as f:
                json.dump(self.pending_changes, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error guardando cambios pendientes: {e}")

    def read_file(self, relative_path):
        """Lee un archivo del proyecto"""
        file_path = self.project_path / relative_path
        if not file_path.exists():
            return None, f"Archivo no encontrado: {relative_path}"
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content, None
        except Exception as e:
            return None, f"Error leyendo archivo: {e}"

    def list_python_files(self):
        """Lista todos los archivos Python del proyecto"""
        try:
            py_files = list(self.project_path.glob("*.py"))
            return [f.name for f in py_files], None
        except Exception as e:
            return None, f"Error listando archivos: {e}"

    def propose_change(self, relative_path, new_content, description=""):
        """Propone un cambio a un archivo"""
        original_content, error = self.read_file(relative_path)
        if error:
            return False, error

        # VALIDACIÓN DE SEGURIDAD: detectar si el código es significativamente más corto
        original_lines = len(original_content.splitlines())
        new_lines = len(new_content.splitlines())

        if new_lines < original_lines * 0.5:
            # El código nuevo tiene menos del 50% de líneas del original
            warning = f"⚠️ ADVERTENCIA: El código propuesto ({new_lines} líneas) es mucho más corto que el original ({original_lines} líneas)."
            print(f"\n{'='*60}")
            print(warning)
            print("Esto puede indicar que el LLM devolvió solo un fragmento en lugar del archivo completo.")
            print("El cambio NO se ha guardado. Intenta nuevamente especificando 'archivo completo'.")
            print(f"{'='*60}\n")
            return False, f"Código propuesto incompleto: {new_lines} vs {original_lines} líneas. El cambio fue rechazado automáticamente."

        change_id = f"{relative_path}_{int(time.time())}"
        self.pending_changes[change_id] = {
            "file": relative_path,
            "original": original_content,
            "proposed": new_content,
            "description": description,
            "timestamp": time.time()
        }
        self._save_pending_changes()
        return True, change_id

    def get_pending_changes(self):
        """Obtiene todos los cambios pendientes"""
        return self.pending_changes

    def approve_change(self, change_id, auto_commit=False):
        """Aprueba y aplica un cambio pendiente (sin commit automático por defecto)"""
        if change_id not in self.pending_changes:
            return False, "Cambio no encontrado"

        change = self.pending_changes[change_id]
        file_path = self.project_path / change["file"]
        
        # DEBUG: Mostrar información detallada
        print(f"\n🔧 [DEBUG] Aplicando cambio {change_id}:")
        print(f"   📁 Ruta proyecto: {self.project_path}")
        print(f"   📄 Archivo destino: {change['file']}")
        print(f"   🎯 Ruta completa: {file_path}")
        print(f"   📊 Tamaño original: {len(change['original'])} bytes")
        print(f"   📊 Tamaño propuesto: {len(change['proposed'])} bytes")
        print(f"   📊 Líneas original: {len(change['original'].splitlines())}")
        print(f"   📊 Líneas propuesto: {len(change['proposed'].splitlines())}")
        print(f"   🔍 Archivo existe: {file_path.exists()}")

        try:
            # Verificar que el archivo existe
            if not file_path.exists():
                return False, f"Archivo no encontrado: {file_path}"

            # Guardar backup
            backup_path = str(file_path) + ".backup"
            print(f"   💾 Creando backup en: {backup_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                original = f.read()
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original)
            print(f"   ✅ Backup creado ({len(original)} bytes)")

            # Aplicar cambio
            print(f"   ✏️  Escribiendo cambio al archivo...")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(change["proposed"])
            
            # Verificar que se escribió correctamente
            with open(file_path, 'r', encoding='utf-8') as f:
                written = f.read()
            print(f"   ✅ Archivo escrito ({len(written)} bytes)")
            
            if len(written) != len(change["proposed"]):
                print(f"   ⚠️  ADVERTENCIA: Tamaño no coincide! Esperado: {len(change['proposed'])}, Escrito: {len(written)}")
            else:
                print(f"   ✅ Verificación exitosa - tamaño coincide")

            result_msg = f"✅ Cambio aplicado a {change['file']} ({len(written)} bytes escritos)"

            # Solo hacer commit si se solicita explícitamente
            if auto_commit:
                result = self._git_commit(change["file"], change.get("description", "Actualización via asistente"))
                result_msg += f" | {result}"
            else:
                result_msg += " (SIN COMMIT - cambios en working directory)"

            # Eliminar de pendientes
            del self.pending_changes[change_id]
            self._save_pending_changes()

            return True, result_msg
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, f"Error aplicando cambio: {e}"

    def reject_change(self, change_id):
        """Rechaza y elimina un cambio pendiente"""
        if change_id not in self.pending_changes:
            return False, "Cambio no encontrado"

        del self.pending_changes[change_id]
        self._save_pending_changes()
        return True, "Cambio rechazado y eliminado"

    def _git_commit(self, file_path, message):
        """Hace commit del cambio en git"""
        try:
            # Cambiar al directorio del proyecto
            original_dir = os.getcwd()
            os.chdir(self.project_path)

            # Añadir archivo
            subprocess.run(["git", "add", file_path], check=True, capture_output=True)

            # Commit
            result = subprocess.run(
                ["git", "commit", "-m", f"[Asistente] {message}"],
                capture_output=True,
                text=True
            )

            os.chdir(original_dir)

            if result.returncode == 0:
                return f"✅ Cambio aplicado y commiteado: {message}"
            else:
                return f"⚠️ Archivo actualizado pero commit falló: {result.stderr}"
        except Exception as e:
            return f"⚠️ Archivo actualizado pero error en git: {e}"

    def generate_diff(self, change_id):
        """Genera un diff del cambio propuesto"""
        if change_id not in self.pending_changes:
            return None

        change = self.pending_changes[change_id]
        original_lines = change["original"].splitlines(keepends=True)
        proposed_lines = change["proposed"].splitlines(keepends=True)

        diff = difflib.unified_diff(
            original_lines,
            proposed_lines,
            fromfile=f"a/{change['file']}",
            tofile=f"b/{change['file']}"
        )
        return "".join(diff)
