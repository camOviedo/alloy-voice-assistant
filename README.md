# Sample AI assistant

You need an `OPENAI_API_KEY` and a `GOOGLE_API_KEY` to run this code. Store them in a `.env` file in the root directory of the project, or set them as environment variables.


If you are running the code on Apple Silicon, run the following command:

```
$ brew install portaudio
```

Create a virtual environment, update pip, and install the required packages:

```
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -U pip
$ pip install -r requirements.txt
```

Run the assistant:

```
$ python3 assistant.py
```

# Asistente de Código con Visión - Interfaz Web

Asistente multi-agente con análisis visual mediante interfaz web moderna (Chainlit).

## 🚀 Inicio Rápido

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Ejecutar interfaz web

```bash
chainlit run app.py
```

La interfaz estará disponible en `http://localhost:8000`

---

## 🎨 Características de la Interfaz Web

### Pantalla Principal (Chat)
- **Input de prompts**: Escribe directamente en el chat
- **Captura de pantalla**: Botón "📷 Capturar Pantalla" cuando se detecta modo visión
- **Acciones rápidas**: Botones para listar archivos, ver cambios pendientes

### Visualización del Workflow (Sidebar)
Chainlit muestra automáticamente los pasos del workflow en la sidebar:

1. **📋 Coordinator**: Muestra la decisión y razonamiento
2. **👁️ Vision Agent**: Análisis de imágenes capturadas
3. **💻 Code Agent**: Análisis del código
4. **✏️ Editor Agent**: Código generado/modificado
5. **🔍 Reviewer Agent**: Revisión y aprobación
6. **📊 Métricas**: Tokens consumidos por agente

### Comandos Disponibles

| Comando | Descripción |
|---------|-------------|
| `muestra archivo.py` | Ver contenido de archivo |
| `lista archivos` | Listar archivos Python del proyecto |
| `modifica archivo.py para...` | Proponer modificación |
| `cambios pendientes` | Ver cambios propuestos |
| `aprueba cambio [ID]` | Aplicar cambio |
| `rechaza cambio [ID]` | Descartar cambio |

### Modo Visión

Cuando escribes prompts con palabras como "pantalla", "imagen", "mira", "qué ves":

1. El sistema detecta necesidad de visión
2. Aparece botón "📷 Capturar Pantalla"
3. Al hacer clic, captura localmente usando Python (mss/ScreenStream)
4. Las imágenes se guardan en `captures/` (privado, nunca salen de tu máquina)
5. El workflow analiza las imágenes

---

## 🔒 Privacidad

- **Captura 100% local**: Usa `mss` + `ScreenStream` en Python, no APIs del navegador
- **Imágenes locales**: Guardadas en `captures/`, nunca en servidores externos
- **LLM local**: Todo el procesamiento usa Ollama local

---

## 📁 Estructura

```
├── app.py              # Punto de entrada Chainlit
├── chainlit.md         # Configuración de la UI
├── graph/workflow.py   # Workflow multi-agente
├── agents/             # Agentes (coordinator, vision, code, editor, reviewer)
├── captures/           # Imágenes capturadas (local)
└── sugerencias_pendientes/  # Cambios propuestos
```

---

## 🛠️ Modo Terminal (Legacy)

Si necesitas usar la interfaz por terminal:

```bash
python main.py
```

Pero la interfaz web es el método recomendado.

---

## Requisitos

- Python 3.10+
- Ollama corriendo localmente
- Navegador web moderno

## Instalación inicial (si es necesario)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If you are running the code on Apple Silicon:
```
brew install portaudio