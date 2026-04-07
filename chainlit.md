# Chainlit Configuration
# Este archivo configura la apariencia y comportamiento de la interfaz

name: "Asistente de Código con Visión"
description: "Interfaz web para el asistente multi-agente con análisis visual"

# UI Configuration
ui:
  name: "🤖 Asistente de Código"
  description: "Multi-agente con visión - Modo Web"
  hide_cot: false  # Mostrar chain of thought (pasos del workflow)
  default_collapse_content: false
  default_collapse_cot: false

# Features
features:
  # Habilitar modo multi-usuario (cada usuario tiene su sesión)
  multi_user: false
  # Habilitar historia de chat
  chat_history: true
  # Habilitar feedback
  feedback: false

# Branding (opcional)
# custom_theme:
#   theme_color: "#4F46E5"
#   background_color: "#FFFFFF"
