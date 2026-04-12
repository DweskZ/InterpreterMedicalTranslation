# Clinic Translate

Traductor **EN ↔ ES** en tiempo real para intérpretes médicos. Captura el audio del sistema (loopback WASAPI), lo transcribe y muestra la traducción al idioma opuesto en una ventana siempre visible.

Dos motores de transcripción disponibles:

| Motor | Modo | Ventajas | Requisitos |
|-------|------|----------|------------|
| **Whisper** (default) | Local, GPU/CPU | Sin costo, sin internet, rolling context | NVIDIA GPU recomendada |
| **Deepgram Nova-3** | Cloud, WebSocket streaming | Detecta EN/ES automáticamente (code-switching) | API key + internet |

La traducción usa **Google Translate** (natural, gratis vía `deep_translator`) con fallback a **Argos Translate** (offline) si no hay internet.

> **Aviso:** herramienta asistencial. No sustituye criterio profesional. En entornos sanitarios revisar siempre la salida.

## Requisitos

- Windows 10/11
- Python **3.10+**
- NVIDIA GPU + CUDA recomendado (funciona en CPU con más latencia)
- ~16 GB RAM mínimo; en GPU 6 GB VRAM probar `--model base`

## Instalación rápida

```powershell
# 1. Clonar o copiar el proyecto
cd E:\AITrasncriptRealTime

# 2. Crear entorno virtual
python -m venv .venv

# 3. Activar el entorno virtual
.\.venv\Scripts\Activate.ps1

# 4. Instalar dependencias
python -m pip install -r requirements.txt

# 5. Descargar paquetes de idioma Argos (una sola vez, requiere internet)
python clinic_translate.py --setup-langs
```

## Configuración de Deepgram (opcional)

Si quieres usar Deepgram Nova-3 como motor de transcripción:

1. Crea una cuenta en [deepgram.com](https://deepgram.com) (tiene $200 de crédito gratis)
2. Crea un archivo `.env` en la raíz del proyecto:

```
DEEPGRAM_API_KEY=tu_api_key_aqui
```

## Uso

```powershell
# Iniciar con Whisper (default, local)
.\run.ps1 --model base

# Iniciar directamente con Deepgram (cloud)
.\run.ps1 --backend deepgram

# También puedes cambiar el backend desde la interfaz gráfica en cualquier momento
```

### Interfaz

- **Panel izquierdo — TRANSCRIPCIÓN:** todo lo que se escucha, en el idioma original (inglés o español)
- **Panel derecho — TRADUCCIÓN:** la traducción al idioma opuesto
- **Toolbar:** selector de backend (Whisper/Deepgram), modelo Whisper, normalización de audio, botón Limpiar

### Opciones de línea de comandos

| Opción | Descripción |
|--------|-------------|
| `--backend whisper` | Motor local (default) |
| `--backend deepgram` | Motor cloud (requiere `.env` con API key) |
| `--model base` | Modelo Whisper: `tiny`, `base`, `small`, `medium` (con/sin `.en`) |
| `--chunk-seconds 3` | Ventana de captura en segundos |
| `--max-history 50` | Máximo de líneas visibles por panel |
| `--no-vad` | Desactivar filtro VAD |
| `--device N` | Elegir dispositivo loopback específico |
| `--list-devices` | Ver dispositivos disponibles |

## Otra PC

```powershell
# Clonar el repo
git clone <url-del-repo>
cd AITrasncriptRealTime

# Crear venv e instalar
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python clinic_translate.py --setup-langs

# Ejecutar
.\run.ps1 --model base
```

## Solución de problemas

- **`RuntimeError: Library cublas64_12.dll is not found`** — Ejecutar `pip install -r requirements.txt` (incluye CUDA runtime). Alternativa: instalar [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads). Si falla GPU se usa CPU automáticamente.
- **Sin texto / error de audio** — Verificar que el altavoz predeterminado de Windows es la salida por la que suena el audio. Usar `--list-devices` para ver dispositivos.
- **Permisos PowerShell** — `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
- **Aviso Hugging Face (`HF_TOKEN`)** — No es error. Opcional: crear token en [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) y `$env:HF_TOKEN = "hf_..."` antes de ejecutar.
