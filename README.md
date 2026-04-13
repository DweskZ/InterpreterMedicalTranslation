# Clinic Translate

Traductor **EN ↔ ES** en tiempo real para intérpretes médicos. Captura el audio del sistema (loopback WASAPI), lo transcribe y muestra la traducción al idioma opuesto en una ventana siempre visible.

## Motores de transcripción

| Motor | Modo | Ventajas | Requisitos |
|-------|------|----------|------------|
| **Whisper** (default) | Local, GPU/CPU | Sin costo, sin internet, rolling context | NVIDIA GPU recomendada |
| **Deepgram Nova-3** | Cloud, WebSocket | Detecta EN/ES automáticamente, muy preciso | API key ($200 gratis) |
| **AssemblyAI** | Cloud, WebSocket | Code-switching EN/ES nativo, 333h gratis/mes | API key (free tier) |

## Motores de traducción

| Motor | Calidad | Costo | Requisitos |
|-------|---------|-------|------------|
| **Google Translate** (default) | Buena | Gratis | Internet |
| **DeepL** | La mejor para EN/ES | Gratis (500K chars/mes) | API key gratis en deepl.com |
| **GPT-4o-mini** | Contexto médico nativo | ~$0.003 por consulta | API key de OpenAI |

Siempre con fallback a **Argos Translate** (offline) si no hay internet.

> **Aviso:** herramienta asistencial. No sustituye criterio profesional.

## Requisitos

- Windows 10/11
- Python **3.10+**
- NVIDIA GPU + CUDA recomendado (funciona en CPU con más latencia)
- ~16 GB RAM mínimo; en GPU 6 GB VRAM probar `--model base`

## Instalación rápida

```powershell
# 1. Permisos de PowerShell (una sola vez)
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# 2. Clonar o copiar el proyecto
cd E:\AITrasncriptRealTime

# 3. Crear entorno virtual
python -m venv .venv

# 4. Activar el entorno virtual
.\.venv\Scripts\Activate.ps1

# 5. Instalar dependencias
python -m pip install -r requirements.txt

# 6. Descargar paquetes de idioma Argos (una sola vez, requiere internet)
python clinic_translate.py --setup-langs
```

## Configuración de APIs (opcional)

Crea un archivo `.env` en la raíz del proyecto con las API keys que quieras usar:

```env
# Transcripción cloud (elige una o ambas)
DEEPGRAM_API_KEY=tu_key_aqui       # deepgram.com ($200 gratis)
ASSEMBLYAI_API_KEY=tu_key_aqui     # assemblyai.com (333h gratis/mes)

# Traducción mejorada (elige una o ambas)
DEEPL_API_KEY=tu_key_aqui          # deepl.com/pro-api (500K chars gratis/mes)
OPENAI_API_KEY=tu_key_aqui         # platform.openai.com (~$0.003 por consulta)
```

No necesitas todas. Sin `.env`, la app funciona con Whisper + Google Translate (todo gratis).

## Uso

```powershell
# Whisper local (default)
.\run.ps1 --model base

# Deepgram cloud
.\run.ps1 --backend deepgram

# AssemblyAI cloud
.\run.ps1 --backend assemblyai
```

El motor de transcripción y traducción también se puede cambiar desde la interfaz gráfica en cualquier momento.

### Interfaz

- **Panel izquierdo — TRANSCRIPCIÓN:** todo lo que se escucha (inglés o español)
- **Panel derecho — TRADUCCIÓN:** la traducción al idioma opuesto
- **Toolbar:** backend (Whisper/Deepgram/AssemblyAI), modelo Whisper, motor de traducción (Google/DeepL/OpenAI), normalizar audio, limpiar

### Opciones CLI

| Opción | Descripción |
|--------|-------------|
| `--backend whisper` | Motor local (default) |
| `--backend deepgram` | Deepgram Nova-3 (requiere API key en `.env`) |
| `--backend assemblyai` | AssemblyAI Universal Streaming (requiere API key en `.env`) |
| `--model base` | Modelo Whisper: `tiny`, `base`, `small`, `medium` (con/sin `.en`) |
| `--chunk-seconds 3` | Ventana de captura en segundos |
| `--max-history 50` | Máximo de líneas visibles por panel |
| `--no-vad` | Desactivar filtro VAD |
| `--device N` | Elegir dispositivo loopback específico |
| `--list-devices` | Ver dispositivos disponibles |

## Otra PC

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
git clone <url-del-repo>
cd InterpreterMedicalTranslation
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python clinic_translate.py --setup-langs
.\run.ps1 --model base
```

## Solución de problemas

- **`RuntimeError: Library cublas64_12.dll is not found`** — `pip install -r requirements.txt` (incluye CUDA runtime). Si falla GPU se usa CPU automáticamente.
- **Sin texto / error de audio** — Verificar altavoz predeterminado. Usar `--list-devices`.
- **Permisos PowerShell** — `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
- **Aviso Hugging Face** — No es error. Opcional: `$env:HF_TOKEN = "hf_..."` antes de ejecutar.
