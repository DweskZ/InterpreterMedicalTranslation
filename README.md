# Clinic Translate (local)

Asistente **100 % local** (en uso): captura el audio que suena en **Windows** (loopback del altavoz predeterminado), transcribe en **inglés** con **faster-whisper** y muestra **español** con **Argos Translate** en una ventana siempre visible.

**Aviso:** texto asistencial, no sustituye criterio profesional; en entornos sanitarios revisar siempre la salida.

## Requisitos

- Windows 10/11, Python **3.10+**
- NVIDIA + CUDA recomendado (en CPU también funciona, con más latencia)
- ~16 GB RAM mínimo práctico; en GPU 6 GB VRAM probar `--model base` o `small`

## Instalación

En PowerShell:

```powershell
cd e:\AITrasncriptRealTime
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

En **Windows**, `requirements.txt` instala `nvidia-cublas-cu12` y **`nvidia-cuda-runtime-cu12`** (sin `cudart64_12.dll` la DLL de cuBLAS no carga). El script también **anteponde** esas carpetas al `PATH`. Si tu `.venv` es antiguo: `python -m pip install -r requirements.txt`.

En este repo ya se creó `.venv` en la máquina de desarrollo; en otra PC repite los mismos pasos.

**Atajo:** con el venv listo puedes usar `.\run.ps1` (equivale a invocar `clinic_translate.py` con el Python del venv).

**Una vez** (descarga del par idioma Argos; requiere internet):

```powershell
.\.venv\Scripts\Activate.ps1
python clinic_translate.py --setup-langs
```

La primera ejecución de Whisper puede descargar el modelo (también requiere internet si no está en caché).

## Uso

1. Pon el **altavoz predeterminado** de Windows como la salida por la que oyes la llamada o el vídeo.
2. Reproduce audio en **inglés** por ese dispositivo.
3. Ejecuta:

```powershell
.\run.ps1 --model base
# o: .\.venv\Scripts\python.exe clinic_translate.py --model base
```

La ventana debe mostrar cada pocos segundos el nivel en dBFS aunque no haya voz: si se queda muy bajo (por debajo de unos -60) mientras reproduces un vídeo, el audio no está yendo al altavoz predeterminado que Windows usa para el loopback (FxSound, otro dispositivo, volumen en cero, etc.).

Opciones útiles:

| Opción | Descripción |
|--------|-------------|
| `--model tiny` | Menor VRAM/RAM, peor calidad |
| `--model small` | Mejor calidad si la GPU aguanta |
| `--chunk-seconds 2.5` | Ventanas más cortas (más carga) |
| `--vad` | Filtro de voz de Whisper ON (por defecto OFF: mejor para audio del PC) |

## Otra PC (p. ej. tu amigo)

Copia la carpeta del proyecto (o clona el repo), crea `.venv` allí, `pip install -r requirements.txt`, `--setup-langs` y prueba con el mismo comando. Los modelos se guardan en caché de usuario (Whisper/Argos).

**Nota:** `argostranslate` instala dependencias pesadas (p. ej. `torch`, `spacy`) solo para su pipeline interno; la traducción en sí sigue siendo local. La instalación puede tardar varios minutos.

### Aviso de Hugging Face (“unauthenticated requests” / `HF_TOKEN`)

Al descargar el modelo de Whisper, Hugging Face puede mostrar un aviso de peticiones sin token. **No es un error**: el programa puede seguir descargando. Si quieres límites más altos y menos mensajes, crea un token en [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) y en PowerShell, antes de ejecutar:

```powershell
$env:HF_TOKEN = "hf_xxxxxxxx"
.\run.ps1
```

(O usa `hf auth login` si tienes la CLI de Hugging Face instalada.)

## Problemas frecuentes

- **`RuntimeError: Library cublas64_12.dll is not found`:** suele faltar **`cudart64_12.dll`** (runtime CUDA), no solo cuBLAS. `requirements.txt` ya incluye `nvidia-cublas-cu12` y `nvidia-cuda-runtime-cu12`; ejecuta `python -m pip install -r requirements.txt` y reinicia la terminal. El programa registra rutas y actualiza `PATH`. Alternativa: instalar [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads). Si falla la GPU, se usa **CPU** (más lento).

- **`[Audio] '_Speaker' object has no attribute 'recorder'`:** corregido en versiones recientes: en Windows el loopback se toma como micrófono virtual (`Loopback`), no del objeto `Speaker`. Actualiza `clinic_translate.py` desde este repo.
- **`fromstring is removed, use frombuffer`:** viene de **soundcard** con **NumPy 2**; el script aplica un parche al arrancar. Si aún falla, prueba `pip install "numpy<2"` en el venv.
- **`NoneType` ... integer / fallos raros de audio:** en Windows la captura WASAPI (**soundcard**) usa **COM**; hay que inicializarlo en **el mismo hilo** que graba. El worker llama `CoInitializeEx` al arrancar. Si aún falla, el código intenta un **respaldo con sounddevice** (PortAudio). Prueba también desactivar FxSound o cambiar el altavoz predeterminado.
- **Sin texto / error de audio:** comprueba que hay salida por el altavoz predeterminado y que el volumen no está silenciado.
- **CUDA / GPU:** actualiza drivers NVIDIA; si falla, el script intenta CPU con `int8`.
- **Permisos PowerShell:** `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` si no deja activar el venv.
