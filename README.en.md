# OpenAI Realtime API Compatible Server

[中文](README.md) | English

A local WebSocket server that mirrors the OpenAI Realtime API protocol, so you can swap OpenAI with local or third‑party model providers while keeping the client mostly unchanged.

## ✨ Features

- 🔄 **Protocol-compatible**: Mirrors OpenAI Realtime API style (URL, JSON events, audio encoding)
- 🔌 **Pluggable backends**: Uses an internal pipeline to connect STT/LLM/TTS providers (Deepgram, Ollama/Llama, ElevenLabs, SiliconFlow, etc.)
- 🚀 **Minimal client changes**: Usually only change `baseUrl` to point to this server
- 🎤 **Built-in Server VAD**: Integrates VAD (Silero when available) for hands-free “open mic” mode
- 🎙️ **Terminal client included**: A full TUI client for voice interaction
- 🌟 **SiliconFlow supported**: Faster & cheaper in mainland China; see [SILICONFLOW.md](SILICONFLOW.md)

## 📁 Project Structure

```
├── main.py                 # FastAPI server entry
├── config.py               # Config management (.env supported)
├── logger_config.py        # Logging configuration module
├── service_providers.py    # STT/LLM/TTS provider implementations
├── protocol.py             # OpenAI Realtime API protocol definitions
├── transport.py            # WebSocket Transport layer (protocol translator)
├── pipeline_manager.py     # Pipeline manager
├── realtime_session.py     # Session lifecycle manager
├── audio_utils.py          # Audio utilities (resampling/playback, etc.)
├── push_to_talk_app.py     # Terminal client (open-mic mode)
├── test_client.py          # Simple test client
└── requirements.txt        # Python dependencies
```

## 🚀 Quick Start

### 1) Install dependencies

```bash
# Option 1: create a venv (recommended)
python -m venv .venv

# Activate venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2) Configure services (important)

Copy and edit environment configuration:

```bash
cp .env.example .env
```

Recommended for users in mainland China (example):

```bash
LLM_PROVIDER=siliconflow
SILICONFLOW_API_KEY=your_api_key
SILICONFLOW_MODEL=deepseek-ai/DeepSeek-V3.2

TTS_PROVIDER=edge_tts
EDGE_TTS_VOICE=zh-CN-XiaoxiaoNeural
```

More docs:
- [QUICKSTART.md](QUICKSTART.md) (Chinese) – practical recipes
- [SILICONFLOW.md](SILICONFLOW.md) (Chinese) – SiliconFlow setup
- [.env.example](.env.example) – full config template

### 3) Start the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# or
python main.py
```

### 4) Run a client

#### Option A: Terminal UI client (recommended)

```bash
pip install textual sounddevice
python push_to_talk_app.py
```

Notes:
- Speak directly to the microphone; Server VAD detects speech automatically
- Press **Q** to quit
- Default URL: `ws://localhost:8000/v1/realtime`
- You can set `USE_LOCAL_SERVER = False` inside the client to use OpenAI instead

#### Option B: Simple test client

```bash
python test_client.py
python test_client.py -i
```

#### Option C: Use OpenAI SDK (pointing to this server)

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"  # no real key needed for local server
)

async with client.realtime.connect(model="gpt-realtime") as conn:
    ...
```

## 🔧 Architecture

### Data flow

```
Client → OpenAI-style JSON → Transport (translate) → Pipeline
                                            ↓
Client ← OpenAI-style JSON ← Transport (translate) ← (VAD → STT → LLM → TTS)
```

### Key components

1. **Transport** ([transport.py](transport.py))
   - Converts OpenAI-style events to internal frames and back

2. **Pipeline Manager** ([pipeline_manager.py](pipeline_manager.py))
   - VAD / STT / LLM / TTS orchestration

3. **Session Manager** ([realtime_session.py](realtime_session.py))
   - WebSocket session lifecycle; connects Transport ↔ Pipeline

4. **Audio Utilities** ([audio_utils.py](audio_utils.py))
   - Audio resampling (24kHz ↔ 16kHz)
   - Audio buffer management
   - Async audio player for client

## 📊 Supported Services

### STT (Speech-to-Text)
| Provider | Config Value | Notes | API Key |
|----------|--------------|-------|---------|
| **Deepgram** 🌟 | `deepgram` | High quality, 200 min/month free | Required |
| OpenAI Whisper | `openai_whisper` | OpenAI official API | Required |
| Local Whisper | `local_whisper` | Completely free, needs model download | Not required |

### LLM (Language Model)
| Provider | Config Value | Notes | API Key |
|----------|--------------|-------|---------|
| **SiliconFlow** 🌟 | `siliconflow` | Fast in China, ~1/10 OpenAI price | Required |
| OpenAI | `openai` | GPT-4o and other models | Required |
| Ollama | `ollama` | Local, completely free | Not required |

### TTS (Text-to-Speech)
| Provider | Config Value | Notes | API Key |
|----------|--------------|-------|---------|
| **Edge TTS** 🌟 | `edge_tts` | Microsoft Edge TTS, free | Not required |
| ElevenLabs | `elevenlabs` | High quality, 10k chars/month free | Required |
| OpenAI TTS | `openai_tts` | OpenAI official TTS | Required |

🌟 = Recommended

## 📝 Configuration

All configuration is done via `.env` file. See [.env.example](.env.example) for the complete template.

### VAD Configuration (Open-mic mode)
```bash
VAD_THRESHOLD=0.5          # Sensitivity (0.0-1.0), higher = less sensitive
VAD_SILENCE_DURATION_MS=500  # Silence detection duration (ms)
VAD_PREFIX_PADDING_MS=300    # Speech prefix padding (ms)
```

## 📄 License

See [LICENSE](LICENSE).
