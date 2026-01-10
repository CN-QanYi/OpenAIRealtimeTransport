# OpenAI Realtime API 兼容服务器

一个完全复刻 OpenAI Realtime API 协议的本地 WebSocket 服务器，允许你使用本地或第三方模型替代 OpenAI。

## ✨ 特性

- 🔄 **完全兼容**：对外复刻 OpenAI Realtime API 的协议（URL、JSON 事件格式、音频编码）
- 🔌 **可替换后端**：对内使用 Pipecat 管道调用本地或第三方模型（Deepgram、Llama 3、ElevenLabs、硅基流动等）
- 🚀 **零客户端修改**：你的客户端应用只需修改 `baseUrl` 即可连接
- 🎤 **内置 Server VAD**：集成 Pipecat 的 Silero VAD，默认启用自由麦模式，自动检测语音活动
- 🎙️ **终端客户端**：提供完整的终端 UI 客户端，无需按键即可进行语音交互
- 🌟 **支持硅基流动**：国内访问快，价格低廉（约为 OpenAI 的 1/10），详见 [SILICONFLOW.md](SILICONFLOW.md)

## 📁 项目结构

```
├── main.py                 # FastAPI 主服务器
├── config.py               # 配置管理（支持 .env）
├── service_providers.py    # STT/LLM/TTS 服务提供商
├── protocol.py             # OpenAI Realtime API 协议定义
├── transport.py            # WebSocket Transport 层（协议翻译官）
├── pipeline_manager.py     # Pipecat 管道管理器
├── realtime_session.py     # 会话生命周期管理
├── audio_utils.py          # 音频处理工具（重采样、音频播放等）
├── push_to_talk_app.py     # Push-to-Talk 终端客户端（推荐）
├── test_client.py          # 简单测试客户端
└── requirements.txt        # 依赖列表
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 方法1: 创建虚拟环境（推荐）
python -m venv .venv

# 激活虚拟环境
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat
# Linux/Mac:
source .venv/bin/activate

# 安装依赖（使用清华镜像源，速度更快）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 或方法2: 直接安装（如果网络良好）
pip install -r requirements.txt

# 核心依赖包括:
# - fastapi, uvicorn: Web 框架和服务器
# - websockets: WebSocket 支持
# - numpy, scipy: 音频处理
# - pipecat-ai: 提供 Server VAD (Silero VAD) 等音频处理功能
```

### 2. 配置服务（重要！）

```bash
# 复制配置文件模板
cp .env.example .env

# 编辑 .env 文件，配置你的服务
# Windows:
notepad .env
# Linux/Mac:
nano .env
```

**推荐配置**（国内用户）：
```bash
# 使用硅基流动 LLM（快速且便宜）
LLM_PROVIDER=siliconflow
SILICONFLOW_API_KEY=你的_api_key
SILICONFLOW_MODEL=Qwen/Qwen2.5-7B-Instruct

# 使用 Edge TTS（完全免费）
TTS_PROVIDER=edge_tts
EDGE_TTS_VOICE=zh-CN-XiaoxiaoNeural
```

详细配置说明请查看：
- [QUICKSTART.md](QUICKSTART.md) - 快速入门指南
- [SILICONFLOW.md](SILICONFLOW.md) - 硅基流动配置指南
- [.env.example](.env.example) - 完整配置模板

### 3. 启动服务器

```bash
# 确保已激活虚拟环境
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# 开发模式（自动重载）
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 或直接运行
python main.py
```

服务器启动后会显示当前配置：
```
==================================================
当前服务配置:
==================================================
STT 服务: deepgram
LLM 服务: siliconflow
  - 模型: Qwen/Qwen2.5-7B-Instruct
TTS 服务: edge_tts
==================================================
```

### 4. 运行客户端测试

#### 方式 1: 终端 UI 客户端（推荐）

提供完整的终端 UI 界面，支持自由麦模式：

```bash
# 安装客户端依赖
pip install textual sounddevice

# 在新的终端窗口中运行客户端
python push_to_talk_app.py
```

**使用说明：**
- 应用启动后直接对着麦克风说话即可
- Server VAD 自动检测语音开始和结束
- 按 **Q** 键退出应用
- 客户端会自动连接到 `ws://localhost:8000/v1/realtime`
- 可以在 `push_to_talk_app.py` 中设置 `USE_LOCAL_SERVER = False` 切换到 OpenAI 官方 API

**功能特性：**
- ✅ 内置自由麦模式，无需按键操作
- ✅ 实时显示会话 ID
- ✅ 自动语音活动检测
- ✅ 实时显示 AI 响应文本
- ✅ 自动播放 AI 语音响应
- ✅ 完整的 TUI（终端用户界面）

#### 方式 2: 简单测试客户端

```bash
# 自动测试模式
python test_client.py

# 交互模式
python test_client.py -i
```

#### 方式 3: 使用 OpenAI SDK

在你的客户端代码中：

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"  # 本地服务器不需要真实 key
)

async with client.realtime.connect(model="gpt-realtime") as conn:
    # 你的代码...
```

## 🔧 架构设计

### 数据流向

```
客户端 → OpenAI 格式 JSON → Transport (翻译) → Pipecat Pipeline
                                                    ↓
客户端 ← OpenAI 格式 JSON ← Transport (翻译) ← (VAD→STT→LLM→TTS)
```

### 核心组件

1. **Transport 层** (`transport.py`)
   - 接收 OpenAI 格式的客户端事件
   - 转换为 Pipecat 内部帧格式
   - 将输出转换回 OpenAI 格式

2. **Pipeline 管理器** (`pipeline_manager.py`)
   - VAD：语音活动检测
   - STT：语音转文字
   - LLM：语言模型推理
   - TTS：文字转语音

3. **会话管理** (`realtime_session.py`)
   - 管理 WebSocket 会话生命周期
   - 协调 Transport 和 Pipeline

4. **音频处理** (`audio_utils.py`)
   - 音频重采样（24kHz ↔ 16kHz）
   - 音频缓冲区管理
   - 异步音频播放器（用于客户端）

## 📋 支持的事件

### 客户端 → 服务器

| 事件类型 | 描述 |
|---------|------|
| `session.update` | 更新会话配置（VAD 参数、指令等） |
| `input_audio_buffer.append` | 追加音频数据（Server VAD 自动处理） |
| `input_audio_buffer.clear` | 清空音频缓冲区 |
| `conversation.item.create` | 创建对话项 |
| `response.create` | 请求生成响应 |
| `response.cancel` | 取消当前响应 |

### 服务器 → 客户端

| 事件类型 | 描述 |
|---------|------|
| `session.created` | 会话已创建 |
| `session.updated` | 会话已更新 |
| `input_audio_buffer.speech_started` | 检测到语音开始 |
| `input_audio_buffer.speech_stopped` | 检测到语音停止 |
| `response.created` | 响应已创建 |
| `response.audio.delta` | 音频增量 |
| `response.audio_transcript.delta` | 转录增量 |
| `response.done` | 响应完成 |

## ⚙️ 配置说明

所有配置都通过 `.env` 文件管理，支持以下配置项：

### 基础配置
```bash
# 服务器配置
DEBUG=true
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
```

### STT 配置
```bash
STT_PROVIDER=deepgram  # 或 openai_whisper, local_whisper
DEEPGRAM_API_KEY=你的_api_key
DEEPGRAM_MODEL=nova-2
DEEPGRAM_LANGUAGE=zh-CN
```

### LLM 配置
```bash
# 硅基流动（推荐）
LLM_PROVIDER=siliconflow
SILICONFLOW_API_KEY=你的_api_key
SILICONFLOW_MODEL=Qwen/Qwen2.5-7B-Instruct

# 或 OpenAI
# LLM_PROVIDER=openai
# OPENAI_API_KEY=你的_api_key
# OPENAI_MODEL=gpt-4o

# 或 Ollama（本地）
# LLM_PROVIDER=ollama
# OLLAMA_BASE_URL=http://localhost:11434
# OLLAMA_MODEL=llama3:8b
```

### TTS 配置
```bash
TTS_PROVIDER=edge_tts  # 或 elevenlabs, openai_tts
EDGE_TTS_VOICE=zh-CN-XiaoxiaoNeural
```

### VAD 配置（自由麦模式）
```bash
VAD_THRESHOLD=0.5  # 灵敏度 (0.0-1.0)，越高越不敏感
VAD_SILENCE_DURATION_MS=500  # 静音检测时长（毫秒）
VAD_PREFIX_PADDING_MS=300  # 语音前缀填充（毫秒）
```

完整配置选项请查看 [.env.example](.env.example)

**注意**: 自由麦模式需要 Pipecat 提供的 Silero VAD 支持，已包含在 `pipecat-ai` 依赖中。

## 🎯 支持的服务

### STT（语音转文字）
| 服务 | 配置值 | 说明 | API Key |
|------|--------|------|--------|
| **Deepgram** 🌟 | `deepgram` | 高质量，每月 200 分钟免费 | 需要 |
| OpenAI Whisper | `openai_whisper` | OpenAI 官方 API | 需要 |
| 本地 Whisper | `local_whisper` | 完全免费，需下载模型 | 不需要 |

### LLM（语言模型）
| 服务 | 配置值 | 说明 | API Key |
|------|--------|------|--------|
| **硅基流动** 🌟 | `siliconflow` | 国内访问快，价格约 OpenAI 1/10 | 需要 |
| OpenAI | `openai` | GPT-4o 等模型 | 需要 |
| Ollama | `ollama` | 本地运行，完全免费 | 不需要 |

### TTS（文字转语音）
| 服务 | 配置值 | 说明 | API Key |
|------|--------|------|--------|
| **Edge TTS** 🌟 | `edge_tts` | 微软 Edge 浏览器 TTS，完全免费 | 不需要 |
| ElevenLabs | `elevenlabs` | 高质量语音，每月 10000 字符免费 | 需要 |
| OpenAI TTS | `openai_tts` | OpenAI 官方 TTS | 需要 |

🌟 = 推荐选项

### 推荐组合

**最佳性价比**（推荐）：
```bash
STT: Deepgram（每月 200 分钟免费）
LLM: 硅基流动（¥0.35/M tokens）
TTS: Edge TTS（完全免费）
成本: 约 ¥0.05/分钟对话
```

**完全免费**：
```bash
STT: 本地 Whisper
LLM: Ollama
TTS: Edge TTS
成本: ¥0（需要本地计算资源）
```

**高质量**：
```bash
STT: Deepgram
LLM: OpenAI GPT-4o
TTS: ElevenLabs
成本: 较高，适合商业应用
```

## ⚠️ 注意事项

### 音频采样率
- OpenAI 协议使用 **24kHz**
- 大多数 STT 模型使用 **16kHz**
- `audio_utils.py` 自动处理重采样

### 内置 Server VAD（自由麦模式）

服务器内置了 Pipecat 的 Silero VAD，默认启用 `server_vad` 模式，自动检测用户的语音活动：

**工作流程：**
1. 客户端连续发送音频数据（`input_audio_buffer.append`）
2. VAD 自动检测到用户开始说话 → 发送 `input_audio_buffer.speech_started` 事件
3. VAD 检测到用户停止说话 → 发送 `input_audio_buffer.speech_stopped` 事件
4. 服务器自动触发 STT → LLM → TTS 流程
5. 客户端收到 AI 响应的音频和文本

**打断功能：**
- 客户端收到 `speech_started` 事件后应立即停止播放 AI 音频
- 实现自然的对话打断体验

**VAD 参数调优：**
可以通过 `session.update` 事件调整 VAD 参数：
- `threshold`: 灵敏度阈值 (0.0-1.0)
- `silence_duration_ms`: 静音检测时长
- `prefix_padding_ms`: 语音前缀填充

### JSON 格式严格性
`response_id` 和 `item_id` 字段必须存在，使用随机 UUID 填充。

### 客户端音频设备
`push_to_talk_app.py` 需要麦克风和扬声器支持。如遇问题：
- Windows: 确保安装了 `sounddevice` 包
- Mac: 需要 `brew install portaudio`
- Linux: 需要安装 `portaudio19-dev` 和 `python3-pyaudio`

## 🎬 快速演示

### 1. 启动服务器（终端 1）
```bash
python main.py
```
输出：
```
OpenAI Realtime API 兼容服务器启动
WebSocket 端点: ws://localhost:8000/v1/realtime
```

### 2. 启动客户端（终端 2）
```bash
# 启动自由麦客户端
python push_to_talk_app.py
```

### 3. 开始对话

**自由麦模式（唯一模式）：**
1. 客户端启动后自动进入自由麦模式
2. 直接对着麦克风说话（例如："你好，今天天气怎么样？"）
3. Server VAD 自动检测你的语音开始和结束
4. 等待 AI 响应（文本会显示在界面上，音频会自动播放）
5. 可以随时打断 AI 的回答，继续说话

## 🔄 替换为本地/第三方模型

本项目已集成真实的 STT、LLM 和 TTS 服务。你可以通过 `.env` 文件配置使用哪个服务提供商。

### 快速配置

1. 复制环境变量示例文件：
```bash
cp .env.example .env
```

2. 编辑 `.env` 文件，配置你需要的服务：

### STT 服务配置

| 服务提供商 | 说明 | 需要 API Key |
|-----------|------|-------------|
| `deepgram` | Deepgram Nova-2，高准确率（推荐） | ✅ |
| `openai_whisper` | OpenAI Whisper API | ✅ |
| `local_whisper` | 本地 Whisper 模型 | ❌ |

```bash
# Deepgram (推荐)
STT_PROVIDER=deepgram
DEEPGRAM_API_KEY=your_key_here
DEEPGRAM_MODEL=nova-2
DEEPGRAM_LANGUAGE=zh-CN

# 或使用本地 Whisper
STT_PROVIDER=local_whisper
WHISPER_MODEL=base  # base, small, medium, large
```

### LLM 服务配置

| 服务提供商 | 说明 | 需要 API Key |
|-----------|------|-------------|
| `openai` | OpenAI GPT 系列（推荐） | ✅ |
| `ollama` | 本地 Ollama 模型 | ❌ |

```bash
# OpenAI (默认)
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o
OPENAI_BASE_URL=https://api.openai.com/v1

# 或使用 Ollama (本地)
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3:8b
```

### TTS 服务配置

| 服务提供商 | 说明 | 需要 API Key |
|-----------|------|-------------|
| `edge_tts` | Microsoft Edge TTS（免费，推荐） | ❌ |
| `elevenlabs` | ElevenLabs 高质量语音 | ✅ |
| `openai_tts` | OpenAI TTS | ✅ |

```bash
# Edge TTS (免费，推荐)
TTS_PROVIDER=edge_tts
EDGE_TTS_VOICE=zh-CN-XiaoxiaoNeural

# 或使用 ElevenLabs (高质量)
TTS_PROVIDER=elevenlabs
ELEVENLABS_API_KEY=your_key_here
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM

# 或使用 OpenAI TTS
TTS_PROVIDER=openai_tts
OPENAI_TTS_VOICE=alloy
OPENAI_TTS_MODEL=tts-1
```

### 完整 .env 示例

```bash
# ==================== STT 配置 ====================
STT_PROVIDER=deepgram
DEEPGRAM_API_KEY=your_deepgram_api_key
DEEPGRAM_MODEL=nova-2
DEEPGRAM_LANGUAGE=zh-CN

# ==================== LLM 配置 ====================
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=4096
LLM_SYSTEM_PROMPT=你是一个有帮助的AI助手。

# ==================== TTS 配置 ====================
TTS_PROVIDER=edge_tts
EDGE_TTS_VOICE=zh-CN-XiaoxiaoNeural

# ==================== VAD 配置 ====================
VAD_THRESHOLD=0.5
VAD_SILENCE_DURATION_MS=500
VAD_PREFIX_PADDING_MS=300

# ==================== 服务器配置 ====================
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
DEBUG=true
```

### 免费方案（无需 API Key）

如果你没有 API Key，可以使用以下完全免费的配置：

```bash
# 本地 Whisper (需要安装 openai-whisper)
STT_PROVIDER=local_whisper
WHISPER_MODEL=base

# Ollama (需要安装并运行 Ollama 服务)
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3:8b

# Edge TTS (免费)
TTS_PROVIDER=edge_tts
EDGE_TTS_VOICE=zh-CN-XiaoxiaoNeural
```

安装本地模型：
```bash
# 安装本地 Whisper
pip install openai-whisper

# 安装 Ollama (访问 https://ollama.ai)
# 然后下载模型
ollama pull llama3:8b
```

## 🐛 故障排除

### 客户端无法连接
```bash
# 检查服务器是否运行
curl http://localhost:8000/health

# 应返回: {"status":"healthy","active_sessions":0}
```

### 音频设备问题
```python
# 测试音频设备
python -c "import sounddevice as sd; print(sd.query_devices())"
```

### 模块导入错误
```bash
# 重新安装依赖
pip install --upgrade -r requirements.txt
```

### WebSocket 连接断开
- 检查防火墙设置
- 确保端口 8000 未被占用
- 查看服务器日志以获取详细错误信息

## 📊 性能优化

### 音频重采样
- 安装 `soxr` 以获得更高质量的重采样：
```bash
pip install soxr
```

### 减少延迟
- 调整 `config.py` 中的 VAD 参数
- 使用更快的本地模型
- 减小音频缓冲区大小

## 🔜 后续计划

- [x] 完整的协议实现
- [x] 自由麦终端客户端
- [x] 音频处理和重采样
- [x] 内置 Server VAD (Silero VAD)，纯自由麦模式
- [x] 集成真实的 STT 服务（Deepgram/Whisper/本地 Whisper）
- [x] 集成真实的 LLM 服务（OpenAI/Ollama）
- [x] 集成真实的 TTS 服务（ElevenLabs/Edge TTS/OpenAI TTS）
- [x] 支持 .env 环境变量配置
- [ ] 支持函数调用
- [ ] 支持多模态输入
- [ ] Docker 部署支持

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

### 开发指南
1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

MIT License
