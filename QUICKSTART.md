# 快速入门指南

## 🚀 三种运行方案

### 方案 1: 完全免费方案（推荐新手）

使用免费的开源服务，无需任何 API Key：

1. **安装 Ollama（本地 LLM）**
   ```bash
   # 访问 https://ollama.ai 下载安装
   # 安装后运行:
   ollama pull llama3:8b
   ```

2. **配置 .env 文件**
   ```bash
   STT_PROVIDER=local_whisper
   WHISPER_MODEL=base
   
   LLM_PROVIDER=ollama
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama3:8b
   
   TTS_PROVIDER=edge_tts
   EDGE_TTS_VOICE=zh-CN-XiaoxiaoNeural
   ```

3. **首次运行会下载 Whisper 模型**
   ```bash
   python main.py
   ```

**优点**: 完全免费，数据隐私
**缺点**: 首次需要下载模型，Whisper 识别较慢

---

### 方案 2: 混合方案（推荐）

使用免费的 Edge TTS + 付费的高质量 STT/LLM：

1. **配置 .env 文件**
   ```bash
   # Deepgram STT (每月 200 分钟免费额度)
   STT_PROVIDER=deepgram
   DEEPGRAM_API_KEY=你的_deepgram_key
   
   # 硅基流动 LLM (国内访问快，价格便宜，推荐！)
   LLM_PROVIDER=siliconflow
   SILICONFLOW_API_KEY=你的_siliconflow_key
   SILICONFLOW_MODEL=Qwen/Qwen2.5-7B-Instruct
   
   # 或使用 OpenAI LLM
   # LLM_PROVIDER=openai
   # OPENAI_API_KEY=你的_openai_key
   # OPENAI_MODEL=gpt-4o-mini
   
   # Edge TTS (完全免费)
   TTS_PROVIDER=edge_tts
   EDGE_TTS_VOICE=zh-CN-XiaoxiaoNeural
   ```

2. **启动服务**
   ```bash
   python main.py
   ```

**优点**: 性能好，成本低，国内访问快
**缺点**: 需要申请 API Key

---

### 方案 3: 高质量方案

使用最优质的商业服务：

1. **配置 .env 文件**
   ```bash
   # Deepgram STT
   STT_PROVIDER=deepgram
   DEEPGRAM_API_KEY=你的_deepgram_key
   
   # OpenAI LLM
   LLM_PROVIDER=openai
   OPENAI_API_KEY=你的_openai_key
   OPENAI_MODEL=gpt-4o
   
   # ElevenLabs TTS (高质量语音)
   TTS_PROVIDER=elevenlabs
   ELEVENLABS_API_KEY=你的_elevenlabs_key
   ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM
   ```

2. **启动服务**
   ```bash
   python main.py
   ```

**优点**: 最高质量，最快速度
**缺点**: 成本较高

---

## 📝 获取 API Keys

### 硅基流动 (LLM) - 推荐！🌟
1. 访问 https://siliconflow.cn/
2. 注册账号（新用户有免费额度）
3. 创建 API Key
4. **优势**: 国内访问快，价格约为 OpenAI 的 1/10

### Deepgram (STT)
1. 访问 https://console.deepgram.com/
2. 注册账号（每月 200 分钟免费额度）
3. 创建 API Key

### OpenAI (LLM)
1. 访问 https://platform.openai.com/
2. 注册账号并充值
3. 创建 API Key

### ElevenLabs (TTS)
1. 访问 https://elevenlabs.io/
2. 注册账号（每月 10,000 字符免费额度）
3. 创建 API Key

---

## 🎯 运行步骤

### 1. 安装依赖

```bash
# 激活虚拟环境
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
# 或
.venv\Scripts\activate.bat    # Windows CMD
# 或
source .venv/bin/activate      # Linux/Mac

# 安装所有依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 复制示例配置
cp .env.example .env

# 编辑 .env 文件，填入你的配置
notepad .env  # Windows
# 或
nano .env     # Linux/Mac
```

### 3. 启动服务器

```bash
# 开发模式（自动重载）
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 或直接运行
python main.py
```

### 4. 启动客户端

**打开新的终端窗口：**

```bash
# 激活虚拟环境
.\.venv\Scripts\Activate.ps1

# 运行客户端
python push_to_talk_app.py
```

### 5. 开始对话

- 直接对着麦克风说话即可
- Server VAD 自动检测语音开始和结束
- AI 响应会自动播放
- 按 **Q** 键退出

---

## 🔍 验证配置

运行以下命令查看当前配置：

```bash
python -c "from config import config, print_config; print_config()"
```

输出示例：
```
==================================================
当前服务配置:
==================================================
STT 服务: deepgram
  - API Key: abc1****xyz9
  - 模型: nova-2
  - 语言: zh-CN
LLM 服务: openai
  - API Key: sk-1****xyz9
  - 模型: gpt-4o
  - Base URL: https://api.openai.com/v1
TTS 服务: edge_tts
  - 声音: zh-CN-XiaoxiaoNeural
VAD 配置:
  - 阈值: 0.5
  - 静音时长: 500ms
==================================================
```

---

## ❗ 常见问题

### 1. 缺少 API Key

**问题**: 启动服务器时提示 API Key 未设置

**解决**: 检查 `.env` 文件是否存在并正确配置了 API Key

### 2. Whisper 模型下载慢

**问题**: 首次使用本地 Whisper 时下载很慢

**解决**: 
- 使用科学上网工具
- 或手动下载模型后放置到 `~/.cache/whisper/` 目录

### 3. Ollama 连接失败

**问题**: `LLM_PROVIDER=ollama` 时提示连接失败

**解决**: 
1. 确保 Ollama 服务正在运行
2. 检查 `OLLAMA_BASE_URL` 是否正确
3. 运行 `ollama list` 确认模型已下载

### 4. 麦克风无法识别

**问题**: 客户端无法捕获音频

**解决**:
```bash
# 查看可用音频设备
python -c "import sounddevice as sd; print(sd.query_devices())"

# 在 push_to_talk_app.py 中指定设备 ID
```

---

## 💡 推荐配置

| 场景 | STT | LLM | TTS | 成本 |
|------|-----|-----|-----|------|
| 开发测试 | local_whisper | ollama | edge_tts | 免费 |
| 日常使用 | deepgram | siliconflow 🌟 | edge_tts | 很低 |
| 生产环境 | deepgram | siliconflow/openai | edge_tts | 低/中 |
| 演示展示 | deepgram | openai | elevenlabs | 高 |

🌟 **推荐**: 硅基流动（SiliconFlow）- 国内访问快，价格低，详见 [SILICONFLOW.md](SILICONFLOW.md)

---

## 📞 需要帮助？

- 查看 [README.md](README.md) 了解详细文档
- 查看 [.env.example](.env.example) 了解所有配置选项
- 提交 Issue: https://github.com/CN-QanYi/OpenAIRealtimeTransport/issues
