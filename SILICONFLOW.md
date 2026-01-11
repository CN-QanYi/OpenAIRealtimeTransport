# 硅基流动 (SiliconFlow) 配置指南

## 简介

硅基流动是一个国内的 AI 服务提供商，提供高性价比的 LLM 服务。兼容 OpenAI API 格式，访问速度快，价格实惠。

## 特点

✅ **国内访问快** - 服务器在国内，无需科学上网
✅ **价格实惠** - 比 OpenAI 便宜很多
✅ **API 兼容** - 完全兼容 OpenAI API 格式
✅ **模型丰富** - 支持 Qwen、DeepSeek、Yi 等多个模型

## 快速开始

### 1. 获取 API Key

1. 访问 https://siliconflow.cn/
2. 注册账号
3. 进入控制台创建 API Key
4. 新用户通常有免费额度

### 2. 配置环境变量

编辑 `.env` 文件：

```bash
# 使用硅基流动
LLM_PROVIDER=siliconflow
SILICONFLOW_API_KEY=sk-your-api-key-here
SILICONFLOW_MODEL=Qwen/Qwen2.5-7B-Instruct
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
```

### 3. 启动服务

```bash
python main.py
```

服务器启动时会显示：
```
LLM 服务: siliconflow
  - API Key: sk-ab****xyz9
  - 模型: Qwen/Qwen2.5-7B-Instruct
  - Base URL: https://api.siliconflow.cn/v1
```

## 推荐模型

| 模型 | 说明 | 适用场景 |
|------|------|----------|
| `Qwen/Qwen2.5-7B-Instruct` | 通义千问 7B | 日常对话，推荐 ⭐ |
| `Qwen/Qwen2.5-72B-Instruct` | 通义千问 72B | 复杂任务，高质量 |
| `deepseek-ai/DeepSeek-V2.5` | DeepSeek V2.5 | 代码生成，推理 |
| `01-ai/Yi-1.5-34B-Chat` | Yi 34B | 中文对话优化 |
| `meta-llama/Llama-3.1-70B-Instruct` | Llama 3.1 70B | 英文对话 |

## 完整配置示例

### 方案 1: 入门配置（推荐新手）

```bash
# STT: 本地 Whisper（免费）
STT_PROVIDER=local_whisper
WHISPER_MODEL=base

# LLM: 硅基流动（便宜快速）
LLM_PROVIDER=siliconflow
SILICONFLOW_API_KEY=sk-your-key
SILICONFLOW_MODEL=Qwen/Qwen2.5-7B-Instruct

# TTS: Edge TTS（免费）
TTS_PROVIDER=edge_tts
EDGE_TTS_VOICE=zh-CN-XiaoxiaoNeural
```

### 方案 2: 平衡配置

```bash
# STT: Deepgram（高质量）
STT_PROVIDER=deepgram
DEEPGRAM_API_KEY=your-key

# LLM: 硅基流动
LLM_PROVIDER=siliconflow
SILICONFLOW_API_KEY=sk-your-key
SILICONFLOW_MODEL=Qwen/Qwen2.5-72B-Instruct

# TTS: Edge TTS（免费）
TTS_PROVIDER=edge_tts
EDGE_TTS_VOICE=zh-CN-XiaoxiaoNeural
```

### 方案 3: 高质量配置

```bash
# STT: Deepgram
STT_PROVIDER=deepgram
DEEPGRAM_API_KEY=your-key

# LLM: 硅基流动（大模型）
LLM_PROVIDER=siliconflow
SILICONFLOW_API_KEY=sk-your-key
SILICONFLOW_MODEL=Qwen/Qwen2.5-72B-Instruct

# TTS: ElevenLabs（高质量）
TTS_PROVIDER=elevenlabs
ELEVENLABS_API_KEY=your-key
```

## 价格参考

以硅基流动为例（仅供参考，以官网为准）：

| 模型 | 输入价格 | 输出价格 |
|------|---------|---------|
| Qwen2.5-7B | ¥0.35/M tokens | ¥0.35/M tokens |
| Qwen2.5-72B | ¥4.13/M tokens | ¥4.13/M tokens |
| DeepSeek-V2.5 | ¥1.33/M tokens | ¥1.33/M tokens |

相比之下，OpenAI GPT-4o 的价格约为：
- 输入: ¥35/M tokens
- 输出: ¥105/M tokens

**硅基流动的价格约为 OpenAI 的 1/10！**

## 测试连接

运行以下命令测试配置：

```bash
python -c "from service_providers import ServiceFactory; provider = ServiceFactory.create_llm_provider('siliconflow'); print('SiliconFlow LLM provider created successfully!')"
```

## 切换模型

如果想切换到其他模型，只需修改 `.env` 中的 `SILICONFLOW_MODEL`：

```bash
# 使用 DeepSeek V2.5
SILICONFLOW_MODEL=deepseek-ai/DeepSeek-V2.5

# 使用 Qwen 72B
SILICONFLOW_MODEL=Qwen/Qwen2.5-72B-Instruct

# 使用 Yi 34B
SILICONFLOW_MODEL=01-ai/Yi-1.5-34B-Chat
```

重启服务器即可生效。

## 常见问题

### Q: 硅基流动支持哪些功能？
A: 支持文本生成、流式输出、对话历史等所有基本功能，与 OpenAI API 完全兼容。

### Q: 可以同时配置多个 LLM 提供商吗？
A: 同一时间只能使用一个提供商。如需切换，修改 `LLM_PROVIDER` 并重启服务器。

### Q: 硅基流动的服务稳定吗？
A: 作为国内主流的 AI 服务提供商，服务稳定性良好。建议配置重试机制。

### Q: 是否支持函数调用（Function Calling）？
A: 硅基流动支持函数调用，但需要模型本身支持该特性（如 Qwen2.5 系列）。

## 技术实现

硅基流动使用与 OpenAI 相同的 API 格式，因此直接复用了 `OpenAILLMProvider`：

```python
# 在 service_providers.py 中
"siliconflow": lambda: OpenAILLMProvider(
    api_key=os.getenv("SILICONFLOW_API_KEY", ""),
    model=os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
    base_url=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
    temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
    max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096"))
)
```

## 更多信息

- 官方网站: https://siliconflow.cn/
- API 文档: https://docs.siliconflow.cn/
- 模型列表: https://siliconflow.cn/models

## 推荐搭配

🎯 **最佳性价比组合**：
```
STT: Deepgram (每月200分钟免费)
LLM: SiliconFlow Qwen2.5-7B (便宜快速)
TTS: Edge TTS (完全免费)
```

💰 **成本**: 约 ¥0.05/分钟对话
⚡ **速度**: 快
🎯 **质量**: 优秀

享受高质量的 AI 语音对话服务！
