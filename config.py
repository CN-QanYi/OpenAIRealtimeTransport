"""
配置文件 - 存放所有服务配置参数
支持从 .env 文件加载配置
"""
import os
from dataclasses import dataclass, field
from typing import Optional

# 尝试加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class AudioConfig:
    """音频配置"""
    # OpenAI Realtime API 使用的采样率
    OPENAI_SAMPLE_RATE: int = 24000
    # Pipecat/STT 内部使用的采样率
    INTERNAL_SAMPLE_RATE: int = 16000
    # 声道数
    CHANNELS: int = 1
    # 音频格式
    SAMPLE_WIDTH: int = 2  # 16-bit PCM
    # 每帧音频的时长（毫秒）
    FRAME_DURATION_MS: int = 20


@dataclass
class VADConfig:
    """语音活动检测配置（内置 Server VAD，自由麦模式）"""
    # VAD 类型: 固定为 "server_vad"，启用自由麦模式
    type: str = "server_vad"
    # 静音检测阈值（毫秒），超过此时长的静音将触发语音结束
    silence_duration_ms: int = field(default_factory=lambda: int(os.getenv("VAD_SILENCE_DURATION_MS", "500")))
    # VAD 灵敏度阈值 (0.0-1.0)，越高越不敏感
    threshold: float = field(default_factory=lambda: float(os.getenv("VAD_THRESHOLD", "0.5")))
    # 语音前缀填充时长（毫秒），保留语音开始前的音频
    prefix_padding_ms: int = field(default_factory=lambda: int(os.getenv("VAD_PREFIX_PADDING_MS", "300")))
    # 启用内置 VAD
    enabled: bool = True


@dataclass
class STTConfig:
    """语音转文字配置"""
    # STT 服务提供商: "deepgram", "openai_whisper", "local_whisper"
    provider: str = field(default_factory=lambda: os.getenv("STT_PROVIDER", "deepgram"))
    
    # Deepgram 配置
    deepgram_api_key: str = field(default_factory=lambda: os.getenv("DEEPGRAM_API_KEY", ""))
    deepgram_model: str = field(default_factory=lambda: os.getenv("DEEPGRAM_MODEL", "nova-2"))
    deepgram_language: str = field(default_factory=lambda: os.getenv("DEEPGRAM_LANGUAGE", "zh-CN"))
    
    # 本地 Whisper 配置
    whisper_model: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "base"))


@dataclass
class LLMConfig:
    """语言模型配置"""
    # LLM 服务提供商: "openai", "ollama", "siliconflow"
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))
    
    # OpenAI 配置
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o"))
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    
    # Ollama 配置
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3:8b"))
    
    # 硅基流动配置
    siliconflow_api_key: str = field(default_factory=lambda: os.getenv("SILICONFLOW_API_KEY", ""))
    siliconflow_model: str = field(default_factory=lambda: os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    siliconflow_base_url: str = field(default_factory=lambda: os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"))
    
    # 通用配置
    temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.7")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "4096")))
    system_prompt: str = field(default_factory=lambda: os.getenv("LLM_SYSTEM_PROMPT", "你是一个有帮助的AI助手。请用简洁的语言回答问题。"))


@dataclass
class TTSConfig:
    """文字转语音配置"""
    # TTS 服务提供商: "elevenlabs", "edge_tts", "openai_tts"
    provider: str = field(default_factory=lambda: os.getenv("TTS_PROVIDER", "edge_tts"))
    
    # ElevenLabs 配置
    elevenlabs_api_key: str = field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", ""))
    elevenlabs_voice_id: str = field(default_factory=lambda: os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"))
    elevenlabs_model: str = field(default_factory=lambda: os.getenv("ELEVENLABS_MODEL", "eleven_monolingual_v1"))
    
    # Edge TTS 配置 (免费)
    edge_tts_voice: str = field(default_factory=lambda: os.getenv("EDGE_TTS_VOICE", "zh-CN-XiaoxiaoNeural"))
    
    # OpenAI TTS 配置
    openai_tts_voice: str = field(default_factory=lambda: os.getenv("OPENAI_TTS_VOICE", "alloy"))
    openai_tts_model: str = field(default_factory=lambda: os.getenv("OPENAI_TTS_MODEL", "tts-1"))


@dataclass
class ServerConfig:
    """服务器配置"""
    # 主机地址
    host: str = field(default_factory=lambda: os.getenv("SERVER_HOST", "0.0.0.0"))
    # 端口号
    port: int = field(default_factory=lambda: int(os.getenv("SERVER_PORT", "8000")))
    # WebSocket 端点路径
    ws_path: str = "/v1/realtime"
    # 是否启用调试模式
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "true").lower() == "true")


@dataclass
class Config:
    """主配置类"""
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


# 全局配置实例
config = Config()


def print_config():
    """打印当前配置（隐藏敏感信息）"""
    def mask_key(key: str) -> str:
        if key and len(key) > 8:
            return key[:4] + "****" + key[-4:]
        return "****" if key else "(未设置)"
    
    print("=" * 50)
    print("当前服务配置:")
    print("=" * 50)
    print(f"STT 服务: {config.stt.provider}")
    if config.stt.provider == "deepgram":
        print(f"  - API Key: {mask_key(config.stt.deepgram_api_key)}")
        print(f"  - 模型: {config.stt.deepgram_model}")
        print(f"  - 语言: {config.stt.deepgram_language}")
    elif config.stt.provider == "local_whisper":
        print(f"  - 模型: {config.stt.whisper_model}")
    
    print(f"LLM 服务: {config.llm.provider}")
    if config.llm.provider == "openai":
        print(f"  - API Key: {mask_key(config.llm.openai_api_key)}")
        print(f"  - 模型: {config.llm.openai_model}")
        print(f"  - Base URL: {config.llm.openai_base_url}")
    elif config.llm.provider == "ollama":
        print(f"  - Base URL: {config.llm.ollama_base_url}")
        print(f"  - 模型: {config.llm.ollama_model}")
    elif config.llm.provider == "siliconflow":
        print(f"  - API Key: {mask_key(config.llm.siliconflow_api_key)}")
        print(f"  - 模型: {config.llm.siliconflow_model}")
        print(f"  - Base URL: {config.llm.siliconflow_base_url}")
    
    print(f"TTS 服务: {config.tts.provider}")
    if config.tts.provider == "elevenlabs":
        print(f"  - API Key: {mask_key(config.tts.elevenlabs_api_key)}")
        print(f"  - Voice ID: {config.tts.elevenlabs_voice_id}")
    elif config.tts.provider == "edge_tts":
        print(f"  - 声音: {config.tts.edge_tts_voice}")
    elif config.tts.provider == "openai_tts":
        print(f"  - 声音: {config.tts.openai_tts_voice}")
    
    print(f"VAD 配置:")
    print(f"  - 阈值: {config.vad.threshold}")
    print(f"  - 静音时长: {config.vad.silence_duration_ms}ms")
    print("=" * 50)
