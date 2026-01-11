"""
管道管理器 - 管理 Pipecat 管道的创建和配置
提供一个简化的接口来构建语音处理管道
"""
import asyncio
from typing import Optional, Callable, Awaitable, List, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from config import config, print_config
from service_providers import (
    ServiceFactory, 
    BaseSTTProvider, 
    BaseLLMProvider, 
    BaseTTSProvider
)
from logger_config import get_logger

logger = get_logger(__name__)


# ==================== 帧类型定义 ====================
# 由于 Pipecat 可能未安装，我们定义自己的帧类型

@dataclass
class Frame:
    """基础帧类型"""
    pass


@dataclass
class AudioFrame(Frame):
    """音频帧"""
    audio: bytes
    sample_rate: int = 16000
    num_channels: int = 1


@dataclass
class InputAudioFrame(AudioFrame):
    """输入音频帧（来自用户）"""
    pass


@dataclass
class OutputAudioFrame(AudioFrame):
    """输出音频帧（发送给用户）"""
    pass


@dataclass
class TextFrame(Frame):
    """文本帧"""
    text: str


@dataclass
class TranscriptionFrame(TextFrame):
    """转录文本帧（STT 输出）"""
    pass


@dataclass
class LLMResponseFrame(TextFrame):
    """LLM 响应帧"""
    pass


@dataclass
class TTSAudioFrame(OutputAudioFrame):
    """TTS 输出音频帧"""
    pass


@dataclass
class UserStartedSpeakingFrame(Frame):
    """用户开始说话事件帧"""
    timestamp_ms: int = 0


@dataclass
class UserStoppedSpeakingFrame(Frame):
    """用户停止说话事件帧"""
    timestamp_ms: int = 0


@dataclass
class BotStartedSpeakingFrame(Frame):
    """机器人开始说话事件帧"""
    pass


@dataclass
class BotStoppedSpeakingFrame(Frame):
    """机器人停止说话事件帧"""
    pass


@dataclass 
class EndFrame(Frame):
    """结束帧，表示处理完成"""
    pass


# ==================== 服务抽象基类 ====================

class BaseService(ABC):
    """服务基类"""
    
    @abstractmethod
    async def process(self, frame: Frame) -> Optional[Frame]:
        """处理帧"""
        pass


class VADService(BaseService):
    """语音活动检测服务 - 使用 Pipecat 的 Silero VAD"""
    
    def __init__(self, threshold: float = 0.5, 
                 silence_duration_ms: int = 500,
                 prefix_padding_ms: int = 300):
        self.threshold = threshold
        self.silence_duration_ms = silence_duration_ms
        self.prefix_padding_ms = prefix_padding_ms
        self._is_speaking = False
        self._silence_frames = 0
        self._on_speech_start: Optional[Callable[[], Awaitable[None]]] = None
        self._on_speech_end: Optional[Callable[[], Awaitable[None]]] = None
        
        # 尝试加载 Pipecat 的 Silero VAD
        self._silero_vad = None
        self._use_silero = False
        self._silero_available = False  # 标记 Silero 库是否可用
        self._silero_sample_rate = None  # 当前 Silero VAD 配置的采样率
        self._prev_vad_state = None  # 保存上一次的 VAD 状态用于检测转换
        self._VADState = None  # 存储 VADState 枚举，避免重复导入
        self._SileroVADAnalyzer = None  # 存储 SileroVADAnalyzer 类，避免重复导入
        try:
            from pipecat.vad.silero import SileroVADAnalyzer, VADState
            self._VADState = VADState  # 存储枚举类
            self._SileroVADAnalyzer = SileroVADAnalyzer  # 存储类引用
            self._silero_available = True
            logger.info("✅ Pipecat Silero VAD 可用，将在收到第一帧时根据采样率初始化")
        except ImportError:
            logger.warning("⚠️  Pipecat Silero VAD 不可用，使用简单的能量检测 VAD")
        except Exception as e:
            logger.warning(f"⚠️  Silero VAD 导入失败: {e}，使用简单的能量检测 VAD")
    
    def on_speech_start(self, callback: Callable[[], Awaitable[None]]):
        """设置语音开始回调"""
        self._on_speech_start = callback
        return self
    
    def on_speech_end(self, callback: Callable[[], Awaitable[None]]):
        """设置语音结束回调"""
        self._on_speech_end = callback
        return self
    
    async def process(self, frame: Frame) -> Optional[Frame]:
        """处理音频帧，检测语音活动"""
        if not isinstance(frame, InputAudioFrame):
            return frame
        
        import numpy as np
        audio_array = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32)
        
        if len(audio_array) == 0:
            return frame
        
        # 尝试初始化或重新配置 Silero VAD（根据采样率）
        if self._silero_available and not self._use_silero:
            # Silero VAD 可用但尚未初始化，根据当前帧的采样率初始化
            if frame.sample_rate in (8000, 16000):
                try:
                    self._silero_vad = self._SileroVADAnalyzer(
                        confidence=self.threshold,
                        min_silence_duration_ms=self.silence_duration_ms,
                        padding_duration_ms=self.prefix_padding_ms,
                        sample_rate=frame.sample_rate
                    )
                    self._use_silero = True
                    self._silero_sample_rate = frame.sample_rate
                    self._prev_vad_state = self._VADState.QUIET
                    logger.info(f"✅ Silero VAD 已初始化，采样率: {frame.sample_rate} Hz")
                except Exception as e:
                    logger.warning(f"⚠️  Silero VAD 初始化失败: {e}，使用简单的能量检测 VAD")
                    self._silero_available = False
            else:
                logger.warning(
                    f"Silero VAD 不支持采样率 {frame.sample_rate} Hz (仅支持 8000/16000 Hz)，"
                    f"使用能量检测 VAD"
                )
                self._silero_available = False
        
        # 使用 Silero VAD 或回退到简单的能量检测
        if self._use_silero and self._silero_vad:
            # 检查采样率是否与初始化时一致
            if frame.sample_rate != self._silero_sample_rate:
                if frame.sample_rate in (8000, 16000):
                    # 采样率改变但仍在支持范围内，重新初始化 Silero VAD
                    try:
                        self._silero_vad = self._SileroVADAnalyzer(
                            confidence=self.threshold,
                            min_silence_duration_ms=self.silence_duration_ms,
                            padding_duration_ms=self.prefix_padding_ms,
                            sample_rate=frame.sample_rate
                        )
                        self._silero_sample_rate = frame.sample_rate
                        self._prev_vad_state = self._VADState.QUIET
                        # 保留 _is_speaking 和 _silence_frames 以保持会话连续性
                        logger.info(f"✅ Silero VAD 已重新初始化，新采样率: {frame.sample_rate} Hz")
                    except Exception as e:
                        logger.warning(f"⚠️  Silero VAD 重新初始化失败: {e}，回退到能量检测 VAD")
                        # 禁用 Silero 但保留当前说话状态以保持会话连续性
                        self._use_silero = False
                        self._silero_vad = None
                        self._silence_frames = 0
                        # 注意：不重置 _is_speaking，保持当前状态
                else:
                    logger.warning(
                        f"Silero VAD 不支持采样率 {frame.sample_rate} Hz (仅支持 8000/16000 Hz)，"
                        f"回退到能量检测 VAD"
                    )
                    # 禁用 Silero 但保留当前说话状态以保持会话连续性
                    self._use_silero = False
                    self._silero_vad = None
                    self._silence_frames = 0
                    # 注意：不重置 _is_speaking，保持当前状态
            else:
                try:
                    # 将 float32 音频转换为 PCM16 (int16) 字节格式
                    audio_int16 = np.clip(audio_array, -32768, 32767).astype(np.int16)
                    audio_bytes = audio_int16.tobytes()
                    
                    # Silero VAD 分析（异步调用），返回 VADState 枚举
                    vad_state = await self._silero_vad.analyze_audio(audio_bytes)
                    
                    # 检测状态转换
                    # QUIET → SPEAKING: 用户开始说话
                    if self._prev_vad_state == self._VADState.QUIET and vad_state == self._VADState.SPEAKING:
                        self._is_speaking = True
                        if self._on_speech_start:
                            await self._on_speech_start()
                        self._prev_vad_state = vad_state
                        return UserStartedSpeakingFrame()
                    
                    # SPEAKING → QUIET: 用户停止说话
                    elif self._prev_vad_state == self._VADState.SPEAKING and vad_state == self._VADState.QUIET:
                        self._is_speaking = False
                        if self._on_speech_end:
                            await self._on_speech_end()
                        self._prev_vad_state = vad_state
                        return UserStoppedSpeakingFrame()
                    
                    # 更新状态
                    self._prev_vad_state = vad_state
                    return frame
                except Exception:
                    logger.exception("Silero VAD 处理错误，回退到能量检测")
                    # 禁用 Silero VAD，但保留当前的说话状态和静音计数器
                    # 这样能量检测 VAD 可以接管并在适当时候发出正确的停止事件
                    # 不要重置 _is_speaking 和 _silence_frames，以保持会话连续性
                    # 并确保下游 STT 缓冲区不会被意外中断
                    self._use_silero = False
                    self._silero_vad = None
                    # 注意：保留 _is_speaking 和 _silence_frames 的当前值
                    # 能量检测 VAD 将从当前状态继续监控，并在检测到
                    # 静音超过阈值时正确发出 UserStoppedSpeakingFrame
                    logger.info(
                        f"VAD 状态转移到能量检测: is_speaking={self._is_speaking}, "
                        f"silence_frames={self._silence_frames}"
                    )
                    # 立即使用当前帧继续进行能量检测处理（不返回，继续执行下面的代码）
        
        # 回退：简单的能量检测 VAD
        # 计算 RMS 能量
        rms = np.sqrt(np.mean(audio_array ** 2))
        
        # 使用动态阈值，避免底噪触发
        # 32768 是 Int16 的最大值
        # 0.5 (默认) * 10000 = 5000 (约为 -16dB IFS)
        base_threshold = max(self.threshold * 10000, 500)
        is_speech = rms > base_threshold
        
        if is_speech and not self._is_speaking:
            self._is_speaking = True
            self._silence_frames = 0
            if self._on_speech_start:
                await self._on_speech_start()
            return UserStartedSpeakingFrame()
        
        elif not is_speech and self._is_speaking:
            self._silence_frames += 1
            # 根据静音时长判断是否结束
            frame_duration_ms = len(audio_array) / frame.sample_rate * 1000
            if self._silence_frames * frame_duration_ms > self.silence_duration_ms:
                self._is_speaking = False
                self._silence_frames = 0
                if self._on_speech_end:
                    await self._on_speech_end()
                return UserStoppedSpeakingFrame()
        
        return frame


class STTService(BaseService):
    """语音转文字服务 - 集成真实的 STT 服务提供商"""
    
    def __init__(self, language: str = "zh-CN"):
        self.language = language
        self._audio_buffer = b''
        self._on_transcription: Optional[Callable[[str], Awaitable[None]]] = None
        
        # 从配置创建 STT 提供商
        self._provider: Optional[BaseSTTProvider] = None
        try:
            self._provider = ServiceFactory.create_stt_provider(
                config.stt.provider,
                api_key=config.stt.deepgram_api_key,
                model=config.stt.deepgram_model if config.stt.provider == "deepgram" else config.stt.whisper_model,
                language=config.stt.deepgram_language
            )
            logger.info(f"STT 服务初始化完成: {config.stt.provider}")
        except Exception as e:
            logger.warning(f"STT 服务初始化失败，将使用模拟模式: {e}")
            self._provider = None
    
    def on_transcription(self, callback: Callable[[str], Awaitable[None]]):
        """设置转录回调"""
        self._on_transcription = callback
        return self
    
    async def process(self, frame: Frame) -> Optional[Frame]:
        """处理音频帧，进行语音识别"""
        if isinstance(frame, UserStoppedSpeakingFrame):
            # 用户停止说话，处理累积的音频
            if self._audio_buffer:
                transcription = ""
                
                if self._provider:
                    # 使用真实的 STT 服务
                    try:
                        transcription = await self._provider.transcribe(self._audio_buffer)
                    except Exception as e:
                        logger.error(f"STT 转录失败: {e}")
                        transcription = "[转录失败]"
                else:
                    # 回退到模拟模式
                    transcription = "[模拟转录] 你好，请问有什么可以帮助你的？"
                
                if transcription:
                    if self._on_transcription:
                        await self._on_transcription(transcription)
                    
                    self._audio_buffer = b''
                    return TranscriptionFrame(text=transcription)
                
                self._audio_buffer = b''
        
        elif isinstance(frame, InputAudioFrame):
            self._audio_buffer += frame.audio
        
        return frame


class LLMService(BaseService):
    """大语言模型服务 - 集成真实的 LLM 服务提供商"""
    
    def __init__(self, model: str = "gpt-4o", 
                 instructions: str = "",
                 temperature: float = 0.7):
        self.model = model
        self.instructions = instructions or config.llm.system_prompt
        self.temperature = temperature
        self._on_response_start: Optional[Callable[[], Awaitable[None]]] = None
        self._on_response_chunk: Optional[Callable[[str], Awaitable[None]]] = None
        self._on_response_end: Optional[Callable[[str], Awaitable[None]]] = None
        
        # 从配置创建 LLM 提供商
        self._provider: Optional[BaseLLMProvider] = None
        try:
            if config.llm.provider == "openai":
                self._provider = ServiceFactory.create_llm_provider(
                    "openai",
                    api_key=config.llm.openai_api_key,
                    model=config.llm.openai_model,
                    base_url=config.llm.openai_base_url,
                    temperature=config.llm.temperature,
                    max_tokens=config.llm.max_tokens
                )
            elif config.llm.provider == "ollama":
                self._provider = ServiceFactory.create_llm_provider(
                    "ollama",
                    base_url=config.llm.ollama_base_url,
                    model=config.llm.ollama_model,
                    temperature=config.llm.temperature
                )
            elif config.llm.provider == "siliconflow":
                self._provider = ServiceFactory.create_llm_provider(
                    "siliconflow",
                    api_key=config.llm.siliconflow_api_key,
                    model=config.llm.siliconflow_model,
                    base_url=config.llm.siliconflow_base_url,
                    temperature=config.llm.temperature,
                    max_tokens=config.llm.max_tokens
                )
            logger.info(f"LLM 服务初始化完成: {config.llm.provider}")
        except Exception as e:
            logger.warning(f"LLM 服务初始化失败，将使用模拟模式: {e}")
            self._provider = None
    
    def on_response_start(self, callback: Callable[[], Awaitable[None]]):
        self._on_response_start = callback
        return self
    
    def on_response_chunk(self, callback: Callable[[str], Awaitable[None]]):
        self._on_response_chunk = callback
        return self
    
    def on_response_end(self, callback: Callable[[str], Awaitable[None]]):
        self._on_response_end = callback
        return self
    
    def update_instructions(self, instructions: str):
        """更新系统提示词"""
        self.instructions = instructions
    
    async def process(self, frame: Frame) -> Optional[Frame]:
        """处理转录文本，生成 LLM 响应"""
        if not isinstance(frame, TranscriptionFrame):
            return frame
        
        if self._on_response_start:
            await self._on_response_start()
        
        full_response = ""
        
        if self._provider:
            # 使用真实的 LLM 服务
            try:
                async def on_chunk(text: str):
                    nonlocal full_response
                    full_response += text
                    if self._on_response_chunk:
                        await self._on_response_chunk(text)
                
                full_response = await self._provider.generate_stream(
                    prompt=frame.text,
                    system_prompt=self.instructions,
                    on_chunk=on_chunk
                )
            except Exception as e:
                logger.error(f"LLM 生成失败: {e}")
                full_response = f"抱歉，我遇到了一些问题: {str(e)}"
                if self._on_response_chunk:
                    await self._on_response_chunk(full_response)
        else:
            # 回退到模拟模式
            full_response = f"你好！我收到了你的消息：「{frame.text}」。作为你的AI助手，我很乐意帮助你。请问有什么具体的问题吗？"
            
            # 模拟流式输出
            chunks = [full_response[i:i+10] for i in range(0, len(full_response), 10)]
            
            for chunk in chunks:
                if self._on_response_chunk:
                    await self._on_response_chunk(chunk)
                await asyncio.sleep(0.05)
        
        if self._on_response_end:
            await self._on_response_end(full_response)
        
        return LLMResponseFrame(text=full_response)


class TTSService(BaseService):
    """文字转语音服务 - 集成真实的 TTS 服务提供商"""
    
    def __init__(self, voice: str = "alloy", sample_rate: int = 16000):
        self.voice = voice
        self.sample_rate = sample_rate
        self._on_audio_chunk: Optional[Callable[[bytes], Awaitable[None]]] = None
        self._on_audio_end: Optional[Callable[[], Awaitable[None]]] = None
        
        # 从配置创建 TTS 提供商
        self._provider: Optional[BaseTTSProvider] = None
        try:
            if config.tts.provider == "elevenlabs":
                self._provider = ServiceFactory.create_tts_provider(
                    "elevenlabs",
                    api_key=config.tts.elevenlabs_api_key,
                    voice_id=config.tts.elevenlabs_voice_id,
                    model=config.tts.elevenlabs_model
                )
            elif config.tts.provider == "edge_tts":
                self._provider = ServiceFactory.create_tts_provider(
                    "edge_tts",
                    voice=config.tts.edge_tts_voice
                )
            elif config.tts.provider == "openai_tts":
                self._provider = ServiceFactory.create_tts_provider(
                    "openai_tts",
                    api_key=config.llm.openai_api_key,
                    voice=config.tts.openai_tts_voice,
                    model=config.tts.openai_tts_model,
                    base_url=config.llm.openai_base_url
                )
            logger.info(f"TTS 服务初始化完成: {config.tts.provider}")
        except Exception as e:
            logger.warning(f"TTS 服务初始化失败，将使用模拟模式: {e}")
            self._provider = None
    
    def on_audio_chunk(self, callback: Callable[[bytes], Awaitable[None]]):
        self._on_audio_chunk = callback
        return self
    
    def on_audio_end(self, callback: Callable[[], Awaitable[None]]):
        self._on_audio_end = callback
        return self
    
    async def process(self, frame: Frame) -> Optional[Frame]:
        """处理 LLM 响应，生成语音"""
        if not isinstance(frame, LLMResponseFrame):
            return frame
        
        full_audio = b""
        
        if self._provider:
            # 使用真实的 TTS 服务
            try:
                async def on_chunk(audio_data: bytes):
                    nonlocal full_audio
                    full_audio += audio_data
                    if self._on_audio_chunk:
                        await self._on_audio_chunk(audio_data)
                
                full_audio = await self._provider.synthesize_stream(
                    text=frame.text,
                    on_audio_chunk=on_chunk
                )
            except Exception as e:
                logger.error(f"TTS 合成失败: {e}")
                # 生成静音作为回退
                full_audio = b'\x00' * (self.sample_rate * 2)  # 1 秒静音
                if self._on_audio_chunk:
                    await self._on_audio_chunk(full_audio)
        else:
            # 回退到模拟模式 - 生成正弦波
            import numpy as np
            
            duration_ms = len(frame.text) * 80
            num_samples = int(self.sample_rate * duration_ms / 1000)
            
            t = np.linspace(0, duration_ms / 1000, num_samples, dtype=np.float32)
            audio_float = np.sin(2 * np.pi * 440 * t) * 0.3
            audio_int16 = (audio_float * 32767).astype(np.int16)
            full_audio = audio_int16.tobytes()
            
            # 分块发送
            chunk_size = int(self.sample_rate * 0.1) * 2
            
            for i in range(0, len(full_audio), chunk_size):
                chunk = full_audio[i:i+chunk_size]
                if self._on_audio_chunk:
                    await self._on_audio_chunk(chunk)
                await asyncio.sleep(0.05)
        
        if self._on_audio_end:
            await self._on_audio_end()
        
        return TTSAudioFrame(audio=full_audio, sample_rate=self.sample_rate)


# ==================== 管道管理器 ====================

class PipelineManager:
    """
    管道管理器
    协调 VAD -> STT -> LLM -> TTS 的处理流程
    """
    
    def __init__(self):
        self.vad: Optional[VADService] = None
        self.stt: Optional[STTService] = None
        self.llm: Optional[LLMService] = None
        self.tts: Optional[TTSService] = None
        
        # 回调函数
        self._on_user_speech_start: Optional[Callable[[], Awaitable[None]]] = None
        self._on_user_speech_end: Optional[Callable[[], Awaitable[None]]] = None
        self._on_transcription: Optional[Callable[[str], Awaitable[None]]] = None
        self._on_response_start: Optional[Callable[[], Awaitable[None]]] = None
        self._on_response_text: Optional[Callable[[str], Awaitable[None]]] = None
        self._on_response_audio: Optional[Callable[[bytes], Awaitable[None]]] = None
        self._on_response_end: Optional[Callable[[str], Awaitable[None]]] = None
        
        self._running = False
        self._audio_queue: asyncio.Queue = asyncio.Queue()
    
    def configure(self, 
                  vad_threshold: float = 0.5,
                  vad_silence_ms: int = 500,
                  llm_model: str = "gpt-4o",
                  llm_instructions: str = "",
                  tts_voice: str = "alloy") -> 'PipelineManager':
        """配置管道参数"""
        
        # 创建服务
        self.vad = VADService(
            threshold=vad_threshold,
            silence_duration_ms=vad_silence_ms
        )
        self.stt = STTService(language="zh-CN")
        self.llm = LLMService(
            model=llm_model,
            instructions=llm_instructions
        )
        self.tts = TTSService(voice=tts_voice)
        
        # 连接 VAD 回调
        self.vad.on_speech_start(self._handle_speech_start)
        self.vad.on_speech_end(self._handle_speech_end)
        
        # 连接 STT 回调
        self.stt.on_transcription(self._handle_transcription)
        
        # 连接 LLM 回调
        self.llm.on_response_start(self._handle_response_start)
        self.llm.on_response_chunk(self._handle_response_text)
        self.llm.on_response_end(self._handle_response_end)
        
        # 连接 TTS 回调
        self.tts.on_audio_chunk(self._handle_audio_chunk)
        
        logger.info("管道已配置")
        return self
    
    # ==================== 回调注册 ====================
    
    def on_user_speech_start(self, callback: Callable[[], Awaitable[None]]):
        self._on_user_speech_start = callback
        return self
    
    def on_user_speech_end(self, callback: Callable[[], Awaitable[None]]):
        self._on_user_speech_end = callback
        return self
    
    def on_transcription(self, callback: Callable[[str], Awaitable[None]]):
        self._on_transcription = callback
        return self
    
    def on_response_start(self, callback: Callable[[], Awaitable[None]]):
        self._on_response_start = callback
        return self
    
    def on_response_text(self, callback: Callable[[str], Awaitable[None]]):
        self._on_response_text = callback
        return self
    
    def on_response_audio(self, callback: Callable[[bytes], Awaitable[None]]):
        self._on_response_audio = callback
        return self
    
    def on_response_end(self, callback: Callable[[str], Awaitable[None]]):
        self._on_response_end = callback
        return self
    
    # ==================== 内部回调处理 ====================
    
    async def _handle_speech_start(self):
        if self._on_user_speech_start:
            await self._on_user_speech_start()
    
    async def _handle_speech_end(self):
        if self._on_user_speech_end:
            await self._on_user_speech_end()
    
    async def _handle_transcription(self, text: str):
        if self._on_transcription:
            await self._on_transcription(text)
    
    async def _handle_response_start(self):
        if self._on_response_start:
            await self._on_response_start()
    
    async def _handle_response_text(self, text: str):
        if self._on_response_text:
            await self._on_response_text(text)
    
    async def _handle_audio_chunk(self, audio: bytes):
        if self._on_response_audio:
            await self._on_response_audio(audio)
    
    async def _handle_response_end(self, full_text: str):
        if self._on_response_end:
            await self._on_response_end(full_text)
    
    # ==================== 公共接口 ====================
    
    async def start(self):
        """启动管道"""
        self._running = True
        logger.info("管道已启动")
    
    async def stop(self):
        """停止管道"""
        self._running = False
        logger.info("管道已停止")
    
    async def push_audio(self, audio_bytes: bytes):
        """推送音频数据到管道（通过内置 VAD 自动检测）"""
        if not self._running:
            return
        
        frame = InputAudioFrame(audio=audio_bytes)
        
        # VAD 自动处理语音活动检测
        if self.vad:
            result = await self.vad.process(frame)
            
            # VAD 检测到语音结束，自动触发完整处理流程
            if isinstance(result, UserStoppedSpeakingFrame):
                if self.stt:
                    stt_result = await self.stt.process(result)
                    
                    # STT 完成后自动触发 LLM
                    if isinstance(stt_result, TranscriptionFrame) and self.llm:
                        llm_result = await self.llm.process(stt_result)
                        
                        # LLM 完成后自动触发 TTS
                        if isinstance(llm_result, LLMResponseFrame) and self.tts:
                            await self.tts.process(llm_result)
            
            # 继续收集音频用于 STT
            elif isinstance(result, (InputAudioFrame, UserStartedSpeakingFrame)):
                if self.stt:
                    await self.stt.process(frame)
        else:
            # 没有 VAD 时直接处理（不应该发生，因为 VAD 是内置的）
            logger.warning("VAD 未启用，这不应该发生在内置 VAD 模式下")
    
    def update_instructions(self, instructions: str):
        """更新 LLM 系统提示词"""
        if self.llm:
            self.llm.update_instructions(instructions)
            logger.info("LLM 指令已更新")
    
    async def force_response(self):
        """强制生成响应（用于手动 VAD 模式）"""
        if self.stt:
            # 触发 STT 处理
            stt_result = await self.stt.process(UserStoppedSpeakingFrame())
            
            if isinstance(stt_result, TranscriptionFrame) and self.llm:
                llm_result = await self.llm.process(stt_result)
                
                if isinstance(llm_result, LLMResponseFrame) and self.tts:
                    await self.tts.process(llm_result)
    
    async def cancel_response(self):
        """取消当前响应"""
        # 清空待处理队列
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        logger.info("响应已取消")
