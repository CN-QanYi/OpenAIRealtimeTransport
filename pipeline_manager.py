"""
管道管理器 - 管理 Pipecat 管道的创建和配置
提供一个简化的接口来构建语音处理管道
"""
import asyncio
import logging
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

logger = logging.getLogger(__name__)


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
    """语音活动检测服务 - 基于能量的语音活动检测"""
    
    def __init__(self, threshold: float = 0.5, 
                 silence_duration_ms: int = 500,
                 prefix_padding_ms: int = 300):
        self.threshold = threshold  # 0.0-1.0，越高越不敏感
        self.silence_duration_ms = silence_duration_ms
        self.prefix_padding_ms = prefix_padding_ms
        self._is_speaking = False
        self._accepting_audio = False  # 是否接收音频进行处理
        self._silence_frames = 0
        self._speech_frames = 0  # 连续语音帧计数
        self._min_speech_frames = 3  # 最少连续语音帧数（避免误触发）
        self._on_speech_start: Optional[Callable[[], Awaitable[None]]] = None
        self._on_speech_end: Optional[Callable[[], Awaitable[None]]] = None
        
        # 动态能量阈值（自适应背景噪音）
        self._energy_threshold = 500.0  # 初始阈值
        self._background_energy = 0.0
        self._frame_count = 0
    
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
        
        # 简单的能量检测 VAD（实际应使用 Silero VAD）
        import numpy as np
        audio_array = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32)
        
        if len(audio_array) == 0:
            return frame
        
        # 计算 RMS 能量
        rms = np.sqrt(np.mean(audio_array ** 2))
        
        # 自适应背景噪音估计（仅在非说话状态时更新）
        if not self._is_speaking:
            self._frame_count += 1
            if self._frame_count <= 30:  # 前30帧用于初始化背景噪音
                self._background_energy = (self._background_energy * (self._frame_count - 1) + rms) / self._frame_count
            else:
                # 使用指数移动平均更新背景噪音（慢速更新）
                self._background_energy = 0.95 * self._background_energy + 0.05 * rms
        
        # 动态调整能量阈值（背景噪音 + 用户设置的灵敏度）
        # threshold: 0.5 表示中等灵敏度，阈值 = 背景噪音 * (2 + threshold * 4)
        # threshold: 0.0 -> 2倍背景噪音, 0.5 -> 4倍, 1.0 -> 6倍
        self._energy_threshold = max(300, self._background_energy * (2 + self.threshold * 4))
        
        # 判断是否为语音
        is_speech = rms > self._energy_threshold
        
        if is_speech:
            # 检测到语音能量
            self._speech_frames += 1
            self._silence_frames = 0
            
            if not self._is_speaking:
                # 需要连续几帧都是语音才确认开始说话（避免误触发）
                if self._speech_frames >= self._min_speech_frames:
                    self._is_speaking = True
                    self._accepting_audio = True
                    if self._on_speech_start:
                        await self._on_speech_start()
                    logger.info(f"✅ 用户开始说话 (RMS: {rms:.2f}, 阈值: {self._energy_threshold:.2f}, 背景: {self._background_energy:.2f})")
        else:
            # 检测到静音
            self._speech_frames = 0
            
            if self._is_speaking:
                # 当前处于说话状态，累计静音帧数
                self._silence_frames += 1
                # 根据静音时长判断是否结束
                frame_duration_ms = len(audio_array) / frame.sample_rate * 1000
                accumulated_silence_ms = self._silence_frames * frame_duration_ms
                
                if accumulated_silence_ms > self.silence_duration_ms:
                    # 静音时间超过阈值，判定为说话结束
                    self._is_speaking = False
                    self._accepting_audio = False  # 停止接收音频
                    self._silence_frames = 0
                    self._speech_frames = 0
                    if self._on_speech_end:
                        await self._on_speech_end()
                    logger.info(f"🛑 用户停止说话 (静音时长: {accumulated_silence_ms:.0f}ms, RMS: {rms:.2f})")
                    # 返回停止帧，通知下游停止处理
                    return UserStoppedSpeakingFrame()
        
        # 只在接收音频状态下返回音频帧，否则丢弃
        if self._accepting_audio:
            return frame
        else:
            # 不在说话状态，丢弃音频帧
            return None
    
    def reset(self):
        """重置VAD状态"""
        self._is_speaking = False
        self._accepting_audio = False
        self._silence_frames = 0
        self._speech_frames = 0
        logger.debug("VAD 状态已重置")


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
        # 忽略 None 帧
        if frame is None:
            return None
            
        if isinstance(frame, UserStoppedSpeakingFrame):
            # 用户停止说话，处理累积的音频
            if self._audio_buffer and len(self._audio_buffer) > 1600:  # 至少 0.1 秒音频
                transcription = ""
                
                if self._provider:
                    # 使用真实的 STT 服务
                    try:
                        logger.info(f"开始转录 ({len(self._audio_buffer)} 字节音频)")
                        transcription = await self._provider.transcribe(self._audio_buffer)
                        if transcription:
                            logger.info(f"✅ 转录完成: {transcription}")
                    except Exception as e:
                        logger.error(f"❌ STT 转录失败: {e}")
                        transcription = ""
                else:
                    # 回退到模拟模式
                    logger.warning("使用模拟转录模式")
                    transcription = "[模拟转录] 你好，请问有什么可以帮助你的？"
                
                # 清空缓冲区
                self._audio_buffer = b''
                
                if transcription:
                    if self._on_transcription:
                        await self._on_transcription(transcription)
                    return TranscriptionFrame(text=transcription)
            else:
                # 音频太短，丢弃
                if self._audio_buffer:
                    logger.debug(f"音频太短，丢弃 ({len(self._audio_buffer)} 字节)")
                self._audio_buffer = b''
        
        elif isinstance(frame, InputAudioFrame):
            # 累积音频数据
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
            
            # VAD 返回 None 表示丢弃该帧（非说话状态）
            if result is None:
                return
            
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
                
                # 重置 VAD 状态，准备下一轮检测
                self.vad.reset()
            
            # 继续收集音频用于 STT
            elif isinstance(result, InputAudioFrame):
                if self.stt:
                    await self.stt.process(result)
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
