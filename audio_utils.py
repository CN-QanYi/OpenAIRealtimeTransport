"""
音频处理工具 - 提供音频重采样、缓冲管理和异步播放功能

主要功能:
- 音频重采样 (24kHz <-> 16kHz)
- 异步音频播放器 (用于客户端播放 AI 响应)
- 音频缓冲区管理
"""
import asyncio
import threading
import queue
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ==================== 音频常量 ====================

# OpenAI Realtime API 使用的采样率
SAMPLE_RATE = 24000

# 声道数（单声道）
CHANNELS = 1

# 采样宽度（16-bit PCM = 2 bytes）
SAMPLE_WIDTH = 2

# 内部处理采样率（大多数 STT 模型使用 16kHz）
INTERNAL_SAMPLE_RATE = 16000


# ==================== 音频重采样工具 ====================

def resample_audio(audio_bytes: bytes, from_rate: int, to_rate: int) -> bytes:
    """
    音频重采样
    
    Args:
        audio_bytes: 原始 PCM 音频数据 (16-bit signed)
        from_rate: 原始采样率
        to_rate: 目标采样率
    
    Returns:
        重采样后的 PCM 音频数据
    """
    if from_rate == to_rate:
        return audio_bytes
    
    try:
        import numpy as np
        
        # 将字节转换为 numpy 数组
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # 计算重采样比例
        ratio = to_rate / from_rate
        new_length = int(len(audio_array) * ratio)
        
        # 使用 scipy 进行高质量重采样（如果可用）
        try:
            from scipy import signal
            resampled = signal.resample(audio_array, new_length)
            resampled = np.clip(resampled, -32768, 32767).astype(np.int16)
        except ImportError:
            # 回退到简单的线性插值
            indices = np.linspace(0, len(audio_array) - 1, new_length)
            resampled = np.interp(indices, np.arange(len(audio_array)), audio_array)
            resampled = np.clip(resampled, -32768, 32767).astype(np.int16)
        
        return resampled.tobytes()
        
    except ImportError:
        logger.warning("numpy 未安装，无法进行重采样，返回原始音频")
        return audio_bytes


def resample_to_16k(audio_bytes: bytes, from_rate: int = SAMPLE_RATE) -> bytes:
    """将音频重采样到 16kHz（用于 STT）"""
    return resample_audio(audio_bytes, from_rate, INTERNAL_SAMPLE_RATE)


def resample_to_24k(audio_bytes: bytes, from_rate: int = INTERNAL_SAMPLE_RATE) -> bytes:
    """将音频重采样到 24kHz（用于 OpenAI 协议）"""
    return resample_audio(audio_bytes, from_rate, SAMPLE_RATE)


# ==================== 音频转换器 ====================

class AudioConverter:
    """
    音频格式转换器
    
    负责在客户端格式 (24kHz) 和内部处理格式 (16kHz) 之间转换
    """
    
    def __init__(self):
        """初始化音频转换器"""
        self.client_sample_rate = SAMPLE_RATE  # 24kHz
        self.internal_sample_rate = INTERNAL_SAMPLE_RATE  # 16kHz
    
    def client_to_internal(self, audio_bytes: bytes) -> bytes:
        """
        将客户端音频 (24kHz) 转换为内部处理格式 (16kHz)
        
        Args:
            audio_bytes: 客户端 PCM 音频数据 (24kHz)
        
        Returns:
            内部处理 PCM 音频数据 (16kHz)
        """
        return resample_to_16k(audio_bytes, from_rate=self.client_sample_rate)
    
    def internal_to_client(self, audio_bytes: bytes) -> bytes:
        """
        将内部处理音频 (16kHz) 转换为客户端格式 (24kHz)
        
        Args:
            audio_bytes: 内部处理 PCM 音频数据 (16kHz)
        
        Returns:
            客户端 PCM 音频数据 (24kHz)
        """
        return resample_to_24k(audio_bytes, from_rate=self.internal_sample_rate)


# ==================== 异步音频播放器 ====================

class AudioPlayerAsync:
    """
    异步音频播放器
    
    用于客户端播放从服务器接收的音频响应。
    使用后台线程进行实际播放，避免阻塞事件循环。
    """
    
    def __init__(self, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS):
        """
        初始化音频播放器
        
        Args:
            sample_rate: 采样率
            channels: 声道数
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self._queue = queue.Queue()  # 避免 Python 3.8 下的 TypeError
        self._frame_count = 0
        self._frame_lock = threading.Lock()  # 保护 _frame_count 的锁
        self._playing = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # 启动播放线程
        self._start_playback_thread()
    
    def _start_playback_thread(self) -> None:
        """启动后台播放线程"""
        if self._thread is not None and self._thread.is_alive():
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()
    
    def _playback_loop(self) -> None:
        """后台播放循环"""
        try:
            import sounddevice as sd
        except ImportError:
            logger.error("sounddevice 未安装，无法播放音频")
            return
        
        stream: Optional[sd.OutputStream] = None
        
        try:
            stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='int16',
            )
            stream.start()
            
            while not self._stop_event.is_set():
                try:
                    # 等待音频数据，带超时以检查停止事件
                    data = self._queue.get(timeout=0.1)
                    if data is None:
                        # None 表示停止信号
                        break
                    
                    # 写入音频流
                    import numpy as np
                    audio_array = np.frombuffer(data, dtype=np.int16)
                    stream.write(audio_array)
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"播放音频时出错: {e}")
                    
        except Exception as e:
            logger.error(f"初始化音频流时出错: {e}")
        finally:
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
    
    def add_data(self, audio_bytes: bytes) -> None:
        """
        添加音频数据到播放队列
        
        Args:
            audio_bytes: PCM 音频数据
        """
        self._queue.put(audio_bytes)
        with self._frame_lock:
            self._frame_count += 1
    
    def reset_frame_count(self) -> None:
        """重置帧计数器（用于新的音频响应）"""
        with self._frame_lock:
            self._frame_count = 0
        # 清空队列中的残留数据
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
    
    def stop(self) -> None:
        """停止播放"""
        self._stop_event.set()
        self._queue.put(None)  # 发送停止信号
        if self._thread is not None:
            self._thread.join(timeout=1.0)
    
    @property
    def frame_count(self) -> int:
        """获取已入队的音频块数量（非"已播放"）"""
        with self._frame_lock:
            return self._frame_count
    
    def __del__(self) -> None:
        """析构时停止播放"""
        try:
            if self._thread is not None and threading.current_thread() is self._thread:
                self._stop_event.set()
                return
            self.stop()
        except Exception:
            pass

# ==================== 音频缓冲区 ====================

class AudioBuffer:
    """
    音频缓冲区管理器
    
    用于累积音频数据，支持分块处理。
    """
    
    def __init__(self, sample_rate: int = SAMPLE_RATE, chunk_duration_ms: int = 100):
        """
        初始化音频缓冲区
        
        Args:
            sample_rate: 采样率
            chunk_duration_ms: 每个分块的持续时间（毫秒）
        """
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self._buffer = bytearray()
        self._lock = threading.Lock()
    
    @property
    def chunk_size(self) -> int:
        """每个分块的字节数"""
        samples_per_chunk = int(self.sample_rate * self.chunk_duration_ms / 1000)
        return samples_per_chunk * SAMPLE_WIDTH * CHANNELS
    
    def append(self, audio_bytes: bytes) -> None:
        """添加音频数据到缓冲区"""
        with self._lock:
            self._buffer.extend(audio_bytes)
    
    def get_chunk(self) -> Optional[bytes]:
        """
        获取一个完整的音频分块
        
        Returns:
            完整的音频分块，如果数据不足则返回 None
        """
        with self._lock:
            if len(self._buffer) >= self.chunk_size:
                chunk = bytes(self._buffer[:self.chunk_size])
                del self._buffer[:self.chunk_size]
                return chunk
            return None
    
    def get_all(self) -> bytes:
        """获取所有缓冲的音频数据并清空缓冲区"""
        with self._lock:
            data = bytes(self._buffer)
            self._buffer.clear()
            return data
    
    def clear(self) -> None:
        """清空缓冲区"""
        with self._lock:
            self._buffer.clear()
    
    def __len__(self) -> int:
        """返回缓冲区中的字节数"""
        with self._lock:
            return len(self._buffer)


# ==================== 音频时长计算 ====================

def calculate_audio_duration_ms(audio_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> float:
    """
    计算音频数据的时长 (毫秒)
    
    Args:
        audio_bytes: PCM 音频数据
        sample_rate: 采样率 (默认 24kHz)
    
    Returns:
        音频时长 (毫秒)
    """
    num_samples = len(audio_bytes) // (SAMPLE_WIDTH * CHANNELS)
    duration_ms = (num_samples / sample_rate) * 1000
    return duration_ms
