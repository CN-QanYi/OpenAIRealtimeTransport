"""
协议定义 - OpenAI Realtime API 事件类型和数据结构
完整复刻 OpenAI Realtime API 的协议格式
"""
import uuid
import time
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Union


def generate_id(prefix: str = "evt") -> str:
    """生成唯一 ID"""
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def generate_timestamp() -> int:
    """生成时间戳（毫秒）"""
    return int(time.time() * 1000)


# ==================== 客户端 -> 服务器 事件类型 ====================

class ClientEventType(str, Enum):
    """客户端发送的事件类型"""
    # 会话相关
    SESSION_UPDATE = "session.update"
    
    # 音频缓冲区相关
    INPUT_AUDIO_BUFFER_APPEND = "input_audio_buffer.append"
    INPUT_AUDIO_BUFFER_COMMIT = "input_audio_buffer.commit"  # 已废弃：内置 Server VAD 自动检测，不需要手动提交
    INPUT_AUDIO_BUFFER_CLEAR = "input_audio_buffer.clear"
    
    # 对话相关
    CONVERSATION_ITEM_CREATE = "conversation.item.create"
    CONVERSATION_ITEM_TRUNCATE = "conversation.item.truncate"
    CONVERSATION_ITEM_DELETE = "conversation.item.delete"
    
    # 响应相关
    RESPONSE_CREATE = "response.create"
    RESPONSE_CANCEL = "response.cancel"


# ==================== 服务器 -> 客户端 事件类型 ====================

class ServerEventType(str, Enum):
    """服务器发送的事件类型"""
    # 错误
    ERROR = "error"
    
    # 会话相关
    SESSION_CREATED = "session.created"
    SESSION_UPDATED = "session.updated"
    
    # 音频缓冲区相关
    INPUT_AUDIO_BUFFER_COMMITTED = "input_audio_buffer.committed"
    INPUT_AUDIO_BUFFER_CLEARED = "input_audio_buffer.cleared"
    INPUT_AUDIO_BUFFER_SPEECH_STARTED = "input_audio_buffer.speech_started"
    INPUT_AUDIO_BUFFER_SPEECH_STOPPED = "input_audio_buffer.speech_stopped"
    
    # 对话相关
    CONVERSATION_CREATED = "conversation.created"
    CONVERSATION_ITEM_CREATED = "conversation.item.created"
    CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED = "conversation.item.input_audio_transcription.completed"
    CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_FAILED = "conversation.item.input_audio_transcription.failed"
    CONVERSATION_ITEM_TRUNCATED = "conversation.item.truncated"
    CONVERSATION_ITEM_DELETED = "conversation.item.deleted"
    
    # 响应相关
    RESPONSE_CREATED = "response.created"
    RESPONSE_DONE = "response.done"
    RESPONSE_OUTPUT_ITEM_ADDED = "response.output_item.added"
    RESPONSE_OUTPUT_ITEM_DONE = "response.output_item.done"
    RESPONSE_CONTENT_PART_ADDED = "response.content_part.added"
    RESPONSE_CONTENT_PART_DONE = "response.content_part.done"
    
    # 音频输出相关
    RESPONSE_AUDIO_DELTA = "response.audio.delta"
    RESPONSE_AUDIO_DONE = "response.audio.done"
    RESPONSE_AUDIO_TRANSCRIPT_DELTA = "response.audio_transcript.delta"
    RESPONSE_AUDIO_TRANSCRIPT_DONE = "response.audio_transcript.done"
    
    # 文本输出相关
    RESPONSE_TEXT_DELTA = "response.text.delta"
    RESPONSE_TEXT_DONE = "response.text.done"
    
    # 函数调用相关
    RESPONSE_FUNCTION_CALL_ARGUMENTS_DELTA = "response.function_call_arguments.delta"
    RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE = "response.function_call_arguments.done"
    
    # 速率限制
    RATE_LIMITS_UPDATED = "rate_limits.updated"


# ==================== 数据结构定义 ====================

@dataclass
class AudioFormat:
    """音频格式配置"""
    type: str = "pcm16"  # pcm16, g711_ulaw, g711_alaw
    sample_rate: int = 24000
    channels: int = 1


@dataclass
class TurnDetection:
    """语音活动检测配置"""
    type: str = "server_vad"  # server_vad 或 null
    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500
    create_response: bool = True


@dataclass
class InputAudioTranscription:
    """输入音频转录配置"""
    model: str = "whisper-1"


@dataclass
class SessionConfig:
    """会话配置"""
    id: Optional[str] = None
    object: str = "realtime.session"
    model: str = "gpt-4o-realtime-preview"
    modalities: List[str] = field(default_factory=lambda: ["text", "audio"])
    instructions: str = ""
    voice: str = "alloy"  # alloy, echo, fable, onyx, nova, shimmer
    input_audio_format: str = "pcm16"
    output_audio_format: str = "pcm16"
    input_audio_transcription: Optional[InputAudioTranscription] = None
    turn_detection: Optional[TurnDetection] = field(default_factory=TurnDetection)
    tools: List[Dict[str, Any]] = field(default_factory=list)
    tool_choice: str = "auto"
    temperature: float = 0.8
    max_response_output_tokens: Union[int, str] = "inf"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，排除 None 值"""
        result = {}
        for k, v in asdict(self).items():
            if v is not None:
                if hasattr(v, 'to_dict'):
                    result[k] = v.to_dict()
                else:
                    result[k] = v
        return result


@dataclass
class ConversationItem:
    """对话项"""
    id: str = field(default_factory=lambda: generate_id("item"))
    object: str = "realtime.item"
    type: str = "message"  # message, function_call, function_call_output
    status: str = "completed"  # in_progress, completed, incomplete
    role: str = "user"  # user, assistant, system
    content: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Response:
    """响应对象"""
    id: str = field(default_factory=lambda: generate_id("resp"))
    object: str = "realtime.response"
    status: str = "in_progress"  # in_progress, completed, cancelled, failed, incomplete
    status_details: Optional[Dict[str, Any]] = None
    output: List[ConversationItem] = field(default_factory=list)
    usage: Optional[Dict[str, int]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['output'] = [item.to_dict() if hasattr(item, 'to_dict') else item for item in self.output]
        return result


# ==================== 服务器事件构建器 ====================

class ServerEventBuilder:
    """服务器事件构建器 - 用于生成符合 OpenAI 格式的事件"""
    
    @staticmethod
    def error(message: str, error_type: str = "invalid_request_error", code: Optional[str] = None, 
              event_id: Optional[str] = None) -> Dict[str, Any]:
        """构建错误事件"""
        return {
            "event_id": event_id or generate_id("evt"),
            "type": ServerEventType.ERROR.value,
            "error": {
                "type": error_type,
                "code": code,
                "message": message,
                "param": None,
            }
        }
    
    @staticmethod
    def session_created(session: SessionConfig, event_id: Optional[str] = None) -> Dict[str, Any]:
        """构建会话创建事件"""
        return {
            "event_id": event_id or generate_id("evt"),
            "type": ServerEventType.SESSION_CREATED.value,
            "session": session.to_dict()
        }
    
    @staticmethod
    def session_updated(session: SessionConfig, event_id: Optional[str] = None) -> Dict[str, Any]:
        """构建会话更新事件"""
        return {
            "event_id": event_id or generate_id("evt"),
            "type": ServerEventType.SESSION_UPDATED.value,
            "session": session.to_dict()
        }
    
    @staticmethod
    def input_audio_buffer_speech_started(audio_start_ms: int = 0, 
                                          item_id: Optional[str] = None,
                                          event_id: Optional[str] = None) -> Dict[str, Any]:
        """构建语音开始事件 - 用于触发客户端打断"""
        return {
            "event_id": event_id or generate_id("evt"),
            "type": ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STARTED.value,
            "audio_start_ms": audio_start_ms,
            "item_id": item_id or generate_id("item"),
        }
    
    @staticmethod
    def input_audio_buffer_speech_stopped(audio_end_ms: int = 0,
                                          item_id: Optional[str] = None,
                                          event_id: Optional[str] = None) -> Dict[str, Any]:
        """构建语音停止事件"""
        return {
            "event_id": event_id or generate_id("evt"),
            "type": ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STOPPED.value,
            "audio_end_ms": audio_end_ms,
            "item_id": item_id or generate_id("item"),
        }
    
    @staticmethod
    def input_audio_buffer_committed(previous_item_id: Optional[str] = None,
                                     item_id: Optional[str] = None,
                                     event_id: Optional[str] = None) -> Dict[str, Any]:
        """构建音频缓冲区提交事件"""
        return {
            "event_id": event_id or generate_id("evt"),
            "type": ServerEventType.INPUT_AUDIO_BUFFER_COMMITTED.value,
            "previous_item_id": previous_item_id,
            "item_id": item_id or generate_id("item"),
        }
    
    @staticmethod
    def input_audio_buffer_cleared(event_id: Optional[str] = None) -> Dict[str, Any]:
        """构建音频缓冲区清空事件"""
        return {
            "event_id": event_id or generate_id("evt"),
            "type": ServerEventType.INPUT_AUDIO_BUFFER_CLEARED.value,
        }
    
    @staticmethod
    def conversation_created(conversation_id: Optional[str] = None, event_id: Optional[str] = None) -> Dict[str, Any]:
        """构建对话创建事件"""
        return {
            "event_id": event_id or generate_id("evt"),
            "type": ServerEventType.CONVERSATION_CREATED.value,
            "conversation": {
                "id": conversation_id or generate_id("conv"),
                "object": "realtime.conversation",
            }
        }
    
    @staticmethod
    def conversation_item_created(item: ConversationItem, 
                                  previous_item_id: Optional[str] = None,
                                  event_id: Optional[str] = None) -> Dict[str, Any]:
        """构建对话项创建事件"""
        return {
            "event_id": event_id or generate_id("evt"),
            "type": ServerEventType.CONVERSATION_ITEM_CREATED.value,
            "previous_item_id": previous_item_id,
            "item": item.to_dict() if hasattr(item, 'to_dict') else item,
        }
    
    @staticmethod
    def response_created(response: Response, event_id: Optional[str] = None) -> Dict[str, Any]:
        """构建响应创建事件"""
        return {
            "event_id": event_id or generate_id("evt"),
            "type": ServerEventType.RESPONSE_CREATED.value,
            "response": response.to_dict()
        }
    
    @staticmethod
    def response_done(response: Response, event_id: Optional[str] = None) -> Dict[str, Any]:
        """构建响应完成事件"""
        response.status = "completed"
        return {
            "event_id": event_id or generate_id("evt"),
            "type": ServerEventType.RESPONSE_DONE.value,
            "response": response.to_dict()
        }
    
    @staticmethod
    def response_output_item_added(response_id: str, item: ConversationItem,
                                   output_index: int = 0,
                                   event_id: Optional[str] = None) -> Dict[str, Any]:
        """构建响应输出项添加事件"""
        return {
            "event_id": event_id or generate_id("evt"),
            "type": ServerEventType.RESPONSE_OUTPUT_ITEM_ADDED.value,
            "response_id": response_id,
            "output_index": output_index,
            "item": item.to_dict() if hasattr(item, 'to_dict') else item,
        }
    
    @staticmethod
    def response_output_item_done(response_id: str, item: ConversationItem,
                                  output_index: int = 0,
                                  event_id: Optional[str] = None) -> Dict[str, Any]:
        """构建响应输出项完成事件"""
        item.status = "completed"
        return {
            "event_id": event_id or generate_id("evt"),
            "type": ServerEventType.RESPONSE_OUTPUT_ITEM_DONE.value,
            "response_id": response_id,
            "output_index": output_index,
            "item": item.to_dict() if hasattr(item, 'to_dict') else item,
        }
    
    @staticmethod
    def response_content_part_added(response_id: str, item_id: str,
                                    output_index: int = 0, content_index: int = 0,
                                    part_type: str = "audio",
                                    event_id: Optional[str] = None) -> Dict[str, Any]:
        """构建响应内容部分添加事件"""
        part = {"type": part_type}
        if part_type == "audio":
            part["transcript"] = ""
        elif part_type == "text":
            part["text"] = ""
        
        return {
            "event_id": event_id or generate_id("evt"),
            "type": ServerEventType.RESPONSE_CONTENT_PART_ADDED.value,
            "response_id": response_id,
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "part": part,
        }
    
    @staticmethod
    def response_content_part_done(response_id: str, item_id: str,
                                   output_index: int = 0, content_index: int = 0,
                                   part: Optional[Dict[str, Any]] = None,
                                   event_id: Optional[str] = None) -> Dict[str, Any]:
        """构建响应内容部分完成事件"""
        return {
            "event_id": event_id or generate_id("evt"),
            "type": ServerEventType.RESPONSE_CONTENT_PART_DONE.value,
            "response_id": response_id,
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "part": part or {"type": "audio", "transcript": ""},
        }
    
    @staticmethod
    def response_audio_delta(response_id: str, item_id: str, 
                             delta: str, output_index: int = 0,
                             content_index: int = 0,
                             event_id: Optional[str] = None) -> Dict[str, Any]:
        """构建音频增量事件 - delta 为 Base64 编码的音频数据"""
        return {
            "event_id": event_id or generate_id("evt"),
            "type": ServerEventType.RESPONSE_AUDIO_DELTA.value,
            "response_id": response_id,
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "delta": delta,
        }
    
    @staticmethod
    def response_audio_done(response_id: str, item_id: str,
                            output_index: int = 0, content_index: int = 0,
                            event_id: Optional[str] = None) -> Dict[str, Any]:
        """构建音频完成事件"""
        return {
            "event_id": event_id or generate_id("evt"),
            "type": ServerEventType.RESPONSE_AUDIO_DONE.value,
            "response_id": response_id,
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
        }
    
    @staticmethod
    def response_audio_transcript_delta(response_id: str, item_id: str,
                                        delta: str, output_index: int = 0,
                                        content_index: int = 0,
                                        event_id: Optional[str] = None) -> Dict[str, Any]:
        """构建音频转录增量事件"""
        return {
            "event_id": event_id or generate_id("evt"),
            "type": ServerEventType.RESPONSE_AUDIO_TRANSCRIPT_DELTA.value,
            "response_id": response_id,
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "delta": delta,
        }
    
    @staticmethod
    def response_audio_transcript_done(response_id: str, item_id: str,
                                       transcript: str, output_index: int = 0,
                                       content_index: int = 0,
                                       event_id: Optional[str] = None) -> Dict[str, Any]:
        """构建音频转录完成事件"""
        return {
            "event_id": event_id or generate_id("evt"),
            "type": ServerEventType.RESPONSE_AUDIO_TRANSCRIPT_DONE.value,
            "response_id": response_id,
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "transcript": transcript,
        }
    
    @staticmethod
    def response_text_delta(response_id: str, item_id: str,
                            delta: str, output_index: int = 0,
                            content_index: int = 0,
                            event_id: Optional[str] = None) -> Dict[str, Any]:
        """构建文本增量事件"""
        return {
            "event_id": event_id or generate_id("evt"),
            "type": ServerEventType.RESPONSE_TEXT_DELTA.value,
            "response_id": response_id,
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "delta": delta,
        }
    
    @staticmethod
    def response_text_done(response_id: str, item_id: str,
                           text: str, output_index: int = 0,
                           content_index: int = 0,
                           event_id: Optional[str] = None) -> Dict[str, Any]:
        """构建文本完成事件"""
        return {
            "event_id": event_id or generate_id("evt"),
            "type": ServerEventType.RESPONSE_TEXT_DONE.value,
            "response_id": response_id,
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "text": text,
        }
    
    @staticmethod
    def rate_limits_updated(rate_limits: Optional[List[Dict[str, Any]]] = None,
                            event_id: Optional[str] = None) -> Dict[str, Any]:
        """构建速率限制更新事件"""
        return {
            "event_id": event_id or generate_id("evt"),
            "type": ServerEventType.RATE_LIMITS_UPDATED.value,
            "rate_limits": rate_limits or [
                {"name": "requests", "limit": 100, "remaining": 99, "reset_seconds": 60},
                {"name": "tokens", "limit": 100000, "remaining": 99000, "reset_seconds": 60},
            ]
        }


# 导出便捷访问
EventBuilder = ServerEventBuilder
