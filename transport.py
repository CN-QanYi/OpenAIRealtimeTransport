"""
OpenAI Realtime API 兼容的 Transport 层
充当"翻译官"，在 OpenAI 协议和 Pipecat 内部帧之间进行转换
"""
import base64
import json
import asyncio
import logging
from typing import Any, Dict, Optional, Callable, Awaitable, Tuple
from dataclasses import dataclass, field

from fastapi import WebSocket, WebSocketDisconnect

from protocol import (
    ClientEventType, ServerEventType, ServerEventBuilder,
    SessionConfig, TurnDetection, ConversationItem, Response,
    generate_id
)
from audio_utils import AudioConverter, AudioBuffer, calculate_audio_duration_ms
from config import config

logger = logging.getLogger(__name__)


@dataclass
class TransportState:
    """Transport 状态"""
    session: SessionConfig = field(default_factory=SessionConfig)
    conversation_id: str = field(default_factory=lambda: generate_id("conv"))
    current_response: Optional[Response] = None
    current_item: Optional[ConversationItem] = None
    is_speaking: bool = False
    audio_start_ms: int = 0
    total_audio_ms: int = 0


class OpenAIRealtimeTransport:
    """
    OpenAI Realtime API 兼容的 WebSocket Transport
    
    主要职责:
    1. 接收客户端的 OpenAI 格式 JSON 事件
    2. 将音频数据转换并传递给 Pipecat 管道
    3. 接收 Pipecat 管道的输出帧
    4. 将输出转换为 OpenAI 格式事件发送给客户端
    """
    
    def __init__(self, websocket: WebSocket):
        """
        初始化 Transport
        
        Args:
            websocket: FastAPI WebSocket 连接
        """
        self.ws = websocket
        self.state = TransportState()
        self.audio_converter = AudioConverter()
        self.audio_buffer = AudioBuffer()
        
        # 事件回调
        self._on_audio_frame: Optional[Callable[[bytes], Awaitable[None]]] = None
        self._on_session_update: Optional[Callable[[SessionConfig], Awaitable[None]]] = None
        self._on_response_create: Optional[Callable[[], Awaitable[None]]] = None
        self._on_response_cancel: Optional[Callable[[], Awaitable[None]]] = None
        self._on_conversation_item: Optional[Callable[[ConversationItem], Awaitable[None]]] = None
        
        # 控制标志
        self._running = False
        self._closed = False
        
        logger.info(f"Transport 已创建，会话 ID: {self.state.session.id}")
    
    # ==================== 回调注册 ====================
    
    def on_audio_frame(self, callback: Callable[[bytes], Awaitable[None]]):
        """注册音频帧回调"""
        self._on_audio_frame = callback
        return self
    
    def on_session_update(self, callback: Callable[[SessionConfig], Awaitable[None]]):
        """注册会话更新回调"""
        self._on_session_update = callback
        return self
    
    def on_response_create(self, callback: Callable[[], Awaitable[None]]):
        """注册响应创建回调"""
        self._on_response_create = callback
        return self
    
    def on_response_cancel(self, callback: Callable[[], Awaitable[None]]):
        """注册响应取消回调"""
        self._on_response_cancel = callback
        return self
    
    def on_conversation_item(self, callback: Callable[[ConversationItem], Awaitable[None]]):
        """注册对话项创建回调"""
        self._on_conversation_item = callback
        return self
    
    # ==================== 生命周期管理 ====================
    
    async def start(self) -> None:
        """启动 Transport，发送初始化事件"""
        self._running = True
        
        # 生成会话 ID
        self.state.session.id = generate_id("sess")
        
        # 发送会话创建事件
        await self._send_event(
            ServerEventBuilder.session_created(self.state.session)
        )
        
        # 发送对话创建事件
        await self._send_event(
            ServerEventBuilder.conversation_created(self.state.conversation_id)
        )
        
        logger.info(f"会话已创建: {self.state.session.id}")
    
    async def run(self) -> None:
        """运行主循环，处理客户端消息"""
        try:
            while self._running:
                message = await self.ws.receive_text()
                await self._handle_client_message(message)
        except WebSocketDisconnect:
            logger.info("WebSocket 连接已断开")
        except Exception as e:
            logger.error(f"Transport 运行错误: {e}")
            await self._send_error(str(e))
        finally:
            await self.close()
    
    async def close(self) -> None:
        """关闭 Transport"""
        if self._closed:
            return
        
        self._running = False
        self._closed = True
        self.audio_buffer.clear()
        
        logger.info(f"Transport 已关闭: {self.state.session.id}")
    
    # ==================== 客户端消息处理 ====================
    
    async def _handle_client_message(self, message: str) -> None:
        """处理客户端发送的消息"""
        try:
            event = json.loads(message)
            event_type = event.get("type", "")
            event_id = event.get("event_id")
            
            # logger.debug(f"收到客户端事件: {event_type}")  # 减少噪音
            
            # 根据事件类型分发处理
            handlers = {
                ClientEventType.SESSION_UPDATE.value: self._handle_session_update,
                ClientEventType.INPUT_AUDIO_BUFFER_APPEND.value: self._handle_audio_append,
                # INPUT_AUDIO_BUFFER_COMMIT 已移除：内置 Server VAD 自动检测，不需要手动提交
                ClientEventType.INPUT_AUDIO_BUFFER_CLEAR.value: self._handle_audio_clear,
                ClientEventType.CONVERSATION_ITEM_CREATE.value: self._handle_conversation_item_create,
                ClientEventType.CONVERSATION_ITEM_TRUNCATE.value: self._handle_conversation_item_truncate,
                ClientEventType.CONVERSATION_ITEM_DELETE.value: self._handle_conversation_item_delete,
                ClientEventType.RESPONSE_CREATE.value: self._handle_response_create,
                ClientEventType.RESPONSE_CANCEL.value: self._handle_response_cancel,
            }
            
            handler = handlers.get(event_type)
            if handler:
                await handler(event)
            else:
                logger.warning(f"未知事件类型: {event_type}")
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析错误: {e}")
            await self._send_error("Invalid JSON format")
        except Exception as e:
            logger.error(f"处理消息错误: {e}")
            await self._send_error(str(e))
    
    async def _handle_session_update(self, event: Dict[str, Any]) -> None:
        """处理会话更新事件"""
        session_data = event.get("session", {})
        
        # 更新会话配置
        if "instructions" in session_data:
            self.state.session.instructions = session_data["instructions"]
        if "voice" in session_data:
            self.state.session.voice = session_data["voice"]
        if "modalities" in session_data:
            self.state.session.modalities = session_data["modalities"]
        if "temperature" in session_data:
            self.state.session.temperature = session_data["temperature"]
        if "max_response_output_tokens" in session_data:
            self.state.session.max_response_output_tokens = session_data["max_response_output_tokens"]
        if "input_audio_format" in session_data:
            self.state.session.input_audio_format = session_data["input_audio_format"]
        if "output_audio_format" in session_data:
            self.state.session.output_audio_format = session_data["output_audio_format"]
        
        # 更新 VAD 配置
        if "turn_detection" in session_data:
            td = session_data["turn_detection"]
            if td is None:
                self.state.session.turn_detection = None
            else:
                self.state.session.turn_detection = TurnDetection(
                    type=td.get("type", "server_vad"),
                    threshold=td.get("threshold", 0.5),
                    prefix_padding_ms=td.get("prefix_padding_ms", 300),
                    silence_duration_ms=td.get("silence_duration_ms", 500),
                    create_response=td.get("create_response", True),
                )
        
        # 更新工具配置
        if "tools" in session_data:
            self.state.session.tools = session_data["tools"]
        if "tool_choice" in session_data:
            self.state.session.tool_choice = session_data["tool_choice"]
        
        # 更新输入转录配置
        if "input_audio_transcription" in session_data:
            iat = session_data["input_audio_transcription"]
            if iat:
                from protocol import InputAudioTranscription
                self.state.session.input_audio_transcription = InputAudioTranscription(
                    model=iat.get("model", "whisper-1")
                )
        
        # 发送更新确认
        await self._send_event(
            ServerEventBuilder.session_updated(self.state.session)
        )
        
        # 触发回调
        if self._on_session_update:
            await self._on_session_update(self.state.session)
        
        logger.info("会话配置已更新")
    
    async def _handle_audio_append(self, event: Dict[str, Any]) -> None:
        """处理音频追加事件"""
        audio_b64 = event.get("audio", "")
        if not audio_b64:
            return
        
        try:
            # 解码 Base64 音频
            audio_bytes = base64.b64decode(audio_b64)
            
            # 添加到缓冲区
            self.audio_buffer.append(audio_bytes)
            
            # 更新总音频时长
            self.state.total_audio_ms += calculate_audio_duration_ms(audio_bytes)
            
            # 重采样：24kHz -> 16kHz
            internal_audio = self.audio_converter.client_to_internal(audio_bytes)
            
            # 触发音频帧回调
            if self._on_audio_frame:
                await self._on_audio_frame(internal_audio)
                
        except Exception as e:
            logger.error(f"音频处理错误: {e}")
    
    async def _handle_audio_clear(self, event: Dict[str, Any]) -> None:
        """处理音频缓冲区清空事件"""
        self.audio_buffer.clear()
        
        await self._send_event(
            ServerEventBuilder.input_audio_buffer_cleared()
        )
        
        logger.info("音频缓冲区已清空")
    
    async def _handle_conversation_item_create(self, event: Dict[str, Any]) -> None:
        """处理对话项创建事件"""
        item_data = event.get("item", {})
        
        item = ConversationItem(
            id=item_data.get("id", generate_id("item")),
            type=item_data.get("type", "message"),
            role=item_data.get("role", "user"),
            content=item_data.get("content", [])
        )
        
        await self._send_event(
            ServerEventBuilder.conversation_item_created(
                item,
                previous_item_id=self.state.current_item.id if self.state.current_item else None
            )
        )
        
        self.state.current_item = item
        
        # 触发回调
        if self._on_conversation_item:
            await self._on_conversation_item(item)
        
        logger.info(f"对话项已创建: {item.id}")
    
    async def _handle_conversation_item_truncate(self, event: Dict[str, Any]) -> None:
        """处理对话项截断事件"""
        item_id = event.get("item_id")
        content_index = event.get("content_index", 0)
        audio_end_ms = event.get("audio_end_ms", 0)
        
        await self._send_event({
            "event_id": generate_id("evt"),
            "type": ServerEventType.CONVERSATION_ITEM_TRUNCATED.value,
            "item_id": item_id,
            "content_index": content_index,
            "audio_end_ms": audio_end_ms,
        })
        
        logger.info(f"对话项已截断: {item_id}")
    
    async def _handle_conversation_item_delete(self, event: Dict[str, Any]) -> None:
        """处理对话项删除事件"""
        item_id = event.get("item_id")
        
        await self._send_event({
            "event_id": generate_id("evt"),
            "type": ServerEventType.CONVERSATION_ITEM_DELETED.value,
            "item_id": item_id,
        })
        
        logger.info(f"对话项已删除: {item_id}")
    
    async def _handle_response_create(self, event: Dict[str, Any]) -> None:
        """处理响应创建事件"""
        # 触发回调让管道开始生成响应
        if self._on_response_create:
            await self._on_response_create()
        
        logger.info("响应创建请求已收到")
    
    async def _handle_response_cancel(self, event: Dict[str, Any]) -> None:
        """处理响应取消事件"""
        if self.state.current_response:
            self.state.current_response.status = "cancelled"
            
            await self._send_event(
                ServerEventBuilder.response_done(self.state.current_response)
            )
        
        # 触发回调
        if self._on_response_cancel:
            await self._on_response_cancel()
        
        logger.info("响应已取消")
    
    # ==================== 向客户端发送事件 ====================
    
    async def _send_event(self, event: Dict[str, Any]) -> None:
        """发送事件到客户端"""
        if self._closed:
            return
        
        try:
            await self.ws.send_json(event)
            # logger.debug(f"发送事件: {event.get('type')}")  # 减少噪音
        except Exception as e:
            logger.error(f"发送事件失败: {e}")
    
    async def _send_error(self, message: str, error_type: str = "server_error") -> None:
        """发送错误事件"""
        await self._send_event(
            ServerEventBuilder.error(message, error_type)
        )
    
    # ==================== 管道输出处理（供外部调用） ====================
    
    async def send_speech_started(self, audio_start_ms: Optional[int] = None) -> None:
        """
        发送语音开始事件
        用于触发客户端打断正在播放的音频
        """
        self.state.is_speaking = True
        self.state.audio_start_ms = audio_start_ms or self.state.total_audio_ms
        
        item_id = generate_id("item")
        
        await self._send_event(
            ServerEventBuilder.input_audio_buffer_speech_started(
                audio_start_ms=self.state.audio_start_ms,
                item_id=item_id
            )
        )
        
        logger.info(f"语音开始: {self.state.audio_start_ms}ms")
    
    async def send_speech_stopped(self, audio_end_ms: Optional[int] = None) -> None:
        """发送语音停止事件"""
        self.state.is_speaking = False
        
        await self._send_event(
            ServerEventBuilder.input_audio_buffer_speech_stopped(
                audio_end_ms=audio_end_ms or self.state.total_audio_ms,
                item_id=self.state.current_item.id if self.state.current_item else generate_id("item")
            )
        )
        
        logger.info("语音停止")
    
    async def begin_response(self) -> Tuple[str, str]:
        """
        开始新的响应
        
        Returns:
            (response_id, item_id) 元组
        """
        # 创建响应对象
        response = Response()
        self.state.current_response = response
        
        # 创建输出项
        item = ConversationItem(
            type="message",
            role="assistant",
            status="in_progress",
            content=[]
        )
        self.state.current_item = item
        response.output.append(item)
        
        # 发送响应创建事件
        await self._send_event(
            ServerEventBuilder.response_created(response)
        )
        
        # 发送输出项添加事件
        await self._send_event(
            ServerEventBuilder.response_output_item_added(
                response_id=response.id,
                item=item,
                output_index=0
            )
        )
        
        # 发送内容部分添加事件
        await self._send_event(
            ServerEventBuilder.response_content_part_added(
                response_id=response.id,
                item_id=item.id,
                output_index=0,
                content_index=0,
                part_type="audio"
            )
        )
        
        logger.info(f"响应开始: {response.id}")
        
        return response.id, item.id
    
    async def send_audio_delta(self, audio_bytes: bytes, response_id: str, item_id: str) -> None:
        """
        发送音频增量
        
        Args:
            audio_bytes: 内部格式的音频数据（16kHz）
            response_id: 响应 ID
            item_id: 项目 ID
        """
        # 重采样：16kHz -> 24kHz
        client_audio = self.audio_converter.internal_to_client(audio_bytes)
        
        # Base64 编码
        audio_b64 = base64.b64encode(client_audio).decode('utf-8')
        
        # 发送音频增量事件
        await self._send_event(
            ServerEventBuilder.response_audio_delta(
                response_id=response_id,
                item_id=item_id,
                delta=audio_b64,
                output_index=0,
                content_index=0
            )
        )
    
    async def send_transcript_delta(self, text: str, response_id: str, item_id: str) -> None:
        """发送转录文本增量"""
        await self._send_event(
            ServerEventBuilder.response_audio_transcript_delta(
                response_id=response_id,
                item_id=item_id,
                delta=text,
                output_index=0,
                content_index=0
            )
        )
    
    async def send_text_delta(self, text: str, response_id: str, item_id: str) -> None:
        """发送文本增量（纯文本模式）"""
        await self._send_event(
            ServerEventBuilder.response_text_delta(
                response_id=response_id,
                item_id=item_id,
                delta=text,
                output_index=0,
                content_index=0
            )
        )
    
    async def end_response(self, transcript: str = "") -> None:
        """结束当前响应"""
        if not self.state.current_response:
            return
        
        response = self.state.current_response
        item = self.state.current_item
        
        # 确保 item 存在
        if not item:
            logger.warning("完成响应时没有当前项")
            return
        
        # 发送音频完成事件
        await self._send_event(
            ServerEventBuilder.response_audio_done(
                response_id=response.id,
                item_id=item.id,
                output_index=0,
                content_index=0
            )
        )
        
        # 发送转录完成事件
        await self._send_event(
            ServerEventBuilder.response_audio_transcript_done(
                response_id=response.id,
                item_id=item.id,
                transcript=transcript,
                output_index=0,
                content_index=0
            )
        )
        
        # 发送内容部分完成事件
        await self._send_event(
            ServerEventBuilder.response_content_part_done(
                response_id=response.id,
                item_id=item.id,
                output_index=0,
                content_index=0,
                part={"type": "audio", "transcript": transcript}
            )
        )
        
        # 发送输出项完成事件
        await self._send_event(
            ServerEventBuilder.response_output_item_done(
                response_id=response.id,
                item=item,
                output_index=0
            )
        )
        
        # 发送响应完成事件
        response.status = "completed"
        response.usage = {
            "total_tokens": 100,  # 模拟值
            "input_tokens": 50,
            "output_tokens": 50,
        }
        
        await self._send_event(
            ServerEventBuilder.response_done(response)
        )
        
        # 发送速率限制更新
        await self._send_event(
            ServerEventBuilder.rate_limits_updated()
        )
        
        self.state.current_response = None
        
        logger.info(f"响应完成: {response.id}")
    
    async def cancel_response(self) -> None:
        """取消当前响应"""
        if self.state.current_response:
            self.state.current_response.status = "cancelled"
            await self._send_event(
                ServerEventBuilder.response_done(self.state.current_response)
            )
            self.state.current_response = None
            logger.info("响应已取消")
