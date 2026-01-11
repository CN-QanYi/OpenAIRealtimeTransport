#!/usr/bin/env uv run
"""
Realtime API 终端应用 - 自由麦模式

这是一个使用 Textual 框架构建的终端用户界面 (TUI) 应用，
展示了如何使用 OpenAI Realtime API 进行语音交互。

运行要求:
- 安装 `uv` 包管理器
- 设置 `OPENAI_API_KEY` 环境变量(使用OpenAI时)
- Mac 系统需要: `brew install portaudio ffmpeg`

运行方式:
`python push_to_talk_app.py`

使用说明:
- 直接对着麦克风说话，Server VAD 会自动检测语音开始和结束
- 按 Q 键退出应用

依赖包:
- textual: 终端 UI 框架
- numpy: 数值计算
- pyaudio: 音频处理
- pydub: 音频转换
- sounddevice: 音频设备访问
- openai[realtime]: OpenAI SDK 及 Realtime 支持
"""
####################################################################
#
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "textual",
#     "numpy",
#     "pyaudio",
#     "pydub",
#     "sounddevice",
#     "openai[realtime]",
# ]
#
# [tool.uv.sources]
# openai = { path = "../../", editable = true }
# ///
from __future__ import annotations

import base64
import asyncio
import json
from typing import Any, cast, Optional
from typing_extensions import override

from textual import events
from audio_utils import CHANNELS, SAMPLE_RATE, AudioPlayerAsync
from textual.app import App, ComposeResult
from textual.widgets import Button, Static, RichLog
from textual.reactive import reactive
from textual.containers import Container

# 本地服务器配置
LOCAL_SERVER_URL = "ws://localhost:8000/v1/realtime"
USE_LOCAL_SERVER = True  # 设置为 True 使用本地服务器，False 使用 OpenAI

# 根据配置选择导入方式
if USE_LOCAL_SERVER:
    import websockets
    from websockets.asyncio.client import ClientConnection
else:
    from openai import AsyncOpenAI
    from openai.types.realtime.session import Session  # type: ignore
    from openai.resources.realtime.realtime import AsyncRealtimeConnection


class SessionDisplay(Static):
    """A widget that shows the current session ID."""

    session_id = reactive("")

    @override
    def render(self) -> str:
        return f"Session ID: {self.session_id}" if self.session_id else "Connecting..."


class AudioStatusIndicator(Static):
    """A widget that shows the current audio status."""

    @override
    def render(self) -> str:
        return "🎤 自由麦模式: 持续监听中... (按 Q 退出)"


class RealtimeApp(App[None]):
    CSS = """
        Screen {
            background: #1a1b26;  /* Dark blue-grey background */
        }

        Container {
            border: double rgb(91, 164, 91);
        }

        Horizontal {
            width: 100%;
        }

        #input-container {
            height: 5;  /* Explicit height for input container */
            margin: 1 1;
            padding: 1 2;
        }

        Input {
            width: 80%;
            height: 3;  /* Explicit height for input */
        }

        Button {
            width: 20%;
            height: 3;  /* Explicit height for button */
        }

        #bottom-pane {
            width: 100%;
            height: 82%;  /* Reduced to make room for session display */
            border: round rgb(205, 133, 63);
            content-align: center middle;
        }

        #status-indicator {
            height: 3;
            content-align: center middle;
            background: #2a2b36;
            border: solid rgb(91, 164, 91);
            margin: 1 1;
        }

        #session-display {
            height: 3;
            content-align: center middle;
            background: #2a2b36;
            border: solid rgb(91, 164, 91);
            margin: 1 1;
        }

        Static {
            color: white;
        }
    """

    should_send_audio: asyncio.Event
    audio_player: AudioPlayerAsync
    last_audio_item_id: str | None
    connection: Any  # WebSocket 连接或 OpenAI 连接
    session: Any  # 会话对象
    connected: asyncio.Event
    ws: Optional[ClientConnection] if USE_LOCAL_SERVER else None  # type: ignore

    def __init__(self) -> None:
        super().__init__()
        self.connection = None
        self.session = None
        self.ws = None
        # 配置客户端
        if not USE_LOCAL_SERVER:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI()
        self.audio_player = AudioPlayerAsync()
        self.last_audio_item_id = None
        self.connected = asyncio.Event()

    @override
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        with Container():
            yield SessionDisplay(id="session-display")
            yield AudioStatusIndicator(id="status-indicator")
            yield RichLog(id="bottom-pane", wrap=True, highlight=True, markup=True)

    async def on_mount(self) -> None:
        self.run_worker(self.handle_realtime_connection())
        self.run_worker(self.send_mic_audio())

    async def handle_realtime_connection(self) -> None:
        """处理 Realtime 连接 - 支持本地服务器和 OpenAI"""
        if USE_LOCAL_SERVER:
            await self._handle_local_server_connection()
        else:
            await self._handle_openai_connection()

    async def _handle_local_server_connection(self) -> None:
        """连接到本地服务器"""
        try:
            self.ws = await websockets.connect(LOCAL_SERVER_URL)
            self.connection = self.ws
            self.connected.set()
            
            # 发送会话更新请求
            await self.ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "turn_detection": {"type": "server_vad"},
                    "modalities": ["audio", "text"],
                }
            }))
            
            acc_items: dict[str, Any] = {}
            
            # 接收事件循环
            async for message in self.ws:
                try:
                    event = json.loads(message)
                    event_type = event.get("type", "")
                    
                    if event_type == "session.created":
                        session_id = event.get("session", {}).get("id", "unknown")
                        session_display = self.query_one(SessionDisplay)
                        session_display.session_id = session_id
                        continue
                    
                    if event_type == "session.updated":
                        continue
                    
                    if event_type == "response.audio.delta":
                        item_id = event.get("item_id", "")
                        delta = event.get("delta", "")
                        
                        if item_id != self.last_audio_item_id:
                            self.audio_player.reset_frame_count()
                            self.last_audio_item_id = item_id
                        
                        if delta:
                            bytes_data = base64.b64decode(delta)
                            self.audio_player.add_data(bytes_data)
                        continue
                    
                    if event_type == "response.audio_transcript.delta":
                        item_id = event.get("item_id", "")
                        delta = event.get("delta", "")
                        
                        if item_id not in acc_items:
                            acc_items[item_id] = delta
                        else:
                            acc_items[item_id] = acc_items[item_id] + delta
                        
                        bottom_pane = self.query_one("#bottom-pane", RichLog)
                        bottom_pane.clear()
                        bottom_pane.write(acc_items[item_id])
                        continue
                    
                    # 处理其他事件类型
                    if event_type == "error":
                        error_msg = event.get("error", {}).get("message", "Unknown error")
                        bottom_pane = self.query_one("#bottom-pane", RichLog)
                        bottom_pane.write(f"[red]错误: {error_msg}[/red]")
                        continue
                        
                except json.JSONDecodeError:
                    continue
                    
        except Exception as e:
            bottom_pane = self.query_one("#bottom-pane", RichLog)
            bottom_pane.write(f"[red]连接错误: {e}[/red]")

    async def _handle_openai_connection(self) -> None:
        """连接到 OpenAI Realtime API"""
        async with self.client.realtime.connect(model="gpt-realtime") as conn:
            self.connection = conn
            self.connected.set()

            # note: this is the default and can be omitted
            # if you want to manually handle VAD yourself, then set `'turn_detection': None`
            await conn.session.update(
                session={
                    "audio": {
                        "input": {"turn_detection": {"type": "server_vad"}},
                    },
                    "model": "gpt-realtime",
                    "type": "realtime",
                }
            )

            acc_items: dict[str, Any] = {}

            async for event in conn:
                if event.type == "session.created":
                    self.session = event.session
                    session_display = self.query_one(SessionDisplay)
                    assert event.session.id is not None
                    session_display.session_id = event.session.id
                    continue

                if event.type == "session.updated":
                    self.session = event.session
                    continue

                if event.type == "response.output_audio.delta":
                    if event.item_id != self.last_audio_item_id:
                        self.audio_player.reset_frame_count()
                        self.last_audio_item_id = event.item_id

                    bytes_data = base64.b64decode(event.delta)
                    self.audio_player.add_data(bytes_data)
                    continue

                if event.type == "response.output_audio_transcript.delta":
                    try:
                        text = acc_items[event.item_id]
                    except KeyError:
                        acc_items[event.item_id] = event.delta
                    else:
                        acc_items[event.item_id] = text + event.delta

                    # Clear and update the entire content because RichLog otherwise treats each delta as a new line
                    bottom_pane = self.query_one("#bottom-pane", RichLog)
                    bottom_pane.clear()
                    bottom_pane.write(acc_items[event.item_id])
                    continue

    async def _get_connection(self) -> Any:
        await self.connected.wait()
        assert self.connection is not None
        return self.connection

    async def send_mic_audio(self) -> None:
        import sounddevice as sd  # type: ignore

        sent_audio = False

        device_info = sd.query_devices()
        print(device_info)

        read_size = int(SAMPLE_RATE * 0.02)

        stream = sd.InputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            dtype="int16",
        )
        stream.start()

        status_indicator = self.query_one(AudioStatusIndicator)

        try:
            while True:
                if stream.read_available < read_size:
                    await asyncio.sleep(0)
                    continue

                data, _ = stream.read(read_size)

                connection = await self._get_connection()
                
                if USE_LOCAL_SERVER:
                    # 本地服务器：直接发送 JSON 消息
                    if not sent_audio:
                        try:
                            await connection.send(json.dumps({"type": "response.cancel"}))
                        except:
                            pass
                        sent_audio = True
                    
                    # 发送音频数据（自由麦模式下持续发送，Server VAD会自动检测）
                    audio_b64 = base64.b64encode(cast(Any, data)).decode("utf-8")
                    await connection.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": audio_b64
                    }))
                else:
                    # OpenAI SDK
                    if not sent_audio:
                        asyncio.create_task(connection.send({"type": "response.cancel"}))
                        sent_audio = True

                    await connection.input_audio_buffer.append(audio=base64.b64encode(cast(Any, data)).decode("utf-8"))

                await asyncio.sleep(0)
        except KeyboardInterrupt:
            pass
        finally:
            stream.stop()
            stream.close()

    async def on_key(self, event: events.Key) -> None:
        """Handle key press events."""
        if event.key == "q":
            self.exit()
            return


if __name__ == "__main__":
    app = RealtimeApp()
    app.run()
