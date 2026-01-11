"""
测试客户端 - 用于验证 OpenAI Realtime API 兼容服务器
模拟 OpenAI SDK 的行为，发送和接收事件
"""
import asyncio
import json
import base64
import numpy as np
import websockets
from typing import Optional, Any


class RealtimeTestClient:
    """
    测试客户端
    用于验证服务器是否正确实现了 OpenAI Realtime API 协议
    """
    
    def __init__(self, url: str = "ws://localhost:8000/v1/realtime"):
        self.url = url
        self.ws: Any = None  # websockets.WebSocketClientProtocol
        self.session_id: Optional[str] = None
        self._running = False
    
    async def connect(self):
        """连接到服务器"""
        print(f"正在连接到 {self.url}...")
        self.ws = await websockets.connect(self.url)
        self._running = True
        print("已连接!")
    
    async def disconnect(self):
        """断开连接"""
        self._running = False
        if self.ws:
            await self.ws.close()
            print("已断开连接")
    
    async def send_event(self, event: dict):
        """发送事件"""
        if self.ws:
            await self.ws.send(json.dumps(event))
            print(f">>> 发送: {event['type']}")
    
    async def receive_events(self):
        """接收事件循环"""
        try:
            while self._running and self.ws:
                message = await asyncio.wait_for(self.ws.recv(), timeout=1.0)
                event = json.loads(message)
                await self._handle_event(event)
        except asyncio.TimeoutError:
            pass
        except websockets.ConnectionClosed:
            print("连接已关闭")
            self._running = False
    
    async def _handle_event(self, event: dict):
        """处理接收到的事件"""
        event_type = event.get("type", "unknown")
        
        print(f"<<< 收到: {event_type}")
        
        if event_type == "session.created":
            self.session_id = event.get("session", {}).get("id")
            print(f"    会话 ID: {self.session_id}")
        
        elif event_type == "session.updated":
            print(f"    会话已更新")
        
        elif event_type == "conversation.created":
            print(f"    对话 ID: {event.get('conversation', {}).get('id')}")
        
        elif event_type == "response.created":
            print(f"    响应 ID: {event.get('response', {}).get('id')}")
        
        elif event_type == "response.audio.delta":
            delta = event.get("delta", "")
            audio_bytes = base64.b64decode(delta) if delta else b""
            print(f"    音频增量: {len(audio_bytes)} 字节")
        
        elif event_type == "response.audio_transcript.delta":
            print(f"    转录增量: {event.get('delta', '')}")
        
        elif event_type == "response.done":
            status = event.get("response", {}).get("status")
            print(f"    响应完成: {status}")
        
        elif event_type == "input_audio_buffer.speech_started":
            print(f"    检测到语音开始")
        
        elif event_type == "input_audio_buffer.speech_stopped":
            print(f"    检测到语音停止")
        
        elif event_type == "error":
            error = event.get("error", {})
            print(f"    错误: {error.get('message')}")
    
    # ==================== 测试用例 ====================
    
    async def test_session_update(self):
        """测试会话更新"""
        print("\n=== 测试: 会话更新 ===")
        
        await self.send_event({
            "type": "session.update",
            "session": {
                "instructions": "你是一个有帮助的AI助手。",
                "voice": "alloy",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "silence_duration_ms": 500
                }
            }
        })
        
        await asyncio.sleep(0.5)
    
    async def test_audio_input(self):
        """测试音频输入"""
        print("\n=== 测试: 音频输入 ===")
        
        # 生成模拟音频（1秒的正弦波）
        sample_rate = 24000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        
        # 生成 440Hz 正弦波
        audio_float = np.sin(2 * np.pi * 440 * t) * 0.5
        audio_int16 = (audio_float * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        # 分块发送
        chunk_size = int(sample_rate * 0.1) * 2  # 100ms
        
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i+chunk_size]
            audio_b64 = base64.b64encode(chunk).decode('utf-8')
            
            await self.send_event({
                "type": "input_audio_buffer.append",
                "audio": audio_b64
            })
            
            await asyncio.sleep(0.05)
        
        print(f"    已发送 {len(audio_bytes)} 字节音频")
    
    async def test_audio_commit(self):
        """测试音频提交（手动 VAD 模式）"""
        print("\n=== 测试: 音频提交 ===")
        
        await self.send_event({
            "type": "input_audio_buffer.commit"
        })
        
        await asyncio.sleep(0.5)
    
    async def test_response_create(self):
        """测试响应创建"""
        print("\n=== 测试: 响应创建 ===")
        
        await self.send_event({
            "type": "response.create"
        })
        
        # 等待响应完成
        await asyncio.sleep(3)
    
    async def test_conversation_item(self):
        """测试对话项创建"""
        print("\n=== 测试: 对话项创建 ===")
        
        await self.send_event({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "你好，请介绍一下你自己。"
                    }
                ]
            }
        })
        
        await asyncio.sleep(0.5)
    
    async def test_response_cancel(self):
        """测试响应取消"""
        print("\n=== 测试: 响应取消 ===")
        
        # 先创建响应
        await self.send_event({
            "type": "response.create"
        })
        
        await asyncio.sleep(0.5)
        
        # 然后取消
        await self.send_event({
            "type": "response.cancel"
        })
        
        await asyncio.sleep(0.5)


async def run_tests():
    """运行所有测试"""
    client = RealtimeTestClient()
    
    try:
        await client.connect()
        
        # 启动接收循环
        receive_task = asyncio.create_task(client.receive_events())
        
        # 等待初始化事件
        await asyncio.sleep(1)
        
        # 运行测试
        await client.test_session_update()
        await asyncio.sleep(0.5)
        
        await client.test_conversation_item()
        await asyncio.sleep(0.5)
        
        await client.test_audio_input()
        await asyncio.sleep(0.5)
        
        await client.test_response_create()
        await asyncio.sleep(3)
        
        await client.test_response_cancel()
        await asyncio.sleep(1)
        
        print("\n=== 所有测试完成 ===")
        
    except Exception as e:
        print(f"测试错误: {e}")
    finally:
        await client.disconnect()


async def interactive_client():
    """交互式客户端"""
    client = RealtimeTestClient()
    
    try:
        await client.connect()
        
        # 启动接收循环
        receive_task = asyncio.create_task(client.receive_events())
        
        print("\n交互式模式 - 可用命令:")
        print("  1 - 更新会话")
        print("  2 - 发送测试音频")
        print("  3 - 提交音频")
        print("  4 - 创建响应")
        print("  5 - 取消响应")
        print("  6 - 创建对话项")
        print("  q - 退出")
        
        while True:
            cmd = await asyncio.get_event_loop().run_in_executor(None, input, "\n请输入命令: ")
            
            if cmd == "1":
                await client.test_session_update()
            elif cmd == "2":
                await client.test_audio_input()
            elif cmd == "3":
                await client.test_audio_commit()
            elif cmd == "4":
                await client.test_response_create()
            elif cmd == "5":
                await client.test_response_cancel()
            elif cmd == "6":
                await client.test_conversation_item()
            elif cmd.lower() == "q":
                break
            else:
                print("未知命令")
            
            await asyncio.sleep(0.5)
        
    except Exception as e:
        print(f"错误: {e}")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    import sys
    
    print("OpenAI Realtime API 兼容服务器 - 测试客户端")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "-i":
        # 交互模式
        asyncio.run(interactive_client())
    else:
        # 自动测试模式
        asyncio.run(run_tests())
