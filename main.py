"""
OpenAI Realtime API 兼容服务器
完全复刻 OpenAI Realtime API 的协议，支持替换为本地或第三方模型

使用方法:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

客户端连接:
    将 OpenAI SDK 的 baseUrl 修改为 ws://localhost:8000 即可
"""
import os
from contextlib import asynccontextmanager
from typing import Optional
from urllib.parse import urlparse

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import config, print_config
from realtime_session import session_manager, RealtimeSession
from logger_config import setup_logging, get_logger

# 配置日志
setup_logging(
    level="DEBUG" if config.server.debug else "INFO",
    use_color=True
)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    logger.info("=" * 60)
    logger.info("OpenAI Realtime API 兼容服务器启动")
    logger.info(f"WebSocket 端点: ws://localhost:{config.server.port}{config.server.ws_path}")
    logger.info("=" * 60)
    print_config()  # 打印当前配置
    
    yield
    
    # 关闭时
    logger.info("服务器正在关闭...")


# 创建 FastAPI 应用
app = FastAPI(
    title="OpenAI Realtime API 兼容服务器",
    description="完全复刻 OpenAI Realtime API 协议的本地服务器",
    version="0.1.0",
    lifespan=lifespan
)

# 添加 CORS 中间件
# 从环境变量加载允许的来源，多个来源用逗号分隔
# 例如: CORS_ORIGINS="http://localhost:3000,http://localhost:8080"
_cors_origins_env = os.getenv("CORS_ORIGINS", "")
_cors_allow_credentials = True


def _parse_and_validate_cors_origins(env_value: str, *, debug: bool) -> tuple[list[str], bool]:
    """解析并校验 CORS_ORIGINS.

    Returns:
        (allowed_origins, allow_credentials)
    """
    raw = (env_value or "")
    stripped = raw.strip()

    # 空字符串或仅空白: 明确记录采用默认行为
    if not stripped:
        if debug:
            logger.warning(
                "CORS: 未配置 CORS_ORIGINS (为空/仅空格), DEBUG 模式将使用 allow_origins=['*'];"
                "allow_credentials 已禁用 (浏览器 CORS 规范要求)."
            )
            return ["*"], False

        defaults = ["http://localhost:3000", "http://localhost:8000"]
        logger.info(
            "CORS: 未配置 CORS_ORIGINS (为空/仅空格), 生产模式将使用默认来源: %s",
            defaults,
        )
        return defaults, True

    # 非空: 解析并校验 URL 格式
    candidates = [origin.strip() for origin in stripped.split(",") if origin.strip()]
    if not candidates:
        # 例如 env_value = "," 或 " , " 等, 仅包含分隔符/空条目, 视为未配置
        if debug:
            logger.warning(
                "CORS: CORS_ORIGINS 仅包含分隔符/空条目, DEBUG 模式将使用 allow_origins=['*'];"
                "allow_credentials 已禁用 (浏览器 CORS 规范要求)."
            )
            return ["*"], False

        defaults = ["http://localhost:3000", "http://localhost:8000"]
        logger.info(
            "CORS: CORS_ORIGINS 仅包含分隔符/空条目, 生产模式将使用默认来源: %s",
            defaults,
        )
        return defaults, True
    invalid: list[str] = []
    for origin in candidates:
        parsed = urlparse(origin)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            invalid.append(origin)

    if invalid:
        raise ValueError(
            "CORS_ORIGINS 配置包含无效 URL (需要 http/https 且包含主机名), 请修正: "
            + ", ".join(invalid)
        )

    logger.info("CORS: 允许的来源: %s", candidates)
    return candidates, True

_cors_allowed_origins, _cors_allow_credentials = _parse_and_validate_cors_origins(
    _cors_origins_env,
    debug=config.server.debug,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allowed_origins,
    allow_credentials=_cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== HTTP 端点 ====================

@app.get("/")
async def root():
    """根端点 - 服务器状态"""
    return {
        "status": "running",
        "service": "OpenAI Realtime API Compatible Server",
        "version": "0.1.0",
        "endpoints": {
            "websocket": f"ws://localhost:{config.server.port}/v1/realtime",
            "health": "/health",
            "sessions": "/v1/sessions"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "active_sessions": session_manager.active_count
    }


@app.get("/v1/sessions")
async def list_sessions():
    """列出活跃会话"""
    return {
        "object": "list",
        "data": [
            {"id": sid, "status": "active"} 
            for sid in session_manager.list_session_ids()
        ],
        "count": session_manager.active_count
    }


# ==================== WebSocket 端点 ====================

@app.websocket("/v1/realtime")
async def websocket_realtime(
    websocket: WebSocket,
    model: Optional[str] = Query(default="gpt-4o-realtime-preview", description="模型名称"),
):
    """
    OpenAI Realtime API 兼容的 WebSocket 端点
    
    支持的查询参数:
    - model: 模型名称（默认: gpt-4o-realtime-preview）
    
    协议:
    - 完全兼容 OpenAI Realtime API 的 JSON 事件格式
    - 音频格式: PCM16, 24kHz, 单声道
    """
    # 验证请求（可选）
    # 在生产环境中，你可能需要验证 Authorization header
    
    await websocket.accept()
    logger.info(f"新的 WebSocket 连接，模型: {model}")
    
    session: Optional[RealtimeSession] = None
    
    try:
        # 创建会话，传递模型参数
        session = await session_manager.create_session(websocket, model=model)
        
        # 运行会话主循环
        await session.run()
        
    except WebSocketDisconnect:
        logger.info("客户端断开连接")
    except Exception as e:
        logger.exception(f"WebSocket 错误: {e}")
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except (SystemExit, KeyboardInterrupt):
            raise
        except Exception as close_err:
            logger.warning(f"关闭 WebSocket 时出错: {close_err}")
    finally:
        # 清理会话（不调用 session.stop，由 session.run() 的 finally 块负责）
        if session:
            await session_manager.remove_session(session.state.session_id)


# 备用端点，支持带模型参数的路径
@app.websocket("/v1/realtime/{model_path:path}")
async def websocket_realtime_with_model(
    websocket: WebSocket,
    model_path: str,
):
    """支持路径参数指定模型的 WebSocket 端点"""
    await websocket.accept()
    logger.info(f"新的 WebSocket 连接，模型路径: {model_path}")
    
    session: Optional[RealtimeSession] = None
    
    try:
        # 使用路径中的模型名称
        session = await session_manager.create_session(websocket, model=model_path)
        await session.run()
    except WebSocketDisconnect:
        logger.info("客户端断开连接")
    except Exception as e:
        logger.exception(f"WebSocket 错误: {e}")
    finally:
        # 清理会话（不调用 session.stop，由 session.run() 的 finally 块负责）
        if session:
            await session_manager.remove_session(session.state.session_id)


# ==================== 模拟 OpenAI REST API 端点 ====================

@app.post("/v1/chat/completions")
async def chat_completions():
    """模拟 Chat Completions API（用于兼容性测试）"""
    return JSONResponse(
        status_code=501,
        content={
            "error": {
                "message": "此服务器仅支持 Realtime API，请使用 WebSocket 连接",
                "type": "not_implemented",
                "code": "realtime_only"
            }
        }
    )


@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    return {
        "object": "list",
        "data": [
            {
                "id": "gpt-4o-realtime-preview",
                "object": "model",
                "created": 1699999999,
                "owned_by": "local",
                "capabilities": {
                    "realtime": True,
                    "audio": True,
                    "text": True
                }
            },
            {
                "id": "gpt-4o-realtime-preview-2024-10-01",
                "object": "model",
                "created": 1699999999,
                "owned_by": "local",
                "capabilities": {
                    "realtime": True,
                    "audio": True,
                    "text": True
                }
            }
        ]
    }


# ==================== 错误处理 ====================

from protocol import generate_id

@app.exception_handler(Exception)
async def global_exception_handler(_request, exc):
    """全局异常处理"""
    # 生成唯一的请求 ID 用于关联日志
    request_id = generate_id("err")
    
    # 记录完整的异常和堆栈跟踪
    logger.exception(f"未处理的异常 [request_id={request_id}]: {exc}")
    
    # 返回通用错误响应，不泄露内部细节
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "internal_error",
                "code": "server_error",
                "request_id": request_id
            }
        }
    )


# ==================== 启动入口 ====================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.debug,
        log_level="debug" if config.server.debug else "info"
    )