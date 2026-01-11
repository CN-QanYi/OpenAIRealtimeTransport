"""
日志配置模块 - 提供统一的日志格式和配置
"""
import logging
import sys


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器（Windows 兼容）"""
    
    # ANSI 颜色代码
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 紫色
    }
    RESET = '\033[0m'
    
    # 模块名颜色
    MODULE_COLOR = '\033[94m'  # 蓝色
    
    def __init__(self, use_color: bool = True):
        """
        初始化格式化器
        
        Args:
            use_color: 是否使用颜色（Windows 10+ 支持）
        """
        self.use_color = use_color and self._supports_color()
        
        # 简化的格式：时间 | 级别 | 模块 | 消息
        fmt = '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
        datefmt = '%H:%M:%S'
        
        super().__init__(fmt, datefmt)
    
    @staticmethod
    def _supports_color() -> bool:
        """检测终端是否支持颜色"""
        # Windows 10+ 支持 ANSI 颜色
        if sys.platform == 'win32':
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                # 启用 ANSI 转义序列
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
                return True
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                # 捕获所有其他异常（如 AttributeError, OSError 等）
                # 在旧版 Windows 或无权限时可能失败
                return False
        return True
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录"""
        if not self.use_color:
            return super().format(record)
        
        # 保存原始级别名
        levelname_orig = record.levelname
        
        # 添加颜色
        color = self.COLORS.get(record.levelname, '')
        if color:
            record.levelname = f"{color}{record.levelname}{self.RESET}"
        
        # 模块名添加颜色
        name_orig = record.name
        record.name = f"{self.MODULE_COLOR}{record.name}{self.RESET}"
        
        # 格式化
        result = super().format(record)
        
        # 恢复原始值
        record.levelname = levelname_orig
        record.name = name_orig
        
        return result


def setup_logging(level: str = "INFO", use_color: bool = True) -> None:
    """
    配置全局日志
    
    Args:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        use_color: 是否使用彩色输出
    """
    # 转换级别字符串
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # 移除现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    # 设置格式化器
    formatter = ColoredFormatter(use_color=use_color)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    root_logger.addHandler(console_handler)
    
    # 调整第三方库日志级别（减少噪音）
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称（通常使用 __name__）
    
    Returns:
        配置好的日志记录器
    """
    return logging.getLogger(name)
