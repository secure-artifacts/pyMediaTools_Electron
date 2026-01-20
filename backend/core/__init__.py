"""
核心模块初始化 - 统一媒体工具包
包含字幕对齐、媒体转换等核心逻辑
"""

# 字幕工具
from .subtitle_utils import LANGUAGES, word_split_by, change_language, get_language
from .subtitle_utils import format_timestamp, read_text_file, read_text_with_google_doc
from .subtitle_utils import read_object_from_json, wirite_to_path
from .subtitle_alignment import audio_subtitle_search_diffent_strong

# gladia_api 需要延迟导入（因为有复杂的依赖）
try:
    from .gladia_api import transcribe_audio_from_gladia
except ImportError:
    transcribe_audio_from_gladia = None

from .srt_parse import SrtParse, timecodeToMilliseconds, millisecondsToTimecode
from .srt_to_fcpxml import SrtsToFcpxml
from .srt_to_fcpxml import SrtsToFcpxml

# 媒体转换配置
try:
    from .config import MODES
except ImportError:
    MODES = {}

__all__ = [
    # 字幕工具
    "LANGUAGES",
    "word_split_by", 
    "change_language",
    "get_language",
    "format_timestamp",
    "read_text_file",
    "read_text_with_google_doc",
    "read_object_from_json",
    "wirite_to_path",
    "audio_subtitle_search_diffent_strong",
    "transcribe_audio_from_gladia",
    "SrtParse",
    "timecodeToMilliseconds",
    "millisecondsToTimecode",
    "SrtsToFcpxml",
    # 媒体转换
    "MODES",
]