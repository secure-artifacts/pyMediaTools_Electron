#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pyMediaTools Python 后端服务
提供 REST API 供 Electron 前端调用
"""
import os
import sys
import json
import time
import tempfile
import threading
import re
import subprocess
import shutil

# Windows 控制台 UTF-8 编码支持
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    # 设置环境变量
    os.environ['PYTHONIOENCODING'] = 'utf-8'

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

# 添加核心模块路径（Windows embedded Python 可能不会自动加入脚本目录）
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
for path in (BACKEND_DIR, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

# 导入核心模块
try:
    from core.subtitle_utils import LANGUAGES, change_language, get_language, read_text_with_google_doc, read_object_from_json
    from core.subtitle_alignment import audio_subtitle_search_diffent_strong
    from core.gladia_api import transcribe_audio_from_gladia
    from core.srt_parse import SrtParse
except ImportError:
    # 如果导入失败，尝试从 pyMediaTools_Unified 导入
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'pyMediaTools_Unified'))
    from pyMediaTools.core.subtitle_utils import LANGUAGES, change_language, get_language, read_text_with_google_doc, read_object_from_json
    from pyMediaTools.core.subtitle_alignment import audio_subtitle_search_diffent_strong
    from pyMediaTools.core.gladia_api import transcribe_audio_from_gladia
    from pyMediaTools.core.srt_parse import SrtParse

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

# 全局处理 OPTIONS 预检请求
@app.before_request
def handle_preflight():
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response

# 全局状态
processing_status = {
    "is_processing": False,
    "progress": "",
    "result": None,
    "error": None
}

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({"status": "ok", "message": "Python backend is running"})

@app.route('/assets/<path:filename>', methods=['GET'])
def serve_assets(filename):
    """提供前端预览使用的资产文件"""
    assets_dir = os.path.join(os.path.dirname(__file__), '..', 'assets')
    return send_from_directory(assets_dir, filename)

# 临时文件上传目录
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route('/api/file/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    """上传文件到服务器临时目录（解决浏览器无法访问本地路径问题）"""
    if request.method == 'OPTIONS':
        return '', 204
    
    if 'file' not in request.files:
        return jsonify({"error": "没有文件"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "文件名为空"}), 400
    
    # 保存到临时目录
    import uuid
    safe_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    save_path = os.path.join(UPLOAD_DIR, safe_name)
    file.save(save_path)
    
    return jsonify({
        "success": True,
        "path": save_path,
        "name": file.filename
    })

@app.route('/api/open-folder', methods=['POST', 'OPTIONS'])
def open_folder():
    """在系统文件管理器中打开文件夹"""
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.json or {}
    folder_path = data.get('path', '')
    
    if not folder_path:
        return jsonify({"error": "缺少文件夹路径"}), 400
    
    # 展开 ~ 为用户目录
    if folder_path.startswith('~'):
        folder_path = os.path.expanduser(folder_path)
    
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        return jsonify({"error": f"文件夹不存在: {folder_path}"}), 404
    
    # 用系统命令打开文件夹
    import subprocess
    import platform
    
    try:
        if platform.system() == 'Darwin':  # macOS
            subprocess.run(['open', folder_path], check=True)
        elif platform.system() == 'Windows':
            subprocess.run(['explorer', folder_path], check=True)
        else:  # Linux
            subprocess.run(['xdg-open', folder_path], check=True)
        
        return jsonify({"success": True, "path": folder_path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/file/download', methods=['GET'])
def download_file():
    """下载后端生成的文件"""
    file_path = request.args.get('path', '')
    if not file_path:
        return jsonify({"error": "缺少文件路径"}), 400
    
    from urllib.parse import unquote
    file_path = unquote(file_path)
    
    if not os.path.exists(file_path):
        return jsonify({"error": f"文件不存在: {file_path}"}), 404
    
    # 提取原始文件名（去掉 UUID 前缀）
    filename = os.path.basename(file_path)
    if '_' in filename and len(filename.split('_')[0]) == 8:
        # 去掉 UUID 前缀
        filename = '_'.join(filename.split('_')[1:])
    
    return send_file(file_path, as_attachment=True, download_name=filename)

@app.route('/api/file/download-zip', methods=['POST', 'OPTIONS'])
def download_zip():
    """打包多个文件为 ZIP 下载"""
    if request.method == 'OPTIONS':
        return '', 204
    
    import zipfile
    import io
    from urllib.parse import unquote
    
    data = request.json
    files = data.get('files', [])
    
    if not files:
        return jsonify({"error": "缺少文件列表"}), 400
    
    # 创建内存中的 ZIP 文件
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in files:
            file_path = unquote(file_path)
            if not os.path.exists(file_path):
                continue
            
            # 去掉 UUID 前缀的文件名
            filename = os.path.basename(file_path)
            if '_' in filename and len(filename.split('_')[0]) == 8:
                filename = '_'.join(filename.split('_')[1:])
            
            zf.write(file_path, filename)
    
    zip_buffer.seek(0)
    
    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name='converted_files.zip'
    )

@app.route('/api/file/proxy', methods=['GET'])
def proxy_local_file():
    """代理本地文件访问，解决浏览器 file:// 安全限制"""
    file_path = request.args.get('path', '')
    if not file_path:
        return jsonify({"error": "缺少文件路径"}), 400
    
    # URL 解码
    from urllib.parse import unquote
    file_path = unquote(file_path)
    
    if not os.path.exists(file_path):
        return jsonify({"error": f"文件不存在: {file_path}"}), 404
    
    # 获取文件扩展名确定 MIME 类型
    ext = os.path.splitext(file_path)[1].lower()
    mime_types = {
        '.mp3': 'audio/mpeg',
        '.wav': 'audio/wav',
        '.m4a': 'audio/mp4',
        '.aac': 'audio/aac',
        '.ogg': 'audio/ogg',
        '.flac': 'audio/flac',
        '.mp4': 'video/mp4',
        '.mov': 'video/quicktime',
        '.mkv': 'video/x-matroska',
        '.avi': 'video/x-msvideo',
        '.webm': 'video/webm'
    }
    mime_type = mime_types.get(ext, 'application/octet-stream')
    
    return send_file(file_path, mimetype=mime_type)

@app.route('/api/languages', methods=['GET'])
def get_languages():
    """获取支持的语言列表"""
    languages = [{"code": k, "name": v["name"], "language": v["language"]} for k, v in LANGUAGES.items()]
    return jsonify(languages)

@app.route('/api/status', methods=['GET'])
def get_status():
    """获取处理状态"""
    return jsonify(processing_status)

@app.route('/api/subtitle/generate-with-file', methods=['POST', 'OPTIONS'])
def generate_subtitle_with_file():
    """生成字幕（支持文件上传，用于批量处理）- 同步处理"""
    if request.method == 'OPTIONS':
        return '', 204
    
    # 获取上传的文件
    if 'audio_file' not in request.files:
        return jsonify({"error": "缺少音频文件"}), 400
    
    audio_file = request.files['audio_file']
    if not audio_file.filename:
        return jsonify({"error": "文件名为空"}), 400
    
    # 保存临时文件
    temp_dir = tempfile.gettempdir()
    temp_audio_path = os.path.join(temp_dir, f"batch_{int(time.time())}_{audio_file.filename}")
    audio_file.save(temp_audio_path)
    
    # 获取其他参数
    source_text = request.form.get('source_text', '')
    translate_text = request.form.get('translate_text', '')
    language = request.form.get('language', 'en')
    audio_cut_length = float(request.form.get('audio_cut_length', 5.0))
    
    try:
        gladia_keys = json.loads(request.form.get('gladia_keys', '[]'))
    except:
        gladia_keys = []
    
    gen_merge_srt = request.form.get('gen_merge_srt', '').lower() == 'true'
    source_up_order = request.form.get('source_up_order', '').lower() == 'true'
    export_fcpxml = request.form.get('export_fcpxml', '').lower() == 'true'
    seamless_fcpxml = request.form.get('seamless_fcpxml', '').lower() == 'true'
    
    if not source_text:
        os.remove(temp_audio_path)
        return jsonify({"error": "缺少原文本"}), 400
    
    try:
        # 创建文本临时文件
        source_text_path = os.path.join(temp_dir, f"source_{int(time.time())}.txt")
        with open(source_text_path, 'w', encoding='utf-8') as f:
            f.write(source_text)
        
        translate_text_dict = {}
        translate_path = None
        if translate_text:
            translate_path = os.path.join(temp_dir, f"translate_{int(time.time())}.txt")
            with open(translate_path, 'w', encoding='utf-8') as f:
                f.write(translate_text)
            translate_text_dict["翻译文本"] = {
                "filename": "翻译文本",
                "filepath": translate_path
            }
        
        current_language = change_language(language) if language in [v["name"] for v in LANGUAGES.values()] else language
        lang_en_name = get_language(current_language)
        
        # 使用原始音频文件名（不含临时前缀）
        original_filename = audio_file.filename
        file_name = os.path.splitext(original_filename)[0]
        log_dir = "./log"
        os.makedirs(log_dir, exist_ok=True)
        
        generation_subtitle_array_path = f"./log/{current_language}_{file_name}_audio_text_whittime.json"
        generation_subtitle_text_path = f"./log/{current_language}_{file_name}_finally.txt"
        
        # 如果是 JSON 文件，直接处理
        if temp_audio_path.lower().endswith(".json"):
            with open(temp_audio_path, 'r', encoding='utf-8') as file:
                audio_json = json.load(file)
            
            if "result" in audio_json:
                transcription = audio_json["result"].get("transcription", {})
            else:
                transcription = audio_json.get("transcription", {})
            
            word_time_info = transcription.get("utterances", [])
            
            all_text = ""
            new_word_time_info = []
            
            for single in word_time_info:
                new_single = {
                    "audio_start": single["start"],
                    "audio_end": single["end"],
                    "text": single["text"],
                    "words": []
                }
                
                words = single.get("words", [])
                for word in words:
                    all_text += " " + word["word"].strip()
                    word_info = {
                        "word": word["word"].strip(),
                        "start": word["start"],
                        "end": word["end"],
                        "score": word.get("confidence", 0)
                    }
                    new_single["words"].append(word_info)
                
                new_word_time_info.append(new_single)
            
            with open(generation_subtitle_array_path, 'w', encoding='utf-8') as f:
                json.dump(new_word_time_info, f, indent=4, ensure_ascii=False)
            
            with open(generation_subtitle_text_path, 'w', encoding='utf-8') as file:
                file.write(all_text.lstrip())
                
        elif not os.path.exists(generation_subtitle_array_path):
            # 通过 Gladia 转录
            progress_generator = transcribe_audio_from_gladia(
                temp_audio_path,
                gladia_keys,
                lang_en_name,
                generation_subtitle_array_path,
                generation_subtitle_text_path,
                audio_cut_length
            )
            
            for progress in progress_generator:
                print(f"批量处理进度: {progress}")
        
        # 读取生成的数据
        if not os.path.exists(generation_subtitle_array_path) or not os.path.exists(generation_subtitle_text_path):
            raise Exception("生成文件发生错误")
        
        generation_subtitle_array = read_object_from_json(generation_subtitle_array_path)
        
        with open(generation_subtitle_text_path, 'r', encoding='utf-8') as f:
            generation_subtitle_text = f.read().strip()
        
        source_text_with_info = read_text_with_google_doc(source_text_path)
        
        # 处理翻译文本
        for k, value in translate_text_dict.items():
            translate_text_path_item = value["filepath"]
            translate_text_with_info = read_text_with_google_doc(translate_text_path_item)
            translate_text_dict[k]["translate_text_with_info"] = translate_text_with_info
            translate_text_dict[k]["trans_srt"] = ""
        
        # 创建输出文件夹（桌面/字幕输出_日期）
        import datetime
        date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = os.path.join(os.path.expanduser("~/Desktop"), f"字幕输出_{date_str}")
        os.makedirs(output_folder, exist_ok=True)
        directory = output_folder
        
        result = audio_subtitle_search_diffent_strong(
            current_language, directory, file_name,
            generation_subtitle_array, generation_subtitle_text,
            source_text_with_info, translate_text_dict,
            gen_merge_srt, source_up_order,
            export_fcpxml, seamless_fcpxml
        )
        
        # 收集生成的文件
        generated_files = []
        # 原文字幕 - 新格式: {文件名}_{语言}_source.srt
        source_srt = os.path.join(directory, f"{file_name}_{current_language}_source.srt")
        if os.path.exists(source_srt):
            generated_files.append(source_srt)
        # 译文字幕
        for k in translate_text_dict.keys():
            trans_srt = os.path.join(directory, f"{file_name}_{current_language}_{k.replace('.txt', '')}_translate.srt")
            if os.path.exists(trans_srt):
                generated_files.append(trans_srt)
        # 合并字幕
        merge_srt = os.path.join(directory, f"{file_name}_{current_language}_merge.srt")
        if os.path.exists(merge_srt):
            generated_files.append(merge_srt)
        # FCPXML
        fcpxml_path = os.path.join(directory, f"{file_name}_{current_language}.fcpxml")
        if os.path.exists(fcpxml_path):
            generated_files.append(fcpxml_path)
        
        # 清理临时文件
        try:
            os.remove(temp_audio_path)
            os.remove(source_text_path)
            if translate_path:
                os.remove(translate_path)
        except:
            pass
        
        return jsonify({
            "message": "处理完成",
            "result": result if result else "成功",
            "files": generated_files
        })
        
    except Exception as e:
        import traceback
        print(f"generate-with-file 错误: {e}")
        traceback.print_exc()
        # 清理临时文件
        try:
            os.remove(temp_audio_path)
        except:
            pass
        return jsonify({"error": str(e)}), 500

@app.route('/api/subtitle/download-zip', methods=['POST', 'OPTIONS'])
def download_subtitle_zip():
    """打包下载字幕文件"""
    if request.method == 'OPTIONS':
        return '', 204
    
    import zipfile
    from io import BytesIO
    
    data = request.json or {}
    files = data.get('files', [])
    
    if not files:
        return jsonify({"error": "没有文件"}), 400
    
    # 创建 ZIP
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in files:
            if os.path.exists(file_path):
                zf.write(file_path, os.path.basename(file_path))
    
    zip_buffer.seek(0)
    
    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'subtitles_{int(time.time())}.zip'
    )

@app.route('/api/subtitle/generate', methods=['POST'])
def generate_subtitle():
    """生成字幕"""
    global processing_status
    
    if processing_status["is_processing"]:
        return jsonify({"error": "正在处理中，请稍候"}), 400
    
    data = request.json
    
    # 必需参数
    audio_path = data.get('audio_path')
    source_text = data.get('source_text')
    language = data.get('language', 'en')
    
    # 可选参数
    translate_text = data.get('translate_text', '')
    gladia_keys = data.get('gladia_keys', [])
    audio_cut_length = data.get('audio_cut_length', 5.0)
    gen_merge_srt = data.get('gen_merge_srt', False)
    source_up_order = data.get('source_up_order', False)
    export_fcpxml = data.get('export_fcpxml', False)
    seamless_fcpxml = data.get('seamless_fcpxml', False)
    
    if not audio_path or not source_text:
        return jsonify({"error": "缺少必需参数: audio_path 和 source_text"}), 400
    
    # 在线程中处理
    def process():
        global processing_status
        processing_status = {
            "is_processing": True,
            "progress": "开始处理...",
            "result": None,
            "error": None
        }
        
        try:
            # 创建临时文件保存文本
            temp_dir = tempfile.gettempdir()
            source_text_path = os.path.join(temp_dir, "source_text.txt")
            with open(source_text_path, 'w', encoding='utf-8') as f:
                f.write(source_text)
            
            # 处理翻译文本
            translate_text_dict = {}
            if translate_text:
                translate_path = os.path.join(temp_dir, "translate_text.txt")
                with open(translate_path, 'w', encoding='utf-8') as f:
                    f.write(translate_text)
                translate_text_dict["翻译文本"] = {
                    "filename": "翻译文本",
                    "filepath": translate_path
                }
            
            current_language = change_language(language) if language in [v["name"] for v in LANGUAGES.values()] else language
            lang_en_name = get_language(current_language)
            
            file_name = os.path.splitext(os.path.basename(audio_path))[0]
            log_dir = "./log"
            os.makedirs(log_dir, exist_ok=True)
            
            generation_subtitle_array_path = f"./log/{current_language}_{file_name}_audio_text_whittime.json"
            generation_subtitle_text_path = f"./log/{current_language}_{file_name}_finally.txt"
            
            # 如果是 JSON 文件，直接处理
            if audio_path.lower().endswith(".json"):
                processing_status["progress"] = "处理JSON文件..."
                with open(audio_path, 'r', encoding='utf-8') as file:
                    audio_json = json.load(file)
                
                if "result" in audio_json:
                    transcription = audio_json["result"].get("transcription", {})
                else:
                    transcription = audio_json.get("transcription", {})
                
                word_time_info = transcription.get("utterances", [])
                
                all_text = ""
                new_word_time_info = []
                
                for single in word_time_info:
                    new_single = {
                        "audio_start": single["start"],
                        "audio_end": single["end"],
                        "text": single["text"],
                        "words": []
                    }
                    
                    words = single.get("words", [])
                    for word in words:
                        all_text += " " + word["word"].strip()
                        word_info = {
                            "word": word["word"].strip(),
                            "start": word["start"],
                            "end": word["end"],
                            "score": word.get("confidence", 0)
                        }
                        new_single["words"].append(word_info)
                    
                    new_word_time_info.append(new_single)
                
                with open(generation_subtitle_array_path, 'w', encoding='utf-8') as f:
                    json.dump(new_word_time_info, f, indent=4, ensure_ascii=False)
                
                with open(generation_subtitle_text_path, 'w', encoding='utf-8') as file:
                    file.write(all_text.lstrip())
                    
            elif not os.path.exists(generation_subtitle_array_path):
                # 通过 Gladia 转录
                processing_status["progress"] = "通过 Gladia 转录音频..."
                
                progress_generator = transcribe_audio_from_gladia(
                    audio_path,
                    gladia_keys,
                    lang_en_name,
                    generation_subtitle_array_path,
                    generation_subtitle_text_path,
                    audio_cut_length
                )
                
                for progress in progress_generator:
                    processing_status["progress"] = str(progress)
            
            # 读取生成的数据
            if not os.path.exists(generation_subtitle_array_path) or not os.path.exists(generation_subtitle_text_path):
                raise Exception("生成文件发生错误")
            
            generation_subtitle_array = read_object_from_json(generation_subtitle_array_path)
            
            with open(generation_subtitle_text_path, 'r', encoding='utf-8') as f:
                generation_subtitle_text = f.read().strip()
            
            source_text_with_info = read_text_with_google_doc(source_text_path)
            
            # 处理翻译文本
            for k, value in translate_text_dict.items():
                translate_text_path = value["filepath"]
                translate_text_with_info = read_text_with_google_doc(translate_text_path)
                translate_text_dict[k]["translate_text_with_info"] = translate_text_with_info
                translate_text_dict[k]["trans_srt"] = ""
            
            # 执行对齐
            directory = os.path.dirname(audio_path)
            result = audio_subtitle_search_diffent_strong(
                current_language, directory, file_name,
                generation_subtitle_array, generation_subtitle_text,
                source_text_with_info, translate_text_dict,
                gen_merge_srt, source_up_order,
                export_fcpxml, seamless_fcpxml
            )
            
            processing_status["progress"] = "完成"
            processing_status["result"] = str(result)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            processing_status["error"] = str(e)
        finally:
            processing_status["is_processing"] = False
    
    thread = threading.Thread(target=process)
    thread.start()
    
    return jsonify({"message": "开始处理", "status": "processing"})

@app.route('/api/srt/adjust', methods=['POST'])
def adjust_srt():
    """调整 SRT 时间"""
    data = request.json
    
    src_path = data.get('src_path')
    interval_time = data.get('interval_time', 1.0)
    char_time = data.get('char_time', 0.1)
    min_char_count = data.get('min_char_count', 20)
    scale = data.get('scale', 1.0)
    ignore = data.get('ignore', '?—:„";/!')
    
    if not src_path:
        return jsonify({"error": "缺少必需参数: src_path"}), 400
    
    try:
        srt_info = SrtParse(src_path, ignore)
        srt_info.updateSrt(interval_time, char_time, min_char_count, scale)
        
        new_path = src_path.replace('.srt', '_new.srt')
        srt_info.write(new_path)
        
        return jsonify({
            "message": "调整完成",
            "output_path": new_path
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/settings/gladia-keys', methods=['GET', 'POST', 'OPTIONS'])
def gladia_keys():
    """管理 Gladia API Keys"""
    if request.method == 'OPTIONS':
        return '', 204
    
    keys_file = os.path.join(os.path.dirname(__file__), 'gladia_keys.json')
    
    if request.method == 'GET':
        if os.path.exists(keys_file):
            with open(keys_file, 'r') as f:
                return jsonify(json.load(f))
        return jsonify({"keys": []})
    
    elif request.method == 'POST':
        data = request.json
        with open(keys_file, 'w') as f:
            json.dump(data, f)
        return jsonify({"message": "保存成功"})

@app.route('/api/settings/elevenlabs', methods=['GET', 'POST', 'OPTIONS'])
def elevenlabs_settings():
    """管理 ElevenLabs API Keys"""
    if request.method == 'OPTIONS':
        return '', 204
    
    settings_file = os.path.join(os.path.dirname(__file__), 'elevenlabs_settings.json')
    
    if request.method == 'GET':
        data = {}
        if os.path.exists(settings_file):
            with open(settings_file, 'r') as f:
                data = json.load(f)

        keys = data.get('api_keys') or []
        if isinstance(keys, str):
            keys = [keys]
        if not keys:
            single_key = data.get('api_key', '')
            if single_key:
                keys = [single_key]

        keys = [k.strip() for k in keys if isinstance(k, str) and k.strip()]
        return jsonify({
            "api_key": keys[0] if keys else "",
            "api_keys": keys
        })
    
    elif request.method == 'POST':
        data = request.json or {}
        keys = data.get('api_keys') or []
        if isinstance(keys, str):
            keys = [keys]
        if not keys:
            single_key = data.get('api_key', '')
            if single_key:
                keys = [single_key]

        keys = [k.strip() for k in keys if isinstance(k, str) and k.strip()]
        payload = {
            "api_key": keys[0] if keys else "",
            "api_keys": keys
        }
        with open(settings_file, 'w') as f:
            json.dump(payload, f)
        return jsonify({"message": "保存成功"})

@app.route('/api/settings/elevenlabs/keys', methods=['GET', 'POST', 'DELETE', 'PUT', 'OPTIONS'])
def manage_elevenlabs_keys():
    """管理 ElevenLabs API Keys（带状态）"""
    if request.method == 'OPTIONS':
        return '', 204
    
    settings_file = os.path.join(os.path.dirname(__file__), 'elevenlabs_settings.json')
    
    def load_data():
        if os.path.exists(settings_file):
            with open(settings_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_data(data):
        with open(settings_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    if request.method == 'GET':
        # 获取所有 key（包含停用的）
        keys_data = _load_elevenlabs_keys(include_disabled=True)
        return jsonify({"keys": keys_data})
    
    elif request.method == 'POST':
        # 添加新 key
        data = request.json or {}
        new_key = data.get('key', '').strip()
        if not new_key:
            return jsonify({"error": "Key 不能为空"}), 400
        
        file_data = load_data()
        keys_data = file_data.get('keys_with_status') or []
        
        # 检查是否已存在
        for item in keys_data:
            if item.get('key') == new_key:
                return jsonify({"error": "Key 已存在"}), 400
        
        keys_data.append({
            "key": new_key,
            "enabled": True,
            "manual_disabled": False,
            "auto_disabled": False,
            "auto_disabled_reason": ""
        })
        file_data['keys_with_status'] = keys_data
        save_data(file_data)
        return jsonify({"message": "添加成功"})
    
    elif request.method == 'DELETE':
        # 删除 key
        data = request.json or {}
        index = data.get('index')
        if index is None:
            return jsonify({"error": "缺少 index"}), 400
        
        file_data = load_data()
        keys_data = file_data.get('keys_with_status') or []
        
        if 0 <= index < len(keys_data):
            keys_data.pop(index)
            file_data['keys_with_status'] = keys_data
            save_data(file_data)
            return jsonify({"message": "删除成功"})
        else:
            return jsonify({"error": "索引无效"}), 400
    
    elif request.method == 'PUT':
        # 更新 key 状态（启用/停用）或调整顺序
        data = request.json or {}
        action = data.get('action')
        
        file_data = load_data()
        # 使用统一的加载函数，确保兼容旧格式
        keys_data = _load_elevenlabs_keys(include_disabled=True)
        if not keys_data:
            return jsonify({"error": "没有 API Key"}), 400
        
        if action == 'toggle':
            index = data.get('index')
            if index is not None and 0 <= index < len(keys_data):
                new_enabled = not keys_data[index].get('enabled', True)
                keys_data[index]['enabled'] = new_enabled
                # 手动操作优先级最高，避免被自动恢复/自动停用逻辑覆盖
                keys_data[index]['manual_disabled'] = (not new_enabled)
                if new_enabled:
                    keys_data[index]['auto_disabled'] = False
                    keys_data[index]['auto_disabled_reason'] = ""
                file_data['keys_with_status'] = keys_data
                save_data(file_data)
                return jsonify({"message": "状态已更新", "enabled": keys_data[index]['enabled']})
        
        elif action == 'move':
            from_idx = data.get('from')
            to_idx = data.get('to')
            if from_idx is not None and to_idx is not None:
                if 0 <= from_idx < len(keys_data) and 0 <= to_idx < len(keys_data):
                    item = keys_data.pop(from_idx)
                    keys_data.insert(to_idx, item)
                    file_data['keys_with_status'] = keys_data
                    save_data(file_data)
                    return jsonify({"message": "顺序已更新"})
        
        elif action == 'reorder':
            # 完整重排序
            new_order = data.get('keys')
            if new_order:
                file_data['keys_with_status'] = new_order
                save_data(file_data)
                return jsonify({"message": "顺序已更新"})
        
        return jsonify({"error": "无效操作"}), 400


@app.route('/api/settings/replace-rules', methods=['GET', 'POST', 'OPTIONS'])
def replace_rules():
    """管理翻译替换规则"""
    if request.method == 'OPTIONS':
        return '', 204
    
    rules_file = os.path.join(os.path.dirname(__file__), 'replace_rules.json')
    
    if request.method == 'GET':
        if os.path.exists(rules_file):
            with open(rules_file, 'r') as f:
                return jsonify(json.load(f))
        return jsonify({"rules": {}})
    
    elif request.method == 'POST':
        data = request.json
        with open(rules_file, 'w') as f:
            json.dump(data, f)
        return jsonify({"message": "保存成功"})

def _load_elevenlabs_keys(include_disabled=False):
    """加载 API Keys，默认只返回启用的"""
    settings_file = os.path.join(os.path.dirname(__file__), 'elevenlabs_settings.json')
    data = {}
    if os.path.exists(settings_file):
        with open(settings_file, 'r') as f:
            data = json.load(f)

    # 新格式：带状态的 key 列表
    keys_data = data.get('keys_with_status') or []
    
    # 兼容旧格式
    if not keys_data:
        old_keys = data.get('api_keys') or []
        if isinstance(old_keys, str):
            old_keys = [old_keys]
        if not old_keys:
            single_key = data.get('api_key', '')
            if single_key:
                old_keys = [single_key]
        # 转换为新格式
        keys_data = [{"key": k.strip(), "enabled": True} for k in old_keys if isinstance(k, str) and k.strip()]
    
    if include_disabled:
        return keys_data
    
    # 只返回启用的 key
    return [item["key"] for item in keys_data if item.get("enabled", True) and item.get("key")]

def _select_elevenlabs_key(keys, key_index=None, rotate_index=None):
    if not keys:
        return ''

    idx = None
    if key_index is not None and str(key_index).strip() != '':
        try:
            idx = int(key_index)
        except (TypeError, ValueError):
            raise ValueError("API Key 索引无效")
        if idx >= 1:
            idx -= 1
    elif rotate_index is not None:
        idx = rotate_index % len(keys)
    else:
        idx = 0

    if idx < 0 or idx >= len(keys):
        raise ValueError("API Key 索引超出范围")

    return keys[idx]

def _set_elevenlabs_key_enabled(api_key, enabled, reason="", source="auto"):
    """按 key 值更新启用状态，返回是否发生变更。
    source:
      - auto: 由后端自动逻辑触发（会记录 auto_disabled）
      - manual: 人工触发（会写入 manual_disabled）
    """
    if not api_key:
        return False

    settings_file = os.path.join(os.path.dirname(__file__), 'elevenlabs_settings.json')
    data = {}
    if os.path.exists(settings_file):
        with open(settings_file, 'r') as f:
            data = json.load(f)

    keys_data = data.get('keys_with_status') or []
    if not keys_data:
        # 兼容旧格式并升级到新格式
        old_keys = data.get('api_keys') or []
        if isinstance(old_keys, str):
            old_keys = [old_keys]
        if not old_keys:
            single_key = data.get('api_key', '')
            if single_key:
                old_keys = [single_key]
        keys_data = [{
            "key": k.strip(),
            "enabled": True,
            "manual_disabled": False,
            "auto_disabled": False,
            "auto_disabled_reason": ""
        } for k in old_keys if isinstance(k, str) and k.strip()]

    changed = False
    for item in keys_data:
        if item.get('key') == api_key:
            # 兼容旧数据结构
            if 'manual_disabled' not in item:
                item['manual_disabled'] = False
                changed = True
            if 'auto_disabled' not in item:
                item['auto_disabled'] = False
                changed = True
            if 'auto_disabled_reason' not in item:
                item['auto_disabled_reason'] = ""
                changed = True

            # 手动停用状态下，自动恢复请求应被忽略
            if source == "auto" and enabled and item.get('manual_disabled', False):
                break

            current_enabled = item.get('enabled', True)
            if current_enabled != enabled:
                item['enabled'] = enabled
                changed = True

            if source == "manual":
                manual_disabled = (not enabled)
                if item.get('manual_disabled') != manual_disabled:
                    item['manual_disabled'] = manual_disabled
                    changed = True
                if enabled:
                    if item.get('auto_disabled', False):
                        item['auto_disabled'] = False
                        changed = True
                    if item.get('auto_disabled_reason'):
                        item['auto_disabled_reason'] = ""
                        changed = True
            else:
                # auto source
                if not enabled:
                    if item.get('auto_disabled') != True:
                        item['auto_disabled'] = True
                        changed = True
                    if reason and item.get('auto_disabled_reason') != reason:
                        item['auto_disabled_reason'] = reason
                        changed = True
                else:
                    if item.get('auto_disabled', False):
                        item['auto_disabled'] = False
                        changed = True
                    if item.get('auto_disabled_reason'):
                        item['auto_disabled_reason'] = ""
                        changed = True
            break

    if changed:
        data['keys_with_status'] = keys_data
        with open(settings_file, 'w') as f:
            json.dump(data, f, indent=2)
        action = "停用" if not enabled else "启用"
        key_prefix = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) >= 12 else api_key
        if reason:
            print(f"[ElevenLabs] 已自动{action} Key {key_prefix}，原因: {reason}")
        else:
            print(f"[ElevenLabs] 已自动{action} Key {key_prefix}")

    return changed

def _is_elevenlabs_key_retryable_error(error_message):
    """判断该错误是否应自动切换到下一个 Key。"""
    msg = (error_message or "").lower()
    retry_tokens = [
        "api 错误[401]",
        "api 错误[403]",
        "api 错误[429]",
        "api 错误[422]",
        "quota_exceeded",
        "insufficient",
        "character_limit",
        "credit",
        "api 错误: 401",
        "api 错误: 403",
        "api 错误: 429",
        "status code: 401",
        "status code: 403",
        "status code: 429",
        "unauthorized",
        "forbidden",
        "invalid api key",
        "invalid_api_key",
        "too many requests",
        "detected_unusual_activity",
        "unusual_activity",
        "subscription",
        "plan",
        "permission",
        "not available for your",
        "not allowed",
        "model_not_available",
        "model_not_supported",
        "voice_not_found",
        "voice not found",
        "you do not have access to this voice",
        "does not have access",
        "unsupported model",
        "feature_not_available",
    ]
    return any(token in msg for token in retry_tokens)

def _should_auto_disable_elevenlabs_key(error_message):
    """判断该错误是否应把当前 Key 自动标记为停用。"""
    msg = (error_message or "").lower()
    disable_tokens = [
        "quota_exceeded",
        "insufficient",
        "character_limit",
        "credit",
        "insufficient characters",
        "api 错误[401]",
        "api 错误[403]",
        "api 错误: 401",
        "api 错误: 403",
        "status code: 401",
        "status code: 403",
        "unauthorized",
        "forbidden",
        "invalid api key",
        "invalid_api_key",
        "account_suspended",
        "account_disabled",
    ]
    return any(token in msg for token in disable_tokens)

def _parse_elevenlabs_error(response):
    """统一解析 ElevenLabs 错误结构，返回 (message, detail_status, detail_code, http_status)。"""
    message = response.text
    detail_status = ""
    detail_code = ""
    http_status = response.status_code
    try:
        error_data = response.json()
        detail = error_data.get("detail")
        if isinstance(detail, dict):
            detail_status = str(detail.get("status") or "")
            detail_code = str(detail.get("code") or "")
            message = detail.get("message", message)
        elif isinstance(detail, str) and detail:
            message = detail
    except Exception:
        pass
    return message, detail_status, detail_code, http_status

def _request_elevenlabs_tts_with_rotation(keys, voice_id, text, model_id, stability, output_format, key_index=None):
    """按优先 Key + 自动轮换策略请求 TTS，成功返回 (audio_bytes, used_key)。"""
    if not keys:
        raise RuntimeError("未配置 API Key")

    preferred_key = None
    if key_index is not None and str(key_index).strip() != '':
        preferred_key = _select_elevenlabs_key(keys, key_index)

    if preferred_key:
        keys_to_try = [preferred_key] + [k for k in keys if k != preferred_key]
    else:
        keys_to_try = list(keys)

    last_err = None
    for api_key in keys_to_try:
        try:
            audio_bytes = _request_elevenlabs_tts(
                api_key, voice_id, text, model_id, stability, output_format
            )
            return audio_bytes, api_key
        except Exception as exc:
            err_msg = str(exc)
            last_err = exc
            # 非 Key 层错误（如 voice_id 无效）直接抛出，不继续轮换
            if not _is_elevenlabs_key_retryable_error(err_msg):
                raise

            # 对明确不可用的 Key 自动停用，避免后续继续打到它
            if _should_auto_disable_elevenlabs_key(err_msg):
                try:
                    _set_elevenlabs_key_enabled(api_key, False, err_msg)
                except Exception as disable_exc:
                    print(f"[ElevenLabs] 自动停用 Key 失败: {disable_exc}")
            continue

    if last_err is not None:
        raise RuntimeError(f"所有可用 Key 均尝试失败，最后错误: {last_err}")
    raise RuntimeError("所有可用 Key 均尝试失败")

def _build_tts_save_path(text, output_format, tag, seq_prefix=""):
    import datetime
    import uuid

    desktop = os.path.expanduser("~/Desktop")
    ext = "mp3" if "mp3" in output_format else "wav"
    prefix = _build_text_prefix(text)
    name_prefix = f"{prefix}_" if prefix else ""
    return os.path.join(desktop, f"{seq_prefix}{name_prefix}{tag}_{datetime.date.today()}_{str(uuid.uuid4())[:4]}.{ext}")

def _delete_oldest_custom_voice(api_key):
    """删除最旧的自定义音色，返回被删除的音色信息"""
    import requests
    
    headers = {"xi-api-key": api_key}
    
    # 获取音色列表
    response = requests.get("https://api.elevenlabs.io/v1/voices", headers=headers, timeout=15)
    if response.status_code != 200:
        return None
    
    voices = response.json().get("voices", [])
    
    # 筛选自定义音色（category 为 'cloned', 'generated' 或 'professional'，这些都占用自定义音色位置）
    custom_voices = [v for v in voices if v.get("category") in ["cloned", "generated", "professional"]]
    
    if not custom_voices:
        return None
    
    # 按创建时间排序，删除最旧的
    # 如果没有 created_at 字段，就删除第一个
    oldest = custom_voices[0]
    
    # 删除
    delete_response = requests.delete(
        f"https://api.elevenlabs.io/v1/voices/{oldest['voice_id']}",
        headers=headers,
        timeout=15
    )
    
    if delete_response.status_code == 200:
        return oldest
    return None


def _request_elevenlabs_tts(api_key, voice_id, text, model_id, stability, output_format, auto_delete_on_limit=True):
    import requests
    import time

    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": 0.75
        }
    }

    def do_request():
        return requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format={output_format}",
            headers=headers,
            json=payload
        )

    response = do_request()

    if response.status_code != 200:
        error_msg, detail_status, detail_code, http_status = _parse_elevenlabs_error(response)
        
        # 检测是否是音色数量限制错误
        merged_error = f"{error_msg} {detail_status} {detail_code}".lower()
        is_limit_error = (
            "maximum amount of custom voices" in merged_error
            or "voice_limit" in merged_error
            or detail_status == "voice_limit_reached"
        )
        
        if is_limit_error and auto_delete_on_limit:
            print(f"[TTS自动删除] 检测到音色数量限制，尝试删除最旧的音色...")
            deleted_voice = _delete_oldest_custom_voice(api_key)
            
            if deleted_voice:
                deleted_name = deleted_voice.get('name', '未知')
                print(f"[TTS自动删除] 已删除音色: {deleted_name}")
                
                # 等待 API 同步
                time.sleep(1)
                
                # 重试请求
                retry_response = do_request()
                
                if retry_response.status_code == 200:
                    print(f"[TTS自动删除] 重试成功！")
                    return retry_response.content
                else:
                    # 重试失败，返回新的结构化错误
                    retry_error, retry_status, retry_code, retry_http = _parse_elevenlabs_error(retry_response)
                    raise RuntimeError(
                        f"API 错误[{retry_http}][{retry_status or '-'}][{retry_code or '-'}] "
                        f"(已自动删除音色「{deleted_name}」但仍失败): {retry_error}"
                    )
            else:
                raise RuntimeError(
                    f"API 错误[{http_status}][{detail_status or '-'}][{detail_code or '-'}]: {error_msg} "
                    f"(尝试自动删除音色失败，可能没有可删除的自定义音色)"
                )
        
        raise RuntimeError(f"API 错误[{http_status}][{detail_status or '-'}][{detail_code or '-'}]: {error_msg}")

    return response.content

@app.route('/api/elevenlabs/voices', methods=['GET', 'OPTIONS'])
def get_elevenlabs_voices():
    """获取 ElevenLabs 语音列表 - 包括用户声音和热门社区声音"""
    if request.method == 'OPTIONS':
        return '', 204

    keys = _load_elevenlabs_keys()
    if not keys:
        return jsonify({"voices": [], "error": "未配置 API Key"})

    try:
        api_key = _select_elevenlabs_key(keys, request.args.get('key_index'))
    except ValueError as exc:
        return jsonify({"voices": [], "error": str(exc)})
    
    voices_list = []
    
    try:
        import requests
        headers = {"xi-api-key": api_key, "Accept": "application/json"}
        
        # 1. 获取用户自己的声音 (克隆的 + 官方预设)
        response = requests.get("https://api.elevenlabs.io/v1/voices", headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            for v in data.get("voices", []):
                voice_id = v.get("voice_id")
                name = v.get("name", "Unknown")
                preview_url = v.get("preview_url", "")
                category = v.get("category", "premade")
                
                # 可删除的类型: cloned, generated, professional
                can_delete = category in ["cloned", "generated", "professional"]
                
                # 标注来源
                if category == "cloned":
                    display_name = f"[克隆] {name}"
                elif category == "generated":
                    display_name = f"[生成] {name}"
                elif category == "professional":
                    display_name = f"[专业] {name}"
                else:
                    display_name = f"[官方] {name}"
                
                if voice_id:
                    voices_list.append({
                        "voice_id": voice_id,
                        "name": display_name,
                        "preview_url": preview_url,
                        "can_delete": can_delete,
                        "category": category
                    })
        
        # 注意：社区声音 (shared-voices) 功能已禁用
        # 原因：ElevenLabs 免费用户无法通过 API 使用库中的声音
        # 如需启用，请升级到 Starter 或更高套餐
        # 
        # 原代码已注释：
        # try:
        #     shared_response = requests.get(
        #         "https://api.elevenlabs.io/v1/shared-voices",
        #         headers=headers,
        #         params={"page_size": 50, "sort": "trending"},
        #         timeout=15
        #     )
        #     ...
        # except Exception as e:
        #     print(f"获取社区声音失败: {e}")
        
        return jsonify({"voices": voices_list})
    except Exception as e:
        return jsonify({"voices": [], "error": str(e)})

@app.route('/api/elevenlabs/search', methods=['POST', 'OPTIONS'])
def search_elevenlabs_voices():
    """搜索 ElevenLabs 声音库"""
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.json or {}
    search_term = data.get('search_term', '')
    
    if not search_term:
        return jsonify({"error": "缺少搜索关键词"}), 400
    
    keys = _load_elevenlabs_keys()
    if not keys:
        return jsonify({"voices": [], "error": "未配置 API Key"})

    try:
        api_key = _select_elevenlabs_key(keys, data.get('key_index'))
    except ValueError as exc:
        return jsonify({"voices": [], "error": str(exc)})
    
    try:
        import requests
        headers = {"xi-api-key": api_key}
        # 搜索共享声音库，增加到50个结果
        response = requests.get(
            f"https://api.elevenlabs.io/v1/shared-voices?search={search_term}&page_size=50",
            headers=headers,
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            voices = [{
                "voice_id": v.get("voice_id") or v.get("public_owner_id"), 
                "name": v["name"],
                "preview_url": v.get("preview_url", ""),
                "public_owner_id": v.get("public_owner_id", v.get("voice_id"))
            } for v in data.get("voices", [])]
            return jsonify({"voices": voices})
        else:
            return jsonify({"voices": [], "error": f"API 错误: {response.status_code}"})
    except Exception as e:
        return jsonify({"voices": [], "error": str(e)})

@app.route('/api/elevenlabs/add-voice', methods=['POST', 'OPTIONS'])
def add_elevenlabs_voice():
    """将社区声音添加到用户的库"""
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.json or {}
    public_voice_id = data.get('public_voice_id', '')
    name = data.get('name', 'My Voice')
    auto_delete = data.get('auto_delete', True)  # 默认启用自动删除
    
    if not public_voice_id:
        return jsonify({"error": "缺少 public_voice_id"}), 400
    
    keys = _load_elevenlabs_keys()
    if not keys:
        return jsonify({"error": "未配置 API Key"}), 400

    try:
        api_key = _select_elevenlabs_key(keys, data.get('key_index'))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    
    def try_add_voice():
        """尝试添加音色"""
        import requests
        headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
        response = requests.post(
            f"https://api.elevenlabs.io/v1/voices/add/{public_voice_id}",
            headers=headers,
            json={"new_name": name},
            timeout=15
        )
        return response
    
    try:
        import requests
        import time
        
        response = try_add_voice()
        
        if response.status_code == 200:
            result = response.json()
            new_voice_id = result.get("voice_id", public_voice_id)
            return jsonify({"success": True, "voice_id": new_voice_id, "name": name})
        else:
            # 检测是否是声音数量限制错误
            is_limit_error = False
            error_message = ""
            try:
                error_data = response.json()
                detail = error_data.get("detail", {})
                if isinstance(detail, dict) and detail.get("status") == "voice_limit_reached":
                    is_limit_error = True
                    error_message = detail.get("message", "已达到声音数量上限")
                # 也检查错误信息文本
                elif "maximum amount of custom voices" in response.text:
                    is_limit_error = True
                    error_message = "已达到声音数量上限"
            except:
                if "maximum amount of custom voices" in response.text:
                    is_limit_error = True
                    error_message = "已达到声音数量上限"
            
            # 如果是数量限制且启用自动删除
            if is_limit_error and auto_delete:
                print(f"[自动删除] 检测到音色数量限制，尝试删除最旧的音色...")
                deleted_voice = _delete_oldest_custom_voice(api_key)
                
                if deleted_voice:
                    deleted_name = deleted_voice.get('name', '未知')
                    print(f"[自动删除] 已删除音色: {deleted_name}")
                    
                    # 等待 API 同步
                    time.sleep(1)
                    
                    # 重试添加
                    retry_response = try_add_voice()
                    
                    if retry_response.status_code == 200:
                        result = retry_response.json()
                        new_voice_id = result.get("voice_id", public_voice_id)
                        return jsonify({
                            "success": True, 
                            "voice_id": new_voice_id, 
                            "name": name,
                            "auto_deleted": deleted_name,
                            "message": f"已自动删除旧音色「{deleted_name}」并成功添加新音色"
                        })
                    else:
                        return jsonify({
                            "error": f"自动删除后仍然添加失败: {retry_response.text}",
                            "auto_deleted": deleted_name
                        }), retry_response.status_code
                else:
                    return jsonify({
                        "error": "voice_limit_reached",
                        "message": f"{error_message}，且无法自动删除（可能没有可删除的自定义音色）"
                    }), 400
            
            # 不自动删除或不是限制错误，返回原始错误
            if is_limit_error:
                return jsonify({
                    "error": "voice_limit_reached",
                    "message": error_message
                }), 400
                
            return jsonify({"error": f"添加失败: {response.text}"}), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/elevenlabs/delete-voice', methods=['POST', 'OPTIONS'])
def delete_elevenlabs_voice():
    """删除用户的自定义声音"""
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.json or {}
    voice_id = data.get('voice_id', '')
    
    if not voice_id:
        return jsonify({"error": "缺少 voice_id"}), 400
    
    keys = _load_elevenlabs_keys()
    if not keys:
        return jsonify({"error": "未配置 API Key"}), 400

    try:
        api_key = _select_elevenlabs_key(keys, data.get('key_index'))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    
    try:
        import requests
        headers = {"xi-api-key": api_key}
        response = requests.delete(
            f"https://api.elevenlabs.io/v1/voices/{voice_id}",
            headers=headers,
            timeout=15
        )
        
        if response.status_code == 200:
            return jsonify({"success": True, "voice_id": voice_id})
        else:
            return jsonify({"error": f"删除失败: {response.text}"}), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/elevenlabs/quota', methods=['GET', 'OPTIONS'])
def get_elevenlabs_quota():
    """获取 ElevenLabs 额度"""
    if request.method == 'OPTIONS':
        return '', 204
    
    keys = _load_elevenlabs_keys()
    if not keys:
        return jsonify({"usage": -1, "limit": -1, "error": "未配置 API Key"})

    try:
        api_key = _select_elevenlabs_key(keys, request.args.get('key_index'))
    except ValueError as exc:
        return jsonify({"usage": -1, "limit": -1, "error": str(exc)})
    
    try:
        import requests
        headers = {"xi-api-key": api_key}
        response = requests.get("https://api.elevenlabs.io/v1/user/subscription", headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            usage = data.get("character_count", 0)
            limit = data.get("character_limit", 0)
            return jsonify({"usage": usage, "limit": limit})
        else:
            return jsonify({"usage": -1, "limit": -1, "error": f"API 错误: {response.status_code}"})
    except Exception as e:
        return jsonify({"usage": -1, "limit": -1, "error": str(e)})

@app.route('/api/elevenlabs/all-quotas', methods=['GET', 'OPTIONS'])
def get_all_elevenlabs_quotas():
    """获取所有 API Key 的额度（包括停用的）"""
    if request.method == 'OPTIONS':
        return '', 204
    
    # 获取所有 key（包括停用的）
    keys_data = _load_elevenlabs_keys(include_disabled=True)
    if not keys_data:
        return jsonify({"keys": [], "error": "未配置 API Key"})
    
    import requests
    results = []
    settings_file = os.path.join(os.path.dirname(__file__), 'elevenlabs_settings.json')
    keys_changed = False
    
    for i, key_info in enumerate(keys_data):
        key = key_info.get('key', '') if isinstance(key_info, dict) else key_info
        enabled = key_info.get('enabled', True) if isinstance(key_info, dict) else True
        manual_disabled = key_info.get('manual_disabled', False) if isinstance(key_info, dict) else False
        auto_disabled = key_info.get('auto_disabled', False) if isinstance(key_info, dict) else False
        
        if not key:
            continue
            
        try:
            headers = {"xi-api-key": key}
            response = requests.get("https://api.elevenlabs.io/v1/user/subscription", headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                usage = data.get("character_count", 0)
                limit = data.get("character_limit", 0)
                remaining = limit - usage
                
                # 自动停用余额不足 200 的 key（仅影响非手动停用项）
                if remaining < 200 and enabled and not manual_disabled:
                    keys_data[i]['enabled'] = False
                    keys_data[i]['auto_disabled'] = True
                    keys_data[i]['auto_disabled_reason'] = f"remaining<{200}"
                    keys_changed = True
                    enabled = False
                    auto_disabled = True
                # 自动恢复：如果此前是自动停用且余额恢复，则自动启用
                elif remaining >= 200 and (not enabled) and auto_disabled and not manual_disabled:
                    keys_data[i]['enabled'] = True
                    keys_data[i]['auto_disabled'] = False
                    keys_data[i]['auto_disabled_reason'] = ""
                    keys_changed = True
                    enabled = True
                    auto_disabled = False
                
                results.append({
                    "index": i + 1,
                    "key_prefix": key[:8] + "..." + key[-4:],
                    "usage": usage,
                    "limit": limit,
                    "remaining": remaining,
                    "percent": round(usage / limit * 100, 1) if limit > 0 else 0,
                    "enabled": enabled,
                    "manual_disabled": manual_disabled,
                    "auto_disabled": auto_disabled
                })
            else:
                results.append({
                    "index": i + 1,
                    "key_prefix": key[:8] + "..." + key[-4:],
                    "error": f"API 错误: {response.status_code}",
                    "enabled": enabled,
                    "manual_disabled": manual_disabled,
                    "auto_disabled": auto_disabled
                })
        except Exception as e:
            results.append({
                "index": i + 1,
                "key_prefix": key[:8] + "..." + key[-4:],
                "error": str(e),
                "enabled": enabled,
                "manual_disabled": manual_disabled,
                "auto_disabled": auto_disabled
            })
    
    # 保存自动停用的更改
    if keys_changed:
        try:
            with open(settings_file, 'r') as f:
                file_data = json.load(f)
            file_data['keys_with_status'] = keys_data
            with open(settings_file, 'w') as f:
                json.dump(file_data, f, indent=2)
        except:
            pass
    
    return jsonify({"keys": results})

@app.route('/api/elevenlabs/tts', methods=['POST', 'OPTIONS'])
def elevenlabs_tts():
    """ElevenLabs 文本转语音"""
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.json or {}
    text = data.get('text', '')
    voice_id = data.get('voice_id', '')
    model_id = data.get('model_id', 'eleven_multilingual_v2')
    stability = data.get('stability', 0.5)
    output_format = data.get('output_format', 'mp3_44100_128')
    save_path = data.get('save_path', '')
    
    if not text or not voice_id:
        return jsonify({"error": "缺少必需参数"}), 400

    keys = _load_elevenlabs_keys()
    if not keys:
        return jsonify({"error": "未配置 API Key"}), 400

    try:
        stability_val = float(stability)
        if stability_val > 1:
            stability_val = stability_val / 100.0
        stability_val = max(0.0, min(1.0, stability_val))

        audio_bytes, used_key = _request_elevenlabs_tts_with_rotation(
            keys,
            voice_id,
            text,
            model_id,
            stability_val,
            output_format,
            key_index=data.get('key_index')
        )
        used_prefix = f"{used_key[:8]}...{used_key[-4:]}" if len(used_key) >= 12 else used_key
        print(f"[ElevenLabs TTS] 使用 Key: {used_prefix}")

        if not save_path:
            save_path = _build_tts_save_path(text, output_format, "tts")

        with open(save_path, 'wb') as f:
            f.write(audio_bytes)

        return jsonify({
            "message": "生成成功",
            "file_path": save_path
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/elevenlabs/tts-batch', methods=['POST', 'OPTIONS'])
def elevenlabs_tts_batch():
    """ElevenLabs 批量文本转语音"""
    if request.method == 'OPTIONS':
        return '', 204

    data = request.json or {}
    items = data.get('items', [])
    default_model = data.get('default_model_id', 'eleven_multilingual_v2')
    default_stability = data.get('default_stability', 0.5)
    default_output_format = data.get('output_format', 'mp3_44100_128')
    enable_circuit_breaker = data.get('enable_circuit_breaker', False)  # 风控熔断开关

    if not isinstance(items, list) or len(items) == 0:
        return jsonify({"error": "缺少批量任务 items"}), 400

    all_keys = _load_elevenlabs_keys()
    if not all_keys:
        return jsonify({"error": "未配置 API Key"}), 400

    results = [None] * len(items)  # 预分配结果数组
    success = 0
    failed_tasks = []  # 失败任务队列
    exhausted_keys = set()  # 已用尽的 Key
    circuit_breaker_triggered = False

    import time

    def try_generate(idx, item, preferred_key=None):
        """尝试生成音频，失败时返回错误类型"""
        nonlocal success, circuit_breaker_triggered
        
        text = item.get('text', '')
        voice_id = item.get('voice_id', '')
        model_id = item.get('model_id') or default_model
        stability = item.get('stability', default_stability)
        output_format = item.get('output_format', default_output_format)
        save_path = item.get('save_path', '')
        
        if not text or not voice_id:
            return {"index": idx, "error": "缺少 text 或 voice_id"}, "invalid"
        
        try:
            stability_val = float(stability)
            if stability_val > 1:
                stability_val = stability_val / 100.0
            stability_val = max(0.0, min(1.0, stability_val))
        except (TypeError, ValueError):
            return {"index": idx, "error": "稳定度参数无效"}, "invalid"
        
        # 选择可用的 Key
        available_keys = [k for k in all_keys if k not in exhausted_keys]
        if not available_keys:
            return {"index": idx, "error": "所有 API Key 余额不足"}, "all_exhausted"
        
        keys_to_try = available_keys if preferred_key is None else [preferred_key] + [k for k in available_keys if k != preferred_key]
        
        for api_key in keys_to_try:
            try:
                audio_bytes = _request_elevenlabs_tts(
                    api_key, voice_id, text, model_id, stability_val, output_format
                )
                
                if not save_path:
                    if len(items) > 1:
                        seq_num = item.get('seq_num', idx + 1)
                        seq_prefix = f"{seq_num:02d}_"
                    else:
                        seq_prefix = ""
                    save_path = _build_tts_save_path(text, output_format, "tts", seq_prefix)
                
                with open(save_path, 'wb') as f:
                    f.write(audio_bytes)
                
                return {"index": idx, "file_path": save_path}, "success"
                
            except Exception as exc:
                err_text = str(exc)
                err_msg = err_text.lower()
                
                # 风控检测
                if "detected_unusual_activity" in err_msg or "unusual_activity" in err_msg:
                    if enable_circuit_breaker:
                        circuit_breaker_triggered = True
                        return {"index": idx, "error": "触发风控保护，已停止所有任务"}, "circuit_breaker"
                    else:
                        # 不启用熔断时，标记这个 Key 并继续尝试其他 Key
                        exhausted_keys.add(api_key)
                        continue

                # Key 级可轮换错误（余额不足/鉴权失败/限流等）
                if _is_elevenlabs_key_retryable_error(err_text):
                    if _should_auto_disable_elevenlabs_key(err_text):
                        try:
                            _set_elevenlabs_key_enabled(api_key, False, err_text)
                        except Exception as disable_exc:
                            print(f"[ElevenLabs Batch] 自动停用 Key 失败: {disable_exc}")
                    exhausted_keys.add(api_key)
                    continue  # 尝试下一个 Key
                
                # 其他错误
                return {"index": idx, "error": str(exc)}, "error"
        
        # 所有 Key 都试过了
        return {"index": idx, "error": "所有可用 Key 余额不足或出错"}, "all_exhausted"

    # 第一轮：正常处理所有任务
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            results[idx] = {"index": idx, "error": "任务格式无效"}
            continue
        
        if circuit_breaker_triggered:
            results[idx] = {"index": idx, "error": "由于风控熔断已跳过"}
            continue
        
        key_index = item.get('key_index')
        preferred_key = None
        if key_index is not None and str(key_index).strip() != '':
            try:
                preferred_key = _select_elevenlabs_key(all_keys, key_index)
            except ValueError:
                preferred_key = None
        
        result, status = try_generate(idx, item, preferred_key)
        results[idx] = result
        
        if status == "success":
            success += 1
        elif status == "circuit_breaker":
            break
        elif status in ["all_exhausted", "error"]:
            failed_tasks.append((idx, item))
        
        time.sleep(0.5)  # 降低请求频率

    # 第二轮：重试失败的任务（用所有 Key 再试一次）
    if failed_tasks and not circuit_breaker_triggered:
        # 重置已用尽的 Key 列表，给所有 Key 再一次机会
        exhausted_keys.clear()
        
        for idx, item in failed_tasks:
            if circuit_breaker_triggered:
                break
            
            time.sleep(1)  # 等待一会儿
            result, status = try_generate(idx, item)
            results[idx] = result
            
            if status == "success":
                success += 1
                # 从失败任务中移除不需要了

    return jsonify({
        "message": f"完成 {success}/{len(items)}" + (" (风控已熔断)" if circuit_breaker_triggered else ""),
        "results": results,
        "success": success,
        "failed": len(items) - success,
        "circuit_breaker_triggered": circuit_breaker_triggered
    })

@app.route('/api/elevenlabs/tts-workflow', methods=['POST', 'OPTIONS'])
def elevenlabs_tts_workflow():
    """一键配音工作流：生成音频 + 智能拆分 + 对齐字幕"""
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.json
    text = data.get('text', '')
    voice_id = data.get('voice_id', '')
    task_index = data.get('task_index', 0)
    need_split = data.get('need_split', True)
    max_duration = float(data.get('max_duration', 29.0))
    subtitle_text = data.get('subtitle_text', '')
    export_mp4 = data.get('export_mp4', False)
    export_fcpxml = data.get('export_fcpxml', True)  # 默认导出 FCPXML
    seamless_fcpxml = data.get('seamless_fcpxml', True)  # 默认无缝字幕
    
    if not text or not voice_id:
        return jsonify({"error": "缺少必要参数"}), 400

    # 兜底互斥：黑屏 MP4 模式下不进行智能拆分
    if export_mp4 and need_split:
        print(f"[一键配音] task_index={task_index} 检测到 mp4+split 同时开启，已强制关闭拆分")
        need_split = False
    
    print(f"[一键配音] task_index={task_index}, need_split={need_split}, export_mp4={export_mp4}, export_fcpxml={export_fcpxml}")
    
    # 获取 API Key（启用状态）
    api_keys = _load_elevenlabs_keys()
    if not api_keys:
        return jsonify({"error": "未配置 API Key"}), 400
    
    try:
        # 创建输出文件夹
        import datetime
        
        # 如果未指定输出目录，使用下载文件夹+日期_一键配音
        output_dir = data.get('output_dir', '').strip()
        if not output_dir:
            today = datetime.date.today().strftime('%Y-%m-%d')
            downloads_folder = os.path.expanduser('~/Downloads')
            output_dir = os.path.join(downloads_folder, f'{today}_一键配音')
        
        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 提取前15个单词作为文件名前缀
        import re
        # 移除 SSML 标签 <...> 和方括号标签 [...]
        clean_text = re.sub(r'<[^>]+>', '', text)
        clean_text = re.sub(r'\[[^\]]+\]', '', clean_text)  # 移除 [reverent] 等标签
        # 提取前15个单词
        words = clean_text.split()[:15]
        text_prefix = '_'.join(words)[:60]  # 限制长度
        # 只保留 ASCII 字母数字和下划线（避免编码问题）
        text_prefix = ''.join(c for c in text_prefix if c.isascii() and (c.isalnum() or c in ' _-')).strip()
        text_prefix = text_prefix.replace(' ', '_')
        # 如果清理后为空，使用默认前缀
        if not text_prefix:
            text_prefix = 'audio'
        
        # 任务编号前缀：01-文案前缀_日期
        date_suffix = datetime.date.today().strftime('%m%d')
        task_prefix = f"{task_index + 1:02d}-{text_prefix}_{date_suffix}"
        
        # 直接创建三类分组目录
        # 视频文案直接放在 _视频文案/ 下，不创建任务子文件夹
        video_group = os.path.join(output_dir, '_视频文案')
        audio_group = os.path.join(output_dir, '_音频字幕', task_prefix)
        metadata_group = os.path.join(output_dir, '_metadata', task_prefix)
        os.makedirs(video_group, exist_ok=True)
        os.makedirs(audio_group, exist_ok=True)
        os.makedirs(metadata_group, exist_ok=True)
        
        # Step 1: 生成音频
        model_id = data.get('model_id', 'eleven_v3')  # 默认使用 v3 模型（支持 [pause] 等标签）
        stability = float(data.get('stability', 0.5))
        output_format = data.get('output_format', 'mp3_44100_128')
        
        audio_bytes, used_key = _request_elevenlabs_tts_with_rotation(
            api_keys, voice_id, text, model_id, stability, output_format,
            key_index=data.get('key_index')
        )
        used_prefix = f"{used_key[:8]}...{used_key[-4:]}" if len(used_key) >= 12 else used_key
        print(f"[一键配音] 任务 {task_index + 1} 使用 Key: {used_prefix}")
        
        # 文件命名：01-文案-source.mp3（直接写入音频分组）
        source_path = os.path.join(audio_group, f'{task_prefix}-source.mp3')
        with open(source_path, 'wb') as f:
            f.write(audio_bytes)
        
        segments = []
        
        # Step 2: 智能拆分（如果需要）
        if need_split:
            try:
                import numpy as np
                from pydub import AudioSegment
                
                audio = AudioSegment.from_file(source_path)
                total_duration = len(audio) / 1000.0
                
                if total_duration > max_duration:
                    # 简化的分割逻辑
                    samples = np.array(audio.get_array_of_samples())
                    sample_rate = audio.frame_rate
                    
                    cut_points = [0.0]
                    current_pos = 0.0
                    
                    while current_pos < total_duration:
                        if total_duration - current_pos <= max_duration:
                            cut_points.append(total_duration)
                            break
                        
                        search_limit = current_pos + max_duration
                        search_start = max(current_pos + 5, search_limit - 10)
                        search_end = min(search_limit, total_duration)
                        
                        # 找静音点
                        start_sample = int(search_start * sample_rate)
                        end_sample = int(search_end * sample_rate)
                        window_size = int(sample_rate * 0.1)
                        
                        min_vol = float('inf')
                        best_cut = search_limit
                        
                        for pos in range(start_sample, end_sample, window_size // 2):
                            window = samples[pos:pos+window_size]
                            if len(window) > 0:
                                vol = np.sqrt(np.mean(window.astype(float)**2))
                                if vol < min_vol:
                                    min_vol = vol
                                    best_cut = pos / sample_rate
                        
                        if best_cut - current_pos < 5.0:
                            best_cut = search_limit
                        
                        cut_points.append(best_cut)
                        current_pos = best_cut
                    
                    # 导出分段
                    for i in range(len(cut_points) - 1):
                        start_ms = int(cut_points[i] * 1000)
                        end_ms = int(cut_points[i + 1] * 1000)
                        segment = audio[start_ms:end_ms]
                        
                        # 命名：01-文案-part_01.mp3（直接写入音频分组）
                        part_path = os.path.join(audio_group, f'{task_prefix}-part_{i+1:02d}.mp3')
                        segment.export(part_path, format='mp3', bitrate='192k')
                        
                        segments.append({
                            "index": i + 1,
                            "start": cut_points[i],
                            "end": cut_points[i + 1],
                            "path": part_path
                        })
                else:
                    segments = [{"index": 1, "start": 0, "end": total_duration, "path": source_path}]
                    
            except Exception as e:
                print(f"拆分失败: {e}")
                segments = []
        
        # Step 3: 生成字幕（使用现有字幕对齐工具，必须有 Gladia Key）
        srt_path = None
        if subtitle_text:
            try:
                # 获取 Gladia Keys
                gladia_keys_file = os.path.join(os.path.dirname(__file__), 'gladia_keys.json')
                gladia_keys = []
                if os.path.exists(gladia_keys_file):
                    with open(gladia_keys_file, 'r') as f:
                        gladia_data = json.load(f)
                        gladia_keys = gladia_data.get('keys', [])
                
                if not gladia_keys:
                    raise Exception("未配置 Gladia API Key，请在设置中配置后再试")
                
                # 使用 Gladia 转录和现有对齐工具
                # 断行字幕文本直接保存到视频文案分组
                subtitle_text_filename = f'{task_prefix}.txt'
                source_text_path = os.path.join(video_group, subtitle_text_filename)
                with open(source_text_path, 'w', encoding='utf-8') as f:
                    f.write(subtitle_text)
                
                # 转录中间产物直接保存到 metadata 分组
                file_name = os.path.splitext(os.path.basename(source_path))[0]
                generation_subtitle_array_path = os.path.join(metadata_group, f'{file_name}_audio_text_withtime.json')
                generation_subtitle_text_path = os.path.join(metadata_group, f'{file_name}_transcription.txt')
                
                # 通过 Gladia 转录
                # Gladia 需要完整的语言名称，不是简写
                current_language = 'english'  # 默认英语
                progress_generator = transcribe_audio_from_gladia(
                    source_path,
                    gladia_keys,
                    current_language,
                    generation_subtitle_array_path,
                    generation_subtitle_text_path,
                    5.0  # audio_cut_length
                )
                
                for progress in progress_generator:
                    print(f"Gladia 转录进度: {progress}")
                
                # 读取转录结果
                if os.path.exists(generation_subtitle_array_path) and os.path.exists(generation_subtitle_text_path):
                    generation_subtitle_array = read_object_from_json(generation_subtitle_array_path)
                    
                    with open(generation_subtitle_text_path, 'r', encoding='utf-8') as f:
                        generation_subtitle_text = f.read().strip()
                    
                    source_text_with_info = read_text_with_google_doc(source_text_path)
                    
                    # 执行对齐
                    target_srt_path = os.path.join(audio_group, f'{task_prefix}-subtitle.srt')
                    target_fcpxml_path = os.path.join(metadata_group, f'{task_prefix}-subtitle.fcpxml')

                    audio_subtitle_search_diffent_strong(
                        current_language, metadata_group, file_name,
                        generation_subtitle_array, generation_subtitle_text,
                        source_text_with_info, {},  # 无翻译文本
                        False, False,  # gen_merge_srt, source_up_order
                        export_fcpxml, seamless_fcpxml,  # 导出 FCPXML（默认启用）
                        source_srt_path=target_srt_path,
                        fcpxml_path=target_fcpxml_path
                    )

                    if os.path.exists(target_srt_path):
                        srt_path = target_srt_path
                else:
                    raise Exception("Gladia 转录失败，无法生成字幕")
                        
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"字幕生成失败: {e}")
        
        # Step 4: 生成黑屏 MP4（如果需要）
        mp4_path = None
        if export_mp4:
            try:
                # 命名：01-文案-black_stereo.mp4（直接写入视频文案分组）
                mp4_path = os.path.join(video_group, f'{task_prefix}.mp4')
                
                # 先获取音频时长
                duration_cmd = [
                    'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1', source_path
                ]
                duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
                audio_duration = float(duration_result.stdout.strip())
                
                # 使用 ffmpeg 生成黑屏视频，精确指定时长
                cmd = [
                    'ffmpeg', '-y',
                    '-f', 'lavfi', '-i', f'color=c=black:s=1920x1080:r=30:d={audio_duration}',
                    '-i', source_path,
                    '-c:v', 'libx264', '-preset', 'fast',
                    '-c:a', 'aac', '-ac', '2', '-b:a', '192k',
                    '-shortest',
                    mp4_path
                ]
                subprocess.run(cmd, check=True, capture_output=True)
            except Exception as e:
                print(f"MP4 生成失败: {e}")
        
        return jsonify({
            "audio_path": source_path,
            "output_folder": output_dir,
            "task_prefix": task_prefix,
            "segments": segments,
            "segment_count": len(segments)
        })
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/elevenlabs/sfx', methods=['POST', 'OPTIONS'])
def elevenlabs_sfx():
    """ElevenLabs 音效生成"""
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.json or {}
    prompt = data.get('prompt', '')
    duration = data.get('duration', 5)
    save_path = data.get('save_path', '')
    
    if not prompt:
        return jsonify({"error": "缺少音效描述"}), 400
    
    keys = _load_elevenlabs_keys()
    if not keys:
        return jsonify({"error": "未配置 API Key"}), 400

    try:
        api_key = _select_elevenlabs_key(keys, data.get('key_index'))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    
    try:
        import requests
        
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "text": prompt,
            "duration_seconds": duration
        }
        
        response = requests.post(
            "https://api.elevenlabs.io/v1/sound-generation",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            # 确定保存路径
            if not save_path:
                save_path = _build_tts_save_path(prompt, "mp3_44100_128", "sfx")
            
            # 保存文件
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            return jsonify({
                "message": "生成成功",
                "file_path": save_path
            })
        else:
            error_msg = response.text
            try:
                error_data = response.json()
                error_msg = error_data.get("detail", {}).get("message", response.text)
            except:
                pass
            return jsonify({"error": f"API 错误: {error_msg}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/srt/seamless', methods=['POST', 'OPTIONS'])
def seamless_srt():
    """生成无缝 SRT"""
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.json
    src_path = data.get('src_path')
    
    if not src_path:
        return jsonify({"error": "缺少必需参数: src_path"}), 400
    
    try:
        srt_info = SrtParse(src_path, '')
        
        # 使每条字幕的结束时间等于下一条的开始时间
        for i in range(len(srt_info.srt_data) - 1):
            srt_info.srt_data[i]['end'] = srt_info.srt_data[i + 1]['start']
        
        new_path = src_path.replace('.srt', '_seamless.srt')
        srt_info.write(new_path)
        
        return jsonify({
            "message": "生成完成",
            "output_path": new_path
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/srt/compute-char-time', methods=['POST', 'OPTIONS'])
def compute_char_time():
    """计算字符时间"""
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.json
    ref_path = data.get('ref_path')
    interval_time = data.get('interval_time', 1.0)
    
    if not ref_path:
        return jsonify({"error": "缺少必需参数: ref_path"}), 400
    
    try:
        srt_info = SrtParse(ref_path, '')
        # 计算平均字符时间
        total_time = 0
        total_chars = 0
        
        for item in srt_info.srt_data:
            duration = item['end'] - item['start'] - interval_time * 1000
            if duration > 0:
                total_time += duration
                total_chars += item.get('char_count', len(item.get('text', '')))
        
        char_time = (total_time / total_chars / 1000) if total_chars > 0 else 0.1
        
        return jsonify({
            "char_time": char_time
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

_INVALID_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')

def _sanitize_filename(name, max_len=60):
    safe = _INVALID_FILENAME_CHARS.sub('_', name)
    safe = re.sub(r"\s+", "_", safe).strip("._ ")
    if not safe:
        return ""
    if len(safe) > max_len:
        safe = safe[:max_len].rstrip("._ ")
    return safe

def _build_text_prefix(text, max_words=10):
    if not text:
        return ""
    cleaned = text.strip()
    if not cleaned:
        return ""

    words = re.findall(r"\S+", cleaned)
    if len(words) >= 2:
        raw = "_".join(words[:max_words])
    else:
        raw = words[0][:max_words]

    return _sanitize_filename(raw)

def _apply_logo_override(pos, override):
    if not isinstance(override, dict):
        return pos

    def _int_or_default(value, default):
        if value is None or value == "":
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    updated = {
        'x': _int_or_default(override.get('x'), pos['x']),
        'y': _int_or_default(override.get('y'), pos['y']),
        'w': _int_or_default(override.get('width'), pos['w']),
        'h': _int_or_default(override.get('height'), pos['h'])
    }

    if updated['w'] <= 0 or updated['h'] <= 0:
        raise ValueError("Logo 宽高必须大于 0")

    return updated

def _parse_timecode(token):
    token = token.strip()
    if not token:
        raise ValueError("裁切时间点不能为空。")

    parts = token.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds

    raise ValueError(f"无法解析时间点: {token}")

def _parse_cut_points(raw):
    if not raw:
        return []

    normalized = raw.replace("，", ",")
    tokens = re.split(r"[,\s;]+", normalized.strip())
    points = []

    for token in tokens:
        if not token:
            continue
        seconds = _parse_timecode(token)
        if seconds < 0:
            raise ValueError(f"时间点不能为负数: {token}")
        if seconds == 0:
            continue
        points.append(seconds)

    return sorted(set(points))

def _build_segments(cut_points):
    segments = []
    start = 0.0

    for point in cut_points:
        if point <= start:
            continue
        segments.append((start, point))
        start = point

    segments.append((start, None))
    return segments

def _get_audio_duration(file_path):
    """获取音频/视频文件的时长（秒）"""
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            file_path
        ], capture_output=True, text=True, check=True, timeout=30)
        dur = float(result.stdout.strip())
        print(f"[DEBUG] 获取时长: {file_path} -> {dur}s")
        return dur
    except Exception as e:
        print(f"[DEBUG] 获取时长失败: {file_path} -> {e}")
        return None

def _build_black_mp4_cmd(file_path, output_path, start, duration, size="1280x720", fps=24):
    # 获取精确时长
    if duration is None:
        audio_duration = _get_audio_duration(file_path)
        if audio_duration:
            duration = audio_duration - start
    
    print(f"[DEBUG] 生成黑屏命令: start={start}, duration={duration}")
    
    # 关键：在 color 源中使用 d= 参数指定精确时长
    # 先视频源（带时长限制），后音频源
    if duration and duration > 0:
        # 有精确时长时使用
        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi', '-i', f'color=c=black:s={size}:r={fps}:d={duration:.3f}',
            '-ss', f"{start:.3f}", '-i', file_path,
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-b:a', '192k',
            '-map', '0:v', '-map', '1:a',
            output_path
        ]
    else:
        # 无法获取时长时用 -shortest
        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi', '-i', f'color=c=black:s={size}:r={fps}',
            '-ss', f"{start:.3f}", '-i', file_path,
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-b:a', '192k',
            '-map', '0:v', '-map', '1:a',
            '-shortest',
            output_path
        ]
    
    return cmd

@app.route('/api/media/convert', methods=['POST', 'OPTIONS'])
def media_convert():
    """媒体转换 - 支持多种模式"""
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.json
    files = data.get('files', [])
    mode = data.get('mode', 'h264')
    output_dir = data.get('output_dir', '')
    cut_points_map = data.get('cut_points_map', {})
    
    if not files:
        return jsonify({"error": "没有选择文件"}), 400
    
    # 定义模式配置
    modes_config = {
        # Logo 添加模式 (竖屏 1080x1920) - 尺寸缩小到 2/3
        'hailuo': {
            'output_ext': '_hailuo.mp4',
            'logo_path': 'assets/Hailuo.png',
            'logo_pos': {'x': 590, 'y': 1850, 'w': 317, 'h': 60}
        },
        'vidu': {
            'output_ext': '_vidu.mp4',
            'logo_path': 'assets/vidu.png',
            'logo_pos': {'x': 700, 'y': 1850, 'w': 240, 'h': 60}
        },
        'veo': {
            'output_ext': '_veo.mp4',
            'logo_path': 'assets/Veo.png',
            'logo_pos': {'x': 700, 'y': 1850, 'w': 240, 'h': 60}
        },
        'heygen': {
            'output_ext': '_heygen.mp4',
            'logo_path': 'assets/HeyGen.png',
            'logo_pos': {'x': 700, 'y': 1850, 'w': 240, 'h': 60}
        },
        'dream': {
            'output_ext': '_dream.mp4',
            'logo_path': 'assets/Dream.png',
            'logo_pos': {'x': 700, 'y': 1850, 'w': 240, 'h': 60}
        },
        'custom_logo': {
            'output_ext': '_custom_logo.mp4',
            'type': 'custom_logo'
        },
        'ai_generated': {
            'output_ext': '_ai.mp4',
            'logo_path': 'assets/AI_Generated.png',
            'logo_pos': {'x': 820, 'y': 10, 'w': 253, 'h': 40}  # 右上角
        },
        # 其他模式
        'image': {'output_ext': '', 'type': 'watermark'},
        'h264': {'output_ext': '_h264.mp4', 'type': 'encode'},
        'x264': {'output_ext': '_x264.mp4', 'type': 'encode_crf'},
        'dnxhr': {'output_ext': '_dnxhr.mov', 'type': 'dnxhr'},
        'dnxhr_hqx': {'output_ext': '_dnxhr_hqx.mov', 'type': 'dnxhr_10bit'},
        'png': {'output_ext': '.png', 'type': 'image'},
        'mp3': {'output_ext': '.mp3', 'type': 'audio'},
        'wav': {'output_ext': '.wav', 'type': 'audio'},
        'audio_black': {'output_ext': '_black.mp4', 'type': 'audio_black'},
        'audio_split': {'output_ext': '', 'type': 'audio_split'},
    }
    
    # custom_logo 和 watermark 是特殊模式，不在预定义配置中
    if mode not in modes_config and mode not in ['custom_logo', 'watermark']:
        return jsonify({"error": f"不支持的模式: {mode}"}), 400
    
    try:
        import subprocess
        converted = []
        config = modes_config.get(mode, {})
        
        # 获取 assets 目录路径
        assets_dir = os.path.join(os.path.dirname(__file__), '..', 'assets')
        
        for file_path in files:
            if not os.path.exists(file_path):
                continue
            
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            ext = os.path.splitext(file_path)[1].lower()
            out_dir = output_dir if output_dir else os.path.dirname(file_path)
            
            # Logo 添加模式(预设)
            if mode in ['hailuo', 'vidu', 'veo', 'heygen', 'dream', 'ai_generated']:
                logo_path = os.path.join(assets_dir, os.path.basename(config['logo_path']))
                pos = config['logo_pos']
                try:
                    pos = _apply_logo_override(pos, data.get('logo_override', {}))
                except ValueError as exc:
                    return jsonify({"error": str(exc)}), 400
                output_path = os.path.join(out_dir, f"{base_name}{config['output_ext']}")
                
                # 先缩放视频到 1080x1920，再叠加 logo
                filter_str = (
                    f"scale=1080:1920:force_original_aspect_ratio=decrease,"
                    f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2[v];"
                    f"[1:v]scale={pos['w']}:{pos['h']}[logo];"
                    f"[v][logo]overlay={pos['x']}:{pos['y']}"
                )
                
                if os.path.exists(logo_path):
                    cmd = [
                        'ffmpeg', '-y', '-i', file_path, '-i', logo_path,
                        '-filter_complex', filter_str,
                        '-c:v', 'libx264', '-c:a', 'aac', '-b:a', '128k',
                        output_path
                    ]
                else:
                    # Logo 不存在，只做格式转换
                    cmd = ['ffmpeg', '-y', '-i', file_path, '-c:v', 'libx264', '-c:a', 'aac', output_path]
            
            # 自定义 Logo 模式
            elif mode == 'custom_logo':
                custom_logo_data = data.get('custom_logo', {})
                logo_path = custom_logo_data.get('path', '')
                try:
                    pos_x = int(custom_logo_data.get('x', 590))
                    pos_y = int(custom_logo_data.get('y', 1810))
                    logo_w = int(custom_logo_data.get('width', 400))
                    logo_h = int(custom_logo_data.get('height', 90))
                except (TypeError, ValueError):
                    return jsonify({"error": "自定义 Logo 参数无效"}), 400

                if logo_w <= 0 or logo_h <= 0:
                    return jsonify({"error": "自定义 Logo 宽高必须大于 0"}), 400

                output_ext = config.get('output_ext', '_custom_logo.mp4')
                output_path = os.path.join(out_dir, f"{base_name}{output_ext}")
                
                if logo_path and os.path.exists(logo_path):
                    filter_str = (
                        f"scale=1080:1920:force_original_aspect_ratio=decrease,"
                        f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2[v];"
                        f"[1:v]scale={logo_w}:{logo_h}[logo];"
                        f"[v][logo]overlay={pos_x}:{pos_y}"
                    )
                    cmd = [
                        'ffmpeg', '-y', '-i', file_path, '-i', logo_path,
                        '-filter_complex', filter_str,
                        '-c:v', 'libx264', '-c:a', 'aac', '-b:a', '128k',
                        output_path
                    ]
                else:
                    return jsonify({"error": "自定义 Logo 文件不存在"}), 400
            
            # 完整水印模式（支持字体、颜色、描边、阴影、位置）
            elif mode == 'watermark':
                wm = data.get('watermark', {})
                text = wm.get('text', 'AI Created')
                font_size = wm.get('font_size', 24)
                color = wm.get('color', '#ffffff').replace('#', '')
                opacity = wm.get('opacity', 1)
                has_stroke = wm.get('stroke', False)
                stroke_color = wm.get('stroke_color', '#000000').replace('#', '')
                stroke_width = wm.get('stroke_width', 2)
                has_shadow = wm.get('shadow', False)
                pos_x = wm.get('x', 'w-tw-10')
                pos_y = wm.get('y', '10')
                font_family = wm.get('font', 'Arial')
                
                # 构建 drawtext 滤镜字符串
                # 转义特殊字符
                escaped_text = text.replace("'", "\\'").replace(":", "\\:")
                
                # 颜色格式转换为 FFmpeg 格式 (RRGGBB@AA)
                alpha_hex = format(int(opacity * 255), '02x')
                color_ffmpeg = f"{color}@0x{alpha_hex}"
                
                # 添加字体
                drawtext = f"drawtext=text='{escaped_text}':fontsize={font_size}:fontcolor=0x{color_ffmpeg}:x={pos_x}:y={pos_y}"
                
                # 尝试使用指定字体
                if font_family:
                    drawtext += f":font='{font_family}'"
                
                # 添加描边 (borderw + bordercolor)
                if has_stroke:
                    drawtext += f":borderw={stroke_width}:bordercolor=0x{stroke_color}"
                
                # 添加阴影 (shadowx, shadowy, shadowcolor)
                if has_shadow:
                    drawtext += ":shadowx=2:shadowy=2:shadowcolor=black@0x80"
                
                output_ext = ext if ext in ['.mp4', '.mov', '.mkv'] else '.mp4'
                output_path = os.path.join(out_dir, f"{base_name}_watermark{output_ext}")
                
                if ext in ['.png', '.jpg', '.jpeg']:
                    # 图片
                    cmd = ['ffmpeg', '-y', '-i', file_path, '-vf', drawtext, output_path]
                else:
                    # 视频
                    cmd = ['ffmpeg', '-y', '-i', file_path, '-vf', drawtext, '-c:v', 'libx264', '-c:a', 'aac', output_path]
            
            # 水印模式
            elif mode == 'image':
                output_ext = ext if ext in ['.mp4', '.png', '.jpg', '.jpeg'] else '.mp4'
                output_path = os.path.join(out_dir, f"{base_name}_watermark{output_ext}")
                
                if ext in ['.png', '.jpg', '.jpeg']:
                    # 图片添加文字水印
                    cmd = [
                        'ffmpeg', '-y', '-i', file_path,
                        '-vf', "drawtext=text='AI Created':fontsize=24:fontcolor=white:x=w-tw-10:y=40",
                        output_path
                    ]
                else:
                    # 视频添加文字水印
                    cmd = [
                        'ffmpeg', '-y', '-i', file_path,
                        '-vf', "drawtext=text='AI Created':fontsize=24:fontcolor=white:x=w-tw-10:y=40",
                        '-c:v', 'libx264', '-c:a', 'aac',
                        output_path
                    ]
            
            # H.264 转换
            elif mode == 'h264':
                output_path = os.path.join(out_dir, f"{base_name}{config['output_ext']}")
                cmd = ['ffmpeg', '-y', '-i', file_path, '-c:v', 'libx264', '-c:a', 'aac', '-avoid_negative_ts', 'make_zero', output_path]
            
            # H.264 压缩
            elif mode == 'x264':
                output_path = os.path.join(out_dir, f"{base_name}{config['output_ext']}")
                cmd = ['ffmpeg', '-y', '-i', file_path, '-c:v', 'libx264', '-crf', '20', '-c:a', 'aac', '-avoid_negative_ts', 'make_zero', output_path]
            
            # DNxHR 转换
            elif mode == 'dnxhr':
                output_path = os.path.join(out_dir, f"{base_name}{config['output_ext']}")
                cmd = ['ffmpeg', '-y', '-i', file_path, '-c:v', 'dnxhd', '-profile:v', 'dnxhr_hq', '-c:a', 'pcm_s16le', '-avoid_negative_ts', 'make_zero', output_path]
            
            # DNxHR 10bit
            elif mode == 'dnxhr_hqx':
                output_path = os.path.join(out_dir, f"{base_name}{config['output_ext']}")
                cmd = ['ffmpeg', '-y', '-i', file_path, '-c:v', 'dnxhd', '-profile:v', 'dnxhr_hqx', '-c:a', 'pcm_s16le', '-avoid_negative_ts', 'make_zero', output_path]
            
            # PNG 转换
            elif mode == 'png':
                output_path = os.path.join(out_dir, f"{base_name}{config['output_ext']}")
                cmd = ['ffmpeg', '-y', '-i', file_path, output_path]
            
            # MP3 转换
            elif mode == 'mp3':
                output_path = os.path.join(out_dir, f"{base_name}{config['output_ext']}")
                cmd = ['ffmpeg', '-y', '-i', file_path, '-vn', '-acodec', 'libmp3lame', '-b:a', '192k', output_path]
            
            # WAV 转换
            elif mode == 'wav':
                output_path = os.path.join(out_dir, f"{base_name}{config['output_ext']}")
                cmd = ['ffmpeg', '-y', '-i', file_path, '-vn', '-acodec', 'pcm_s16le', output_path]

            # 音频转黑屏 MP4
            elif mode == 'audio_black':
                output_path = os.path.join(out_dir, f"{base_name}{config['output_ext']}")
                cmd = _build_black_mp4_cmd(file_path, output_path, 0.0, None)

            # 音频裁切导出
            elif mode == 'audio_split':
                cut_points_raw = ''
                if isinstance(cut_points_map, dict):
                    cut_points_raw = cut_points_map.get(file_path, '')
                if not cut_points_raw:
                    cut_points_raw = data.get('cut_points', '')
                export_mp3 = bool(data.get('export_mp3', True))
                export_mp4 = bool(data.get('export_mp4', True))

                if not export_mp3 and not export_mp4:
                    return jsonify({"error": "请至少选择一种导出格式"}), 400

                # 没有裁切点时，直接转换整个文件（不裁切）
                if not cut_points_raw.strip():
                    if export_mp3:
                        mp3_path = os.path.join(out_dir, f"{base_name}.mp3")
                        cmd = [
                            'ffmpeg', '-y', '-i', file_path,
                            '-vn', '-c:a', 'libmp3lame', '-b:a', '192k', '-ac', '2',
                            mp3_path
                        ]
                        subprocess.run(cmd, check=True, capture_output=True)
                        converted.append(mp3_path)
                    
                    if export_mp4:
                        mp4_path = os.path.join(out_dir, f"{base_name}_black.mp4")
                        cmd = _build_black_mp4_cmd(file_path, mp4_path, 0, None)
                        subprocess.run(cmd, check=True, capture_output=True)
                        converted.append(mp4_path)
                    
                    continue

                try:
                    cut_points = _parse_cut_points(cut_points_raw)
                except ValueError as exc:
                    return jsonify({"error": str(exc)}), 400

                segments = _build_segments(cut_points)

                # MP3 按裁切点分段
                if export_mp3:
                    for idx, (start, end) in enumerate(segments, start=1):
                        duration = None if end is None else max(end - start, 0)
                        if duration is not None and duration <= 0:
                            continue

                        suffix = f"_part{idx:02d}"
                        mp3_path = os.path.join(out_dir, f"{base_name}{suffix}.mp3")
                        cmd = [
                            'ffmpeg', '-y',
                            '-ss', f"{start:.3f}", '-i', file_path,
                            '-vn', '-c:a', 'libmp3lame', '-b:a', '192k', '-ac', '2'
                        ]
                        if duration is not None:
                            cmd.extend(['-t', f"{duration:.3f}"])
                        cmd.append(mp3_path)
                        subprocess.run(cmd, check=True, capture_output=True)
                        converted.append(mp3_path)

                # 黑屏 MP4 只生成一个完整的（原始音频，不裁切）
                if export_mp4:
                    mp4_path = os.path.join(out_dir, f"{base_name}_black.mp4")
                    cmd = _build_black_mp4_cmd(file_path, mp4_path, 0, None)
                    subprocess.run(cmd, check=True, capture_output=True)
                    converted.append(mp4_path)

                continue

            else:
                continue
            
            subprocess.run(cmd, check=True, capture_output=True)
            converted.append(output_path)
        
        # 获取每个文件的时长
        files_with_duration = []
        for f in converted:
            dur = _get_audio_duration(f)
            files_with_duration.append({
                "path": f,
                "duration": round(dur, 2) if dur else None
            })
        
        return jsonify({
            "message": f"成功转换 {len(converted)} 个文件",
            "files": converted,
            "files_info": files_with_duration
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/media/waveform', methods=['POST', 'OPTIONS'])
def get_waveform():
    """获取音/视频文件的波形峰值数据，用于前端波形可视化"""
    if request.method == 'OPTIONS':
        return '', 204

    data = request.json
    file_path = data.get('file_path', '')
    num_peaks = int(data.get('num_peaks', 300))

    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "文件不存在"}), 400

    try:
        import struct

        # 使用 FFmpeg 提取单声道、低采样率的原始 float32 音频数据
        cmd = [
            'ffmpeg', '-hide_banner', '-i', file_path,
            '-ac', '1', '-ar', '8000',
            '-f', 'f32le', '-'
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)

        raw_data = result.stdout
        num_samples = len(raw_data) // 4  # float32 = 4 bytes

        if num_samples == 0:
            duration = _get_audio_duration(file_path) or 0
            return jsonify({"peaks": [], "duration": round(duration, 3), "num_peaks": 0})

        samples = struct.unpack(f'{num_samples}f', raw_data)

        # 计算峰值
        block_size = max(1, num_samples // num_peaks)
        peaks = []
        for i in range(min(num_peaks, num_samples // max(block_size, 1))):
            start_idx = i * block_size
            end_idx = min(start_idx + block_size, num_samples)
            block = samples[start_idx:end_idx]
            peak = max(abs(s) for s in block) if block else 0
            peaks.append(peak)

        # 归一化
        max_peak = max(peaks) if peaks else 1
        if max_peak > 0:
            peaks = [round(p / max_peak, 4) for p in peaks]

        duration = _get_audio_duration(file_path) or (num_samples / 8000)

        print(f"[波形] {os.path.basename(file_path)}: {len(peaks)} peaks, {duration:.1f}s")
        return jsonify({
            "peaks": peaks,
            "duration": round(duration, 3),
            "num_peaks": len(peaks)
        })

    except subprocess.TimeoutExpired:
        return jsonify({"error": "波形提取超时"}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"波形提取失败: {str(e)}"}), 500


@app.route('/api/media/trim', methods=['POST', 'OPTIONS'])
def media_trim():
    """精确裁切视频/音频文件"""
    if request.method == 'OPTIONS':
        return '', 204

    data = request.json
    file_path = data.get('file_path', '')
    start_time = float(data.get('start', 0))
    end_time = float(data.get('end', 0))
    output_dir = data.get('output_dir', '')

    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "文件不存在"}), 400
    if end_time <= start_time:
        return jsonify({"error": "结束时间必须大于开始时间"}), 400

    try:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        ext = os.path.splitext(file_path)[1].lower()
        out_dir = output_dir if output_dir else os.path.dirname(file_path)

        duration = end_time - start_time
        start_str = _format_scene_time(start_time)
        end_str = _format_scene_time(end_time)
        output_filename = f"{base_name}_trimmed_{start_str.replace(':', '.')}-{end_str.replace(':', '.')}{ext}"
        output_path = os.path.join(out_dir, output_filename)

        precise = data.get('precise', True)  # 默认精确模式

        if precise:
            # ===== 精确模式：-ss 放在 -i 之后（先解码再定位，帧级精度）=====
            # 音频用 aac，视频用 libx264 crf 18（高质量）
            cmd = [
                'ffmpeg', '-y',
                '-i', file_path,
                '-ss', f'{start_time:.3f}',
                '-to', f'{end_time:.3f}',
                '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
                '-c:a', 'aac', '-b:a', '192k',
                '-avoid_negative_ts', 'make_zero',
                output_path
            ]
            print(f"[裁切-精确] {base_name} [{start_str} -> {end_str}] => {output_filename}")
        else:
            # ===== 快速模式：-ss 放在 -i 之前（关键帧级，很快但不精确）=====
            cmd = [
                'ffmpeg', '-y',
                '-ss', f'{start_time:.3f}',
                '-i', file_path,
                '-t', f'{duration:.3f}',
                '-c', 'copy',
                '-avoid_negative_ts', 'make_zero',
                output_path
            ]
            print(f"[裁切-快速] {base_name} [{start_str} -> {end_str}] => {output_filename}")

        subprocess.run(cmd, check=True, capture_output=True, timeout=300)

        # 获取输出文件时长
        out_duration = _get_audio_duration(output_path)

        return jsonify({
            "message": f"裁切完成: {output_filename}",
            "output_path": output_path,
            "output_filename": output_filename,
            "duration": round(out_duration, 3) if out_duration else round(duration, 3),
            "mode": "精确" if precise else "快速"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"裁切失败: {str(e)}"}), 500


@app.route('/api/media/scene-detect', methods=['POST', 'OPTIONS'])
def scene_detect():
    """场景剪切检测 - 使用 FFmpeg 的 scdet/select 滤镜检测视频中的场景变化"""
    if request.method == 'OPTIONS':
        return '', 204

    data = request.json
    file_path = data.get('file_path', '')
    threshold = float(data.get('threshold', 0.3))  # 灵敏度, 0.0~1.0, 越小越灵敏
    min_interval = float(data.get('min_interval', 0.5))  # 场景最小间隔（秒），过滤过于密集的检测结果

    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "文件不存在"}), 400

    try:
        # 获取视频时长
        duration = _get_audio_duration(file_path)
        if not duration:
            return jsonify({"error": "无法获取视频时长"}), 400

        # 获取视频帧率
        fps_result = subprocess.run([
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            file_path
        ], capture_output=True, text=True, timeout=30)
        fps_str = fps_result.stdout.strip()
        try:
            if '/' in fps_str:
                num, den = fps_str.split('/')
                fps = float(num) / float(den)
            else:
                fps = float(fps_str)
        except Exception:
            fps = 30.0

        # 获取视频分辨率
        res_result = subprocess.run([
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=p=0',
            file_path
        ], capture_output=True, text=True, timeout=30)
        resolution = res_result.stdout.strip()

        print(f"[场景检测] 开始分析: {file_path}, 阈值={threshold}, 最小间隔={min_interval}s, FPS={fps:.2f}")

        # 使用 FFmpeg select 滤镜检测场景变化
        # select='gt(scene,THRESHOLD)' 检测场景变化分数大于阈值的帧
        # 输出每一帧的时间戳和场景分数
        cmd = [
            'ffmpeg', '-hide_banner',
            '-i', file_path,
            '-vf', f"select='gt(scene,{threshold})',showinfo",
            '-f', 'null', '-'
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600  # 10分钟超时
        )

        # 从 stderr 中解析 showinfo 的输出（FFmpeg 的 showinfo 滤镜输出到 stderr）
        scene_points = []
        last_time = -min_interval  # 用于去重过近的场景点

        for line in result.stderr.splitlines():
            if 'showinfo' in line and 'pts_time' in line:
                # 解析 pts_time
                import re as _re
                pts_match = _re.search(r'pts_time:\s*([0-9.]+)', line)
                if pts_match:
                    pts_time = float(pts_match.group(1))
                    # 跳过开头（0~0.5秒的通常是视频开始，不是真正的场景切换）
                    if pts_time < 0.3:
                        continue
                    # 如果距离上个场景点太近，跳过
                    if pts_time - last_time < min_interval:
                        continue
                    # 解析场景分数（如果有）
                    scene_score = None
                    # showinfo 不直接输出 scene score，只输出被 select 选中的帧
                    scene_points.append({
                        "time": round(pts_time, 3),
                        "time_str": _format_scene_time(pts_time)
                    })
                    last_time = pts_time

        print(f"[场景检测] 检测到 {len(scene_points)} 个场景切换点")

        # 构建片段信息
        segments = []
        boundaries = [0] + [p["time"] for p in scene_points] + [duration]
        for i in range(len(boundaries) - 1):
            seg_start = boundaries[i]
            seg_end = boundaries[i + 1]
            seg_duration = seg_end - seg_start
            segments.append({
                "index": i + 1,
                "start": round(seg_start, 3),
                "end": round(seg_end, 3),
                "start_str": _format_scene_time(seg_start),
                "end_str": _format_scene_time(seg_end),
                "duration": round(seg_duration, 3),
                "duration_str": _format_scene_time(seg_duration)
            })

        return jsonify({
            "message": f"检测到 {len(scene_points)} 个场景切换点，共 {len(segments)} 个片段",
            "file": file_path,
            "duration": round(duration, 3),
            "fps": round(fps, 2),
            "resolution": resolution,
            "threshold": threshold,
            "scene_points": scene_points,
            "segments": segments
        })

    except subprocess.TimeoutExpired:
        return jsonify({"error": "场景检测超时（超过 10 分钟）"}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"场景检测失败: {str(e)}"}), 500


def _format_scene_time(seconds):
    """将秒数格式化为 HH:MM:SS.mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:06.3f}"
    return f"{m:02d}:{s:06.3f}"


@app.route('/api/media/scene-split', methods=['POST', 'OPTIONS'])
def scene_split():
    """按场景切换点将视频拆分为独立片段"""
    if request.method == 'OPTIONS':
        return '', 204

    data = request.json
    file_path = data.get('file_path', '')
    segments = data.get('segments', [])
    output_dir = data.get('output_dir', '')

    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "文件不存在"}), 400
    if not segments:
        return jsonify({"error": "没有指定要导出的片段"}), 400

    try:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        ext = os.path.splitext(file_path)[1].lower()
        out_dir = output_dir if output_dir else os.path.dirname(file_path)
        scene_output_dir = os.path.join(out_dir, f"{base_name}_scenes")
        os.makedirs(scene_output_dir, exist_ok=True)

        exported = []
        total = len(segments)

        for i, seg in enumerate(segments):
            start = float(seg.get('start', 0))
            end = float(seg.get('end', 0))
            seg_duration = end - start

            if seg_duration <= 0:
                continue

            idx = seg.get('index', i + 1)
            output_filename = f"{base_name}_scene{idx:03d}{ext}"
            output_path = os.path.join(scene_output_dir, output_filename)

            # 使用 ffmpeg 精确切割（重编码，帧级精度）
            cmd = [
                'ffmpeg', '-y',
                '-i', file_path,
                '-ss', f'{start:.3f}',
                '-to', f'{end:.3f}',
                '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
                '-c:a', 'aac', '-b:a', '192k',
                '-avoid_negative_ts', 'make_zero',
                output_path
            ]

            print(f"[场景拆分-精确] ({i+1}/{total}) {output_filename} [{_format_scene_time(start)} -> {_format_scene_time(end)}]")
            subprocess.run(cmd, check=True, capture_output=True, timeout=300)
            exported.append({
                "path": output_path,
                "filename": output_filename,
                "index": idx,
                "start": start,
                "end": end,
                "duration": round(seg_duration, 3)
            })

        return jsonify({
            "message": f"成功导出 {len(exported)} 个片段到 {scene_output_dir}",
            "output_dir": scene_output_dir,
            "files": exported
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"场景拆分失败: {str(e)}"}), 500


@app.route('/api/audio/smart-split-analyze', methods=['POST', 'OPTIONS'])
def smart_split_analyze():
    """智能分析音频分割点（基于静音检测）"""
    if request.method == 'OPTIONS':
        return '', 204
    
    # 支持两种方式：上传文件 或 传递文件路径
    file_path = None
    temp_file_path = None
    
    if 'audio_file' in request.files:
        # 上传文件方式
        audio_file = request.files['audio_file']
        if audio_file.filename:
            import tempfile
            fd, temp_file_path = tempfile.mkstemp(suffix=os.path.splitext(audio_file.filename)[1])
            os.close(fd)
            audio_file.save(temp_file_path)
            file_path = temp_file_path
        max_duration = float(request.form.get('max_duration', 29.0))
    else:
        # JSON 方式
        data = request.json or {}
        file_path = data.get('file_path', '')
        max_duration = float(data.get('max_duration', 29.0))
    
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "文件不存在"}), 400
    
    try:
        import numpy as np
        try:
            from moviepy.editor import AudioFileClip
        except ImportError:
            from moviepy.audio.io.AudioFileClip import AudioFileClip
        
        audio = AudioFileClip(file_path)
        total_duration = audio.duration
        
        def find_best_cut_point(clip, search_start, search_end, fps=22050):
            """在指定范围内找音量最低点"""
            try:
                if hasattr(clip, "subclipped"):
                    subclip = clip.subclipped(search_start, search_end)
                else:
                    subclip = clip.subclip(search_start, search_end)
                arr = subclip.to_soundarray(fps=fps)
            except Exception as e:
                return search_end, 0.0
            
            if len(arr) == 0:
                return search_end, 0.0
            
            window_size = int(fps * 0.1)
            if window_size == 0:
                window_size = 1
            
            volumes = []
            timestamps = []
            
            for i in range(0, len(arr), window_size):
                chunk = arr[i:i+window_size]
                if len(chunk) == 0:
                    continue
                rms = np.sqrt(np.mean(chunk**2))
                volumes.append(rms)
                timestamps.append(search_start + i / float(fps))
            
            if not volumes:
                return search_end, 0.0
            
            volumes = np.array(volumes)
            timestamps = np.array(timestamps)
            
            min_vol = np.min(volumes)
            threshold = min_vol + 0.005
            candidates_indices = np.where(volumes <= threshold)[0]
            
            if len(candidates_indices) > 0:
                best_time = timestamps[candidates_indices[-1]]
                best_vol = volumes[candidates_indices[-1]]
            else:
                min_idx = np.argmin(volumes)
                best_time = timestamps[min_idx]
                best_vol = volumes[min_idx]
            
            return best_time, float(best_vol)
        
        # 计算分割点
        cut_points = [0.0]
        current_pos = 0.0
        
        while current_pos < total_duration:
            if total_duration - current_pos <= max_duration:
                cut_points.append(total_duration)
                break
            
            search_limit = current_pos + max_duration
            search_start = max(current_pos + 5, search_limit - 10)
            search_end = min(search_limit, total_duration)
            
            best_cut, vol = find_best_cut_point(audio, search_start, search_end)
            
            if best_cut - current_pos < 5.0:
                best_cut = search_limit
            
            cut_points.append(best_cut)
            current_pos = best_cut
        
        audio.close()
        
        # 构建分段信息
        segments = []
        for i in range(len(cut_points) - 1):
            start = cut_points[i]
            end = cut_points[i + 1]
            segments.append({
                "index": i + 1,
                "start": round(start, 2),
                "end": round(end, 2),
                "duration": round(end - start, 2)
            })
        
        # 清理临时文件
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass
        
        return jsonify({
            "total_duration": round(total_duration, 2),
            "max_duration": max_duration,
            "cut_points": [round(p, 2) for p in cut_points],
            "segments": segments,
            "segment_count": len(segments)
        })
        
    except Exception as e:
        # 清理临时文件
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/video/analyze', methods=['POST', 'OPTIONS'])
def analyze_video():
    """分析视频链接 - 支持单个视频和播放列表"""
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.json
    url = data.get('url', '')
    
    if not url:
        return jsonify({"error": "缺少视频链接"}), 400
    
    try:
        import yt_dlp
        
        ydl_opts = {
            'quiet': True,
            'extract_flat': 'in_playlist',
            'dump_single_json': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        
        # 处理播放列表
        if 'entries' in info and info.get('entries'):
            entries = list(info['entries'])
            return jsonify({
                "title": info.get('title', '播放列表'),
                "entries": [{
                    "title": e.get('title', 'Unknown'),
                    "url": e.get('url') or e.get('webpage_url'),
                    "webpage_url": e.get('webpage_url') or e.get('url'),
                    "duration": e.get('duration', 0),
                    "thumbnail": e.get('thumbnail', '')
                } for e in entries if e]
            })
        else:
            # 单个视频
            duration_secs = info.get('duration', 0)
            return jsonify({
                "title": info.get('title', '未知'),
                "url": info.get('webpage_url') or url,
                "webpage_url": info.get('webpage_url') or url,
                "duration": duration_secs,
                "thumbnail": info.get('thumbnail', '')
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/video/download', methods=['POST', 'OPTIONS'])
def download_video():
    """下载单个视频"""
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.json
    url = data.get('url', '')
    quality = data.get('quality', 'best')
    output_dir = data.get('output_dir', os.path.expanduser('~/Downloads'))
    download_subtitle = data.get('download_subtitle', False)
    
    if not url:
        return jsonify({"error": "缺少视频链接"}), 400
    
    try:
        import yt_dlp
        
        ydl_opts = {
            'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
        }
        
        if quality == '1080p':
            ydl_opts['format'] = 'bestvideo[height<=1080]+bestaudio/best'
        elif quality == '720p':
            ydl_opts['format'] = 'bestvideo[height<=720]+bestaudio/best'
        elif quality == '480p':
            ydl_opts['format'] = 'bestvideo[height<=480]+bestaudio/best'
        else:
            ydl_opts['format'] = 'bestvideo+bestaudio/best'
        
        if download_subtitle:
            ydl_opts['writesubtitles'] = True
            ydl_opts['subtitleslangs'] = ['en', 'zh-Hans']
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        return jsonify({
            "message": "下载完成",
            "output_path": output_dir
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/video/download-batch', methods=['POST', 'OPTIONS'])
def download_video_batch():
    """批量下载视频 - 支持播放列表"""
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.json
    items = data.get('items', [])
    options = data.get('options', {})
    output_dir = data.get('output_dir', '') or os.path.expanduser('~/Downloads')
    
    if not items:
        return jsonify({"error": "没有要下载的视频"}), 400
    
    try:
        import yt_dlp
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        is_audio_only = options.get('audio_only', False)
        target_ext = options.get('ext', 'mp4')
        quality = options.get('quality', 'best')
        subtitles = options.get('subtitles', False)
        sub_lang = options.get('sub_lang', 'en')
        
        ydl_opts = {
            'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'ignoreerrors': True,
        }
        
        if is_audio_only:
            ydl_opts['format'] = 'bestaudio/best'
            ydl_opts['postprocessors'] = [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': target_ext,
                'preferredquality': '192',
            }]
        else:
            if quality.lower() == 'best':
                ydl_opts['format'] = 'bestvideo+bestaudio/best'
            else:
                try:
                    h = int(str(quality).lower().replace('p', ''))
                    ydl_opts['format'] = f'bestvideo[height<={h}]+bestaudio/best[height<={h}]'
                except:
                    ydl_opts['format'] = 'bestvideo+bestaudio/best'
            
            if target_ext in ['mp4', 'mkv', 'webm', 'mov']:
                ydl_opts['merge_output_format'] = target_ext
        
        if subtitles:
            ydl_opts['writesubtitles'] = True
            ydl_opts['subtitleslangs'] = [sub_lang, 'en', 'zh-Hans']
        
        # 下载所有视频
        urls = [item.get('url') for item in items if item.get('url')]
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download(urls)
        
        return jsonify({
            "message": f"成功下载 {len(urls)} 个视频",
            "output_path": output_dir,
            "count": len(urls)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Python backend server on port 5001...")
    app.run(host='127.0.0.1', port=5001, debug=False, threaded=True)
