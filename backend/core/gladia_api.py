"""
Gladia API 接口 - 从 SW_GenSubTitle/Gladia_API.py 移植
"""
import os
import pathlib
from datetime import datetime
import json
import requests
import time
import shutil
import platform
import subprocess
import random
from pydub import AudioSegment
from pydub.silence import detect_silence
try:
    from utils import get_ffmpeg_exe
except ImportError:
    # 如果直接导入失败，尝试通过 shutil 查找
    import shutil
    def get_ffmpeg_exe():
        return shutil.which('ffmpeg') or 'ffmpeg'

GLADIA_API_URL = "https://api.gladia.io/audio/text/audio-transcription/"

API_keys = []
cur_api_key = ""


def get_next_api_key():
    """获取下一个API key"""
    global API_keys, cur_api_key

    if len(API_keys) == 0:
        return ""
    
    if cur_api_key == "":
        return API_keys[0]
    
    cur_index = API_keys.index(cur_api_key)
    next_index = cur_index + 1
    if next_index < len(API_keys):
        return API_keys[next_index]
    
    return ""


# 支持的语言
languages = [
    "afrikaans", "albanian", "amharic", "arabic", "armenian", "assamese",
    "azerbaijani", "bashkir", "basque", "belarusian", "bengali", "bosnian",
    "breton", "bulgarian", "burmese", "catalan", "chinese", "croatian", "czech",
    "danish", "dutch", "english", "estonian", "faroese", "finnish", "flemish",
    "french", "galician", "georgian", "german", "greek", "gujarati",
    "haitian creole", "hausa", "hawaiian", "hebrew", "hindi", "hungarian",
    "icelandic", "igbo", "iloko", "indonesian", "irish", "italian", "japanese",
    "javanese", "kannada", "kazakh", "khmer", "korean", "lao", "latin",
    "latvian", "letzeburgesch", "lingala", "lithuanian", "luxembourgish",
    "macedonian", "malagasy", "malay", "malayalam", "maltese", "maori",
    "marathi", "moldavian", "mongolian", "nepali", "norwegian", "nynorsk",
    "occitan", "oriya", "panjabi", "pashto", "persian", "polish", "portuguese",
    "punjabi", "romanian", "russian", "sanskrit", "serbian", "shona", "sindhi",
    "sinhala", "slovak", "slovenian", "somali", "spanish", "sundanese",
    "swahili", "swedish", "tagalog", "tajik", "tamil", "tatar", "telugu", "thai",
    "tibetan", "turkish", "turkmen", "ukrainian", "urdu", "uzbek", "valencian",
    "vietnamese", "welsh", "yiddish", "yoruba"
]


def extract_audio_from_video(video_path, output_path, audio_format="mp3"):
    """从视频文件中提取音频"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    audio_path = f"{output_path}/{pathlib.Path(video_path).stem}.{audio_format}"
    os.makedirs(output_path, exist_ok=True)
    
    ffmpeg_path = get_ffmpeg_exe()

    if audio_format == "wav":
        cmd = [
            ffmpeg_path,
            "-y",
            "-i", video_path,
            "-vn",
            "-ar", "32000",
            "-ac", "1",
            audio_path
        ]
    else:
        cmd = [
            ffmpeg_path,
            "-y",
            "-i", video_path,
            "-vn",
            "-ar", "44100",
            "-ac", "1",
            "-b:a", "192k",
            audio_path
        ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg 提取音频失败:\n{result.stderr.decode()}")

    return audio_path


def split_audio_on_silence(audio_path, output_dir, min_minutes=20.0, max_minutes=50.0,
                           silence_thresh=None, min_silence_len=500, audio_format="mp3"):
    """按静音切分长音频"""
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1)
    total_ms = len(audio)
    min_ms = min_minutes * 60 * 1000
    max_ms = max_minutes * 60 * 1000
    base_name = pathlib.Path(audio_path).stem
    os.makedirs(output_dir, exist_ok=True)

    if silence_thresh is None:
        silence_thresh = audio.dBFS - 14

    silences = detect_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    silence_points = [(start + end)//2 for start, end in silences]

    segments_ms = []
    start = 0

    for split_point in silence_points:
        if split_point - start >= min_ms:
            while split_point - start > max_ms:
                mid = start + max_ms
                segments_ms.append((start, mid))
                start = mid
            segments_ms.append((start, split_point))
            start = split_point

    if start < total_ms:
        segments_ms.append((start, total_ms))

    if len(segments_ms) >= 2:
        last_start, last_end = segments_ms[-1]
        last_dur = last_end - last_start
        if last_dur < 60_000:
            print(f"最后一段太短 ({last_dur/1000:.1f}s)，合并到前一段")
            prev_start, prev_end = segments_ms[-2]
            segments_ms[-2] = (prev_start, last_end)
            segments_ms.pop(-1)

    segments = []
    for idx, (seg_start, seg_end) in enumerate(segments_ms, 1):
        segment = audio[seg_start:seg_end]
        segment_path = os.path.join(output_dir, f"{base_name}_part{idx}.{audio_format}")
        export_kwargs = {"format": audio_format}
        if audio_format != "wav":
            export_kwargs["bitrate"] = "192k"
        segment.export(segment_path, **export_kwargs)
        segments.append((segment_path, len(segment)/1000))

    return segments


def transcribe_local_audio(file_path, api_key="", language_behaviour="automatic single language",
                           language="", diarization=False, toggle_word_timestamps=False,
                           output_format="json"):
    """转录本地音频文件"""
    global cur_api_key
    
    if language_behaviour == "manual" and language == "":
        language_behaviour = "automatic single language"
    
    if not os.path.exists(file_path):
        print(f"错误: 文件未找到 {file_path}")
        return None
        
    headers = {"x-gladia-key": api_key}
    file_name = os.path.basename(file_path)
    
    try:
        with open(file_path, 'rb') as f:
            files = {'audio': (file_name, f, 'audio/mpeg')}
            payload = {
                "language_behaviour": language_behaviour,
                "diarization": str(diarization).lower(),
                "toggle_word_timestamps": str(toggle_word_timestamps).lower(),
                "output_format": output_format,
                "language": language
            }
            if diarization:
                payload["diarization_max_speakers"] = 2
                
            print(f"\n正在上传并转录本地文件: {file_path} (单词时间戳: {toggle_word_timestamps})...")
            response = requests.post(GLADIA_API_URL, headers=headers, files=files, data=payload)
            
            if 200 <= response.status_code < 300:
                result_json = response.json()
                if 'prediction' in result_json or ('output' in result_json and output_format != "json"):
                    return result_json
                elif 'result_url' in result_json:
                    result_url = result_json['result_url']
                    print(f"文件已提交进行异步处理。结果 URL: {result_url}")
                    return poll_for_result(result_url, api_key)
                else:
                    print("API 响应中未找到 'prediction'/'output' 或 'result_url'。")
                    print(json.dumps(result_json, indent=2, ensure_ascii=False))
                    return None
            else:
                print(f"API 请求错误: {response.status_code}")
                try:
                    error_info = response.json()
                    print("错误详情:", error_info)
                    message = error_info.get("message", "")

                    if "limit exceeded" in message:
                        print("Gladia达到限制，切换下一个api key重试。")
                        cur_api_key = get_next_api_key()
                        if cur_api_key == "":
                            print("无可用api key，无法继续。")
                        else:
                            return transcribe_local_audio(
                                file_path,
                                api_key=cur_api_key,
                                language_behaviour="manual",
                                language=language,
                                diarization=False,
                                toggle_word_timestamps=True,
                                output_format="json"
                            )
                except json.JSONDecodeError:
                    print("错误详情 (非JSON):", response.text)
                return None
    except Exception as e:
        print(f"发生意外错误: {e}")
        return None


def poll_for_result(result_url, api_key, poll_interval_seconds=10, max_attempts=60):
    """轮询异步任务结果"""
    headers = {"x-gladia-key": api_key, "Accept": "application/json"}
    print(f"开始轮询结果 URL: {result_url}")
    
    for attempt in range(max_attempts):
        print(f"轮询尝试 {attempt + 1}/{max_attempts}...")
        try:
            response = requests.get(result_url, headers=headers)
            if response.status_code == 200:
                result_json = response.json()
                status = result_json.get('status', '').lower()
                if ('prediction' in result_json or 'output' in result_json) and (status == 'done' or not status):
                    print("异步转录完成！")
                    return result_json
                elif status == 'processing':
                    print(f"状态: {status}. 等待 {poll_interval_seconds} 秒后重试...")
                elif status == 'error':
                    print("转录任务出错。")
                    print(json.dumps(result_json, indent=2, ensure_ascii=False))
                    return None
                else:
                    print(f"当前状态: '{status}'. 等待 {poll_interval_seconds} 秒后重试...")
            elif response.status_code == 202:
                print(f"服务器接受请求，仍在处理 (状态码 {response.status_code})。等待 {poll_interval_seconds} 秒后重试...")
            else:
                print(f"轮询时发生错误: {response.status_code}")
                try:
                    print("错误详情:", response.json())
                except json.JSONDecodeError:
                    print("错误详情 (非JSON):", response.text)
                return None
        except requests.exceptions.RequestException as e:
            print(f"轮询时发生网络错误: {e}")
        time.sleep(poll_interval_seconds)
    print("已达到最大轮询次数，转录可能仍在进行中或已失败。")
    return None


def get_json_result(transcribe_result, last_result, full_text, start_time):
    """处理转录结果"""
    if not transcribe_result:
        print("结果为空，无法打印或保存。")
        return False
        
    if 'prediction' in transcribe_result:
        prediction = transcribe_result['prediction']
        if isinstance(prediction, list) and prediction:
            for i, item in enumerate(prediction):
                audio_start = item.get('time_begin', 0) + start_time
                audio_end = item.get('time_end', 0) + start_time
                transcript_part = {
                    "text": item.get('transcription', ''),
                    "audio_start": audio_start,
                    "audio_end": audio_end,
                    "duration": audio_end - audio_start,
                    "words": [],
                }
                
                words = transcript_part["words"]
                if 'words' in item and item['words']:
                    for word_info in item['words']:
                        word = word_info.get('word', '')
                        full_text.append(word.strip())
                        
                        word_dict = {
                            "word": word.strip(),
                            "start": word_info.get('time_begin', 0.0) + start_time,
                            "end": word_info.get('time_end', 0.0) + start_time,
                            "score": word_info.get('confidence', 0.0),
                        }
                        words.append(word_dict)
                else:
                    print("  (此片段无单词级别时间戳)")
                    
                last_result.append(transcript_part)

            print(f"\n--- 完整转录 ---")
            return True
    return False


def transcribe_audio_from_gladia(media_path, api_keys, language, json_path, txt_path, min_minutes=5.0):
    """通过Gladia转录音频的对外接口"""
    print(language)
    if language not in languages:
        print(f"不支持的语种 {language}")
        yield f"不支持的语种 {language}"
        return None
    
    output_path = "./gladia_tmp"

    global API_keys, cur_api_key
    API_keys = api_keys
    if cur_api_key == "":
        cur_api_key = get_next_api_key()
    elif cur_api_key not in API_keys:
        cur_api_key = ""
        cur_api_key = get_next_api_key()
        
    if cur_api_key == "":
        print("无可用Gladia Key")
        yield "无可用Gladia Key，请添加Gladia key。"
        raise RuntimeError("无可用Gladia Key，请添加Gladia key。")
    
    # 如果是视频，就先提取里面的音频
    if media_path.lower().endswith((".mp4", ".mov", ".mkv", ".flv", ".avi", ".wmv")):
        print("提取音频")
        yield "提取音频"
        audio_path = extract_audio_from_video(media_path, output_path)
    else:
        audio_path = media_path
    print(audio_path)
    
    # 拆分音频
    audios_list = []
    print("切分音频")
    yield "切分音频"
    audios_list = split_audio_on_silence(audio_path, output_path, min_minutes=min_minutes)
    
    # 开始转录
    cur_start_time = 0
    last_result = []
    full_text_list = []
    print("开始转录音频")
    yield "开始转录音频"
    cur_index = 1
    
    for audio_segm_path, duration in audios_list:
        print(f"{audio_segm_path}: {duration:.1f} 秒")
        yield f"{audio_segm_path}: {cur_index}/{len(audios_list)}"
        cur_index += 1
        result_word_ts = None
        
        for i in range(3):
            result_word_ts = transcribe_local_audio(
                audio_segm_path,
                api_key=cur_api_key,
                language_behaviour="manual",
                language=language,
                diarization=False,
                toggle_word_timestamps=True,
                output_format="json"
            )

            if result_word_ts:
                break
            else:
                print(f"转录失败，自动重试{i}")
                
        if result_word_ts:
            ret = get_json_result(result_word_ts, last_result, full_text_list, cur_start_time)
            if not ret:
                print("转录结果有问题")
                yield "转录结果有问题"
                raise RuntimeError("转录结果有问题")
        else:
            print("转录失败----")
            yield "转录失败----"
            raise RuntimeError("转录失败----")
            
        cur_start_time += duration
    
    # 直接保存最终的json和txt文件
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(last_result, f, indent=2, ensure_ascii=False)

    full_text = " ".join(full_text_list)

    with open(txt_path, 'w', encoding='utf-8') as file:
        file.write(full_text)
        
    try:
        shutil.rmtree(output_path)
    except:
        pass
       
    return "转录完成"
