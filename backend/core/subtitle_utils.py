"""
字幕工具函数 - 从 SW_GenSubTitle/utils.py 移植
"""
import os
import json
import re
import unicodedata

# 模型支持语言
LANGUAGES = {
    "en": { "code":"en", "language": "english", "name": "英语"},
    "zh": { "code":"zh", "language": "chinese", "name": "中文"},
    "de": { "code":"de", "language": "german", "name": "德语"},
    "es": { "code":"es", "language": "spanish", "name": "西班牙语"},
    "ru": { "code":"ru", "language": "russian", "name": "俄语"},
    "ko": { "code":"ko", "language": "korean", "name": "韩语"},
    "fr": { "code":"fr", "language": "french", "name": "法语"},
    "ja": { "code":"ja", "language": "japanese", "name": "日语"},
    "pt": { "code":"pt", "language": "portuguese", "name": "葡萄牙语"},
    "tr": { "code":"tr", "language": "turkish", "name": "土耳其语"},
    "pl": { "code":"pl", "language": "polish", "name": "波兰语"},
    "ca": { "code":"ca", "language": "catalan", "name": "加泰罗尼亚语"},
    "nl": { "code":"nl", "language": "dutch", "name": "荷兰语"},
    "ar": { "code":"ar", "language": "arabic", "name": "阿拉伯语"},
    "sv": { "code":"sv", "language": "swedish", "name": "瑞典语"},
    "it": { "code":"it", "language": "italian", "name": "意大利语"},
    "id": { "code":"id", "language": "indonesian", "name": "印尼语"},
    "hi": { "code":"hi", "language": "hindi", "name": "印地语"},
    "fi": { "code":"fi", "language": "finnish", "name": "芬兰语"},
    "vi": { "code":"vi", "language": "vietnamese", "name": "越南语"},
    "he": { "code":"he", "language": "hebrew", "name": "希伯来语"},
    "uk": { "code":"uk", "language": "ukrainian", "name": "乌克兰语"},
    "el": { "code":"el", "language": "greek", "name": "希腊语"},
    "ms": { "code":"ms", "language": "malay", "name": "马来语"},
    "cs": { "code":"cs", "language": "czech", "name": "捷克语"},
    "ro": { "code":"ro", "language": "romanian", "name": "罗马尼亚语"},
    "da": { "code":"da", "language": "danish", "name": "丹麦语"},
    "hu": { "code":"hu", "language": "hungarian", "name": "匈牙利语"},
    "ta": { "code":"ta", "language": "tamil", "name": "泰米尔语"},
    "no": { "code":"no", "language": "norwegian", "name": "挪威语"},
    "th": { "code":"th", "language": "thai", "name": "泰语"},
    "ur": { "code":"ur", "language": "urdu", "name": "乌尔都语"},
    "hr": { "code":"hr", "language": "croatian", "name": "克罗地亚语"},
    "bg": { "code":"bg", "language": "bulgarian", "name": "保加利亚语"},
    "lt": { "code":"lt", "language": "lithuanian", "name": "立陶宛语"},
    "la": { "code":"la", "language": "latin", "name": "拉丁语"},
    "mi": { "code":"mi", "language": "maori", "name": "毛利语"},
    "ml": { "code":"ml", "language": "malayalam", "name": "马拉雅拉姆语"},
    "cy": { "code":"cy", "language": "welsh", "name": "威尔士语"},
    "sk": { "code":"sk", "language": "slovak", "name": "斯洛伐克语"},
    "te": { "code":"te", "language": "telugu", "name": "泰卢固语"},
    "fa": { "code":"fa", "language": "persian", "name": "波斯语"},
    "lv": { "code":"lv", "language": "latvian", "name": "拉脱维亚语"},
    "bn": { "code":"bn", "language": "bengali", "name": "孟加拉语"},
    "sr": { "code":"sr", "language": "serbian", "name": "塞尔维亚语"},
    "az": { "code":"az", "language": "azerbaijani", "name": "阿塞拜疆语"},
    "sl": { "code":"sl", "language": "slovenian", "name": "斯洛文尼亚语"},
    "kn": { "code":"kn", "language": "kannada", "name": "卡纳达语"},
    "et": { "code":"et", "language": "estonian", "name": "爱沙尼亚语"},
    "mk": { "code":"mk", "language": "macedonian", "name": "马其顿语"},
    "br": { "code":"br", "language": "breton", "name": "布列塔尼语"},
    "eu": { "code":"eu", "language": "basque", "name": "巴斯克语"},
    "is": { "code":"is", "language": "icelandic", "name": "冰岛语"},
    "hy": { "code":"hy", "language": "armenian", "name": "亚美尼亚语"},
    "ne": { "code":"ne", "language": "nepali", "name": "尼泊尔语"},
    "mn": { "code":"mn", "language": "mongolian", "name": "蒙古语"},
    "bs": { "code":"bs", "language": "bosnian", "name": "波斯尼亚语"},
    "kk": { "code":"kk", "language": "kazakh", "name": "哈萨克语"},
    "sq": { "code":"sq", "language": "albanian", "name": "阿尔巴尼亚语"},
    "sw": { "code":"sw", "language": "swahili", "name": "斯瓦希里语"},
    "gl": { "code":"gl", "language": "galician", "name": "加利西亚语"},
    "mr": { "code":"mr", "language": "marathi", "name": "马拉地语"},
    "pa": { "code":"pa", "language": "punjabi", "name": "旁遮普语"},
    "si": { "code":"si", "language": "sinhala", "name": "僧伽罗语"},
    "km": { "code":"km", "language": "khmer", "name": "高棉语"},
    "sn": { "code":"sn", "language": "shona", "name": "绍纳语"},
    "yo": { "code":"yo", "language": "yoruba", "name": "约鲁巴语"},
    "so": { "code":"so", "language": "somali", "name": "索马里语"},
    "af": { "code":"af", "language": "afrikaans", "name": "南非荷兰语"},
    "oc": { "code":"oc", "language": "occitan", "name": "奥克语"},
    "ka": { "code":"ka", "language": "georgian", "name": "格鲁吉亚语"},
    "be": { "code":"be", "language": "belarusian", "name": "白俄罗斯语"},
    "tg": { "code":"tg", "language": "tajik", "name": "塔吉克语"},
    "sd": { "code":"sd", "language": "sindhi", "name": "信德语"},
    "gu": { "code":"gu", "language": "gujarati", "name": "古吉拉特语"},
    "am": { "code":"am", "language": "amharic", "name": "阿姆哈拉语"},
    "yi": { "code":"yi", "language": "yiddish", "name": "意第绪语"},
    "lo": { "code":"lo", "language": "lao", "name": "老挝语"},
    "uz": { "code":"uz", "language": "uzbek", "name": "乌兹别克语"},
    "fo": { "code":"fo", "language": "faroese", "name": "法罗语"},
    "ht": { "code":"ht", "language": "haitian creole", "name": "海地克里奥尔语"},
    "ps": { "code":"ps", "language": "pashto", "name": "普什图语"},
    "tk": { "code":"tk", "language": "turkmen", "name": "土库曼语"},
    "nn": { "code":"nn", "language": "nynorsk", "name": "新挪威语"},
    "mt": { "code":"mt", "language": "maltese", "name": "马耳他语"},
    "sa": { "code":"sa", "language": "sanskrit", "name": "梵语"},
    "lb": { "code":"lb", "language": "luxembourgish", "name": "卢森堡语"},
    "my": { "code":"my", "language": "myanmar", "name": "缅甸语"},
    "bo": { "code":"bo", "language": "tibetan", "name": "藏语"},
    "tl": { "code":"tl", "language": "tagalog", "name": "他加禄语"},
    "mg": { "code":"mg", "language": "malagasy", "name": "马达加斯加语"},
    "as": { "code":"as", "language": "assamese", "name": "阿萨姆语"},
    "tt": { "code":"tt", "language": "tatar", "name": "鞑靼语"},
    "haw": { "code":"haw", "language": "hawaiian", "name": "夏威夷语"},
    "ln": { "code":"ln", "language": "lingala", "name": "林加拉语"},
    "ha": { "code":"ha", "language": "hausa", "name": "豪萨语"},
    "ba": { "code":"ba", "language": "bashkir", "name": "巴什基尔语"},
    "jw": { "code":"jw", "language": "javanese", "name": "爪哇语"},
    "su": { "code":"su", "language": "sundanese", "name": "巽他语"},
    "yue": { "code":"yue", "language": "cantonese", "name": "粤语"}
}

word_split_by = {
    "my": " ",
    "ko": " ",
    "vi": " ",
    "es": " ",
    "mn": " ",
    "en": " ",
    "fr": " ",
}


def change_language(selected):
    """根据中文名称获取语言代码"""
    current_language = next(code for code, info in LANGUAGES.items() if info["name"] == selected)
    return current_language


def get_language(current_language):
    """根据语言代码获取语言英文名"""
    x = next(info for info in LANGUAGES.values() if info["code"] == current_language)
    return x["language"]


def wirite_to_path(save, path, name="file", type="srt"):
    """写入文件到指定路径"""
    name = os.path.join(path, f'{name}.{type}')
    with open(name, 'w', encoding='utf-8') as f:
        f.write(save)


def wirite_to_local(save, name, type="json"):
    """写入到本地log目录"""
    if not os.path.exists("log"):
        os.makedirs("log")
    log_json = ""
    try:
        if type == "json":
            log_json = json.dumps(save, indent=4, ensure_ascii=False)
            name = f'./log/{name}.{type}'
        else:
            log_json = save
            name = f'./log/{name}.{type}'
    except Exception as e:
        print(e)
        return
    with open(name, 'w', encoding='utf-8') as f:
        f.write(log_json)


# 标点符号列表
symbols = [
    '.', ',', ';', ':', '?', '!', '-', '(', ')', '[', ']', '{', '}', '`', '~', "'", "„", '"',
    '"', '"', ''', ''', '…', '—', '_', '—', '、', '￥', '《', '》', '「', '」', '『', '』',
    '【', '】', '。', '、', '・', 'ー', '（', '）', '（', '）', '、', '·', '……', '——', "–", ": „",
    '。', '、', '「', '」', '（', '）', '「', '」', '《', '》', '『', '』', '【', '】', " ",
    '、', '。', '！', '？', '；', '：', '（', '）', '［', '］', '｛', '｝', '｜', '／',
    '＊', '＆', '％', '＃', '＠', '＊', '＋', '＝', '＄', '＾', '＿', '｀', '〜', '＼',
    '｜', '．', '＜', '＞', '⟨', '⟩', '«', '»', '‹', '›', '‚', '•', '‣', '⁃', '․', '′', '″', '‴', '‵', '‶', '‷', '§', '¶',
    '⁋', '†', '‡', '⸺', '⸻', '❛', '❜', '❝', '❞', '❡', '❢', '❣', '❯', '❮', '❭', '❬',
    '❱', '❲', '❳', '❴', '❵', "¿", ","
]


def is_only_symbols(text):
    """检查文本是否只包含符号"""
    for char in text:
        if char not in symbols:
            return False
    return True


def remove_symbols(text):
    """移除所有符号"""
    pattern = re.compile('|'.join(re.escape(sym) for sym in symbols))
    cleaned_text = pattern.sub("", text)
    return cleaned_text


def replace_symbols_to_one(text):
    """把所有标点符号替换成句号"""
    pattern = re.compile('|'.join(re.escape(sym) for sym in symbols if sym.strip() != ""))
    cleaned_text = pattern.sub(".", text)
    return cleaned_text


def is_punctuation(text):
    """检查是否为标点"""
    return all(unicodedata.category(char).startswith('P') for char in text)


def read_text_file_remove_break(file_path):
    """读取文件并移除换行"""
    paragraph_text = ""
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    for txt in lines:
        txt = txt.strip()
        if txt:
            paragraph_text += word_split_by["en"] + txt
    return paragraph_text.lstrip()


def read_text_file(file_path):
    """读取整个文件"""
    with open(file_path, 'r', encoding='utf-8') as file:
        paragraph_text = file.read()
    return paragraph_text


def read_text_with_google_doc(file_path, text_replace_dict={}, ignore_case=True, preserve_full_width_spaces=False):
    """读取Google Doc格式的文本文件"""
    document = {
        "title": "",
        "language": '',
        "chapter": "",
        "contents": []
    }
    
    text_replaces_dict = {}
    file_name = os.path.basename(file_path)
    for k, v in text_replace_dict.items():
        lang_name = k
        code = v.get("Code", "")
        if file_name.startswith(lang_name) or \
            (code != "" and file_name.lower().startswith(code.lower())):
            text_replaces_dict = v["Text"]
            break
            
    paragraph_counter = 1
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            content_type = "text"
            if line.startswith("##") or line.endswith("##"):
                if preserve_full_width_spaces:
                    line = re.sub(r"[ \t\r\n\f\v]", word_split_by["en"], line.replace('\n', '').replace('\r', '').replace('##', ''))
                else:
                    line = re.sub(r'\s+', word_split_by["en"], line.replace('\n', '').replace('\r', '').replace('##', ''))
                content_type = "end"
            else:
                if preserve_full_width_spaces:
                    line = re.sub(r"[ \t\r\n\f\v]", word_split_by["en"], line.replace('\n', '').replace('\r', '').replace('##', ''))
                else:
                    line = re.sub(r'\s+', word_split_by["en"], line.replace('\n', '').replace('\r', '').replace('##', ''))
                
            line = line.strip()
            for k, v in text_replaces_dict.items():
                if is_punctuation(k):
                    line = line.replace(k, v)
                elif ignore_case:
                    line = re.sub(f'\\b{k}\\b', v, line, flags=re.IGNORECASE)
                else:
                    line = re.sub(f'\\b{k}\\b', v, line)
                
            content = {
                "paragraph": paragraph_counter,
                "type": content_type,
                "css": '',
                "content": line
            }
            document["contents"].append(content)

            if content_type == "end":
                paragraph_counter += 1

    return document


def read_object_from_json(file_path):
    """读取JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def format_timestamp(seconds: float, always_include_hours: bool = False, 
                    decimal_marker: str = "."):
    """格式化时间戳"""
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )
