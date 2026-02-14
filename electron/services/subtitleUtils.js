/**
 * 字幕工具函数 — 完整移植自 core/subtitle_utils.py
 */
const fs = require('fs');
const path = require('path');

// ==================== 语言配置 ====================
const LANGUAGES = {
    "en": { code: "en", language: "english", name: "英语" },
    "zh": { code: "zh", language: "chinese", name: "中文" },
    "de": { code: "de", language: "german", name: "德语" },
    "es": { code: "es", language: "spanish", name: "西班牙语" },
    "ru": { code: "ru", language: "russian", name: "俄语" },
    "ko": { code: "ko", language: "korean", name: "韩语" },
    "fr": { code: "fr", language: "french", name: "法语" },
    "ja": { code: "ja", language: "japanese", name: "日语" },
    "pt": { code: "pt", language: "portuguese", name: "葡萄牙语" },
    "tr": { code: "tr", language: "turkish", name: "土耳其语" },
    "pl": { code: "pl", language: "polish", name: "波兰语" },
    "ca": { code: "ca", language: "catalan", name: "加泰罗尼亚语" },
    "nl": { code: "nl", language: "dutch", name: "荷兰语" },
    "ar": { code: "ar", language: "arabic", name: "阿拉伯语" },
    "sv": { code: "sv", language: "swedish", name: "瑞典语" },
    "it": { code: "it", language: "italian", name: "意大利语" },
    "id": { code: "id", language: "indonesian", name: "印尼语" },
    "hi": { code: "hi", language: "hindi", name: "印地语" },
    "fi": { code: "fi", language: "finnish", name: "芬兰语" },
    "vi": { code: "vi", language: "vietnamese", name: "越南语" },
    "he": { code: "he", language: "hebrew", name: "希伯来语" },
    "uk": { code: "uk", language: "ukrainian", name: "乌克兰语" },
    "el": { code: "el", language: "greek", name: "希腊语" },
    "ms": { code: "ms", language: "malay", name: "马来语" },
    "cs": { code: "cs", language: "czech", name: "捷克语" },
    "ro": { code: "ro", language: "romanian", name: "罗马尼亚语" },
    "da": { code: "da", language: "danish", name: "丹麦语" },
    "hu": { code: "hu", language: "hungarian", name: "匈牙利语" },
    "ta": { code: "ta", language: "tamil", name: "泰米尔语" },
    "no": { code: "no", language: "norwegian", name: "挪威语" },
    "th": { code: "th", language: "thai", name: "泰语" },
    "ur": { code: "ur", language: "urdu", name: "乌尔都语" },
    "hr": { code: "hr", language: "croatian", name: "克罗地亚语" },
    "bg": { code: "bg", language: "bulgarian", name: "保加利亚语" },
    "lt": { code: "lt", language: "lithuanian", name: "立陶宛语" },
    "la": { code: "la", language: "latin", name: "拉丁语" },
    "mi": { code: "mi", language: "maori", name: "毛利语" },
    "ml": { code: "ml", language: "malayalam", name: "马拉雅拉姆语" },
    "cy": { code: "cy", language: "welsh", name: "威尔士语" },
    "sk": { code: "sk", language: "slovak", name: "斯洛伐克语" },
    "te": { code: "te", language: "telugu", name: "泰卢固语" },
    "fa": { code: "fa", language: "persian", name: "波斯语" },
    "lv": { code: "lv", language: "latvian", name: "拉脱维亚语" },
    "bn": { code: "bn", language: "bengali", name: "孟加拉语" },
    "sr": { code: "sr", language: "serbian", name: "塞尔维亚语" },
    "az": { code: "az", language: "azerbaijani", name: "阿塞拜疆语" },
    "sl": { code: "sl", language: "slovenian", name: "斯洛文尼亚语" },
    "kn": { code: "kn", language: "kannada", name: "卡纳达语" },
    "et": { code: "et", language: "estonian", name: "爱沙尼亚语" },
    "mk": { code: "mk", language: "macedonian", name: "马其顿语" },
    "br": { code: "br", language: "breton", name: "布列塔尼语" },
    "eu": { code: "eu", language: "basque", name: "巴斯克语" },
    "is": { code: "is", language: "icelandic", name: "冰岛语" },
    "hy": { code: "hy", language: "armenian", name: "亚美尼亚语" },
    "ne": { code: "ne", language: "nepali", name: "尼泊尔语" },
    "mn": { code: "mn", language: "mongolian", name: "蒙古语" },
    "bs": { code: "bs", language: "bosnian", name: "波斯尼亚语" },
    "kk": { code: "kk", language: "kazakh", name: "哈萨克语" },
    "sq": { code: "sq", language: "albanian", name: "阿尔巴尼亚语" },
    "sw": { code: "sw", language: "swahili", name: "斯瓦希里语" },
    "gl": { code: "gl", language: "galician", name: "加利西亚语" },
    "mr": { code: "mr", language: "marathi", name: "马拉地语" },
    "pa": { code: "pa", language: "punjabi", name: "旁遮普语" },
    "si": { code: "si", language: "sinhala", name: "僧伽罗语" },
    "km": { code: "km", language: "khmer", name: "高棉语" },
    "sn": { code: "sn", language: "shona", name: "绍纳语" },
    "yo": { code: "yo", language: "yoruba", name: "约鲁巴语" },
    "so": { code: "so", language: "somali", name: "索马里语" },
    "af": { code: "af", language: "afrikaans", name: "南非荷兰语" },
    "oc": { code: "oc", language: "occitan", name: "奥克语" },
    "ka": { code: "ka", language: "georgian", name: "格鲁吉亚语" },
    "be": { code: "be", language: "belarusian", name: "白俄罗斯语" },
    "tg": { code: "tg", language: "tajik", name: "塔吉克语" },
    "sd": { code: "sd", language: "sindhi", name: "信德语" },
    "gu": { code: "gu", language: "gujarati", name: "古吉拉特语" },
    "am": { code: "am", language: "amharic", name: "阿姆哈拉语" },
    "yi": { code: "yi", language: "yiddish", name: "意第绪语" },
    "lo": { code: "lo", language: "lao", name: "老挝语" },
    "uz": { code: "uz", language: "uzbek", name: "乌兹别克语" },
    "fo": { code: "fo", language: "faroese", name: "法罗语" },
    "ht": { code: "ht", language: "haitian creole", name: "海地克里奥尔语" },
    "ps": { code: "ps", language: "pashto", name: "普什图语" },
    "tk": { code: "tk", language: "turkmen", name: "土库曼语" },
    "nn": { code: "nn", language: "nynorsk", name: "新挪威语" },
    "mt": { code: "mt", language: "maltese", name: "马耳他语" },
    "sa": { code: "sa", language: "sanskrit", name: "梵语" },
    "lb": { code: "lb", language: "luxembourgish", name: "卢森堡语" },
    "my": { code: "my", language: "myanmar", name: "缅甸语" },
    "bo": { code: "bo", language: "tibetan", name: "藏语" },
    "tl": { code: "tl", language: "tagalog", name: "他加禄语" },
    "mg": { code: "mg", language: "malagasy", name: "马达加斯加语" },
    "as": { code: "as", language: "assamese", name: "阿萨姆语" },
    "tt": { code: "tt", language: "tatar", name: "鞑靼语" },
    "haw": { code: "haw", language: "hawaiian", name: "夏威夷语" },
    "ln": { code: "ln", language: "lingala", name: "林加拉语" },
    "ha": { code: "ha", language: "hausa", name: "豪萨语" },
    "ba": { code: "ba", language: "bashkir", name: "巴什基尔语" },
    "jw": { code: "jw", language: "javanese", name: "爪哇语" },
    "su": { code: "su", language: "sundanese", name: "巽他语" },
    "yue": { code: "yue", language: "cantonese", name: "粤语" },
};

const wordSplitBy = {
    my: " ", ko: " ", vi: " ", es: " ", mn: " ", en: " ", fr: " ",
};

// ==================== 标点符号 ====================
const symbols = [
    '.', ',', ';', ':', '?', '!', '-', '(', ')', '[', ']', '{', '}', '`', '~', "'", '„', '"',
    '"', '"', '\u2018', '\u2019', '…', '—', '_', '、', '￥', '《', '》', '「', '」', '『', '』',
    '【', '】', '。', '・', 'ー', '（', '）', '·', '……', '——', '–', ': „',
    '！', '？', '；', '：', '［', '］', '｛', '｝', '｜', '／',
    '＊', '＆', '％', '＃', '＠', '＋', '＝', '＄', '＾', '＿', '｀', '〜', '＼',
    '．', '＜', '＞', '⟨', '⟩', '«', '»', '‹', '›', '‚', '•', '‣', '⁃', '․', '′', '″',
    '‴', '‵', '‶', '‷', '§', '¶', '⁋', '†', '‡', '⸺', '⸻', '❛', '❜', '❝', '❞',
    '❡', '❢', '❣', '❯', '❮', '❭', '❬', '❱', '❲', '❳', '❴', '❵', '¿', ',', ' ',
];

// 构建匹配正则（排除空格）
const _symbolsEscaped = symbols.filter(s => s.trim() !== '').map(s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
const _symbolsPattern = new RegExp(_symbolsEscaped.join('|'), 'g');

/** 检查文本是否只包含符号 */
function isOnlySymbols(text) {
    for (const ch of text) {
        if (!symbols.includes(ch)) return false;
    }
    return true;
}

/** 移除所有符号 */
function removeSymbols(text) {
    return text.replace(_symbolsPattern, '');
}

/** 把所有标点符号替换成句号 */
function replaceSymbolsToOne(text) {
    return text.replace(_symbolsPattern, '.');
}

/** 检查是否为标点 */
function isPunctuation(text) {
    // 简化版：检查是否全部为标点字符
    for (const ch of text) {
        if (/[a-zA-Z0-9\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]/.test(ch)) return false;
    }
    return true;
}

// ==================== 语言处理 ====================

/** 根据中文名称获取语言代码 */
function changeLanguage(selected) {
    for (const [code, info] of Object.entries(LANGUAGES)) {
        if (info.name === selected) return code;
    }
    return selected; // 如果找不到，返回原值
}

/** 根据语言代码获取语言英文名 */
function getLanguage(currentLanguage) {
    const info = LANGUAGES[currentLanguage];
    return info ? info.language : currentLanguage;
}

// ==================== 文件操作 ====================

/** 写入文件到指定路径 */
function writeToPath(content, dirPath, name = 'file', ext = 'srt') {
    const filePath = path.join(dirPath, `${name}.${ext}`);
    fs.writeFileSync(filePath, content, 'utf-8');
}

/** 读取 JSON 文件 */
function readObjectFromJson(filePath) {
    return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
}

/** 读取文本文件并移除换行 */
function readTextFileRemoveBreak(filePath) {
    const lines = fs.readFileSync(filePath, 'utf-8').split('\n');
    let result = '';
    for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed) {
            result += wordSplitBy.en + trimmed;
        }
    }
    return result.trimStart();
}

/**
 * 读取 Google Doc 格式的文本文件
 * 支持 ## 标记的段落结尾，文本替换规则
 */
function readTextWithGoogleDoc(filePath, textReplaceDict = {}, ignoreCase = true, preserveFullWidthSpaces = false) {
    const document = {
        title: '',
        language: '',
        chapter: '',
        contents: [],
    };

    // 查找匹配的替换规则
    let textReplacesDict = {};
    const fileName = path.basename(filePath);
    for (const [k, v] of Object.entries(textReplaceDict)) {
        const langName = k;
        const code = v.Code || '';
        if (fileName.startsWith(langName) || (code !== '' && fileName.toLowerCase().startsWith(code.toLowerCase()))) {
            textReplacesDict = v.Text || {};
            break;
        }
    }

    let paragraphCounter = 1;
    const fileContent = fs.readFileSync(filePath, 'utf-8');
    const lines = fileContent.split('\n');

    for (let rawLine of lines) {
        let line = rawLine.trim();
        if (!line) continue;

        let contentType = 'text';

        if (line.startsWith('##') || line.endsWith('##')) {
            line = line.replace(/##/g, '').replace(/\n/g, '').replace(/\r/g, '');
            if (preserveFullWidthSpaces) {
                line = line.replace(/[ \t\r\n\f\v]/g, wordSplitBy.en);
            } else {
                line = line.replace(/\s+/g, wordSplitBy.en);
            }
            contentType = 'end';
        } else {
            line = line.replace(/##/g, '').replace(/\n/g, '').replace(/\r/g, '');
            if (preserveFullWidthSpaces) {
                line = line.replace(/[ \t\r\n\f\v]/g, wordSplitBy.en);
            } else {
                line = line.replace(/\s+/g, wordSplitBy.en);
            }
        }

        line = line.trim();

        // 应用字符替换规则
        for (const [k, v] of Object.entries(textReplacesDict)) {
            if (isPunctuation(k)) {
                // 标点替换：直接替换
                line = line.split(k).join(v);
            } else if (ignoreCase) {
                // 单词边界替换（不区分大小写）
                try {
                    const regex = new RegExp(`\\b${k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'gi');
                    line = line.replace(regex, v);
                } catch { /* 忽略无效正则 */ }
            } else {
                try {
                    const regex = new RegExp(`\\b${k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'g');
                    line = line.replace(regex, v);
                } catch { /* 忽略无效正则 */ }
            }
        }

        document.contents.push({
            paragraph: paragraphCounter,
            type: contentType,
            css: '',
            content: line,
        });

        if (contentType === 'end') {
            paragraphCounter++;
        }
    }

    return document;
}

/** 格式化时间戳为 SRT 格式 (HH:MM:SS,mmm) */
function formatTimestamp(seconds, alwaysIncludeHours = false, decimalMarker = '.') {
    if (seconds < 0) seconds = 0;
    let milliseconds = Math.round(seconds * 1000);
    const hours = Math.floor(milliseconds / 3600000);
    milliseconds -= hours * 3600000;
    const minutes = Math.floor(milliseconds / 60000);
    milliseconds -= minutes * 60000;
    const secs = Math.floor(milliseconds / 1000);
    milliseconds -= secs * 1000;

    const hoursMarker = (alwaysIncludeHours || hours > 0)
        ? `${String(hours).padStart(2, '0')}:` : '';
    return `${hoursMarker}${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}${decimalMarker}${String(milliseconds).padStart(3, '0')}`;
}

module.exports = {
    LANGUAGES,
    wordSplitBy,
    symbols,
    isOnlySymbols,
    removeSymbols,
    replaceSymbolsToOne,
    isPunctuation,
    changeLanguage,
    getLanguage,
    writeToPath,
    readObjectFromJson,
    readTextFileRemoveBreak,
    readTextWithGoogleDoc,
    formatTimestamp,
};
