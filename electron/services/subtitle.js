/**
 * SRT 字幕处理服务
 * 替代 Python 的 SrtParse 和字幕相关 API
 */
const fs = require('fs');
const path = require('path');

// ==================== SRT 解析 ====================

/**
 * 解析 SRT 文件，返回字幕段数组
 * 每段: { index, start, end, text }
 * start/end 均为毫秒
 */
function parseSRT(content) {
    const blocks = content.replace(/\r\n/g, '\n').split(/\n\n+/).filter(b => b.trim());
    const result = [];

    for (const block of blocks) {
        const lines = block.trim().split('\n');
        if (lines.length < 2) continue;

        // 序号行
        const index = parseInt(lines[0].trim());
        if (isNaN(index)) continue;

        // 时间行
        const timeMatch = lines[1].match(
            /(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})/
        );
        if (!timeMatch) continue;

        const startMs = parseInt(timeMatch[1]) * 3600000 + parseInt(timeMatch[2]) * 60000 +
            parseInt(timeMatch[3]) * 1000 + parseInt(timeMatch[4]);
        const endMs = parseInt(timeMatch[5]) * 3600000 + parseInt(timeMatch[6]) * 60000 +
            parseInt(timeMatch[7]) * 1000 + parseInt(timeMatch[8]);

        const text = lines.slice(2).join('\n').trim();
        const charCount = text.replace(/[\s\n]/g, '').length;

        result.push({ index, start: startMs, end: endMs, text, char_count: charCount });
    }

    return result;
}

/** 毫秒转 SRT 时间码 */
function msToTimecode(ms) {
    const h = Math.floor(ms / 3600000);
    const m = Math.floor((ms % 3600000) / 60000);
    const s = Math.floor((ms % 60000) / 1000);
    const msRem = ms % 1000;
    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')},${String(msRem).padStart(3, '0')}`;
}

/** 写 SRT 文件 */
function writeSRT(items, outputPath) {
    const content = items.map((item, i) => {
        const idx = item.index || (i + 1);
        return `${idx}\n${msToTimecode(item.start)} --> ${msToTimecode(item.end)}\n${item.text}`;
    }).join('\n\n') + '\n';
    fs.writeFileSync(outputPath, content, 'utf-8');
}

// ==================== SRT 操作 ====================

/**
 * 调整 SRT 时间
 * 对应 Python 的 srt_info.updateSrt(interval_time, char_time, min_char_count, scale)
 */
function adjustSRT(srcPath, options = {}) {
    const {
        intervalTime = 1.0,
        charTime = 0.1,
        minCharCount = 20,
        scale = 1.0,
        ignore = '?—:„";/!',
    } = options;

    const content = fs.readFileSync(srcPath, 'utf-8');
    const items = parseSRT(content);

    const ignoreSet = new Set(ignore.split(''));

    // 根据参数调整每条字幕的时间
    for (const item of items) {
        // 计算有效字符数（排除忽略字符和空格）
        let effectiveChars = 0;
        for (const ch of item.text) {
            if (!ignoreSet.has(ch) && ch !== ' ' && ch !== '\n') {
                effectiveChars++;
            }
        }

        const charCount = Math.max(effectiveChars, minCharCount);
        const duration = (charCount * charTime + intervalTime) * 1000 * scale;
        item.end = item.start + Math.round(duration);
    }

    const newPath = srcPath.replace('.srt', '_new.srt');
    writeSRT(items, newPath);
    return { message: '调整完成', output_path: newPath };
}

/** 生成无缝 SRT -- 每条字幕的 end 等于下一条的 start */
function seamlessSRT(srcPath) {
    const content = fs.readFileSync(srcPath, 'utf-8');
    const items = parseSRT(content);

    for (let i = 0; i < items.length - 1; i++) {
        items[i].end = items[i + 1].start;
    }

    const newPath = srcPath.replace('.srt', '_seamless.srt');
    writeSRT(items, newPath);
    return { message: '生成完成', output_path: newPath };
}

/** 计算字符时间 */
function computeCharTime(refPath, intervalTime = 1.0) {
    const content = fs.readFileSync(refPath, 'utf-8');
    const items = parseSRT(content);

    let totalTime = 0;
    let totalChars = 0;

    for (const item of items) {
        const duration = item.end - item.start - intervalTime * 1000;
        if (duration > 0) {
            totalTime += duration;
            totalChars += item.char_count;
        }
    }

    const charTime = totalChars > 0 ? (totalTime / totalChars / 1000) : 0.1;
    return { char_time: charTime };
}

module.exports = {
    parseSRT,
    writeSRT,
    msToTimecode,
    adjustSRT,
    seamlessSRT,
    computeCharTime,
};
