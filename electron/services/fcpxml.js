/**
 * SRT 转 FCPXML 模块 — 完整移植自 core/srt_to_fcpxml.py
 * 生成 Final Cut Pro XML 时间线
 */
const fs = require('fs');
const path = require('path');

// ==================== SRT 解析 ====================

/** 解析 SRT 时间码 "HH:MM:SS,mmm" 为毫秒 */
function parseSRTTime(timeStr) {
    const match = timeStr.trim().match(/(\d+):(\d+):(\d+)[,.](\d+)/);
    if (!match) return 0;
    const [, h, m, s, ms] = match;
    return parseInt(h) * 3600000 + parseInt(m) * 60000 + parseInt(s) * 1000 + parseInt(ms);
}

/** 解析 SRT 字符串为条目数组 */
function parseSRTString(srtContent) {
    const entries = [];
    const blocks = srtContent.trim().split(/\n\s*\n/);
    for (const block of blocks) {
        const lines = block.trim().split('\n');
        if (lines.length < 3) continue;
        const timeLine = lines[1];
        const timeParts = timeLine.split('-->');
        if (timeParts.length !== 2) continue;
        const start = parseSRTTime(timeParts[0]);
        const end = parseSRTTime(timeParts[1]);
        const text = lines.slice(2).join('\n');
        entries.push({
            index: parseInt(lines[0]) || entries.length + 1,
            start,   // 毫秒
            end,     // 毫秒
            text,
            duration: end - start,
        });
    }
    return entries;
}

// ==================== FCPXML 生成 ====================

/** 把 SRT 时间 (ms) 根据帧率转换为分数形式的秒字符串 */
function getFractionTime(timeMs, fps = 30) {
    const frame = Math.floor(timeMs / (1000 / fps));
    // 简化分数
    const num = frame * 100;
    const den = fps * 100;
    const g = gcd(Math.abs(num), Math.abs(den));
    return `${num / g}/${den / g}s`;
}

/** 最大公约数 */
function gcd(a, b) {
    if (b === 0) return a;
    return gcd(b, a % b);
}

/** XML 特殊字符转义 */
function xmlEscape(str) {
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&apos;');
}

/**
 * 把多个 SRT 字符串转换到一个 FCPXML 文件中
 * 完整移植自 SrtsToFcpxml
 *
 * @param {string}   sourceSrt       原文 SRT 字符串
 * @param {string[]} transSrts       翻译 SRT 字符串数组
 * @param {string}   savePath        输出 FCPXML 文件路径
 * @param {boolean}  seamlessFcpxml  是否无缝模式
 */
function SrtsToFcpxml(sourceSrt, transSrts, savePath, seamlessFcpxml) {
    const sourceSubs = parseSRTString(sourceSrt);
    const count = sourceSubs.length;
    if (count === 0) {
        console.log('Srt 字幕长度为0');
        return;
    }

    // 读取字幕样式设置
    let subtitleSetting = {};
    const settingPath = path.join(process.cwd(), 'subtitle_pref.json');
    if (fs.existsSync(settingPath)) {
        try {
            subtitleSetting = JSON.parse(fs.readFileSync(settingPath, 'utf-8'));
        } catch { /* 忽略读取错误 */ }
    }
    // 也尝试从 backend 目录读取
    const { getBackendDir } = require('./settings');
    const backendSettingPath = path.join(getBackendDir(), 'subtitle_pref.json');
    if (!Object.keys(subtitleSetting).length && fs.existsSync(backendSettingPath)) {
        try {
            subtitleSetting = JSON.parse(fs.readFileSync(backendSettingPath, 'utf-8'));
        } catch { /* 忽略 */ }
    }

    // 项目名称（从文件名提取）
    const projectName = path.basename(savePath, path.extname(savePath));

    // ---- 构建 XML 字符串（手写 XML，避免引入 XML 库） ----
    let xml = '';
    xml += '<?xml version="1.0" encoding="UTF-8"?>\n';
    xml += '<fcpxml version="1.9">\n';
    xml += '\t<resources>\n';
    xml += '\t\t<format name="FFVideoFormat1080p30" frameDuration="1/30s" width="1920" height="1080" id="r0"/>\n';
    xml += '\t\t<effect name="Basic Title" uid=".../Titles.localized/Bumper:Opener.localized/Basic Title.localized/Basic Title.moti" id="r1"/>\n';
    xml += '\t</resources>\n';
    xml += '\t<library>\n';
    xml += `\t\t<event name="${xmlEscape(projectName)}">\n`;
    xml += `\t\t\t<project name="${xmlEscape(projectName)}">\n`;

    const duration = getFractionTime(sourceSubs[count - 1].end);
    xml += `\t\t\t\t<sequence tcFormat="NDF" tcStart="0/1s" duration="${duration}" format="r0">\n`;
    xml += '\t\t\t\t\t<spine>\n';

    // 保存每个 title 节点的子 title（翻译层）位置
    const titlePositions = [];
    let totalIndex = 0;
    let preSubEnd = 0;

    for (let i = 0; i < count; i++) {
        const sub = sourceSubs[i];

        // 非无缝模式：插入 gap
        if (!seamlessFcpxml) {
            if (preSubEnd < sub.start) {
                const gapOffset = getFractionTime(preSubEnd);
                const gapDuration = getFractionTime(sub.start - preSubEnd);
                xml += `\t\t\t\t\t\t<gap name="Gap" start="3600/1s" offset="${gapOffset}" duration="${gapDuration}"/>\n`;
            }
        }

        let start = sub.start;
        if (seamlessFcpxml && sub.start > 34) {
            start = sub.start - 34;
        }
        const startStr = getFractionTime(start);

        let dur = sub.duration;
        if (seamlessFcpxml && i < count - 1) {
            dur = sourceSubs[i + 1].start - start;
        }
        const durationStr = getFractionTime(dur);

        // 源文本 title
        const srcAlignment = subtitleSetting.source_alignment || 'center';
        const srcFontColor = subtitleSetting.source_fontColor || '1 1 1 1';
        const srcBold = subtitleSetting.source_bold || '0';
        const srcStrokeColor = subtitleSetting.source_strokeColor || '1 1 1 1';
        const srcFont = subtitleSetting.source_font || 'Arial';
        const srcFontSize = subtitleSetting.source_fontSize || '50';
        const srcItalic = subtitleSetting.source_italic || '0';
        const srcStrokeWidth = subtitleSetting.source_strokeWidth || '0';
        const srcLineSpacing = subtitleSetting.source_lineSpacing || '0';
        const srcPosY = subtitleSetting.source_pos || '-45';

        const titleText = sub.text.trim().replace(/@/g, '\n');

        // 记录 title 在 XML 中的位置（用于后续插入翻译层）
        const titleStart = xml.length;

        xml += `\t\t\t\t\t\t<title name="Subtitle" ref="r1" enabled="1" start="${startStr}" offset="${startStr}" duration="${durationStr}">\n`;
        xml += `\t\t\t\t\t\t\t<text roll-up-height="0">\n`;
        xml += `\t\t\t\t\t\t\t\t<text-style ref="ts${totalIndex}">${xmlEscape(titleText)}</text-style>\n`;
        xml += `\t\t\t\t\t\t\t</text>\n`;
        xml += `\t\t\t\t\t\t\t<text-style-def id="ts${totalIndex}">\n`;
        xml += `\t\t\t\t\t\t\t\t<text-style alignment="${srcAlignment}" fontColor="${srcFontColor}" bold="${srcBold}" strokeColor="${srcStrokeColor}" font="${srcFont}" fontSize="${srcFontSize}" italic="${srcItalic}" strokeWidth="${srcStrokeWidth}" lineSpacing="${srcLineSpacing}"/>\n`;
        xml += `\t\t\t\t\t\t\t</text-style-def>\n`;
        xml += `\t\t\t\t\t\t\t<adjust-conform type="fit"/>\n`;
        xml += `\t\t\t\t\t\t\t<adjust-transform scale="1 1" position="0 ${srcPosY}" anchor="0 0"/>\n`;

        titlePositions.push(xml.length); // 记录 </title> 前的位置

        xml += `\t\t\t\t\t\t</title>\n`;

        totalIndex++;
        preSubEnd = sub.end;
    }

    // ---- 添加翻译层 ----
    // 由于需要在每个 title 内部插入子 title，我们需要重新构建
    // 为简化，我们先闭合之前的内容，然后用字符串插入的方式添加翻译层
    // 这里采用重新生成的方式

    if (transSrts && transSrts.length > 0) {
        // 重新生成完整 XML（包含翻译层）
        xml = '';
        xml += '<?xml version="1.0" encoding="UTF-8"?>\n';
        xml += '<fcpxml version="1.9">\n';
        xml += '\t<resources>\n';
        xml += '\t\t<format name="FFVideoFormat1080p30" frameDuration="1/30s" width="1920" height="1080" id="r0"/>\n';
        xml += '\t\t<effect name="Basic Title" uid=".../Titles.localized/Bumper:Opener.localized/Basic Title.localized/Basic Title.moti" id="r1"/>\n';
        xml += '\t</resources>\n';
        xml += '\t<library>\n';
        xml += `\t\t<event name="${xmlEscape(projectName)}">\n`;
        xml += `\t\t\t<project name="${xmlEscape(projectName)}">\n`;
        xml += `\t\t\t\t<sequence tcFormat="NDF" tcStart="0/1s" duration="${duration}" format="r0">\n`;
        xml += '\t\t\t\t\t<spine>\n';

        totalIndex = 0;
        preSubEnd = 0;

        const parsedTransSubs = transSrts.map(s => parseSRTString(s));

        for (let i = 0; i < count; i++) {
            const sub = sourceSubs[i];

            if (!seamlessFcpxml && preSubEnd < sub.start) {
                const gapOffset = getFractionTime(preSubEnd);
                const gapDuration = getFractionTime(sub.start - preSubEnd);
                xml += `\t\t\t\t\t\t<gap name="Gap" start="3600/1s" offset="${gapOffset}" duration="${gapDuration}"/>\n`;
            }

            let start = sub.start;
            if (seamlessFcpxml && sub.start > 34) start = sub.start - 34;
            const startStr = getFractionTime(start);

            let dur = sub.duration;
            if (seamlessFcpxml && i < count - 1) dur = sourceSubs[i + 1].start - start;
            const durationStr = getFractionTime(dur);

            const srcAlignment = subtitleSetting.source_alignment || 'center';
            const srcFontColor = subtitleSetting.source_fontColor || '1 1 1 1';
            const srcBold = subtitleSetting.source_bold || '0';
            const srcStrokeColor = subtitleSetting.source_strokeColor || '1 1 1 1';
            const srcFont = subtitleSetting.source_font || 'Arial';
            const srcFontSize = subtitleSetting.source_fontSize || '50';
            const srcItalic = subtitleSetting.source_italic || '0';
            const srcStrokeWidth = subtitleSetting.source_strokeWidth || '0';
            const srcLineSpacing = subtitleSetting.source_lineSpacing || '0';
            const srcPosY = subtitleSetting.source_pos || '-45';

            const titleText = sub.text.trim().replace(/@/g, '\n');

            xml += `\t\t\t\t\t\t<title name="Subtitle" ref="r1" enabled="1" start="${startStr}" offset="${startStr}" duration="${durationStr}">\n`;
            xml += `\t\t\t\t\t\t\t<text roll-up-height="0">\n`;
            xml += `\t\t\t\t\t\t\t\t<text-style ref="ts${totalIndex}">${xmlEscape(titleText)}</text-style>\n`;
            xml += `\t\t\t\t\t\t\t</text>\n`;
            xml += `\t\t\t\t\t\t\t<text-style-def id="ts${totalIndex}">\n`;
            xml += `\t\t\t\t\t\t\t\t<text-style alignment="${srcAlignment}" fontColor="${srcFontColor}" bold="${srcBold}" strokeColor="${srcStrokeColor}" font="${srcFont}" fontSize="${srcFontSize}" italic="${srcItalic}" strokeWidth="${srcStrokeWidth}" lineSpacing="${srcLineSpacing}"/>\n`;
            xml += `\t\t\t\t\t\t\t</text-style-def>\n`;
            xml += `\t\t\t\t\t\t\t<adjust-conform type="fit"/>\n`;
            xml += `\t\t\t\t\t\t\t<adjust-transform scale="1 1" position="0 ${srcPosY}" anchor="0 0"/>\n`;

            totalIndex++;

            // 翻译层
            let lane = 1;
            for (const transSubs of parsedTransSubs) {
                if (i >= transSubs.length) { lane++; continue; }
                const transSub = transSubs[i];

                let tStart = transSub.start;
                if (seamlessFcpxml && transSub.start > 34) tStart = transSub.start - 34;
                const tStartStr = getFractionTime(tStart);

                let tDur = transSub.duration;
                if (seamlessFcpxml && i < transSubs.length - 1) {
                    tDur = transSubs[i + 1].start - tStart;
                }
                const tDurationStr = getFractionTime(tDur);

                const tAlignment = subtitleSetting.trans_alignment || 'center';
                const tFontColor = subtitleSetting.trans_fontColor || '1 1 1 1';
                const tBold = subtitleSetting.trans_bold || '0';
                const tStrokeColor = subtitleSetting.trans_strokeColor || '1 1 1 1';
                const tFont = subtitleSetting.trans_font || 'Arial';
                const tFontSize = subtitleSetting.trans_fontSize || '50';
                const tItalic = subtitleSetting.trans_italic || '0';
                const tStrokeWidth = subtitleSetting.trans_strokeWidth || '0';
                const tLineSpacing = subtitleSetting.trans_lineSpacing || '0';
                const tPosY = subtitleSetting.trans_pos || '-38';

                const tText = transSub.text.trim().replace(/@/g, '\n');

                xml += `\t\t\t\t\t\t\t<title name="Subtitle" lane="${lane}" ref="r1" enabled="1" start="${tStartStr}" offset="${tStartStr}" duration="${tDurationStr}">\n`;
                xml += `\t\t\t\t\t\t\t\t<text roll-up-height="0">\n`;
                xml += `\t\t\t\t\t\t\t\t\t<text-style ref="ts${totalIndex}">${xmlEscape(tText)}</text-style>\n`;
                xml += `\t\t\t\t\t\t\t\t</text>\n`;
                xml += `\t\t\t\t\t\t\t\t<text-style-def id="ts${totalIndex}">\n`;
                xml += `\t\t\t\t\t\t\t\t\t<text-style alignment="${tAlignment}" fontColor="${tFontColor}" bold="${tBold}" strokeColor="${tStrokeColor}" font="${tFont}" fontSize="${tFontSize}" italic="${tItalic}" strokeWidth="${tStrokeWidth}" lineSpacing="${tLineSpacing}"/>\n`;
                xml += `\t\t\t\t\t\t\t\t</text-style-def>\n`;
                xml += `\t\t\t\t\t\t\t\t<adjust-conform type="fit"/>\n`;
                xml += `\t\t\t\t\t\t\t\t<adjust-transform scale="1 1" position="0 ${tPosY}" anchor="0 0"/>\n`;
                xml += `\t\t\t\t\t\t\t</title>\n`;

                totalIndex++;
                lane++;
            }

            xml += `\t\t\t\t\t\t</title>\n`;
            preSubEnd = sub.end;
        }
    }

    // 闭合标签
    xml += '\t\t\t\t\t</spine>\n';
    xml += '\t\t\t\t</sequence>\n';
    xml += '\t\t\t</project>\n';
    xml += '\t\t</event>\n';
    xml += '\t</library>\n';
    xml += '</fcpxml>\n';

    // 写入文件
    const dir = path.dirname(savePath);
    if (dir) fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(savePath, xml, 'utf-8');
    console.log(`FCPXML 已写入: ${savePath}`);
}

module.exports = {
    parseSRTString,
    parseSRTTime,
    getFractionTime,
    SrtsToFcpxml,
};
