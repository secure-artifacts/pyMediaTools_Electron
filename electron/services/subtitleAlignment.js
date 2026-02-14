/**
 * 字幕对齐核心逻辑 — 完整移植自 core/subtitle_alignment.py
 * 使用 diff-match-patch 实现转录文本与原文本的精确对齐
 */
const DiffMatchPatch = require('diff-match-patch');
const { wordSplitBy, writeToPath, replaceSymbolsToOne } = require('./subtitleUtils');
const { SrtsToFcpxml } = require('./fcpxml');

// ==================== 辅助函数 ====================

/** 格式化秒数为 SRT 时间码 HH:MM:SS,mmm */
function formatTime(seconds) {
    if (seconds < 0) seconds = 0;
    const millis = Math.round((seconds % 1) * 1000);
    let totalSec = Math.floor(seconds);
    let minutes = Math.floor(totalSec / 60);
    totalSec = totalSec % 60;
    const hours = Math.floor(minutes / 60);
    minutes = minutes % 60;
    return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(totalSec).padStart(2, '0')},${String(millis).padStart(3, '0')}`;
}

/** 清理文本 — 替换空白为空格，标点替换为统一符号 */
function cleanText(text) {
    let cleaned = text.trim().replace(/\n/g, '').replace(/\r/g, '');
    cleaned = cleaned.replace(/\s+/g, wordSplitBy.en);
    cleaned = cleaned.toLowerCase();
    cleaned = replaceSymbolsToOne(cleaned);
    return cleaned;
}

// ==================== 核心对齐算法 ====================

/**
 * 处理差异并计算音频位置
 * 完整移植自 process_diffs_with_audio_positions_strong
 */
function processDiffsWithAudioPositionsStrong(params) {
    const {
        diffs,
        source_text_with_no_info,
        translate_text_dict,
        generation_subtitle_text,
        source_text_with_info,
        generation_subtitle_array,
        title,
        directory,
        language,
        gen_merge_srt,
        source_up_order,
        export_fcpxml,
        seamless_fcpxml,
        source_srt_path,
        fcpxml_path,
    } = params;

    let mergeText = '';
    let sourceText = '';
    let generateText = '';
    const sourceStore = {};
    const generateStore = {};
    const allstrings = {};

    function getSourceStore(index) {
        if (!sourceStore[index]) sourceStore[index] = {};
        return sourceStore[index];
    }
    function getGenerateStore(index) {
        if (!generateStore[index]) generateStore[index] = {};
        return generateStore[index];
    }
    function getAllstrings(index) {
        if (!allstrings[index]) allstrings[index] = {};
        return allstrings[index];
    }

    // ---- 生成对应关系 ----
    for (let i = 0; i < diffs.length; i++) {
        const [op, content] = diffs[i];
        const lengthSource = sourceText.length;
        const lengthGenerate = generateText.length;
        const lengthAll = mergeText.length;
        const lengthContent = content.length;

        if (op === 1) {
            // DIFF_INSERT: 原文本有、生成文本没有
            for (let number = 0; number < lengthContent; number++) {
                const indexAll = number + lengthAll;
                const indexSource = lengthSource + number;
                const source = getSourceStore(indexSource);
                const all = getAllstrings(indexAll);
                source.index_in_all = indexAll;
                all.index = indexAll;
                all.source_index = indexSource;
                const g = mergeText + content;
                all.char = source.char = g[indexAll];
            }
            sourceText += content;
        } else if (op === -1) {
            // DIFF_DELETE: 生成文本有、原文本没有
            for (let number = 0; number < lengthContent; number++) {
                const indexAll = number + lengthAll;
                const indexGen = lengthGenerate + number;
                const generate = getGenerateStore(indexGen);
                const all = getAllstrings(indexAll);
                all.index = indexAll;
                generate.index_in_all = indexAll;
                all.gen_index = indexGen;
                const g = mergeText + content;
                all.char = generate.char = g[indexAll];
            }
            generateText += content;
        } else if (op === 0) {
            // DIFF_EQUAL: 两边都有
            for (let number = 0; number < lengthContent; number++) {
                const indexAll = number + lengthAll;
                const indexSource = lengthSource + number;
                const source = getSourceStore(indexSource);
                const all = getAllstrings(indexAll);
                source.index_in_all = indexAll;
                all.index = indexAll;
                all.source_index = indexSource;
                const g = mergeText + content;
                all.char = source.char = g[indexAll];
            }
            sourceText += content;

            for (let number = 0; number < lengthContent; number++) {
                const indexAll = number + lengthAll;
                const indexGen = lengthGenerate + number;
                const generate = getGenerateStore(indexGen);
                const all = getAllstrings(indexAll);
                all.index = indexAll;
                generate.index_in_all = indexAll;
                all.gen_index = indexGen;
                const g = mergeText + content;
                all.char = generate.char = g[indexAll];
            }
            generateText += content;
        }
        mergeText += content;
    }

    if (generateText.length !== generation_subtitle_text.length) {
        return `比较生成文件长度不同${generateText.length}和${generation_subtitle_text.length}`;
    }
    if (sourceText.length !== source_text_with_no_info.length) {
        return `比较源文件长度不同${sourceText.length}和${source_text_with_no_info.length}`;
    }

    const alltextLength = mergeText.length;
    const audioEnd = generation_subtitle_array[generation_subtitle_array.length - 1].audio_end;

    // ---- 从生成文本添加时间戳 ----
    let text = '';
    let istart = true;
    let lastpoint = 0;

    for (const sentences of generation_subtitle_array) {
        for (const word of sentences.words) {
            let currentContent;
            if (istart) {
                istart = false;
                currentContent = word.word;
                word.whitespace = false;
            } else {
                word.whitespace = true;
                currentContent = wordSplitBy.en + word.word;
            }

            const lengthGen = text.length;
            let error = null;

            if (word.start !== undefined) {
                const start = word.start;
                const end = word.end;
                const score = word.score;
                const wordtext = word.word;
                const whitespace = word.whitespace;
                const eva = wordtext.length > 0 ? (end - start) / wordtext.length : 0;
                if (eva > 0.2) {
                    error = `从生成添加时间,长度大${eva}`;
                }
                let newstart = start;
                lastpoint = end;

                for (let number = 0; number < currentContent.length; number++) {
                    const generate = getGenerateStore(lengthGen + number);
                    const indexAll = generate.index_in_all;
                    const all = getAllstrings(indexAll);
                    all.char_gen_add_in_time = currentContent[number];
                    all.wordtext = wordtext;
                    all.score = score;
                    all.eva = eva;

                    if (whitespace && number === 0) {
                        all.audio_start = newstart;
                        all.audio_end = newstart;
                    } else if (number === currentContent.length - 1) {
                        all.audio_start = newstart;
                        all.audio_end = end;
                    } else {
                        all.audio_start = newstart;
                        newstart = Math.round((newstart + eva) * 1000) / 1000;
                        all.audio_end = newstart;
                    }
                    if (error) all.error = error;
                }
            } else {
                word.error = error = '从生成添加时间,没有时间';
                const wordtext = word.word;
                for (let number = 0; number < currentContent.length; number++) {
                    const generate = getGenerateStore(lengthGen + number);
                    const indexAll = generate.index_in_all;
                    const all = getAllstrings(indexAll);
                    all.char_gen_add_in_time = currentContent[number];
                    all.wordtext = wordtext;
                    all.score = 0;
                    all.eva = 0;
                    all.audio_start = lastpoint;
                    all.audio_end = lastpoint;
                    if (error) all.error = error;
                }
            }
            text += currentContent;
        }
    }

    if (text.length !== generation_subtitle_text.length) {
        return `从生成添加时间 长度不同${text.length}和${generation_subtitle_text.length}`;
    }

    // ---- 源文本添加时间戳 ----
    mergeText = '';
    let lastOp = null;

    for (let i = 0; i < diffs.length; i++) {
        const [op, content] = diffs[i];
        const lengthAll = mergeText.length;
        const lengthContent = content.length;

        if (op === 1) {
            if (lastOp === null) {
                let genSource = null;
                let audioDurio = 0.1;
                if (lengthAll + lengthContent >= alltextLength) {
                    audioDurio = audioEnd;
                    genSource = '结束';
                } else {
                    const allEnd = getAllstrings(lengthAll + lengthContent - 1);
                    if (allEnd.audio_start !== undefined) {
                        genSource = allEnd;
                        audioDurio = Math.max(allEnd.audio_start - 0.1, 0);
                    }
                }
                const eva = lengthContent > 0 ? audioDurio / lengthContent : 0;
                let newstart = 0;
                for (let number = 0; number < lengthContent; number++) {
                    const indexAll = number + lengthAll;
                    const all = getAllstrings(indexAll);
                    all.audio_start = newstart;
                    newstart = Math.round((newstart + eva) * 1000) / 1000;
                    all.audio_end = newstart;
                    all.eva = eva;
                    all.from = '生成开头丢失文本';
                    all.gen = genSource;
                }
            } else if (lastOp.op === 0) {
                let genSource = lastOp;
                let newstart = lastOp.all_end.audio_end;
                let audioEndVal = newstart + 0.2;
                if (lengthAll + lengthContent >= alltextLength) {
                    audioEndVal = audioEnd;
                    genSource = '结束';
                } else {
                    const allEnd = getAllstrings(lengthAll + lengthContent - 1);
                    if (allEnd.audio_start !== undefined) {
                        genSource = allEnd;
                        audioEndVal = Math.max(allEnd.audio_start - 0.1, 0);
                    }
                }
                const audioDurio = audioEndVal - newstart;
                const eva = lengthContent > 0 ? audioDurio / lengthContent : 0;
                for (let number = 0; number < lengthContent; number++) {
                    const indexAll = number + lengthAll;
                    const all = getAllstrings(indexAll);
                    all.audio_start = newstart;
                    newstart = Math.round((newstart + eva) * 1000) / 1000;
                    all.audio_end = newstart;
                    all.eva = eva;
                    all.from = '生成丢失单词';
                    all.gen = genSource;
                }
            } else if (lastOp.op === -1) {
                const allStart = lastOp.all_start;
                const allEnd = lastOp.all_end;
                let newstart = allStart.audio_start;
                const audioDurio = allEnd.audio_end - newstart;
                const eva = lengthContent > 0 ? audioDurio / lengthContent : 0;
                for (let number = 0; number < lengthContent; number++) {
                    const indexAll = number + lengthAll;
                    const all = getAllstrings(indexAll);
                    all.audio_start = newstart;
                    newstart = Math.round((newstart + eva) * 1000) / 1000;
                    all.audio_end = newstart;
                    all.eva = eva;
                    all.from = '生成错误文本';
                    all.gen = lastOp;
                }
            }
            lastOp = null;
        } else if (op === -1) {
            const allStart = getAllstrings(lengthAll);
            const allEnd = getAllstrings(lengthAll + lengthContent - 1);
            lastOp = { all_start: allStart, all_end: allEnd, op };
        } else if (op === 0) {
            const allStart = getAllstrings(lengthAll);
            const allEnd = getAllstrings(lengthAll + lengthContent - 1);
            lastOp = { all_start: allStart, all_end: allEnd, op };
        }
        mergeText += content;
    }

    // ---- 检查所有源文件字符串都赋予时间戳 ----
    for (const x of Object.values(allstrings)) {
        if (x.source_index !== undefined && x.audio_start === undefined) {
            return `检查到没有正常赋值${JSON.stringify(x)}`;
        }
    }

    // ---- 每个单词的结尾处添加 0.2 长度 ----
    const sortedKeys = Object.keys(allstrings).map(Number).sort((a, b) => a - b);
    const sortedValues = sortedKeys.map(k => allstrings[k]);
    const les = sortedValues.length;
    for (let i = 0; i < les; i++) {
        if (i > 0 && i < les - 1) {
            const s = sortedValues[i];
            const nextitem = sortedValues[i + 1];
            if (nextitem.audio_start !== undefined && s.audio_end !== undefined) {
                const currentAudioEnd = s.audio_end;
                const nextAudioStart = nextitem.audio_start;
                if (nextAudioStart - currentAudioEnd >= 0.3) {
                    s.audio_end = Math.round((s.audio_end + 0.1) * 1000) / 1000;
                    nextitem.audio_start = Math.round((nextitem.audio_start - 0.1) * 1000) / 1000;
                }
            }
        }
    }

    // ---- 生成字幕文件 ----
    text = '';
    istart = true;
    let index = 0;
    let sourceSrt = '';
    let mergeSrt = '';

    const fs = require('fs');
    const path = require('path');

    for (const content of source_text_with_info.contents) {
        index++;
        let currentContent;
        if (istart) {
            istart = false;
            currentContent = content.content;
        } else {
            currentContent = wordSplitBy.en + content.content;
        }

        if (content.type !== null) {
            const source = getSourceStore(text.length);
            const endsource = getSourceStore(Math.min(text.length + currentContent.length - 1, alltextLength - 1));
            if (endsource.index_in_all === undefined || source.index_in_all === undefined) {
                console.log(`不该发生的错误,找不到存储${JSON.stringify(content)}`);
                text += currentContent;
                continue;
            }
            const start = getAllstrings(source.index_in_all);
            const end = getAllstrings(endsource.index_in_all);

            const srtStart = (index === 1) ? formatTime(0) : formatTime(start.audio_start);
            const srtEnd = formatTime(end.audio_end);

            let mergeTransContent = '';
            for (const [k, value] of Object.entries(translate_text_dict)) {
                const transContents = value.translate_text_with_info?.contents;
                if (!transContents || index - 1 >= transContents.length) continue;
                const transContentText = transContents[index - 1].content;
                mergeTransContent += transContentText + '\n';
                if (value.trans_srt === undefined) value.trans_srt = '';
                value.trans_srt += `${index}\n${srtStart} --> ${srtEnd}\n${transContentText}\n\n`;
            }

            const contentText = content.content;
            sourceSrt += `${index}\n${srtStart} --> ${srtEnd}\n${contentText}\n\n`;

            if (gen_merge_srt && mergeTransContent !== '') {
                mergeTransContent = mergeTransContent.replace(/\n$/, '');
                if (source_up_order) {
                    mergeSrt += `${index}\n${srtStart} --> ${srtEnd}\n${contentText}\n${mergeTransContent}\n\n`;
                } else {
                    mergeSrt += `${index}\n${srtStart} --> ${srtEnd}\n${mergeTransContent}\n${contentText}\n\n`;
                }
            }
        }
        text += currentContent;
    }

    if (text.length !== source_text_with_no_info.length) {
        return `核对源文本段落长度不同${text.length}和${source_text_with_no_info.length}`;
    }

    // ---- 写入原 SRT ----
    if (source_srt_path) {
        const srtDir = path.dirname(source_srt_path);
        if (srtDir) fs.mkdirSync(srtDir, { recursive: true });
        fs.writeFileSync(source_srt_path, sourceSrt, 'utf-8');
    } else {
        writeToPath(sourceSrt, directory, `${title}_${language}_source`, 'srt');
    }

    // ---- 写入翻译 SRT ----
    for (const [k, value] of Object.entries(translate_text_dict)) {
        const transSrt = value.trans_srt || '';
        const filename = k.replace('.txt', '');
        writeToPath(transSrt, directory, `${title}_${language}_${filename}_translate`, 'srt');
    }

    // ---- 写入合并 SRT ----
    if (gen_merge_srt) {
        if (mergeSrt !== '') {
            writeToPath(mergeSrt, directory, `${title}_${language}_merge`, 'srt');
        } else {
            console.log('翻译文本为空，不再生成合并srt文件。');
        }
    }

    // ---- 导出 FCPXML ----
    if (export_fcpxml) {
        let savePath;
        if (fcpxml_path) {
            savePath = fcpxml_path;
            const fcpxmlDir = path.dirname(savePath);
            if (fcpxmlDir) fs.mkdirSync(fcpxmlDir, { recursive: true });
        } else {
            savePath = path.join(directory, `${title}_${language}.fcpxml`);
        }

        const translateSrtList = [];
        for (const value of Object.values(translate_text_dict)) {
            translateSrtList.push(value.trans_srt || '');
        }
        SrtsToFcpxml(sourceSrt, translateSrtList, savePath, seamless_fcpxml);
    }

    return `生成了字幕文件${title}`;
}

// ==================== 主对齐入口 ====================

/**
 * 主对齐函数
 * 完整移植自 audio_subtitle_search_diffent_strong
 *
 * @param {string} currentLanguage      语言代码
 * @param {string} directory            输出目录
 * @param {string} fileName             文件名（不含扩展名）
 * @param {Array}  generationSubtitleArray  Gladia 转录结果（含词级别时间戳）
 * @param {string} generationSubtitleText   转录纯文本
 * @param {Object} sourceTextWithInfo       read_text_with_google_doc 的返回值
 * @param {Object} translateTextDict        翻译文本字典
 * @param {boolean} genMergeSrt             是否生成合并 SRT
 * @param {boolean} sourceUpOrder           源文本在上方
 * @param {boolean} exportFcpxml            是否导出 FCPXML
 * @param {boolean} seamlessFcpxml          FCPXML 是否无缝
 * @param {string}  sourceSrtPath           可选，指定源SRT输出路径
 * @param {string}  fcpxmlPath              可选，指定FCPXML输出路径
 */
function audioSubtitleSearchDifferentStrong(
    currentLanguage, directory, fileName,
    generationSubtitleArray, generationSubtitleText,
    sourceTextWithInfo, translateTextDict,
    genMergeSrt, sourceUpOrder, exportFcpxml, seamlessFcpxml,
    sourceSrtPath = null, fcpxmlPath = null
) {
    // 构建纯文本（不含结构信息）
    let sourceTextWithNoInfo = '';
    for (const content of sourceTextWithInfo.contents) {
        sourceTextWithNoInfo += wordSplitBy.en + content.content;
    }

    // 清理文本
    const cleanedGenText = cleanText(generationSubtitleText);
    const cleanedSourceText = cleanText(sourceTextWithNoInfo);

    // 执行 diff
    const dmp = new DiffMatchPatch();
    dmp.Diff_Timeout = 0;
    const diffs = dmp.diff_main(cleanedGenText, cleanedSourceText);
    dmp.diff_cleanupSemantic(diffs);

    return processDiffsWithAudioPositionsStrong({
        title: fileName,
        diffs,
        directory,
        language: currentLanguage,
        gen_merge_srt: genMergeSrt,
        source_up_order: sourceUpOrder,
        export_fcpxml: exportFcpxml,
        seamless_fcpxml: seamlessFcpxml,
        source_srt_path: sourceSrtPath,
        fcpxml_path: fcpxmlPath,
        source_text_with_no_info: cleanedSourceText,
        translate_text_dict: translateTextDict,
        generation_subtitle_text: cleanedGenText,
        source_text_with_info: sourceTextWithInfo,
        generation_subtitle_array: generationSubtitleArray,
    });
}

module.exports = {
    formatTime,
    cleanText,
    audioSubtitleSearchDifferentStrong,
};
