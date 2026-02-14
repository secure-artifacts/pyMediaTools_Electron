/**
 * 一键配音工作流服务
 * 替代 Python elevenlabs_tts_workflow
 * 生成音频 + 智能拆分 + 字幕对齐 + 黑屏MP4
 */
const fs = require('fs');
const path = require('path');
const os = require('os');
const elevenlabs = require('./elevenlabs');
const ffmpeg = require('./ffmpeg');
const gladia = require('./gladia');
const settings = require('./settings');

/**
 * 一键配音工作流
 */
async function ttsWorkflow(data) {
    const {
        text, voice_id, task_index = 0,
        need_split: rawNeedSplit = true,
        max_duration = 29.0,
        subtitle_text = '',
        export_mp4 = false,
        export_fcpxml = true,
        seamless_fcpxml = true,
        model_id = 'eleven_v3',
        stability = 0.5,
        output_format = 'mp3_44100_128',
        key_index = null,
        output_dir: rawOutputDir = '',
    } = data;

    if (!text || !voice_id) throw new Error('缺少必要参数');

    // 兜底互斥：黑屏 MP4 模式下不进行智能拆分
    let needSplit = rawNeedSplit;
    if (export_mp4 && needSplit) {
        console.log(`[一键配音] task_index=${task_index} 检测到 mp4+split 同时开启，已强制关闭拆分`);
        needSplit = false;
    }

    const apiKeys = elevenlabs.loadKeys();
    if (!apiKeys || apiKeys.length === 0) throw new Error('未配置 API Key');

    // 创建输出文件夹
    const today = new Date();
    const dateStr = today.toISOString().split('T')[0];
    let outputDir = rawOutputDir.trim();
    if (!outputDir) {
        outputDir = path.join(os.homedir(), 'Downloads', `${dateStr}_一键配音`);
    }
    fs.mkdirSync(outputDir, { recursive: true });

    // 提取文本前缀作为文件名
    const cleanText = text.replace(/<[^>]+>/g, '').replace(/\[[^\]]+\]/g, '');
    let textPrefix = cleanText.split(/\s+/).slice(0, 15).join('_').slice(0, 60);
    textPrefix = textPrefix.replace(/[^a-zA-Z0-9\u4e00-\u9fff _-]/g, '').replace(/\s+/g, '_').trim();
    if (!textPrefix) textPrefix = 'audio';

    const dateSuffix = `${String(today.getMonth() + 1).padStart(2, '0')}${String(today.getDate()).padStart(2, '0')}`;
    const taskPrefix = `${String(task_index + 1).padStart(2, '0')}-${textPrefix}_${dateSuffix}`;

    // 创建分组目录
    const videoGroup = path.join(outputDir, '_视频文案');
    const audioGroup = path.join(outputDir, '_音频字幕', taskPrefix);
    const metadataGroup = path.join(outputDir, '_metadata', taskPrefix);
    fs.mkdirSync(videoGroup, { recursive: true });
    fs.mkdirSync(audioGroup, { recursive: true });
    fs.mkdirSync(metadataGroup, { recursive: true });

    // Step 1: 生成音频
    const stabilityVal = Math.max(0, Math.min(1, parseFloat(stability) > 1 ? parseFloat(stability) / 100 : parseFloat(stability)));
    const { audio, usedKey } = await elevenlabs.requestTTSWithRotation(
        apiKeys, voice_id, text, model_id, stabilityVal, output_format, key_index
    );
    const usedPrefix = usedKey.length >= 12 ? `${usedKey.slice(0, 8)}...${usedKey.slice(-4)}` : usedKey;
    console.log(`[一键配音] 任务 ${task_index + 1} 使用 Key: ${usedPrefix}`);

    const sourcePath = path.join(audioGroup, `${taskPrefix}-source.mp3`);
    fs.writeFileSync(sourcePath, audio);

    let segments = [];

    // Step 2: 智能拆分（使用 FFmpeg 静音检测替代 pydub + numpy）
    if (needSplit) {
        try {
            const totalDuration = await ffmpeg.getDuration(sourcePath);
            if (totalDuration && totalDuration > max_duration) {
                const splitResult = await ffmpeg.smartSplitAnalyze(sourcePath, max_duration);

                // 导出分段
                for (let i = 0; i < splitResult.segments.length; i++) {
                    const seg = splitResult.segments[i];
                    const partPath = path.join(audioGroup, `${taskPrefix}-part_${String(i + 1).padStart(2, '0')}.mp3`);

                    await ffmpeg.runCommand('ffmpeg', [
                        '-y', '-i', sourcePath,
                        '-ss', seg.start.toFixed(3),
                        '-t', (seg.end - seg.start).toFixed(3),
                        '-c:a', 'libmp3lame', '-b:a', '192k',
                        partPath
                    ]);

                    segments.push({
                        index: i + 1,
                        start: seg.start,
                        end: seg.end,
                        path: partPath,
                    });
                }
            } else if (totalDuration) {
                segments = [{ index: 1, start: 0, end: totalDuration, path: sourcePath }];
            }
        } catch (e) {
            console.error('拆分失败:', e);
        }
    }

    // Step 3: 生成字幕
    let srtPath = null;
    if (subtitle_text) {
        try {
            const gladiaKeysData = settings.loadGladiaKeys();
            const gladiaKeys = gladiaKeysData.keys || [];

            if (!gladiaKeys || gladiaKeys.length === 0) {
                throw new Error('未配置 Gladia API Key，请在设置中配置后再试');
            }

            // 保存断行字幕文本
            const subtitleTextPath = path.join(videoGroup, `${taskPrefix}.txt`);
            fs.writeFileSync(subtitleTextPath, subtitle_text, 'utf-8');

            // 转录
            const fileName = path.parse(sourcePath).name;
            const arrayPath = path.join(metadataGroup, `${fileName}_audio_text_withtime.json`);
            const textPath = path.join(metadataGroup, `${fileName}_transcription.txt`);

            const result = await gladia.transcribeAudio(sourcePath, gladiaKeys, 'english');

            fs.writeFileSync(arrayPath, JSON.stringify(result.wordTimeInfo, null, 4), 'utf-8');
            fs.writeFileSync(textPath, result.fullText, 'utf-8');

            // 使用完整的 diff-match-patch 对齐算法生成 SRT
            const subtitleUtils = require('./subtitleUtils');
            const { audioSubtitleSearchDifferentStrong } = require('./subtitleAlignment');

            const sourceTextWithInfo = subtitleUtils.readTextWithGoogleDoc(subtitleTextPath);
            const translateTextDict = {};

            const alignResult = audioSubtitleSearchDifferentStrong(
                'en', audioGroup, taskPrefix,
                result.wordTimeInfo, result.fullText,
                sourceTextWithInfo, translateTextDict,
                false, // genMergeSrt
                true,  // sourceUpOrder
                export_fcpxml,
                seamless_fcpxml
            );

            console.log(`[一键配音] 字幕对齐结果: ${alignResult}`);

            const targetSrtPath = path.join(audioGroup, `${taskPrefix}_en_source.srt`);
            if (fs.existsSync(targetSrtPath)) {
                srtPath = targetSrtPath;
            }
        } catch (e) {
            console.error('字幕生成失败:', e);
        }
    }

    // Step 4: 生成黑屏 MP4（如果需要）
    let mp4Path = null;
    if (export_mp4) {
        try {
            mp4Path = path.join(videoGroup, `${taskPrefix}.mp4`);
            await ffmpeg.generateBlackMp4(sourcePath, mp4Path);
        } catch (e) {
            console.error('MP4 生成失败:', e);
        }
    }

    return {
        audio_path: sourcePath,
        output_folder: outputDir,
        task_prefix: taskPrefix,
        segments,
        segment_count: segments.length,
    };
}

/**
 * 简化的 SRT 生成（基于 Gladia 转录结果和字幕文本）
 */
function generateSimpleSRT(wordTimeInfo, subtitleText, outputPath) {
    const lines = subtitleText.split('\n').filter(l => l.trim());
    if (lines.length === 0 || wordTimeInfo.length === 0) return;

    // 收集所有单词
    const allWords = [];
    for (const utt of wordTimeInfo) {
        for (const w of utt.words || []) {
            allWords.push(w);
        }
    }

    if (allWords.length === 0) return;

    // 按字符顺序分配时间到每行
    let wordIdx = 0;
    const srtEntries = [];

    for (let i = 0; i < lines.length; i++) {
        const lineWords = lines[i].trim().split(/\s+/);
        const wordsNeeded = lineWords.length;

        if (wordIdx >= allWords.length) break;

        const startTime = allWords[wordIdx].start;
        const endWordIdx = Math.min(wordIdx + wordsNeeded - 1, allWords.length - 1);
        let endTime = allWords[endWordIdx].end;

        // 如果还有下一行，确保不会重叠
        if (i < lines.length - 1 && endWordIdx + 1 < allWords.length) {
            endTime = Math.min(endTime, allWords[endWordIdx + 1].start);
        }

        srtEntries.push({
            index: i + 1,
            start: Math.round(startTime * 1000),
            end: Math.round(endTime * 1000),
            text: lines[i].trim(),
        });

        wordIdx = endWordIdx + 1;
    }

    // 写入 SRT
    const { writeSRT } = require('./subtitle');
    writeSRT(srtEntries, outputPath);
}

module.exports = {
    ttsWorkflow,
};
