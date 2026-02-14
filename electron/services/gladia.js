/**
 * Gladia 语音转文字服务 — 完整移植自 core/gladia_api.py
 * 支持长音频自动切分（通过 FFmpeg 静音检测）、分段转录、API key 轮换
 */
const fs = require('fs');
const path = require('path');
const os = require('os');
const https = require('https');
const http = require('http');
const { execFile, spawn } = require('child_process');

const GLADIA_API_URL = 'https://api.gladia.io';

// ==================== HTTP 请求工具 ====================

function gladiaRequest(method, urlStr, headers, body, timeout = 120000) {
    return new Promise((resolve, reject) => {
        const url = new URL(urlStr);
        const client = url.protocol === 'https:' ? https : http;
        const options = {
            hostname: url.hostname,
            port: url.port || (url.protocol === 'https:' ? 443 : 80),
            path: url.pathname + url.search,
            method,
            headers,
            timeout,
        };
        const req = client.request(options, (res) => {
            const chunks = [];
            res.on('data', c => chunks.push(c));
            res.on('end', () => {
                resolve({ status: res.statusCode, body: Buffer.concat(chunks), headers: res.headers });
            });
        });
        req.on('timeout', () => { req.destroy(); reject(new Error('Gladia 请求超时')); });
        req.on('error', reject);
        if (body) req.write(body);
        req.end();
    });
}

// ==================== FFmpeg 音频处理 ====================

/**
 * 从视频文件中提取音频（替代 Python extract_audio_from_video）
 */
async function extractAudioFromVideo(videoPath, outputDir, audioFormat = 'mp3') {
    const baseName = path.parse(videoPath).name;
    const audioPath = path.join(outputDir, `${baseName}.${audioFormat}`);
    fs.mkdirSync(outputDir, { recursive: true });

    const ffmpegPath = process.env.FFMPEG_PATH || 'ffmpeg';
    const args = ['-y', '-i', videoPath, '-vn'];
    if (audioFormat === 'wav') {
        args.push('-ar', '32000', '-ac', '1');
    } else {
        args.push('-ar', '44100', '-ac', '1', '-b:a', '192k');
    }
    args.push(audioPath);

    return new Promise((resolve, reject) => {
        execFile(ffmpegPath, args, { timeout: 300000 }, (err, stdout, stderr) => {
            if (err) return reject(new Error(`FFmpeg 提取音频失败: ${stderr || err.message}`));
            resolve(audioPath);
        });
    });
}

/**
 * 获取音频时长（秒）
 */
async function getAudioDuration(filePath) {
    const ffprobePath = process.env.FFPROBE_PATH || 'ffprobe';
    return new Promise((resolve, reject) => {
        execFile(ffprobePath, [
            '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', filePath
        ], { timeout: 30000 }, (err, stdout) => {
            if (err) return reject(err);
            resolve(parseFloat(stdout.trim()) || 0);
        });
    });
}

/**
 * 通过 FFmpeg 检测静音，返回静音中点列表（毫秒）
 * 替代 Python pydub.silence.detect_silence
 */
async function detectSilencePoints(filePath, silenceThreshDb = -30, minSilenceLen = 0.5) {
    const ffmpegPath = process.env.FFMPEG_PATH || 'ffmpeg';
    return new Promise((resolve, reject) => {
        const args = [
            '-i', filePath, '-af',
            `silencedetect=noise=${silenceThreshDb}dB:d=${minSilenceLen}`,
            '-f', 'null', '-'
        ];
        const proc = spawn(ffmpegPath, args, { stdio: ['pipe', 'pipe', 'pipe'] });
        let stderr = '';
        proc.stderr.on('data', d => { stderr += d.toString(); });
        proc.on('close', () => {
            const points = [];
            const startRegex = /silence_start:\s*([\d.]+)/g;
            const endRegex = /silence_end:\s*([\d.]+)/g;
            const starts = [];
            const ends = [];
            let m;
            while ((m = startRegex.exec(stderr))) starts.push(parseFloat(m[1]));
            while ((m = endRegex.exec(stderr))) ends.push(parseFloat(m[1]));

            for (let i = 0; i < Math.min(starts.length, ends.length); i++) {
                const midMs = Math.round(((starts[i] + ends[i]) / 2) * 1000);
                points.push(midMs);
            }
            resolve(points);
        });
        proc.on('error', reject);
    });
}

/**
 * 按静音切分长音频 — 替代 Python split_audio_on_silence
 * 使用 FFmpeg 进行切分
 */
async function splitAudioOnSilence(audioPath, outputDir, minMinutes = 5.0, maxMinutes = 50.0, audioFormat = 'mp3') {
    fs.mkdirSync(outputDir, { recursive: true });
    const baseName = path.parse(audioPath).name;

    // 获取总时长（毫秒）
    const totalDurationSec = await getAudioDuration(audioPath);
    const totalMs = Math.round(totalDurationSec * 1000);
    const minMs = minMinutes * 60 * 1000;
    const maxMs = maxMinutes * 60 * 1000;

    // 如果总时长小于最大限制，无需切分
    if (totalMs <= maxMs) {
        return [{ path: audioPath, duration: totalDurationSec }];
    }

    // 检测静音点
    const silencePoints = await detectSilencePoints(audioPath, -30, 0.5);

    // 计算片段
    const segmentsMs = [];
    let start = 0;

    for (const splitPoint of silencePoints) {
        if (splitPoint - start >= minMs) {
            while (splitPoint - start > maxMs) {
                const mid = start + maxMs;
                segmentsMs.push([start, mid]);
                start = mid;
            }
            segmentsMs.push([start, splitPoint]);
            start = splitPoint;
        }
    }
    if (start < totalMs) {
        segmentsMs.push([start, totalMs]);
    }

    // 合并太短的最后一段
    if (segmentsMs.length >= 2) {
        const last = segmentsMs[segmentsMs.length - 1];
        const lastDur = last[1] - last[0];
        if (lastDur < 60000) {
            const prev = segmentsMs[segmentsMs.length - 2];
            segmentsMs[segmentsMs.length - 2] = [prev[0], last[1]];
            segmentsMs.pop();
        }
    }

    // 如果只有一段，直接返回
    if (segmentsMs.length <= 1) {
        return [{ path: audioPath, duration: totalDurationSec }];
    }

    // 用 FFmpeg 切分每段
    const ffmpegPath = process.env.FFMPEG_PATH || 'ffmpeg';
    const segments = [];

    for (let idx = 0; idx < segmentsMs.length; idx++) {
        const [segStart, segEnd] = segmentsMs[idx];
        const segPath = path.join(outputDir, `${baseName}_part${idx + 1}.${audioFormat}`);
        const startSec = segStart / 1000;
        const durSec = (segEnd - segStart) / 1000;

        const args = ['-y', '-i', audioPath, '-ss', String(startSec), '-t', String(durSec)];
        if (audioFormat !== 'wav') {
            args.push('-ac', '1', '-b:a', '192k');
        } else {
            args.push('-ac', '1', '-ar', '32000');
        }
        args.push(segPath);

        await new Promise((resolve, reject) => {
            execFile(ffmpegPath, args, { timeout: 120000 }, (err) => {
                if (err) return reject(new Error(`FFmpeg 切分音频失败: ${err.message}`));
                resolve();
            });
        });

        segments.push({ path: segPath, duration: durSec });
    }

    return segments;
}

// ==================== Gladia API ====================

/**
 * 上传音频文件到 Gladia v2
 */
async function uploadAudio(apiKey, filePath) {
    const fileData = fs.readFileSync(filePath);
    const fileName = path.basename(filePath);
    const boundary = '----FormBoundary' + Math.random().toString(36).slice(2);

    let body = '';
    body += `--${boundary}\r\n`;
    body += `Content-Disposition: form-data; name="audio"; filename="${fileName}"\r\n`;
    body += `Content-Type: application/octet-stream\r\n\r\n`;

    const bodyStart = Buffer.from(body, 'utf-8');
    const bodyEnd = Buffer.from(`\r\n--${boundary}--\r\n`, 'utf-8');
    const fullBody = Buffer.concat([bodyStart, fileData, bodyEnd]);

    const res = await gladiaRequest('POST', `${GLADIA_API_URL}/v2/upload`, {
        'x-gladia-key': apiKey,
        'Content-Type': `multipart/form-data; boundary=${boundary}`,
        'Content-Length': fullBody.length,
    }, fullBody, 300000);

    if (res.status !== 200 && res.status !== 201) {
        const errText = res.body.toString().slice(0, 500);
        // 检查是否 limit exceeded
        if (errText.includes('limit exceeded') || errText.includes('quota')) {
            throw new Error('LIMIT_EXCEEDED');
        }
        throw new Error(`Gladia 上传失败: ${res.status} - ${errText}`);
    }

    const data = JSON.parse(res.body.toString());
    return data.audio_url || data.url;
}

/**
 * 发起转录请求 (v2)
 */
async function startTranscription(apiKey, audioUrl, language = 'english') {
    // Gladia v2 使用 language_config.languages 数组，且需要语言代码（如 'en'）
    // 如果传入的是英文名，尝试映射到代码
    const { LANGUAGES } = require('./subtitleUtils');
    let langCode = language;
    for (const [code, info] of Object.entries(LANGUAGES)) {
        if (info.language === language || info.name === language) {
            langCode = code;
            break;
        }
    }

    const payloadObj = {
        audio_url: audioUrl,
    };

    // 只在非自动检测时设置语言
    if (langCode && langCode !== 'auto') {
        payloadObj.language_config = {
            languages: [langCode],
        };
    }

    const payload = JSON.stringify(payloadObj);

    const res = await gladiaRequest('POST', `${GLADIA_API_URL}/v2/transcription`, {
        'x-gladia-key': apiKey,
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(payload),
    }, payload, 30000);

    if (res.status !== 200 && res.status !== 201) {
        throw new Error(`Gladia 转录请求失败: ${res.status} - ${res.body.toString().slice(0, 300)}`);
    }

    const data = JSON.parse(res.body.toString());
    return data.result_url || data.id;
}

/**
 * 轮询转录结果
 */
async function pollResult(apiKey, resultUrl, maxAttempts = 60, interval = 10000) {
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
        console.log(`Gladia 轮询尝试 ${attempt + 1}/${maxAttempts}...`);
        const res = await gladiaRequest('GET', resultUrl, {
            'x-gladia-key': apiKey,
            'Accept': 'application/json',
        }, null, 15000);

        if (res.status === 200) {
            const data = JSON.parse(res.body.toString());
            const status = (data.status || '').toLowerCase();
            if ((data.result || data.transcription) && (status === 'done' || status === 'completed' || !status)) {
                console.log('Gladia 异步转录完成！');
                return data;
            } else if (status === 'error') {
                throw new Error(`Gladia 转录出错: ${JSON.stringify(data)}`);
            }
            console.log(`Gladia 状态: ${status}, 等待...`);
        } else if (res.status === 202) {
            console.log('Gladia 仍在处理...');
        } else {
            throw new Error(`Gladia 轮询错误: ${res.status}`);
        }
        await new Promise(r => setTimeout(r, interval));
    }
    throw new Error('Gladia 转录超时: 已达最大轮询次数');
}

/**
 * 解析 Gladia 转录结果为统一格式 — 移植自 get_json_result
 * @param {Object} transcribeResult  Gladia API 返回值
 * @param {Array}  lastResult       累积结果数组
 * @param {Array}  fullTextList     全文词列表
 * @param {number} startTime        时间偏移（秒，用于分段合并）
 */
function getJsonResult(transcribeResult, lastResult, fullTextList, startTime) {
    if (!transcribeResult) return false;

    // v2 API 格式
    const transcription = transcribeResult.result?.transcription || transcribeResult.transcription || {};
    const utterances = transcription.utterances || [];

    if (utterances.length > 0) {
        for (const item of utterances) {
            const audioStart = (item.start || 0) + startTime;
            const audioEnd = (item.end || 0) + startTime;
            const part = {
                text: item.text || '',
                audio_start: audioStart,
                audio_end: audioEnd,
                duration: audioEnd - audioStart,
                words: [],
            };

            const words = item.words || [];
            for (const wordInfo of words) {
                const word = (wordInfo.word || '').trim();
                fullTextList.push(word);
                part.words.push({
                    word,
                    start: (wordInfo.start || 0) + startTime,
                    end: (wordInfo.end || 0) + startTime,
                    score: wordInfo.confidence || 0,
                });
            }
            lastResult.push(part);
        }
        return true;
    }

    // v1 API 格式 (prediction)
    const prediction = transcribeResult.prediction;
    if (Array.isArray(prediction) && prediction.length > 0) {
        for (const item of prediction) {
            const audioStart = (item.time_begin || 0) + startTime;
            const audioEnd = (item.time_end || 0) + startTime;
            const part = {
                text: item.transcription || '',
                audio_start: audioStart,
                audio_end: audioEnd,
                duration: audioEnd - audioStart,
                words: [],
            };

            const words = item.words || [];
            for (const wordInfo of words) {
                const word = (wordInfo.word || '').trim();
                fullTextList.push(word);
                part.words.push({
                    word,
                    start: (wordInfo.time_begin || 0) + startTime,
                    end: (wordInfo.time_end || 0) + startTime,
                    score: wordInfo.confidence || 0,
                });
            }
            lastResult.push(part);
        }
        return true;
    }

    return false;
}

// ==================== 主接口 ====================

/**
 * 完整的转录流程 — 完整移植自 transcribe_audio_from_gladia
 * 支持长音频自动切分、错误重试、API Key 轮换
 *
 * @param {string}   mediaPath    音频/视频文件路径
 * @param {string[]} apiKeys      Gladia API Key 数组
 * @param {string}   language     语言（英文名）
 * @param {string}   jsonPath     结果 JSON 保存路径
 * @param {string}   txtPath      纯文本保存路径
 * @param {number}   minMinutes   最小切分时长（分钟）
 * @param {Function} onProgress   进度回调
 * @returns {Object} { wordTimeInfo, fullText }
 */
async function transcribeAudioFull(mediaPath, apiKeys, language, jsonPath, txtPath, minMinutes = 5.0, onProgress = null) {
    if (!apiKeys || apiKeys.length === 0) {
        throw new Error('无可用 Gladia Key，请添加 Gladia key。');
    }

    const tmpDir = path.join(os.tmpdir(), `gladia_tmp_${Date.now()}`);
    let audioPath = mediaPath;

    // 如果是视频，提取音频
    const videoExts = ['.mp4', '.mov', '.mkv', '.flv', '.avi', '.wmv'];
    if (videoExts.includes(path.extname(mediaPath).toLowerCase())) {
        if (onProgress) onProgress('提取音频');
        audioPath = await extractAudioFromVideo(mediaPath, tmpDir);
    }

    // 切分音频
    if (onProgress) onProgress('切分音频');
    const segments = await splitAudioOnSilence(audioPath, tmpDir, minMinutes);

    // 开始转录
    let curStartTime = 0;
    const lastResult = [];
    const fullTextList = [];
    let curKeyIndex = 0;

    if (onProgress) onProgress('开始转录音频');

    for (let idx = 0; idx < segments.length; idx++) {
        const { path: segPath, duration } = segments[idx];

        if (onProgress) onProgress(`转录进度: ${idx + 1}/${segments.length}`);

        let success = false;

        // 尝试每个 key，每个 key 最多重试 3 次
        for (let keyAttempt = curKeyIndex; keyAttempt < apiKeys.length; keyAttempt++) {
            const apiKey = apiKeys[keyAttempt];

            for (let retry = 0; retry < 3; retry++) {
                try {
                    // 上传
                    const audioUrl = await uploadAudio(apiKey, segPath);

                    // 转录
                    const resultUrl = await startTranscription(apiKey, audioUrl, language);

                    // 轮询结果
                    const result = await pollResult(apiKey, resultUrl);

                    // 解析结果
                    const ok = getJsonResult(result, lastResult, fullTextList, curStartTime);
                    if (!ok) {
                        throw new Error('转录结果有问题');
                    }

                    success = true;
                    curKeyIndex = keyAttempt; // 记住当前 key
                    break;
                } catch (e) {
                    console.error(`Gladia 转录失败 (key ${keyAttempt + 1}, 重试 ${retry + 1}): ${e.message}`);

                    if (e.message === 'LIMIT_EXCEEDED') {
                        console.log('Gladia 达到限制，切换下一个 API key');
                        break; // 跳到下一个 key
                    }

                    // 其他错误继续重试
                    if (retry < 2) {
                        await new Promise(r => setTimeout(r, 2000));
                    }
                }
            }

            if (success) break;
        }

        if (!success) {
            throw new Error('转录失败: 所有 API key 均不可用');
        }

        curStartTime += duration;
    }

    // 保存结果
    if (jsonPath) {
        const jsonDir = path.dirname(jsonPath);
        fs.mkdirSync(jsonDir, { recursive: true });
        fs.writeFileSync(jsonPath, JSON.stringify(lastResult, null, 2), 'utf-8');
    }

    const fullText = fullTextList.join(' ');
    if (txtPath) {
        const txtDir = path.dirname(txtPath);
        fs.mkdirSync(txtDir, { recursive: true });
        fs.writeFileSync(txtPath, fullText, 'utf-8');
    }

    // 清理临时文件
    try {
        if (fs.existsSync(tmpDir)) {
            fs.rmSync(tmpDir, { recursive: true, force: true });
        }
    } catch { /* 忽略清理错误 */ }

    return { wordTimeInfo: lastResult, fullText };
}

/**
 * 简化版转录接口（兼容之前的 API）
 */
async function transcribeAudio(filePath, gladiaKeys, language = 'english', onProgress = null) {
    // 如果传入的 language 是中文名称，转为英文
    const { LANGUAGES } = require('./subtitleUtils');
    let langEn = language;
    for (const info of Object.values(LANGUAGES)) {
        if (info.name === language || info.code === language) {
            langEn = info.language;
            break;
        }
    }

    return transcribeAudioFull(filePath, gladiaKeys, langEn, null, null, 5.0, onProgress);
}

module.exports = {
    transcribeAudio,
    transcribeAudioFull,
    uploadAudio,
    startTranscription,
    pollResult,
    extractAudioFromVideo,
    splitAudioOnSilence,
    getJsonResult,
};
