/**
 * FFmpeg/FFprobe 操作封装
 * 替代 Python 后端中所有 subprocess.run(ffmpeg/ffprobe...) 调用
 */
const { spawn, execFile } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');

// 超时默认值
const DEFAULT_TIMEOUT = 600000; // 10 分钟
const PROBE_TIMEOUT = 30000;    // 30 秒

/**
 * 解析命令路径 - 优先使用环境变量中配置的路径
 */
function resolveCommand(cmd) {
    if (cmd === 'ffmpeg' && process.env.FFMPEG_PATH) return process.env.FFMPEG_PATH;
    if (cmd === 'ffprobe' && process.env.FFPROBE_PATH) return process.env.FFPROBE_PATH;
    return cmd;
}

/**
 * 执行 FFmpeg/FFprobe 命令，返回 Promise
 */
function runCommand(cmd, args, options = {}) {
    const timeout = options.timeout || DEFAULT_TIMEOUT;
    const resolvedCmd = resolveCommand(cmd);
    return new Promise((resolve, reject) => {
        let stdout = '';
        let stderr = '';
        const proc = spawn(resolvedCmd, args, {
            timeout,
            env: { ...process.env, ...options.env },
            cwd: options.cwd,
        });
        proc.stdout.on('data', d => stdout += d.toString());
        proc.stderr.on('data', d => stderr += d.toString());
        proc.on('close', code => {
            if (code === 0 || options.allowNonZero) {
                resolve({ stdout, stderr, code });
            } else {
                reject(new Error(`${cmd} 退出码 ${code}: ${stderr.slice(0, 500)}`));
            }
        });
        proc.on('error', (err) => {
            reject(new Error(`${cmd} 启动失败 (${resolvedCmd}): ${err.message}`));
        });
    });
}


/** 获取音频/视频时长（秒） */
async function getDuration(filePath) {
    // 方法1: 通过 format=duration 获取
    try {
        const { stdout, stderr } = await runCommand('ffprobe', [
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            filePath
        ], { timeout: PROBE_TIMEOUT });
        const dur = parseFloat(stdout.trim());
        if (!isNaN(dur) && dur > 0) return dur;
        console.warn(`[getDuration] format=duration 返回无效值: stdout="${stdout.trim()}", stderr="${(stderr || '').trim()}", file=${filePath}`);
    } catch (e) {
        console.warn(`[getDuration] 方法1失败 (format=duration): ${e.message}, file=${filePath}`);
    }

    // 方法2: 通过 stream=duration 获取（某些容器格式 format 级别没有 duration）
    try {
        const { stdout, stderr } = await runCommand('ffprobe', [
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            filePath
        ], { timeout: PROBE_TIMEOUT });
        const dur = parseFloat(stdout.trim());
        if (!isNaN(dur) && dur > 0) {
            console.log(`[getDuration] 方法2成功 (stream=duration): ${dur}s, file=${filePath}`);
            return dur;
        }
    } catch (e) {
        console.warn(`[getDuration] 方法2失败 (stream=duration): ${e.message}`);
    }

    // 方法3: 用 ffprobe -count_packets 计算时长（最准确但最慢）
    try {
        const { stdout } = await runCommand('ffprobe', [
            '-v', 'error',
            '-count_packets',
            '-show_entries', 'stream=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            filePath
        ], { timeout: PROBE_TIMEOUT * 2 });
        // 可能返回多行（多流），取第一个有效值
        for (const line of stdout.split('\n')) {
            const dur = parseFloat(line.trim());
            if (!isNaN(dur) && dur > 0) {
                console.log(`[getDuration] 方法3成功 (count_packets): ${dur}s, file=${filePath}`);
                return dur;
            }
        }
    } catch (e) {
        console.warn(`[getDuration] 方法3失败 (count_packets): ${e.message}`);
    }

    console.error(`[getDuration] 所有方法均失败, file=${filePath}`);
    return null;
}

/** 获取帧率 */
async function getFrameRate(filePath) {
    try {
        const { stdout } = await runCommand('ffprobe', [
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            filePath
        ], { timeout: PROBE_TIMEOUT });
        const fpsStr = stdout.trim();
        if (fpsStr.includes('/')) {
            const [num, den] = fpsStr.split('/');
            return parseFloat(num) / parseFloat(den);
        }
        return parseFloat(fpsStr) || 30.0;
    } catch {
        return 30.0;
    }
}

/** 获取分辨率 */
async function getResolution(filePath) {
    try {
        const { stdout } = await runCommand('ffprobe', [
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=p=0',
            filePath
        ], { timeout: PROBE_TIMEOUT });
        // ffprobe csv 输出 "1920,1080" → 转为 "1920x1080"
        return stdout.trim().replace(',', 'x');
    } catch {
        return '';
    }
}

/** 获取波形峰值数据 */
async function getWaveform(filePath, numPeaks = 300) {
    const { stdout } = await runCommand('ffmpeg', [
        '-hide_banner', '-i', filePath,
        '-ac', '1', '-ar', '8000',
        '-f', 'f32le', '-'
    ], { timeout: 120000 });

    // stdout 是 Buffer 形式的原始 float32 数据
    const raw = Buffer.from(stdout, 'binary');
    const numSamples = Math.floor(raw.length / 4);
    if (numSamples === 0) return { peaks: [], duration: 0, numPeaks: 0 };

    const samples = [];
    for (let i = 0; i < numSamples; i++) {
        samples.push(raw.readFloatLE(i * 4));
    }

    const blockSize = Math.max(1, Math.floor(numSamples / numPeaks));
    const peaks = [];
    for (let i = 0; i < Math.min(numPeaks, Math.floor(numSamples / blockSize)); i++) {
        const start = i * blockSize;
        const end = Math.min(start + blockSize, numSamples);
        let maxVal = 0;
        for (let j = start; j < end; j++) {
            maxVal = Math.max(maxVal, Math.abs(samples[j]));
        }
        peaks.push(maxVal);
    }

    const maxPeak = Math.max(...peaks) || 1;
    const normalized = peaks.map(p => Math.round((p / maxPeak) * 10000) / 10000);
    const duration = await getDuration(filePath) || (numSamples / 8000);

    return { peaks: normalized, duration: Math.round(duration * 1000) / 1000, numPeaks: normalized.length };
}

/**
 * 获取波形 — 使用 pipe（二进制安全）
 */
function getWaveformBinary(filePath, numPeaks = 300) {
    return new Promise(async (resolve, reject) => {
        const proc = spawn(resolveCommand('ffmpeg'), [
            '-hide_banner', '-i', filePath,
            '-ac', '1', '-ar', '8000',
            '-f', 'f32le', 'pipe:1'
        ], { timeout: 120000 });

        const chunks = [];
        proc.stdout.on('data', chunk => chunks.push(chunk));
        proc.stderr.on('data', () => { }); // 忽略 stderr
        proc.on('close', async (code) => {
            const raw = Buffer.concat(chunks);
            const numSamples = Math.floor(raw.length / 4);
            if (numSamples === 0) {
                const dur = await getDuration(filePath) || 0;
                return resolve({ peaks: [], duration: dur, numPeaks: 0 });
            }

            const blockSize = Math.max(1, Math.floor(numSamples / numPeaks));
            const peaks = [];
            for (let i = 0; i < Math.min(numPeaks, Math.floor(numSamples / blockSize)); i++) {
                const start = i * blockSize;
                const end = Math.min(start + blockSize, numSamples);
                let maxVal = 0;
                for (let j = start; j < end; j++) {
                    const val = Math.abs(raw.readFloatLE(j * 4));
                    if (val > maxVal) maxVal = val;
                }
                peaks.push(maxVal);
            }

            const maxPeak = Math.max(...peaks) || 1;
            const normalized = peaks.map(p => Math.round((p / maxPeak) * 10000) / 10000);
            const duration = await getDuration(filePath) || (numSamples / 8000);

            resolve({
                peaks: normalized,
                duration: Math.round(duration * 1000) / 1000,
                numPeaks: normalized.length
            });
        });
        proc.on('error', reject);
    });
}

/** 场景检测 */
async function sceneDetect(filePath, threshold = 0.3, minInterval = 0.5) {
    const duration = await getDuration(filePath);
    if (!duration) throw new Error(`无法获取视频时长，请检查文件是否有效: ${path.basename(filePath)}`);

    const fps = await getFrameRate(filePath);
    const resolution = await getResolution(filePath);

    const { stderr } = await runCommand('ffmpeg', [
        '-hide_banner', '-i', filePath,
        '-vf', `select='gt(scene,${threshold})',showinfo`,
        '-f', 'null', '-'
    ], { timeout: DEFAULT_TIMEOUT, allowNonZero: true });

    const scenePoints = [];
    let lastTime = -minInterval;
    const ptsRegex = /pts_time:\s*([0-9.]+)/;

    for (const line of stderr.split('\n')) {
        if (line.includes('showinfo') && line.includes('pts_time')) {
            const m = line.match(ptsRegex);
            if (m) {
                const ptsTime = parseFloat(m[1]);
                if (ptsTime < 0.3) continue;
                if (ptsTime - lastTime < minInterval) continue;
                scenePoints.push({
                    time: Math.round(ptsTime * 1000) / 1000,
                    time_str: formatSceneTime(ptsTime)
                });
                lastTime = ptsTime;
            }
        }
    }

    // 构建片段
    const boundaries = [0, ...scenePoints.map(p => p.time), duration];
    const segments = [];
    for (let i = 0; i < boundaries.length - 1; i++) {
        const segStart = boundaries[i];
        const segEnd = boundaries[i + 1];
        const segDur = segEnd - segStart;
        segments.push({
            index: i + 1,
            start: Math.round(segStart * 1000) / 1000,
            end: Math.round(segEnd * 1000) / 1000,
            start_str: formatSceneTime(segStart),
            end_str: formatSceneTime(segEnd),
            duration: Math.round(segDur * 1000) / 1000,
            duration_str: formatSceneTime(segDur)
        });
    }

    return {
        message: `检测到 ${scenePoints.length} 个场景切换点，共 ${segments.length} 个片段`,
        file: filePath,
        duration: Math.round(duration * 1000) / 1000,
        fps: Math.round(fps * 100) / 100,
        resolution,
        threshold,
        scene_points: scenePoints,
        segments
    };
}

/** 场景拆分 */
async function sceneSplit(filePath, segments, outputDir) {
    const baseName = path.parse(filePath).name;
    const ext = path.extname(filePath).toLowerCase();
    const outDir = outputDir || path.dirname(filePath);
    const sceneOutputDir = path.join(outDir, `${baseName}_scenes`);
    fs.mkdirSync(sceneOutputDir, { recursive: true });

    const exported = [];
    for (let i = 0; i < segments.length; i++) {
        const seg = segments[i];
        const start = parseFloat(seg.start || 0);
        const end = parseFloat(seg.end || 0);
        if (end - start <= 0) continue;

        const idx = seg.index || (i + 1);
        const outputFilename = `${baseName}_scene${String(idx).padStart(3, '0')}${ext}`;
        const outputPath = path.join(sceneOutputDir, outputFilename);

        await runCommand('ffmpeg', [
            '-y', '-i', filePath,
            '-ss', start.toFixed(3),
            '-to', end.toFixed(3),
            '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
            '-c:a', 'aac', '-b:a', '192k',
            '-avoid_negative_ts', 'make_zero',
            outputPath
        ]);

        exported.push({
            path: outputPath,
            filename: outputFilename,
            index: idx,
            start, end,
            duration: Math.round((end - start) * 1000) / 1000
        });
    }

    return {
        message: `成功导出 ${exported.length} 个片段到 ${sceneOutputDir}`,
        output_dir: sceneOutputDir,
        files: exported
    };
}

/** 精确裁切 */
async function mediaTrim(filePath, startTime, endTime, outputDir, precise = true) {
    const baseName = path.parse(filePath).name;
    const ext = path.extname(filePath).toLowerCase();
    const outDir = outputDir || path.dirname(filePath);
    const duration = endTime - startTime;
    const startStr = formatSceneTime(startTime);
    const endStr = formatSceneTime(endTime);
    const outputFilename = `${baseName}_trimmed_${startStr.replace(/:/g, '.')}-${endStr.replace(/:/g, '.')}${ext}`;
    const outputPath = path.join(outDir, outputFilename);

    let args;
    if (precise) {
        args = [
            '-y', '-i', filePath,
            '-ss', startTime.toFixed(3), '-to', endTime.toFixed(3),
            '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
            '-c:a', 'aac', '-b:a', '192k',
            '-avoid_negative_ts', 'make_zero',
            outputPath
        ];
    } else {
        args = [
            '-y', '-ss', startTime.toFixed(3),
            '-i', filePath,
            '-t', duration.toFixed(3),
            '-c', 'copy',
            '-avoid_negative_ts', 'make_zero',
            outputPath
        ];
    }

    await runCommand('ffmpeg', args);
    const outDuration = await getDuration(outputPath);

    return {
        message: `裁切完成: ${outputFilename}`,
        output_path: outputPath,
        output_filename: outputFilename,
        duration: outDuration ? Math.round(outDuration * 1000) / 1000 : Math.round(duration * 1000) / 1000,
        mode: precise ? '精确' : '快速'
    };
}

/** 批量提取视频首帧截图 */
async function batchThumbnail(videoFiles, outputDir, format = 'jpg', quality = 2) {
    fs.mkdirSync(outputDir, { recursive: true });
    const results = [];

    for (let i = 0; i < videoFiles.length; i++) {
        const filePath = videoFiles[i];
        const baseName = path.parse(filePath).name;
        const outFile = path.join(outputDir, `${baseName}.${format}`);

        try {
            const args = ['-y', '-i', filePath, '-vframes', '1'];
            if (format === 'jpg') {
                args.push('-q:v', String(quality));
            }
            args.push(outFile);
            await runCommand('ffmpeg', args, { timeout: 30000 });
            results.push({ path: outFile, source: filePath, success: true });
        } catch (e) {
            results.push({ source: filePath, success: false, error: e.message });
        }
    }
    return results;
}

/** 构建黑屏 MP4 命令参数 */
function buildBlackMp4Args(filePath, outputPath, start, duration, size = '1280x720', fps = 24) {
    const args = ['-y'];

    // 黑屏视频源
    if (duration != null) {
        args.push('-f', 'lavfi', '-i', `color=c=black:s=${size}:r=${fps}:d=${duration}`);
    } else {
        args.push('-f', 'lavfi', '-i', `color=c=black:s=${size}:r=${fps}`);
    }

    // 音频输入
    if (start > 0) {
        args.push('-ss', start.toFixed(3));
    }
    args.push('-i', filePath);

    // 混合滤镜：确保双声道
    args.push(
        '-filter_complex',
        `[1:a]aformat=channel_layouts=stereo[stereo];[0:v][stereo]concat=n=1:v=1:a=1[outv][outa]`,
        '-map', '[outv]', '-map', '[outa]'
    );

    // 编码设置
    args.push(
        '-c:v', 'libx264', '-preset', 'fast',
        '-c:a', 'aac', '-ac', '2', '-b:a', '192k',
        '-shortest',
        outputPath
    );

    return args;
}

/** 简化版黑屏 MP4 生成 */
async function generateBlackMp4(filePath, outputPath, start = 0, duration = null) {
    // 获取音频时长（失败时退化为 shortest 模式，不阻断导出）
    let resolvedDuration = duration;
    if (resolvedDuration == null) {
        resolvedDuration = await getDuration(filePath);
    }

    fs.mkdirSync(path.dirname(outputPath), { recursive: true });
    const args = buildBlackMp4Args(filePath, outputPath, start, resolvedDuration, '1920x1080', 30);
    await runCommand('ffmpeg', args);

    if (!fs.existsSync(outputPath)) {
        throw new Error(`黑屏 MP4 未生成: ${outputPath}`);
    }
}

/**
 * 媒体转换
 */
async function mediaConvert(filePath, mode, outDir, options = {}) {
    const baseName = path.parse(filePath).name;
    const ext = path.extname(filePath).toLowerCase();
    fs.mkdirSync(outDir, { recursive: true });

    const modeConfigs = {
        'mp3': { outputExt: '.mp3', type: 'audio' },
        'wav': { outputExt: '.wav', type: 'audio' },
        'aac': { outputExt: '.aac', type: 'audio' },
        'flac': { outputExt: '.flac', type: 'audio' },
        'mp4': { outputExt: '.mp4', type: 'video' },
        'mov': { outputExt: '.mov', type: 'video' },
        'webm': { outputExt: '.webm', type: 'video' },
        'gif': { outputExt: '.gif', type: 'video' },
        'audio_black': { outputExt: '.mp4', type: 'audio_black' },
        'audio_split': { outputExt: '', type: 'audio_split' },
    };

    const config = modeConfigs[mode];
    if (!config) throw new Error(`不支持的转换模式: ${mode}`);

    const results = [];

    if (config.type === 'audio') {
        const outputPath = path.join(outDir, `${baseName}${config.outputExt}`);
        let args;
        switch (mode) {
            case 'mp3':
                args = ['-y', '-i', filePath, '-vn', '-c:a', 'libmp3lame', '-b:a', '192k', '-ac', '2', outputPath];
                break;
            case 'wav':
                args = ['-y', '-i', filePath, '-vn', '-acodec', 'pcm_s16le', outputPath];
                break;
            case 'aac':
                args = ['-y', '-i', filePath, '-vn', '-c:a', 'aac', '-b:a', '192k', outputPath];
                break;
            case 'flac':
                args = ['-y', '-i', filePath, '-vn', '-c:a', 'flac', outputPath];
                break;
        }
        await runCommand('ffmpeg', args);
        results.push(outputPath);
    } else if (config.type === 'video') {
        const outputPath = path.join(outDir, `${baseName}${config.outputExt}`);
        let args;
        switch (mode) {
            case 'mp4':
                args = ['-y', '-i', filePath, '-c:v', 'libx264', '-crf', '23', '-c:a', 'aac', outputPath];
                break;
            case 'mov':
                args = ['-y', '-i', filePath, '-c:v', 'libx264', '-crf', '23', '-c:a', 'aac', outputPath];
                break;
            case 'webm':
                args = ['-y', '-i', filePath, '-c:v', 'libvpx-vp9', '-crf', '30', '-b:v', '0', '-c:a', 'libopus', outputPath];
                break;
            case 'gif':
                args = ['-y', '-i', filePath,
                    '-vf', `fps=10,scale=480:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse`,
                    '-loop', '0', outputPath];
                break;
        }
        await runCommand('ffmpeg', args);
        results.push(outputPath);
    } else if (config.type === 'audio_black') {
        const outputPath = path.join(outDir, `${baseName}.mp4`);
        await generateBlackMp4(filePath, outputPath);
        results.push(outputPath);
    } else if (config.type === 'audio_split') {
        // 音频裁切导出 - 由 mediaConvertBatch 中处理
    }

    return results;
}

/** 静音检测（替代 moviepy 的 silence_detect） */
async function detectSilence(filePath, noiseDb = -30, minDuration = 0.5) {
    const { stderr } = await runCommand('ffmpeg', [
        '-hide_banner', '-i', filePath,
        '-af', `silencedetect=noise=${noiseDb}dB:d=${minDuration}`,
        '-f', 'null', '-'
    ], { timeout: DEFAULT_TIMEOUT, allowNonZero: true });

    const silencePoints = [];
    const startRegex = /silence_start:\s*([0-9.]+)/g;
    const endRegex = /silence_end:\s*([0-9.]+)/g;

    let m;
    const starts = [];
    const ends = [];
    while ((m = startRegex.exec(stderr))) starts.push(parseFloat(m[1]));
    while ((m = endRegex.exec(stderr))) ends.push(parseFloat(m[1]));

    for (let i = 0; i < Math.min(starts.length, ends.length); i++) {
        silencePoints.push({
            start: starts[i],
            end: ends[i],
            mid: (starts[i] + ends[i]) / 2
        });
    }
    return silencePoints;
}

/** 智能分割分析（替代 moviepy 的 split_audio_on_silence） */
async function smartSplitAnalyze(filePath, maxDuration = 29) {
    const totalDuration = await getDuration(filePath);
    if (!totalDuration) throw new Error(`无法获取音频时长，请检查文件是否有效: ${path.basename(filePath)}`);

    // 用 FFmpeg silencedetect 检测静音点
    const silencePoints = await detectSilence(filePath, -30, 0.3);
    const silenceMids = silencePoints.map(s => s.mid);

    // 基于静音点计算分割
    const cutPoints = [0.0];
    let currentPos = 0.0;

    while (currentPos < totalDuration) {
        if (totalDuration - currentPos <= maxDuration) {
            cutPoints.push(totalDuration);
            break;
        }

        const searchLimit = currentPos + maxDuration;
        const searchStart = Math.max(currentPos + 5, searchLimit - 10);

        // 在搜索范围内找最佳静音点
        let bestCut = searchLimit;
        let minVol = Infinity;

        for (const mid of silenceMids) {
            if (mid >= searchStart && mid <= searchLimit) {
                bestCut = mid;
                break; // 取第一个在范围内的静音点
            }
        }

        if (bestCut - currentPos < 5.0) {
            bestCut = searchLimit;
        }

        cutPoints.push(bestCut);
        currentPos = bestCut;
    }

    // 构建分段信息
    const segments = [];
    for (let i = 0; i < cutPoints.length - 1; i++) {
        const start = cutPoints[i];
        const end = cutPoints[i + 1];
        segments.push({
            index: i + 1,
            start: Math.round(start * 100) / 100,
            end: Math.round(end * 100) / 100,
            duration: Math.round((end - start) * 100) / 100
        });
    }

    return {
        total_duration: Math.round(totalDuration * 100) / 100,
        max_duration: maxDuration,
        cut_points: cutPoints.map(p => Math.round(p * 100) / 100),
        segments,
        segment_count: segments.length
    };
}

/** 批量剪辑 — 将一个视频按命名片段列表导出多个文件 */
async function batchCut(filePath, segments, outputDir, precise = true) {
    const baseName = path.parse(filePath).name;
    const ext = path.extname(filePath).toLowerCase();
    const outDir = outputDir || path.dirname(filePath);
    const cutOutputDir = path.join(outDir, `${baseName}_cuts`);
    fs.mkdirSync(cutOutputDir, { recursive: true });

    // 获取视频总时长（用于 "到结尾" 的片段）
    const totalDuration = await getDuration(filePath);

    const exported = [];
    for (let i = 0; i < segments.length; i++) {
        const seg = segments[i];
        const start = parseFloat(seg.start || 0);
        let end = seg.end != null && seg.end !== '' ? parseFloat(seg.end) : null;
        if (end == null && totalDuration) end = totalDuration;
        if (end == null) throw new Error(`片段 #${i + 1}: 无法确定结束时间（无法获取视频总时长）`);
        if (end - start <= 0) continue;

        // 文件名：序号_名称
        const safeName = (seg.name || `片段${i + 1}`).replace(/[/\\:*?"<>|]/g, '_');
        const idx = String(i + 1).padStart(2, '0');
        const outputFilename = `${idx}_${safeName}${ext}`;
        const outputPath = path.join(cutOutputDir, outputFilename);

        let args;
        if (precise) {
            // 精确模式：重编码，帧级精准
            args = [
                '-y', '-i', filePath,
                '-ss', start.toFixed(3), '-to', end.toFixed(3),
                '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
                '-c:a', 'aac', '-b:a', '192k',
                '-avoid_negative_ts', 'make_zero',
                outputPath
            ];
        } else {
            // 快速模式：stream copy
            args = [
                '-y', '-ss', start.toFixed(3),
                '-i', filePath,
                '-to', (end - start).toFixed(3),
                '-c', 'copy',
                '-avoid_negative_ts', 'make_zero',
                outputPath
            ];
        }

        await runCommand('ffmpeg', args);

        exported.push({
            path: outputPath,
            filename: outputFilename,
            name: seg.name || `片段${i + 1}`,
            index: i + 1,
            start, end,
            duration: Math.round((end - start) * 1000) / 1000,
            mode: precise ? '精确' : '快速'
        });
    }

    return {
        message: `成功导出 ${exported.length} 个片段到 ${cutOutputDir}`,
        output_dir: cutOutputDir,
        files: exported,
        mode: precise ? '精确' : '快速'
    };
}

// ==================== 工具函数 ====================

function formatSceneTime(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    if (h > 0) {
        return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${s.toFixed(3).padStart(6, '0')}`;
    }
    return `${String(m).padStart(2, '0')}:${s.toFixed(3).padStart(6, '0')}`;
}

function parseTimecode(token, fps = 25) {
    const parts = token.split(':');
    if (parts.length === 4) {
        // HH:MM:SS:FF (NLE timecode with frames)
        return parseFloat(parts[0]) * 3600 + parseFloat(parts[1]) * 60 + parseFloat(parts[2]) + parseFloat(parts[3]) / fps;
    } else if (parts.length === 3) {
        return parseFloat(parts[0]) * 3600 + parseFloat(parts[1]) * 60 + parseFloat(parts[2]);
    } else if (parts.length === 2) {
        return parseFloat(parts[0]) * 60 + parseFloat(parts[1]);
    }
    return parseFloat(token);
}

function parseCutPoints(raw) {
    if (!raw || !raw.trim()) return [];
    const normalized = raw.replace(/，/g, ',').replace(/；/g, ';');
    const tokens = normalized.split(/[,\s;]+/).filter(t => t.trim());
    return tokens.map(t => {
        const val = parseTimecode(t.trim());
        if (isNaN(val) || val < 0) throw new Error(`无效的分割点: ${t}`);
        return val;
    }).sort((a, b) => a - b);
}

function buildSegments(cutPoints) {
    if (cutPoints.length === 0) return [[0, null]];
    const segments = [];
    segments.push([0, cutPoints[0]]);
    for (let i = 0; i < cutPoints.length - 1; i++) {
        segments.push([cutPoints[i], cutPoints[i + 1]]);
    }
    segments.push([cutPoints[cutPoints.length - 1], null]);
    return segments;
}

module.exports = {
    runCommand,
    getDuration,
    getFrameRate,
    getResolution,
    getWaveformBinary,
    sceneDetect,
    sceneSplit,
    mediaTrim,
    batchThumbnail,
    generateBlackMp4,
    mediaConvert,
    detectSilence,
    smartSplitAnalyze,
    formatSceneTime,
    parseTimecode,
    parseCutPoints,
    buildSegments,
    buildBlackMp4Args,
    batchCut,
};
