/**
 * reels-wysiwyg-export.js — WYSIWYG 混合导出引擎
 *
 * 混合架构：
 *   1. FFmpeg 处理背景视频（循环+淡入淡出+缩放+音频混合）→ 临时背景视频
 *   2. FFmpeg 提取背景帧序列 → JPEG 文件
 *   3. Canvas 逐帧渲染：背景帧 + 全局蒙版 + 字幕（与预览完全相同的 renderer）
 *   4. 合成帧 → JPEG pipe → FFmpeg 编码 → 临时纯视频
 *   5. Web Audio OfflineAudioContext 离线渲染音频（含混响+立体声）→ WAV
 *   6. FFmpeg 简单合并视频 + 预渲染音频 → 最终输出
 *
 * 音频效果完全使用 Web Audio API 渲染，确保导出与预览 100% 一致。
 */

/**
 * Canvas → Raw RGBA Uint8Array（零压缩，专业级画质）
 * 注意：必须用 slice() 创建独立副本，避免 IPC 传输时 SharedArrayBuffer 问题
 */
function _canvasToRawRGBA(canvas) {
    const ctx = canvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    return imageData.data.buffer.slice(0);
}

/**
 * 加载图片为 Image 对象 (返回 Promise)
 */
function _loadImage(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error(`图片加载失败: ${src}`));
        img.src = src;
    });
}

/**
 * AudioBuffer → WAV (PCM 16-bit) ArrayBuffer
 */
function _audioBufferToWav(buffer, maxSamples) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const bitsPerSample = 16;
    const bytesPerSample = bitsPerSample / 8;
    const blockAlign = numChannels * bytesPerSample;
    const numSamples = maxSamples ? Math.min(maxSamples, buffer.length) : buffer.length;
    const dataSize = numSamples * blockAlign;
    const headerSize = 44;
    const totalSize = headerSize + dataSize;

    const wav = new ArrayBuffer(totalSize);
    const view = new DataView(wav);

    const writeStr = (offset, str) => {
        for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    };
    writeStr(0, 'RIFF');
    view.setUint32(4, totalSize - 8, true);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitsPerSample, true);
    writeStr(36, 'data');
    view.setUint32(40, dataSize, true);

    const channels = [];
    for (let ch = 0; ch < numChannels; ch++) {
        channels.push(buffer.getChannelData(ch));
    }

    let offset = headerSize;
    for (let i = 0; i < numSamples; i++) {
        for (let ch = 0; ch < numChannels; ch++) {
            let sample = channels[ch][i];
            sample = Math.max(-1, Math.min(1, sample));
            const int16 = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
            view.setInt16(offset, int16, true);
            offset += bytesPerSample;
        }
    }

    return wav;
}


/**
 * WYSIWYG 导出一个 Reel 任务
 */
async function reelsWysiwygExport(params) {
    let {
        canvas,
        style,
        segments,
        overlays: taskOverlays,
        backgroundPath,
        voicePath,
        outputPath,
        targetWidth = 1080,
        targetHeight = 1920,
        fps = 30,
        voiceVolume = 1.0,
        bgVolume = 0.1,
        loopFade = true,
        loopFadeDur = 1.0,
        customDuration = 0,  // 自定义输出时长（秒），0 = 自动
        bgmPath = '',        // 配乐文件路径
        bgmVolume = 0.3,     // 配乐音量 (0~1)
        bgScale = 100,       // 背景图片缩放 (50~300%)
        bgDurScale = 100,    // 背景素材时长缩放 (10~500%)
        audioDurScale = 100,  // 音频素材时长缩放 (10~500%)
        reverbEnabled = false,
        reverbPreset = 'hall',
        reverbMix = 30,
        stereoWidth = 100,
        audioFxTarget = 'all',
        useGPU = false,
        onProgress,
        onLog,
        isCancelled,
    } = params;

    if (!canvas) throw new Error('需要提供 canvas');
    if (!backgroundPath) throw new Error('缺少背景素材');
    if (!outputPath) throw new Error('缺少输出路径');
    if (!window.electronAPI || !window.electronAPI.reelsComposeWysiwyg) {
        throw new Error('需要 reelsComposeWysiwyg IPC 接口');
    }

    // 允许无字幕（纯覆层模式）
    if (!segments) segments = [];

    const log = (msg) => { if (onLog) onLog(msg); console.log(`[WYSIWYG] ${msg}`); };
    const progress = (v) => { if (onProgress) onProgress(v); };

    canvas.width = targetWidth;
    canvas.height = targetHeight;
    const ctx = canvas.getContext('2d');
    const renderer = new ReelsCanvasRenderer(canvas);

    // 获取时长：优先自定义时长，否则音频时长，否则视频背景时长
    let duration = 0;
    const _audioDurFactor = (audioDurScale || 100) / 100;
    const _bgDurFactor = (bgDurScale || 100) / 100;
    if (customDuration > 0) {
        duration = customDuration;
        log(`使用自定义时长: ${duration}s`);
    } else {
        if (voicePath) {
            log('正在获取音频时长...');
            let rawAudioDur = await window.electronAPI.getMediaDuration(voicePath);
            if (rawAudioDur > 0 && _audioDurFactor !== 1.0) {
                duration = rawAudioDur * _audioDurFactor;
                log(`音频原始时长: ${rawAudioDur.toFixed(2)}s × ${audioDurScale}% = ${duration.toFixed(2)}s`);
            } else {
                duration = rawAudioDur;
            }
        }
        if (!duration || duration <= 0) {
            log('正在获取背景视频时长...');
            let rawBgDur = await window.electronAPI.getMediaDuration(backgroundPath);
            if (rawBgDur > 0 && _bgDurFactor !== 1.0) {
                duration = rawBgDur * _bgDurFactor;
                log(`背景原始时长: ${rawBgDur.toFixed(2)}s × ${bgDurScale}% = ${duration.toFixed(2)}s`);
            } else {
                duration = rawBgDur;
            }
        }
        if (!duration || duration <= 0) {
            // 图片背景无音频时默认 5 秒
            duration = 5;
            log(`无法获取时长，使用默认 ${duration}s`);
        }
    }
    const totalFrames = Math.ceil(duration * fps);
    log(`时长: ${duration.toFixed(2)}s, 帧数: ${totalFrames}, FPS: ${fps}`);
    if (bgScale !== 100) log(`背景缩放: ${bgScale}%`);
    if (bgDurScale !== 100) log(`背景时长缩放: ${bgDurScale}%`);
    if (audioDurScale !== 100) log(`音频时长缩放: ${audioDurScale}%`);

    // ── 按 audioDurScale 缩放字幕时间戳（让字幕跟随音频拉长/缩短）──
    if (_audioDurFactor !== 1.0 && segments && segments.length > 0) {
        log(`字幕时间戳同步缩放 ×${_audioDurFactor.toFixed(2)}`);
        segments = segments.map(seg => ({
            ...seg,
            start: (seg.start || 0) * _audioDurFactor,
            end:   (seg.end   || 0) * _audioDurFactor,
        }));
    }

    // 注意：OfflineAudioContext 在 Electron contextIsolation:true 下会崩溃
    // 所有音频效果由 FFmpeg afir 卷积滤镜处理（使用相同 seeded PRNG 的 IR）

    // ═══ 阶段 1: 让主进程用 FFmpeg 预处理背景 + 提取帧序列 ═══
    log('阶段1: FFmpeg 预处理背景视频（循环+淡入淡出+提取帧）...');
    progress(2);

    const prepResult = await window.electronAPI.reelsComposeWysiwyg('prepare-bg', {
        backgroundPath,
        voicePath,
        targetWidth,
        targetHeight,
        fps,
        duration: duration,
        loopFade,
        loopFadeDur,
        bgScale: bgScale || 100,
    });

    if (!prepResult || prepResult.error) {
        throw new Error(prepResult?.error || '背景预处理失败');
    }
    const framesDir = prepResult.framesDir;
    const totalBgFrames = prepResult.frameCount;
    log(`背景帧提取完成: ${totalBgFrames} 帧 → ${framesDir}`);
    progress(20);

    // ═══ 阶段 2: 启动 FFmpeg 编码器 ═══
    log('阶段2: 启动 FFmpeg 编码器...');
    const sessionId = await window.electronAPI.reelsComposeWysiwyg('start', {
        width: targetWidth,
        height: targetHeight,
        fps,
        outputPath,
        voicePath,
        voiceVolume,
        bgVolume,
        backgroundPath,
        // 如果 voicePath === backgroundPath（voiced_bg 模式），
        // 背景音频已经作为 voice 混入，不要再重复混入
        bgHasAudio: (voicePath && backgroundPath && voicePath === backgroundPath) ? false : !_isImageFile(backgroundPath),
        bgmPath: bgmPath || '',
        bgmVolume: bgmVolume || 0,
        audioDurScale: audioDurScale || 100,
        reverbEnabled,
        reverbPreset,
        reverbMix,
        stereoWidth,
        audioFxTarget,
        useGPU,
    });
    if (!sessionId) throw new Error('FFmpeg 启动失败');

    // 全局蒙版
    const hasMask = !!style.global_mask_enabled;
    const maskColor = style.global_mask_color || '#000000';
    const maskOpacity = style.global_mask_opacity ?? 0.5;


    // ═══ 阶段 3: 逐帧渲染 ═══
    log('阶段3: 逐帧 Canvas 渲染...');
    const t0 = Date.now();

    // 预加载第一帧
    let currentBgImg = null;
    let currentBgIdx = -1;

    try {
        for (let frameIdx = 0; frameIdx < totalFrames; frameIdx++) {
            // ── 取消检查 ──
            if (isCancelled && isCancelled()) {
                log('用户取消导出');
                throw new Error('__CANCELLED__');
            }
            const t = frameIdx / fps;

            // 加载背景帧（从 FFmpeg 提取的 JPEG 序列）
            const bgFrameIdx = Math.min(frameIdx, totalBgFrames - 1);
            if (bgFrameIdx !== currentBgIdx) {
                const framePath = `${framesDir}/frame_${String(bgFrameIdx + 1).padStart(6, '0')}.png`;
                try {
                    currentBgImg = await _loadImage(`file://${framePath}`);
                    currentBgIdx = bgFrameIdx;
                } catch (e) {
                    // 帧文件可能不存在（超出范围），保持最后一帧
                    if (!currentBgImg) {
                        // 没有任何帧，画黑色
                        ctx.fillStyle = '#000000';
                        ctx.fillRect(0, 0, targetWidth, targetHeight);
                    }
                }
            }

            // ── 绘制背景帧 ──
            ctx.fillStyle = '#000000';
            ctx.fillRect(0, 0, targetWidth, targetHeight);
            if (currentBgImg) {
                // 应用背景缩放
                const _userScale = (bgScale || 100) / 100;
                if (Math.abs(_userScale - 1.0) < 0.01) {
                    // 无缩放，直接绘制（FFmpeg 已处理尺寸）
                    ctx.drawImage(currentBgImg, 0, 0, targetWidth, targetHeight);
                } else {
                    // 应用用户缩放: 放大/缩小后居中裁切
                    const srcW = currentBgImg.naturalWidth || targetWidth;
                    const srcH = currentBgImg.naturalHeight || targetHeight;
                    const coverScale = Math.max(targetWidth / srcW, targetHeight / srcH) * _userScale;
                    const drawW = srcW * coverScale;
                    const drawH = srcH * coverScale;
                    const drawX = (targetWidth - drawW) / 2;
                    const drawY = (targetHeight - drawH) / 2;
                    ctx.drawImage(currentBgImg, drawX, drawY, drawW, drawH);
                }
            }

            // ── 全局蒙版 ──
            if (hasMask) {
                ctx.save();
                ctx.globalAlpha = maskOpacity;
                ctx.fillStyle = maskColor;
                ctx.fillRect(0, 0, targetWidth, targetHeight);
                ctx.restore();
            }

            // ── 覆盖层（文案卡片等）──
            if (taskOverlays && taskOverlays.length > 0 && window.ReelsOverlay) {
                for (const ov of taskOverlays) {
                    const ovStart = parseFloat(ov.start || 0);
                    const ovEnd = parseFloat(ov.end || 9999);
                    // scroll 覆层不受 end 限制（动画完成后保持最终位置）
                    if (t >= ovStart && (ov.type === 'scroll' || t <= ovEnd)) {
                        // 导出时不画辅助线和选中框
                        const origSelected = ov._selected;
                        ov._selected = false;
                        ov._exporting = true;
                        ov._exportDuration = duration; // 让 scroll 覆层知道实际导出时长
                        ReelsOverlay.drawOverlay(ctx, ov, t, targetWidth, targetHeight);
                        ov._selected = origSelected;
                        ov._exporting = false;
                    }
                }
            }

            // ── 字幕（与预览完全相同的渲染器）──
            const activeSeg = segments.find(seg => t >= (seg.start || 0) && t <= (seg.end || 0));
            if (activeSeg) {
                renderer.renderSubtitle(style, activeSeg, t, targetWidth, targetHeight);
            }

            // ── AI 水印 ──
            if (typeof _drawWatermarks === 'function') {
                _drawWatermarks(ctx, targetWidth, targetHeight);
            }

            // ── Canvas → Raw RGBA → IPC（零压缩）──
            const rawBuf = _canvasToRawRGBA(canvas);
            const frameResult = await window.electronAPI.reelsComposeWysiwyg('frame', {
                sessionId,
                raw: rawBuf,
            });
            // 兼容旧版 boolean 和新版 { ok, error, detail } 格式
            const frameOk = frameResult === true || (frameResult && frameResult.ok);
            if (!frameOk) {
                const errMsg = (frameResult && frameResult.error) || '未知编码错误';
                const errDetail = (frameResult && frameResult.detail) || '';
                throw new Error(`FFmpeg 写入帧失败 (frame ${frameIdx}): ${errMsg}${errDetail ? '\n' + errDetail.slice(-200) : ''}`);
            }

            // ── 进度 + UI yield ──
            if (frameIdx % Math.max(1, Math.floor(fps / 2)) === 0) {
                const pct = 20 + Math.round((frameIdx / totalFrames) * 65);
                progress(pct);
                const elapsed = (Date.now() - t0) / 1000;
                const fpsActual = (frameIdx + 1) / Math.max(0.1, elapsed);
                const eta = (totalFrames - frameIdx) / Math.max(0.1, fpsActual);
                log(`帧 ${frameIdx + 1}/${totalFrames} (${pct}%) | ${fpsActual.toFixed(1)} fps | 剩余 ~${Math.ceil(eta)}s`);
                await new Promise(r => setTimeout(r, 0)); // yield
            }
        }

        // ═══ 阶段 4: 完成编码 + 混合音频 ═══
        log('阶段4: 完成编码与音频混合...');
        progress(88);
        const result = await window.electronAPI.reelsComposeWysiwyg('finish', { sessionId });
        if (!result || result.error) throw new Error(result?.error || 'FFmpeg 编码完成失败');

        // 清理背景帧
        await window.electronAPI.reelsComposeWysiwyg('cleanup-bg', { framesDir });

        progress(100);
        const totalTime = ((Date.now() - t0) / 1000).toFixed(1);
        log(`导出完成 (${totalTime}s): ${outputPath}`);
        return { output_path: result.output_path || outputPath };

    } catch (e) {
        try { await window.electronAPI.reelsComposeWysiwyg('abort', { sessionId }); } catch (_) { }
        try { await window.electronAPI.reelsComposeWysiwyg('cleanup-bg', { framesDir }); } catch (_) { }
        if (e && e.message === '__CANCELLED__') {
            log('导出已取消，资源已清理');
            return { cancelled: true };
        }
        throw e;
    }
}

function _isImageFile(filePath) {
    const ext = (filePath || '').split('.').pop().toLowerCase();
    return ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'gif'].includes(ext);
}

if (typeof window !== 'undefined') {
    window.reelsWysiwygExport = reelsWysiwygExport;
}
