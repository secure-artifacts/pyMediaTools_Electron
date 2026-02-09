// API 基础 URL
const API_BASE = 'http://127.0.0.1:5001/api';
const API_ORIGIN = API_BASE.replace(/\/api$/, '');

// 当前选中的文件路径
let currentAudioPath = '';
let currentSrtSrcPath = '';
let currentSrtOrgiPath = '';
let currentSrtRefPath = '';
let currentSeamlessSrtPath = '';
let currentMediaFiles = [];
let currentMediaFileInfos = [];
let currentAudioCutPoints = {};
let currentVideoUrl = '';
let backendReady = false;
let settingsAutoLoaded = false;
let replaceRulesCache = null;

// 音频预览状态
let audioPreviewElement = null;
let currentPreviewFilePath = '';

// ElevenLabs 播放器状态
let audioPlayer = null;
let currentAudioPath_elevenlabs = '';

const LOGO_DEFAULTS = {
    hailuo: { x: 590, y: 1810, w: 475, h: 90 },
    vidu: { x: 700, y: 1810, w: 360, h: 90 },
    veo: { x: 700, y: 1810, w: 360, h: 90 },
    heygen: { x: 700, y: 1810, w: 360, h: 90 },
    dream: { x: 700, y: 1810, w: 360, h: 90 },
    ai_generated: { x: 680, y: 20, w: 380, h: 60 },
    custom: { x: 590, y: 1810, w: 400, h: 90 }
};

const LOGO_PRESET_ASSETS = {
    hailuo: 'Hailuo.png',
    vidu: 'vidu.png',
    veo: 'Veo.png',
    heygen: 'HeyGen.png',
    dream: 'Dream.png',
    ai_generated: 'AI_Generated.png'
};

const logoImageCache = new Map();
const voiceCache = new Map();

// Toast 通知系统
function showToast(message, type = 'info', duration = 4000) {
    const existingToast = document.querySelector('.toast');
    if (existingToast) {
        existingToast.remove();
    }

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <span class="toast-icon">${type === 'success' ? '✓' : type === 'error' ? '✕' : 'ℹ'}</span>
        <span class="toast-message">${message}</span>
    `;

    document.body.appendChild(toast);
    setTimeout(() => toast.classList.add('show'), 10);
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initSubTabs();
    initFileInputs();
    initAudioPlayer();
    initMediaModeOptions();
    initBatchTTS();
    initSubtitleBatch();
    loadSettings();
    loadWatermarkSettings();  // 加载保存的水印设置
    checkBackendHealth();
    addToastStyles();

    // 启动心跳检测（每30秒检查一次后端状态）
    startHeartbeat();
});

// 心跳检测，保持后端活跃
let heartbeatInterval = null;
let lastHeartbeatSuccess = true;

function startHeartbeat() {
    // 每30秒发送一次心跳
    heartbeatInterval = setInterval(async () => {
        if (!backendReady) return; // 如果还没连上，不发心跳

        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000);

            const response = await fetch(`${API_BASE}/health`, {
                signal: controller.signal
            });
            clearTimeout(timeoutId);

            if (response.ok) {
                if (!lastHeartbeatSuccess) {
                    // 之前断开了，现在恢复了
                    updateStatus('后端服务已恢复连接', 'success');
                    showToast('后端服务已恢复', 'success');
                    lastHeartbeatSuccess = true;
                }
            } else {
                throw new Error('后端响应异常');
            }
        } catch (error) {
            if (lastHeartbeatSuccess) {
                // 之前正常，现在断开了
                updateStatus('后端服务连接断开，尝试重连...', 'error');
                lastHeartbeatSuccess = false;
                // 重新开始健康检查（会尝试重连）
                healthCheckRetries = 0;
                checkBackendHealth();
            }
        }
    }, 30000); // 30秒
}

// 添加 Toast 样式
function addToastStyles() {
    const style = document.createElement('style');
    style.textContent = `
        .toast {
            position: fixed;
            top: 80px;
            left: 50%;
            transform: translateX(-50%) translateY(-20px);
            padding: 12px 24px;
            background: rgba(0, 0, 0, 0.9);
            border-radius: 8px;
            display: flex;
            align-items: center;
            gap: 10px;
            z-index: 9999;
            opacity: 0;
            transition: all 0.3s ease;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            max-width: 80%;
        }
        .toast.show { opacity: 1; transform: translateX(-50%) translateY(0); }
        .toast-success { border-left: 4px solid #00d9a5; }
        .toast-error { border-left: 4px solid #ff4757; }
        .toast-info { border-left: 4px solid #3498db; }
        .toast-icon { font-size: 16px; font-weight: bold; }
        .toast-success .toast-icon { color: #00d9a5; }
        .toast-error .toast-icon { color: #ff4757; }
        .toast-info .toast-icon { color: #3498db; }
        .toast-message { color: white; font-size: 14px; }
    `;
    document.head.appendChild(style);
}

// 标签页切换
function initTabs() {
    const tabs = document.querySelectorAll('.tab');
    const panels = document.querySelectorAll('.panel');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            panels.forEach(p => p.classList.remove('active'));

            tab.classList.add('active');
            const panelId = tab.dataset.tab + '-panel';
            document.getElementById(panelId).classList.add('active');
        });
    });
}

// 子标签页切换
function initSubTabs() {
    const subTabs = document.querySelectorAll('.sub-tab');

    subTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const parent = tab.parentElement.parentElement;
            parent.querySelectorAll('.sub-tab').forEach(t => t.classList.remove('active'));
            parent.querySelectorAll('.subtab-content').forEach(c => c.classList.remove('active'));

            tab.classList.add('active');
            const contentId = tab.dataset.subtab + '-subtab';
            document.getElementById(contentId).classList.add('active');

            // 刷新对应的预览
            if (contentId === 'media-logo-subtab') {
                setTimeout(updateLogoPreview, 100);
            } else if (contentId === 'media-watermark-subtab') {
                setTimeout(updateWatermarkPreview, 100);
            }
        });
    });
}

// 初始化 Audio 播放器
function initAudioPlayer() {
    audioPlayer = document.getElementById('tts-audio');
    const seekSlider = document.getElementById('seek-slider');
    const btnPlay = document.getElementById('btn-play');

    if (!audioPlayer) return;

    audioPlayer.addEventListener('loadedmetadata', () => {
        seekSlider.max = Math.floor(audioPlayer.duration);
        document.getElementById('total-time').textContent = formatTime(audioPlayer.duration);
        seekSlider.disabled = false;
        btnPlay.disabled = false;
    });

    audioPlayer.addEventListener('timeupdate', () => {
        if (!seekSlider.dragging) {
            seekSlider.value = Math.floor(audioPlayer.currentTime);
            document.getElementById('current-time').textContent = formatTime(audioPlayer.currentTime);
        }
    });

    audioPlayer.addEventListener('ended', () => {
        btnPlay.textContent = '▶ 播放';
        seekSlider.value = 0;
        document.getElementById('current-time').textContent = '00:00';
    });

    seekSlider.addEventListener('input', () => {
        audioPlayer.currentTime = seekSlider.value;
        document.getElementById('current-time').textContent = formatTime(seekSlider.value);
    });

    // 稳定性滑块
    const stabilitySlider = document.getElementById('tts-stability');
    if (stabilitySlider) {
        stabilitySlider.addEventListener('input', (e) => {
            document.getElementById('stability-value').textContent = e.target.value + '%';
        });
    }
}

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

function togglePlayback() {
    const btnPlay = document.getElementById('btn-play');

    if (audioPlayer.paused) {
        audioPlayer.play();
        btnPlay.textContent = '⏸ 暂停';
    } else {
        audioPlayer.pause();
        btnPlay.textContent = '▶ 继续';
    }
}

// 初始化文件输入
function initFileInputs() {
    // 音频文件
    document.getElementById('audio-file-input').addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            currentAudioPath = file.path || file.name;
            document.getElementById('audio-path').value = file.name;
            showToast(`已选择: ${file.name}`, 'success');
        }
    });

    // 原文本文件
    document.getElementById('source-file-input').addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            const reader = new FileReader();
            reader.onload = (ev) => {
                document.getElementById('source-text').value = ev.target.result;
                showToast('原文本已加载', 'success');
            };
            reader.readAsText(file);
        }
    });

    // 翻译文本文件
    document.getElementById('translate-file-input').addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            const reader = new FileReader();
            reader.onload = (ev) => {
                document.getElementById('translate-text').value = ev.target.result;
                showToast('翻译文本已加载', 'success');
            };
            reader.readAsText(file);
        }
    });

    // SRT 文件
    document.getElementById('srt-src-input').addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            currentSrtSrcPath = file.path || file.name;
            document.getElementById('srt-src-path').value = file.name;
        }
    });

    document.getElementById('srt-orgi-input').addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            currentSrtOrgiPath = file.path || file.name;
            document.getElementById('srt-orgi-path').value = file.name;
        }
    });

    document.getElementById('srt-ref-input').addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            currentSrtRefPath = file.path || file.name;
            document.getElementById('srt-ref-path').value = file.name;
        }
    });

    document.getElementById('seamless-srt-input').addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            currentSeamlessSrtPath = file.path || file.name;
            document.getElementById('seamless-srt-path').value = file.name;
        }
    });

    // 媒体文件
    document.getElementById('media-input-file').addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            currentMediaFileInfos = Array.from(e.target.files).map(f => ({
                path: f.path || f.name,
                name: f.name,
                file: f  // 保存 File 对象引用，用于创建 blob URL 播放
            }));
            currentMediaFiles = currentMediaFileInfos.map(item => item.path);
            document.getElementById('media-input-path').value =
                e.target.files.length === 1 ? e.target.files[0].name : `${e.target.files.length} 个文件`;
            renderAudioSplitFileList();
        }
    });

    // 声音搜索回车
    const voiceSearchInput = document.getElementById('voice-search-input');
    if (voiceSearchInput) {
        voiceSearchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                searchVoices();
            }
        });
    }
}

function renderAudioSplitFileList() {
    const list = document.getElementById('audio-split-file-list');
    if (!list) return;

    list.innerHTML = '';

    // 隐藏旧的全局预览播放器
    const globalPlayer = document.getElementById('audio-preview-player');
    if (globalPlayer) globalPlayer.style.display = 'none';

    if (currentMediaFileInfos.length === 0) {
        const hint = document.createElement('p');
        hint.className = 'hint';
        hint.textContent = '请先选择文件。';
        list.appendChild(hint);
        return;
    }

    const nextCutPoints = {};

    currentMediaFileInfos.forEach((file, idx) => {
        // 创建文件卡片
        const card = document.createElement('div');
        card.className = 'audio-file-card';
        card.dataset.idx = idx;
        card.style.cssText = 'background: var(--bg-tertiary); border-radius: 8px; padding: 12px; border: 1px solid rgba(255,255,255,0.05);';

        // 顶部：文件名 + 时长 + 状态
        const header = document.createElement('div');
        header.style.cssText = 'display: flex; align-items: center; gap: 8px; margin-bottom: 8px;';

        const playBtn = document.createElement('button');
        playBtn.className = 'btn btn-secondary';
        playBtn.style.cssText = 'padding: 4px 8px; font-size: 12px;';
        playBtn.textContent = '▶️';
        playBtn.onclick = () => playAudioInCard(idx, file);

        const name = document.createElement('div');
        name.style.cssText = 'flex: 1; font-size: 13px; color: var(--text-primary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;';
        name.textContent = file.name || `文件 ${idx + 1}`;
        name.title = file.name;

        const duration = document.createElement('span');
        duration.className = 'audio-card-duration';
        duration.id = `audio-card-duration-${idx}`;
        duration.style.cssText = 'font-size: 11px; color: var(--text-muted);';
        duration.textContent = '--:--';

        const status = document.createElement('span');
        status.className = 'audio-card-status';
        status.id = `audio-card-status-${idx}`;
        status.style.cssText = 'font-size: 11px; padding: 2px 6px; border-radius: 3px; background: rgba(128,128,128,0.2); color: var(--text-muted);';
        status.textContent = '待分析';

        header.appendChild(playBtn);
        header.appendChild(name);
        header.appendChild(duration);
        header.appendChild(status);

        // 波形图容器
        const waveformContainer = document.createElement('div');
        waveformContainer.style.cssText = 'position: relative; height: 50px; margin-bottom: 8px; background: var(--bg-secondary); border-radius: 4px; overflow: hidden; cursor: pointer;';
        waveformContainer.dataset.idx = idx;

        const canvas = document.createElement('canvas');
        canvas.id = `audio-waveform-${idx}`;
        canvas.style.cssText = 'width: 100%; height: 100%; pointer-events: none;';

        const progress = document.createElement('div');
        progress.id = `audio-progress-${idx}`;
        progress.style.cssText = 'position: absolute; left: 0; top: 0; bottom: 0; width: 0%; background: rgba(102, 126, 234, 0.3); pointer-events: none;';

        // 播放光标
        const cursor = document.createElement('div');
        cursor.id = `audio-cursor-${idx}`;
        cursor.style.cssText = 'position: absolute; top: 0; bottom: 0; width: 2px; background: #ff6b6b; left: 0%; pointer-events: none; display: none;';

        const loading = document.createElement('div');
        loading.id = `audio-loading-${idx}`;
        loading.style.cssText = 'position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; color: var(--text-muted); font-size: 11px;';
        loading.textContent = '加载波形...';

        // 点击/拖拽跳转
        const seekToPosition = (e) => {
            const audio = document.getElementById(`audio-element-${idx}`);
            if (!audio) return;

            // 如果音频还没加载，先加载
            if (!audio.src && file.file) {
                audio.src = URL.createObjectURL(file.file);
            }

            // 获取时长（从音频或从 audioCardData）
            const duration = audio.duration || window.audioCardData?.[idx]?.duration;
            if (!duration) return;

            const rect = waveformContainer.getBoundingClientRect();
            const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
            const ratio = x / rect.width;

            // 如果音频已加载完成，直接跳转
            if (audio.readyState >= 1) {
                audio.currentTime = ratio * duration;
            } else {
                // 等待元数据加载完成后跳转
                audio.onloadedmetadata = () => {
                    audio.currentTime = ratio * audio.duration;
                };
            }

            // 更新光标
            cursor.style.left = (ratio * 100) + '%';
            cursor.style.display = 'block';
            progress.style.width = (ratio * 100) + '%';
        };

        waveformContainer.addEventListener('click', seekToPosition);

        // 拖拽支持
        let isDragging = false;
        waveformContainer.addEventListener('mousedown', (e) => {
            isDragging = true;
            seekToPosition(e);
            e.preventDefault();  // 防止选中文字
        });
        waveformContainer.addEventListener('mousemove', (e) => {
            if (isDragging) {
                seekToPosition(e);
            }
        });
        waveformContainer.addEventListener('mouseup', () => {
            isDragging = false;
        });
        waveformContainer.addEventListener('mouseleave', () => {
            isDragging = false;
        });

        waveformContainer.appendChild(canvas);
        waveformContainer.appendChild(progress);
        waveformContainer.appendChild(cursor);
        waveformContainer.appendChild(loading);

        // 分割点输入
        const cutRow = document.createElement('div');
        cutRow.style.cssText = 'display: flex; align-items: center; gap: 8px;';

        const cutLabel = document.createElement('span');
        cutLabel.style.cssText = 'font-size: 11px; color: var(--text-muted); white-space: nowrap;';
        cutLabel.textContent = '分割点:';

        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'input';
        input.id = `audio-cut-points-${idx}`;
        input.placeholder = '例如: 12.5, 01:10, 02:30.5';
        input.value = currentAudioCutPoints[file.path] || '';
        input.style.cssText = 'flex: 1; padding: 4px 8px; font-size: 12px;';
        input.addEventListener('input', () => {
            currentAudioCutPoints[file.path] = input.value;
        });

        const addCutBtn = document.createElement('button');
        addCutBtn.className = 'btn btn-secondary';
        addCutBtn.style.cssText = 'padding: 4px 8px; font-size: 11px;';
        addCutBtn.textContent = '✂️';
        addCutBtn.title = '在当前播放位置添加分割点';
        addCutBtn.onclick = () => addCutPointToCard(idx, file.path);

        if (currentAudioCutPoints[file.path]) {
            nextCutPoints[file.path] = currentAudioCutPoints[file.path];
        }

        cutRow.appendChild(cutLabel);
        cutRow.appendChild(input);
        cutRow.appendChild(addCutBtn);

        // 隐藏的 audio 元素
        const audio = document.createElement('audio');
        audio.id = `audio-element-${idx}`;
        audio.style.display = 'none';

        card.appendChild(header);
        card.appendChild(waveformContainer);
        card.appendChild(cutRow);
        card.appendChild(audio);
        list.appendChild(card);

        // 异步生成波形
        if (file.file) {
            generateWaveformForCard(idx, file.file);
        }
    });

    currentAudioCutPoints = nextCutPoints;

    // 更新智能分割按钮状态
    if (typeof updateSmartSplitButtonState === 'function') {
        updateSmartSplitButtonState();
    }
}

// 检测是否为视频文件（通过文件扩展名或 MIME 类型）
function isVideoFile(file) {
    const videoExtensions = ['.mp4', '.mov', '.mkv', '.avi', '.wmv', '.flv', '.webm', '.m4v'];
    const fileName = file.name?.toLowerCase() || '';
    const mimeType = file.type?.toLowerCase() || '';

    return videoExtensions.some(ext => fileName.endsWith(ext)) || mimeType.startsWith('video/');
}

// 为单个卡片生成波形
async function generateWaveformForCard(idx, fileObj) {
    const canvas = document.getElementById(`audio-waveform-${idx}`);
    const loading = document.getElementById(`audio-loading-${idx}`);
    const durationEl = document.getElementById(`audio-card-duration-${idx}`);
    if (!canvas) return;

    try {
        // 检测是否为视频文件 - 视频文件无法使用 Web Audio API 解码
        if (isVideoFile(fileObj)) {
            // 对于视频文件，使用 video 元素获取时长
            await generateWaveformForVideo(idx, fileObj, canvas, loading, durationEl);
            return;
        }

        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const arrayBuffer = await fileObj.arrayBuffer();

        let audioBuffer;
        try {
            audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        } catch (decodeError) {
            console.warn('音频解码失败，尝试使用媒体元素:', decodeError.message);
            audioContext.close();
            // 解码失败时，回退到视频处理方式
            await generateWaveformForVideo(idx, fileObj, canvas, loading, durationEl);
            return;
        }

        // 更新时长显示
        if (durationEl) {
            durationEl.textContent = formatTimeAudio(audioBuffer.duration);
        }

        // 获取音频数据
        const channelData = audioBuffer.getChannelData(0);
        const samples = 150;
        const blockSize = Math.floor(channelData.length / samples);
        const peaks = [];

        for (let i = 0; i < samples; i++) {
            let sum = 0;
            for (let j = 0; j < blockSize; j++) {
                sum += Math.abs(channelData[i * blockSize + j]);
            }
            peaks.push(sum / blockSize);
        }

        const maxPeak = Math.max(...peaks);
        const normalizedPeaks = peaks.map(p => p / maxPeak);

        // 保存数据
        if (!window.audioCardData) window.audioCardData = {};
        window.audioCardData[idx] = {
            peaks: normalizedPeaks,
            duration: audioBuffer.duration
        };

        // 绘制波形
        drawWaveform(canvas, normalizedPeaks);
        if (loading) loading.style.display = 'none';

        audioContext.close();
    } catch (error) {
        console.error('波形生成失败:', error);
        if (loading) {
            loading.textContent = '无法加载波形';
            loading.style.color = 'var(--text-muted)';
        }
        // 尝试使用备用方法获取时长
        try {
            await generateWaveformForVideo(idx, fileObj, canvas, loading, durationEl);
        } catch (fallbackError) {
            console.error('备用方法也失败:', fallbackError);
        }
    }
}

// 为视频文件生成简单的占位波形并获取时长
async function generateWaveformForVideo(idx, fileObj, canvas, loading, durationEl) {
    return new Promise((resolve, reject) => {
        const mediaElement = document.createElement('video');
        const blobUrl = URL.createObjectURL(fileObj);

        mediaElement.preload = 'metadata';
        mediaElement.muted = true;

        const timeout = setTimeout(() => {
            URL.revokeObjectURL(blobUrl);
            if (loading) {
                loading.textContent = '视频文件 (无波形)';
                loading.style.fontSize = '10px';
            }
            resolve();
        }, 10000); // 10秒超时

        mediaElement.onloadedmetadata = () => {
            clearTimeout(timeout);
            const duration = mediaElement.duration;

            // 更新时长显示
            if (durationEl && isFinite(duration)) {
                durationEl.textContent = formatTimeAudio(duration);
            }

            // 保存数据（生成简单的占位波形）
            if (!window.audioCardData) window.audioCardData = {};
            const fakePeaks = Array(150).fill(0).map(() => 0.3 + Math.random() * 0.4);
            window.audioCardData[idx] = {
                peaks: fakePeaks,
                duration: duration
            };

            // 绘制简单的占位波形
            drawWaveform(canvas, fakePeaks);

            if (loading) {
                loading.textContent = '🎬 视频';
                loading.style.fontSize = '10px';
                loading.style.background = 'rgba(102, 126, 234, 0.2)';
                loading.style.padding = '2px 6px';
                loading.style.borderRadius = '3px';
                loading.style.position = 'absolute';
                loading.style.top = '4px';
                loading.style.right = '4px';
                loading.style.left = 'auto';
                loading.style.bottom = 'auto';
                loading.style.display = 'block';
            }

            URL.revokeObjectURL(blobUrl);
            resolve();
        };

        mediaElement.onerror = (e) => {
            clearTimeout(timeout);
            console.error('视频元数据加载失败:', e);
            if (loading) {
                loading.textContent = '无法加载';
            }
            URL.revokeObjectURL(blobUrl);
            reject(new Error('视频加载失败'));
        };

        mediaElement.src = blobUrl;
    });
}

// 播放卡片中的音频
function playAudioInCard(idx, file) {
    const audio = document.getElementById(`audio-element-${idx}`);
    const playBtn = document.querySelector(`.audio-file-card[data-idx="${idx}"] button`);
    if (!audio || !file.file) return;

    if (audio.paused) {
        // 停止其他正在播放的
        document.querySelectorAll('.audio-file-card audio').forEach(a => {
            if (a.id !== `audio-element-${idx}`) {
                a.pause();
            }
        });
        document.querySelectorAll('.audio-file-card button').forEach(b => {
            if (b.textContent === '⏸️') b.textContent = '▶️';
        });

        if (!audio.src) {
            audio.src = URL.createObjectURL(file.file);
        }
        audio.play();
        playBtn.textContent = '⏸️';

        // 更新进度和光标
        audio.ontimeupdate = () => {
            const progress = document.getElementById(`audio-progress-${idx}`);
            const cursor = document.getElementById(`audio-cursor-${idx}`);
            if (audio.duration) {
                const ratio = (audio.currentTime / audio.duration * 100);
                if (progress) progress.style.width = ratio + '%';
                if (cursor) {
                    cursor.style.left = ratio + '%';
                    cursor.style.display = 'block';
                }
            }
        };
    } else {
        audio.pause();
        playBtn.textContent = '▶️';
    }
}

// 在卡片当前播放位置添加分割点
function addCutPointToCard(idx, filePath) {
    const audio = document.getElementById(`audio-element-${idx}`);
    const input = document.getElementById(`audio-cut-points-${idx}`);
    if (!audio || !input) return;

    const currentTime = audio.currentTime;
    if (currentTime <= 0) {
        showToast('请先播放音频到目标位置', 'warning');
        return;
    }

    const timeStr = formatTimeAudio(currentTime);
    const existing = input.value.trim();
    input.value = existing ? existing + ', ' + timeStr : timeStr;
    currentAudioCutPoints[filePath] = input.value;

    showToast(`已添加分割点: ${timeStr}`, 'success');
}

// ==================== 音频预览功能 ====================

let currentPreviewBlobUrl = null;

function loadAudioForPreview(filePath, fileName, fileObj) {
    const audio = document.getElementById('audio-preview-element');
    const nameEl = document.getElementById('audio-preview-name');
    const seekSlider = document.getElementById('audio-preview-seek');
    const durationEl = document.getElementById('audio-preview-duration');
    const playBtn = document.getElementById('audio-preview-play');

    if (!audio) return;

    currentPreviewFilePath = filePath;
    nameEl.textContent = fileName || '加载中...';

    // 更新智能分割按钮状态
    if (typeof updateSmartSplitButtonState === 'function') {
        updateSmartSplitButtonState();
    }

    // 释放之前的 blob URL
    if (currentPreviewBlobUrl) {
        URL.revokeObjectURL(currentPreviewBlobUrl);
        currentPreviewBlobUrl = null;
    }

    // 使用 File 对象创建 blob URL（解决浏览器安全限制）
    if (fileObj) {
        currentPreviewBlobUrl = URL.createObjectURL(fileObj);
        audio.src = currentPreviewBlobUrl;
    } else {
        // 回退到后端代理
        audio.src = `${API_BASE}/file/proxy?path=${encodeURIComponent(filePath)}`;
    }

    audio.load();

    audio.onloadedmetadata = () => {
        seekSlider.max = audio.duration;  // 使用精确值
        seekSlider.step = 0.1;  // 更精细的步进
        durationEl.textContent = `00:00 / ${formatTimeAudio(audio.duration)}`;
    };

    audio.ontimeupdate = () => {
        seekSlider.value = audio.currentTime;  // 使用精确浮点值
        durationEl.textContent = `${formatTimeAudio(audio.currentTime)} / ${formatTimeAudio(audio.duration)}`;
        updateWaveformProgress(audio.currentTime, audio.duration);
    };

    audio.onended = () => {
        playBtn.textContent = '▶️';
    };

    // 滑杆拖动
    seekSlider.oninput = () => {
        audio.currentTime = seekSlider.value;
    };

    playBtn.textContent = '▶️';

    // 生成波形
    if (fileObj) {
        generateWaveform(fileObj);
    }

    // 波形点击跳转
    const waveformContainer = document.getElementById('audio-waveform-container');
    waveformContainer.onclick = (e) => {
        const rect = waveformContainer.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const ratio = x / rect.width;
        if (audio.duration) {
            audio.currentTime = ratio * audio.duration;
        }
    };
}

function toggleAudioPreview() {
    const audio = document.getElementById('audio-preview-element');
    const playBtn = document.getElementById('audio-preview-play');

    if (!audio || !audio.src) return;

    if (audio.paused) {
        audio.play();
        playBtn.textContent = '⏸️';
    } else {
        audio.pause();
        playBtn.textContent = '▶️';
    }
}

function addCutPointAtCurrentTime() {
    const audio = document.getElementById('audio-preview-element');
    if (!audio || !currentPreviewFilePath) {
        showToast('请先选择要播放的音频', 'warning');
        return;
    }

    const currentTime = audio.currentTime;
    const timeStr = formatTimeAudio(currentTime);

    // 找到对应文件的输入框
    const fileIdx = currentMediaFileInfos.findIndex(f => f.path === currentPreviewFilePath);
    if (fileIdx === -1) return;

    const input = document.getElementById(`audio-cut-points-${fileIdx}`);
    if (!input) return;

    // 添加裁切点
    const existing = input.value.trim();
    if (existing) {
        input.value = existing + ', ' + timeStr;
    } else {
        input.value = timeStr;
    }

    // 更新缓存
    currentAudioCutPoints[currentPreviewFilePath] = input.value;

    showToast(`已添加裁切点: ${timeStr}`, 'success');
}

function formatTimeAudio(seconds) {
    if (isNaN(seconds)) return '00:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 10);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${ms}`;
}

// 波形进度更新
function updateWaveformProgress(currentTime, duration) {
    const progress = document.getElementById('audio-waveform-progress');
    const cursor = document.getElementById('audio-waveform-cursor');
    if (!progress || !cursor || !duration) return;

    const ratio = currentTime / duration;
    progress.style.width = (ratio * 100) + '%';
    cursor.style.left = (ratio * 100) + '%';
}

// 存储当前波形数据
let currentWaveformData = {
    peaks: [],
    duration: 0,
    canvas: null
};

// 生成音频波形
async function generateWaveform(fileObj) {
    const canvas = document.getElementById('audio-waveform-canvas');
    const loading = document.getElementById('audio-waveform-loading');
    if (!canvas) return;

    loading.style.display = 'flex';

    try {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const arrayBuffer = await fileObj.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

        // 获取音频数据
        const channelData = audioBuffer.getChannelData(0);
        const samples = 200; // 采样点数量
        const blockSize = Math.floor(channelData.length / samples);
        const peaks = [];

        for (let i = 0; i < samples; i++) {
            let sum = 0;
            for (let j = 0; j < blockSize; j++) {
                sum += Math.abs(channelData[i * blockSize + j]);
            }
            peaks.push(sum / blockSize);
        }

        // 归一化
        const maxPeak = Math.max(...peaks);
        const normalizedPeaks = peaks.map(p => p / maxPeak);

        // 保存波形数据
        currentWaveformData = {
            peaks: normalizedPeaks,
            duration: audioBuffer.duration,
            canvas: canvas
        };

        // 绘制波形（初始无分割点）
        drawWaveform(canvas, normalizedPeaks);
        loading.style.display = 'none';

        audioContext.close();
    } catch (error) {
        console.error('波形生成失败:', error);
        loading.textContent = '波形加载失败';
    }
}

function drawWaveform(canvas, peaks, cutPoints = [], totalDuration = 0) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const barWidth = w / peaks.length;
    const midY = h / 2;

    ctx.clearRect(0, 0, w, h);

    // 绘制波形条
    peaks.forEach((peak, i) => {
        const barHeight = peak * (h * 0.8);
        const x = i * barWidth;

        // 渐变颜色：有声音的部分较亮，静音部分较暗
        const intensity = peak;
        const r = Math.floor(102 + intensity * 50);
        const g = Math.floor(126 + intensity * 30);
        const b = Math.floor(234);

        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${0.4 + intensity * 0.5})`;
        ctx.fillRect(x, midY - barHeight / 2, barWidth - 1, barHeight);
    });

    // 绘制分割点标记线
    if (cutPoints.length > 0 && totalDuration > 0) {
        ctx.strokeStyle = '#ff6b6b';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 2]);

        cutPoints.forEach((cutTime, idx) => {
            if (cutTime <= 0 || cutTime >= totalDuration) return;
            const x = (cutTime / totalDuration) * w;

            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, h);
            ctx.stroke();

            // 分割点标签
            ctx.fillStyle = '#ff6b6b';
            ctx.font = '10px sans-serif';
            ctx.fillText(`#${idx + 1}`, x + 2, 10);
        });

        ctx.setLineDash([]);
    }
}

function initMediaModeOptions() {
    // 拖拽文件支持
    const dropZone = document.getElementById('media-drop-zone');
    const fileInput = document.getElementById('media-input-file');

    if (dropZone && fileInput) {
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.style.borderColor = 'var(--accent)';
                dropZone.style.background = 'rgba(255,255,255,0.05)';
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.style.borderColor = 'var(--border-color)';
                dropZone.style.background = '';
            });
        });

        dropZone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                // 触发文件输入的 change 事件
                const dataTransfer = new DataTransfer();
                for (const file of files) {
                    dataTransfer.items.add(file);
                }
                fileInput.files = dataTransfer.files;
                fileInput.dispatchEvent(new Event('change', { bubbles: true }));
            }
        });
    }

    // Logo 预设切换：显示/隐藏自定义 Logo 设置
    const logoPresets = document.querySelectorAll('input[name="logo-preset"]');
    const customLogoOptions = document.getElementById('custom-logo-options');

    logoPresets.forEach(radio => {
        radio.addEventListener('change', () => {
            if (radio.value === 'custom' && radio.checked) {
                customLogoOptions?.classList.remove('hidden');
            } else {
                customLogoOptions?.classList.add('hidden');
            }
            // 自动加载该预设的默认位置
            resetLogoPosition();
        });
    });

    // 自定义 Logo 文件选择
    const customLogoFile = document.getElementById('custom-logo-file');
    if (customLogoFile) {
        customLogoFile.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                const file = e.target.files[0];
                document.getElementById('custom-logo-path').value = file.name;
                // 存储文件路径
                customLogoFile.dataset.filePath = file.path || file.name;
                showToast(`已选择 Logo: ${file.name}`, 'success');
                updateLogoPreview();
            }
        });
    }

    // 水印位置切换
    const watermarkPosition = document.getElementById('watermark-position');
    const watermarkCustomPos = document.getElementById('watermark-custom-pos');

    if (watermarkPosition) {
        watermarkPosition.addEventListener('change', () => {
            if (watermarkPosition.value === 'custom') {
                watermarkCustomPos.style.display = 'flex';
            } else {
                watermarkCustomPos.style.display = 'none';
            }
        });
    }

    // 水印预设文本选择
    const watermarkPreset = document.getElementById('watermark-preset');
    const watermarkText = document.getElementById('watermark-text');

    if (watermarkPreset && watermarkText) {
        watermarkPreset.addEventListener('change', () => {
            if (watermarkPreset.value) {
                watermarkText.value = watermarkPreset.value;
                updateWatermarkPreview();
            }
        });
    }

    // 水印颜色同步（颜色选择器 <-> 文本输入）
    const watermarkColor = document.getElementById('watermark-color');
    const watermarkColorText = document.getElementById('watermark-color-text');

    if (watermarkColor && watermarkColorText) {
        watermarkColor.addEventListener('input', () => {
            watermarkColorText.value = watermarkColor.value;
        });
        watermarkColorText.addEventListener('input', () => {
            if (/^#[0-9A-Fa-f]{6}$/.test(watermarkColorText.value)) {
                watermarkColor.value = watermarkColorText.value;
            }
        });
    }

    // 水印透明度标签
    const watermarkOpacity = document.getElementById('watermark-opacity');
    const opacityLabel = document.getElementById('watermark-opacity-label');

    if (watermarkOpacity && opacityLabel) {
        watermarkOpacity.addEventListener('input', () => {
            opacityLabel.textContent = Math.round(watermarkOpacity.value * 100) + '%';
            updateWatermarkPreview();
        });
    }

    // 防抖函数，避免预览闪烁
    const debounce = (fn, delay = 100) => {
        let timer;
        return (...args) => {
            clearTimeout(timer);
            timer = setTimeout(() => fn(...args), delay);
        };
    };

    const debouncedWatermarkPreview = debounce(updateWatermarkPreview, 150);
    const debouncedLogoPreview = debounce(updateLogoPreview, 150);

    // 为所有水印参数添加变化监听器，自动刷新预览（带防抖）
    const watermarkInputs = [
        'watermark-text', 'watermark-font', 'watermark-fontsize', 'watermark-color',
        'watermark-stroke', 'watermark-stroke-color', 'watermark-stroke-width',
        'watermark-shadow', 'watermark-position', 'watermark-offset-x', 'watermark-offset-y',
        'watermark-opacity'
    ];
    watermarkInputs.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('input', debouncedWatermarkPreview);
            el.addEventListener('change', debouncedWatermarkPreview);
        }
    });

    // 为 Logo 参数添加变化监听器（实时刷新预览）
    const logoInputs = ['logo-pos-x', 'logo-pos-y', 'logo-width', 'logo-height'];
    logoInputs.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('input', debouncedLogoPreview);
            el.addEventListener('change', debouncedLogoPreview);
        }
    });

    const formatModes = document.querySelectorAll('input[name="format-mode"]');
    const audioSplitOptions = document.getElementById('audio-split-options');
    const updateAudioSplitOptions = () => {
        const selected = document.querySelector('input[name="format-mode"]:checked')?.value;
        if (selected === 'audio_split') {
            audioSplitOptions?.classList.remove('hidden');
            renderAudioSplitFileList();
        } else {
            audioSplitOptions?.classList.add('hidden');
        }
    };

    formatModes.forEach(input => {
        input.addEventListener('change', updateAudioSplitOptions);
    });

    updateAudioSplitOptions();

    // 初始化预览
    setTimeout(() => {
        updateLogoPreview();
        updateWatermarkPreview();
    }, 500);
}

// ==================== 预览功能 ====================

function updateLogoPreview() {
    // 获取选中的预设 Logo
    const preset = document.querySelector('input[name="logo-preset"]:checked')?.value || 'hailuo';

    // 获取输入框的实际值
    const posX = parseInt(document.getElementById('logo-pos-x')?.value) || 590;
    const posY = parseInt(document.getElementById('logo-pos-y')?.value) || 1810;
    const logoW = parseInt(document.getElementById('logo-width')?.value) || 400;
    const logoH = parseInt(document.getElementById('logo-height')?.value) || 90;

    // 获取预设标签
    const presetLabels = {
        'hailuo': 'Dream+Hailuo',
        'vidu': 'Dream+Vidu',
        'veo': 'Dream+Veo',
        'heygen': 'Dream+HeyGen',
        'dream': 'Dreamina',
        'ai_generated': 'AI Generated',
        'custom': 'Custom Logo'
    };
    const label = presetLabels[preset] || 'Logo';
    const logoSource = getLogoPreviewSource(preset);

    // 渲染到深色背景
    renderLogoToCanvas('logo-preview-canvas', {
        posX,
        posY,
        logoW,
        logoH,
        label,
        bgType: 'dark',
        sources: logoSource.sources
    });

    // 渲染到浅色开背景
    renderLogoToCanvas('logo-preview-canvas-light', {
        posX,
        posY,
        logoW,
        logoH,
        label,
        bgType: 'light',
        sources: logoSource.sources
    });
}

function getLogoPreviewSource(preset) {
    if (preset === 'custom') {
        const customPath = document.getElementById('custom-logo-file')?.dataset?.filePath;
        if (customPath) {
            return { sources: [normalizeFilePath(customPath)] };
        }
    }

    const assetFile = LOGO_PRESET_ASSETS[preset];
    if (assetFile) {
        const localPath = resolveAssetPath(`../assets/${assetFile}`);
        const apiPath = `${API_ORIGIN}/assets/${encodeURIComponent(assetFile)}`;
        return { sources: [localPath, apiPath] };
    }

    return { sources: [] };
}

function resolveAssetPath(relativePath) {
    try {
        return new URL(relativePath, window.location.href).toString();
    } catch (e) {
        return relativePath;
    }
}

function normalizeFilePath(pathValue) {
    if (!pathValue) return '';
    if (/^file:\/\//i.test(pathValue)) {
        return pathValue;
    }
    if (/^[a-zA-Z]:\\/.test(pathValue)) {
        return `file:///${pathValue.replace(/\\/g, '/')}`;
    }
    if (pathValue.startsWith('/')) {
        return `file://${pathValue}`;
    }
    return pathValue;
}

function renderLogoToCanvas(canvasId, params) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const ctx = canvas.getContext('2d');

    // Retina 支持
    const dpr = window.devicePixelRatio || 1;
    const displayWidth = 135;
    const displayHeight = 240;

    // 设置实际像素尺寸
    canvas.width = displayWidth * dpr;
    canvas.height = displayHeight * dpr;
    canvas.style.width = displayWidth + 'px';
    canvas.style.height = displayHeight + 'px';
    ctx.scale(dpr, dpr);

    const w = displayWidth;
    const h = displayHeight;

    // 清空并绘制背景
    if (params.bgType === 'dark') {
        const gradient = ctx.createLinearGradient(0, 0, w, h);
        gradient.addColorStop(0, '#1a1a2e');
        gradient.addColorStop(0.5, '#16213e');
        gradient.addColorStop(1, '#0f3460');
        ctx.fillStyle = gradient;
    } else {
        const gradient = ctx.createLinearGradient(0, 0, w, h);
        gradient.addColorStop(0, '#f5f7fa');
        gradient.addColorStop(0.5, '#e4e9f2');
        gradient.addColorStop(1, '#c3cfe2');
        ctx.fillStyle = gradient;
    }
    ctx.fillRect(0, 0, w, h);

    // 缩放比例 (1080x1920 -> 135x240)
    const scale = 135 / 1080;

    // 绘制 Logo 占位区域
    const lx = params.posX * scale;
    const ly = params.posY * scale;
    const lw = params.logoW * scale;
    const lh = params.logoH * scale;

    const imgEntry = params.sources && params.sources.length ? getLogoImage(params.sources) : null;
    const canDrawImage = imgEntry && imgEntry.status === 'loaded' && imgEntry.img;

    // Logo 背景
    ctx.fillStyle = params.bgType === 'dark' ? 'rgba(255, 255, 255, 0.12)' : 'rgba(0, 0, 0, 0.08)';
    ctx.fillRect(lx, ly, lw, lh);

    if (canDrawImage) {
        ctx.drawImage(imgEntry.img, lx, ly, lw, lh);
    }

    // Logo 边框
    ctx.strokeStyle = params.bgType === 'dark' ? 'rgba(255, 255, 255, 0.75)' : 'rgba(0, 0, 0, 0.5)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.strokeRect(lx, ly, lw, lh);
    ctx.setLineDash([]);

    if (!canDrawImage) {
        // Logo 文字占位
        ctx.fillStyle = params.bgType === 'dark' ? 'rgba(255, 255, 255, 0.9)' : 'rgba(0, 0, 0, 0.7)';
        const fontSize = Math.max(8, Math.min(18, lh * 0.6));
        ctx.font = `${fontSize}px -apple-system, BlinkMacSystemFont, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(params.label, lx + lw / 2, ly + lh / 2);
    }

    // 尺寸提示
    ctx.fillStyle = params.bgType === 'dark' ? 'rgba(255, 255, 255, 0.75)' : 'rgba(0, 0, 0, 0.6)';
    ctx.font = '9px -apple-system, BlinkMacSystemFont, sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(`W:${params.logoW} H:${params.logoH}`, 6, 6);

    if (!canDrawImage) {
        ctx.fillStyle = params.bgType === 'dark' ? 'rgba(255, 255, 255, 0.6)' : 'rgba(0, 0, 0, 0.45)';
        ctx.font = '9px -apple-system, BlinkMacSystemFont, sans-serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText('预览占位', 6, 18);
    }
}

function getLogoImage(sources) {
    const key = sources.join('|');
    let entry = logoImageCache.get(key);
    if (!entry) {
        const img = new Image();
        entry = { img, status: 'loading', sources, index: 0 };
        img.onload = () => {
            entry.status = 'loaded';
            updateLogoPreview();
        };
        img.onerror = () => {
            if (entry.index + 1 < entry.sources.length) {
                entry.index += 1;
                img.src = entry.sources[entry.index];
                return;
            }
            entry.status = 'error';
            updateLogoPreview();
        };
        img.src = sources[0];
        logoImageCache.set(key, entry);
    }
    return entry;
}

function updateWatermarkPreview() {
    // 获取水印参数
    const text = document.getElementById('watermark-text')?.value || 'AI Created';
    const fontSize = parseInt(document.getElementById('watermark-fontsize')?.value) || 24;
    const color = document.getElementById('watermark-color')?.value || '#ffffff';
    const opacity = parseFloat(document.getElementById('watermark-opacity')?.value) || 1;
    const hasStroke = document.getElementById('watermark-stroke')?.checked || false;
    const strokeColor = document.getElementById('watermark-stroke-color')?.value || '#000000';
    const strokeWidth = parseInt(document.getElementById('watermark-stroke-width')?.value) || 2;
    const hasShadow = document.getElementById('watermark-shadow')?.checked || false;
    const position = document.getElementById('watermark-position')?.value || 'top-right';
    const fontFamily = document.getElementById('watermark-font')?.value || 'Arial';
    const offsetX = parseInt(document.getElementById('watermark-offset-x')?.value) || 10;
    const offsetY = parseInt(document.getElementById('watermark-offset-y')?.value) || 10;

    // 渲染到深色背景
    renderWatermarkToCanvas('watermark-preview-canvas', {
        text, fontSize, color, opacity, hasStroke, strokeColor, strokeWidth,
        hasShadow, position, fontFamily, offsetX, offsetY,
        bgType: 'dark'
    });

    // 渲染到浅色背景
    renderWatermarkToCanvas('watermark-preview-canvas-light', {
        text, fontSize, color, opacity, hasStroke, strokeColor, strokeWidth,
        hasShadow, position, fontFamily, offsetX, offsetY,
        bgType: 'light'
    });
}

function renderWatermarkToCanvas(canvasId, params) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const ctx = canvas.getContext('2d');

    // Retina 支持
    const dpr = window.devicePixelRatio || 1;
    const displayWidth = 135;
    const displayHeight = 240;

    // 设置实际像素尺寸
    canvas.width = displayWidth * dpr;
    canvas.height = displayHeight * dpr;
    canvas.style.width = displayWidth + 'px';
    canvas.style.height = displayHeight + 'px';
    ctx.scale(dpr, dpr);

    const w = displayWidth;
    const h = displayHeight;

    // 清空并绘制背景
    if (params.bgType === 'dark') {
        const gradient = ctx.createLinearGradient(0, 0, w, h);
        gradient.addColorStop(0, '#1a1a2e');
        gradient.addColorStop(0.5, '#16213e');
        gradient.addColorStop(1, '#0f3460');
        ctx.fillStyle = gradient;
    } else {
        const gradient = ctx.createLinearGradient(0, 0, w, h);
        gradient.addColorStop(0, '#f5f7fa');
        gradient.addColorStop(0.5, '#e4e9f2');
        gradient.addColorStop(1, '#c3cfe2');
        ctx.fillStyle = gradient;
    }
    ctx.fillRect(0, 0, w, h);

    // 缩放比例 (1080x1920 -> 135x240)
    const scale = 135 / 1080;
    // 字体使用更大的缩放比例使预览更清晰 (约2倍)
    const fontScale = scale * 2;
    const scaledFontSize = Math.max(params.fontSize * fontScale, 4);
    const scaledOffsetX = params.offsetX * scale;
    const scaledOffsetY = params.offsetY * scale;

    ctx.font = `${scaledFontSize}px "${params.fontFamily}", -apple-system, BlinkMacSystemFont, sans-serif`;
    ctx.globalAlpha = params.opacity;

    // 测量文字宽度
    const textWidth = ctx.measureText(params.text).width;
    const textHeight = scaledFontSize;

    // 计算位置
    let x, y;
    switch (params.position) {
        case 'top-left': x = scaledOffsetX; y = scaledOffsetY + textHeight; break;
        case 'top-right': x = w - textWidth - scaledOffsetX; y = scaledOffsetY + textHeight; break;
        case 'bottom-left': x = scaledOffsetX; y = h - scaledOffsetY; break;
        case 'bottom-right': x = w - textWidth - scaledOffsetX; y = h - scaledOffsetY; break;
        case 'center': x = (w - textWidth) / 2; y = (h + textHeight) / 2; break;
        default: x = w - textWidth - scaledOffsetX; y = scaledOffsetY + textHeight;
    }

    // 阴影
    if (params.hasShadow) {
        ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
        ctx.shadowOffsetX = 2;
        ctx.shadowOffsetY = 2;
        ctx.shadowBlur = 4;
    }

    // 描边
    if (params.hasStroke) {
        ctx.strokeStyle = params.strokeColor;
        ctx.lineWidth = params.strokeWidth * scale;
        ctx.strokeText(params.text, x, y);
    }

    // 文字
    ctx.fillStyle = params.color;
    ctx.fillText(params.text, x, y);

    // 重置
    ctx.globalAlpha = 1;
    ctx.shadowColor = 'transparent';
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;
    ctx.shadowBlur = 0;
}

// ==================== 水印设置保存/加载 ====================

function saveWatermarkSettings() {
    const settings = {
        text: document.getElementById('watermark-text')?.value || 'AI Generated',
        font: document.getElementById('watermark-font')?.value || 'Arial',
        fontSize: document.getElementById('watermark-fontsize')?.value || '24',
        color: document.getElementById('watermark-color')?.value || '#ffffff',
        opacity: document.getElementById('watermark-opacity')?.value || '1',
        stroke: document.getElementById('watermark-stroke')?.checked || false,
        strokeColor: document.getElementById('watermark-stroke-color')?.value || '#000000',
        strokeWidth: document.getElementById('watermark-stroke-width')?.value || '2',
        shadow: document.getElementById('watermark-shadow')?.checked || false,
        position: document.getElementById('watermark-position')?.value || 'top-right'
    };

    localStorage.setItem('watermarkSettings', JSON.stringify(settings));
    showToast('水印设置已保存', 'success');
}

function loadWatermarkSettings() {
    const saved = localStorage.getItem('watermarkSettings');
    if (!saved) return;

    try {
        const settings = JSON.parse(saved);

        if (settings.text) document.getElementById('watermark-text').value = settings.text;
        if (settings.font) document.getElementById('watermark-font').value = settings.font;
        if (settings.fontSize) document.getElementById('watermark-fontsize').value = settings.fontSize;
        if (settings.color) {
            document.getElementById('watermark-color').value = settings.color;
            document.getElementById('watermark-color-text').value = settings.color;
        }
        if (settings.opacity) {
            document.getElementById('watermark-opacity').value = settings.opacity;
            document.getElementById('watermark-opacity-label').textContent = Math.round(settings.opacity * 100) + '%';
        }
        if (settings.stroke !== undefined) document.getElementById('watermark-stroke').checked = settings.stroke;
        if (settings.strokeColor) document.getElementById('watermark-stroke-color').value = settings.strokeColor;
        if (settings.strokeWidth) document.getElementById('watermark-stroke-width').value = settings.strokeWidth;
        if (settings.shadow !== undefined) document.getElementById('watermark-shadow').checked = settings.shadow;
        if (settings.position) document.getElementById('watermark-position').value = settings.position;

        // 更新预览
        setTimeout(updateWatermarkPreview, 100);
    } catch (e) {
        console.error('加载水印设置失败:', e);
    }
}

// ==================== 位置调整辅助函数 ====================

// Logo 位置调整（方向按钮）
function adjustLogoPos(dx, dy) {
    const posX = document.getElementById('logo-pos-x');
    const posY = document.getElementById('logo-pos-y');
    if (posX) posX.value = parseInt(posX.value) + dx;
    if (posY) posY.value = parseInt(posY.value) + dy;
    updateLogoPreview();
}

// 重置 Logo 位置为当前预设默认值
function resetLogoPosition() {
    const preset = document.querySelector('input[name="logo-preset"]:checked')?.value || 'hailuo';
    const cfg = LOGO_DEFAULTS[preset] || LOGO_DEFAULTS.hailuo;

    document.getElementById('logo-pos-x').value = cfg.x;
    document.getElementById('logo-pos-y').value = cfg.y;
    document.getElementById('logo-width').value = cfg.w;
    document.getElementById('logo-height').value = cfg.h;

    // 同步滑块
    const widthRange = document.getElementById('logo-width-range');
    const heightRange = document.getElementById('logo-height-range');
    if (widthRange) widthRange.value = cfg.w;
    if (heightRange) heightRange.value = cfg.h;

    updateLogoPreview();
    showToast(`已重置为 ${preset} 预设位置`, 'success');
}

function getLogoOverrideFromInputs() {
    const preset = document.querySelector('input[name="logo-preset"]:checked')?.value || 'hailuo';
    const defaults = LOGO_DEFAULTS[preset] || LOGO_DEFAULTS.hailuo;

    const xVal = parseInt(document.getElementById('logo-pos-x')?.value);
    const yVal = parseInt(document.getElementById('logo-pos-y')?.value);
    const wVal = parseInt(document.getElementById('logo-width')?.value);
    const hVal = parseInt(document.getElementById('logo-height')?.value);

    return {
        x: Number.isFinite(xVal) ? xVal : defaults.x,
        y: Number.isFinite(yVal) ? yVal : defaults.y,
        width: Number.isFinite(wVal) ? wVal : defaults.w,
        height: Number.isFinite(hVal) ? hVal : defaults.h
    };
}

// 水印偏移调整（方向按钮）
function adjustWatermarkOffset(dx, dy) {
    const offsetX = document.getElementById('watermark-offset-x');
    const offsetY = document.getElementById('watermark-offset-y');
    if (offsetX) offsetX.value = Math.max(0, parseInt(offsetX.value) + dx);
    if (offsetY) offsetY.value = Math.max(0, parseInt(offsetY.value) + dy);
    updateWatermarkPreview();
}

// 检查后端健康状态
let healthCheckRetries = 0;
const MAX_HEALTH_RETRIES = 20; // 最多重试20次（约60秒）

async function checkBackendHealth() {
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);

        const response = await fetch(`${API_BASE}/health`, {
            signal: controller.signal
        });
        clearTimeout(timeoutId);

        if (response.ok) {
            updateStatus('后端服务已连接', 'success');
            backendReady = true;
            healthCheckRetries = 0;
            if (!settingsAutoLoaded) {
                settingsAutoLoaded = true;
                loadSettings(true);
            }
        }
    } catch (error) {
        healthCheckRetries++;

        if (healthCheckRetries >= MAX_HEALTH_RETRIES) {
            updateStatus(`后端服务无法连接 (已重试${healthCheckRetries}次)，请检查 Python 环境`, 'error');
            console.error('后端启动失败，可能原因：1.Python未安装 2.依赖包缺失 3.端口5001被占用');
            // 显示更详细的提示
            showToast('后端服务无法启动，请检查 Python 是否已安装', 'error', 10000);
        } else {
            const remaining = MAX_HEALTH_RETRIES - healthCheckRetries;
            updateStatus(`等待后端服务启动... (${healthCheckRetries}/${MAX_HEALTH_RETRIES})`, 'error');
            setTimeout(checkBackendHealth, 3000);
        }
    }
}

// 更新状态
function updateStatus(text, type = 'normal', elementId = 'status-text') {
    const statusText = document.getElementById(elementId);
    if (statusText) {
        statusText.textContent = text;
        statusText.className = 'status-text';
        if (type === 'error') statusText.classList.add('error');
        if (type === 'processing') statusText.classList.add('processing');
    }
}

function setIndeterminateProgress(elementId, active) {
    const bar = document.getElementById(elementId);
    if (!bar) return;
    bar.classList.toggle('indeterminate', active);
}

// 清空文本
function clearText(targetId) {
    document.getElementById(targetId).value = '';
    showToast('已清空', 'info');
}

// 加载设置
async function loadSettings(autoLoadVoices = false) {
    try {
        const response = await fetch(`${API_BASE}/settings/gladia-keys`);
        const data = await response.json();
        if (data.keys) {
            document.getElementById('gladia-keys').value = data.keys.join('\n');
        }
    } catch (error) {
        // 忽略
    }

    // 加载 ElevenLabs API Keys
    try {
        const response = await fetch(`${API_BASE}/settings/elevenlabs`);
        const data = await response.json();
        const keyTextarea = document.getElementById('elevenlabs-api-keys');
        if (keyTextarea) {
            const keys = Array.isArray(data.api_keys) ? data.api_keys : (data.api_key ? [data.api_key] : []);
            keyTextarea.value = keys.join('\n');
            if (keys.length > 0 && autoLoadVoices && backendReady) {
                loadVoices();
            }
        }
    } catch (error) {
        // 忽略
    }

    // 加载替换规则
    try {
        const response = await fetch(`${API_BASE}/settings/replace-rules`);
        const data = await response.json();
        const rulesTextarea = document.getElementById('replace-rules');
        const langSelect = document.getElementById('replace-language');

        if (!rulesTextarea || !langSelect) return;

        if (typeof data.rules === 'string') {
            if (data.language) {
                langSelect.value = data.language;
            }
            rulesTextarea.value = data.rules || '';
            replaceRulesCache = null;
        } else if (data.rules && typeof data.rules === 'object') {
            replaceRulesCache = data.rules;
            const preferredLang = data.language || langSelect.value;
            if (preferredLang && replaceRulesCache[preferredLang] !== undefined) {
                langSelect.value = preferredLang;
                rulesTextarea.value = replaceRulesCache[preferredLang] || '';
            } else {
                rulesTextarea.value = '';
            }
        } else {
            rulesTextarea.value = '';
            replaceRulesCache = null;
        }
    } catch (error) {
        // 忽略
    }
}

// ==================== 批量字幕对齐功能 ====================

let subtitleBatchTasks = []; // 存储批量任务 {file, fileName, sourceText, translateText}

// 切换批量模式
function toggleSubtitleBatchMode() {
    const batchMode = document.getElementById('subtitle-batch-mode')?.checked;
    const batchSection = document.getElementById('subtitle-batch-section');
    const singleSections = document.querySelectorAll('#subtitle-panel .form-section:not(#subtitle-batch-section):not(:has(#subtitle-batch-mode))');

    // 隐藏/显示 STEP 1-3（单文件模式的输入）
    const step1 = document.querySelector('#audio-path')?.closest('.form-section');
    const step2 = document.querySelector('#source-text')?.closest('.form-section');
    const step3 = document.querySelector('#translate-text')?.closest('.form-section');

    if (batchMode) {
        batchSection?.classList.remove('hidden');
        step1?.classList.add('hidden');
        step2?.classList.add('hidden');
        step3?.classList.add('hidden');
    } else {
        batchSection?.classList.add('hidden');
        step1?.classList.remove('hidden');
        step2?.classList.remove('hidden');
        step3?.classList.remove('hidden');
    }
}

// 初始化批量音频输入
function initSubtitleBatch() {
    const batchInput = document.getElementById('batch-audio-input');
    if (batchInput) {
        batchInput.addEventListener('change', (e) => {
            const files = Array.from(e.target.files || []);
            addAudioFilesToBatch(files);
            e.target.value = ''; // 清空以便再次选择
        });
    }

    // 添加拖拽支持
    const list = document.getElementById('subtitle-batch-list');
    const section = document.getElementById('subtitle-batch-section');

    if (section) {
        section.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
            section.style.background = 'rgba(0, 217, 165, 0.1)';
            section.style.border = '2px dashed #00d9a5';
        });

        section.addEventListener('dragleave', (e) => {
            e.preventDefault();
            e.stopPropagation();
            section.style.background = '';
            section.style.border = '';
        });

        section.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            section.style.background = '';
            section.style.border = '';

            const files = Array.from(e.dataTransfer.files || []);
            const audioFiles = files.filter(f =>
                /\.(mp4|mov|mkv|wav|mp3|m4a|flv|avi|wmv|json)$/i.test(f.name)
            );

            if (audioFiles.length > 0) {
                addAudioFilesToBatch(audioFiles);
            } else {
                showToast('请拖入音频/视频文件', 'error');
            }
        });
    }
}

// 添加音频文件到批量列表
function addAudioFilesToBatch(files) {
    files.forEach(file => {
        const task = {
            file: file,
            fileName: file.name,
            sourceText: '',
            translateText: '',
            status: 'pending',
            duration: null
        };
        subtitleBatchTasks.push(task);

        // 异步获取时长
        getAudioDuration(file).then(duration => {
            task.duration = duration;
            renderSubtitleBatchList();
        });
    });
    renderSubtitleBatchList();
    showToast(`已添加 ${files.length} 个文件`, 'success');
}

// 获取音频时长
function getAudioDuration(file) {
    return new Promise(resolve => {
        const url = URL.createObjectURL(file);
        const audio = new Audio();
        audio.onloadedmetadata = () => {
            URL.revokeObjectURL(url);
            resolve(audio.duration);
        };
        audio.onerror = () => {
            URL.revokeObjectURL(url);
            resolve(null);
        };
        audio.src = url;
    });
}

// 格式化时长
function formatDuration(seconds) {
    if (!seconds || isNaN(seconds)) return '--:--';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// 批量音频播放器
let batchAudioPlayer = null;
let currentPlayingIndex = -1;

function playBatchAudio(idx, btn) {
    const task = subtitleBatchTasks[idx];
    if (!task || !task.file) return;

    // 如果正在播放同一个，停止
    if (currentPlayingIndex === idx && batchAudioPlayer && !batchAudioPlayer.paused) {
        batchAudioPlayer.pause();
        batchAudioPlayer.currentTime = 0;
        btn.textContent = '▶️';
        currentPlayingIndex = -1;
        return;
    }

    // 停止之前的
    if (batchAudioPlayer) {
        batchAudioPlayer.pause();
        // 重置之前按钮
        const allBtns = document.querySelectorAll('.subtitle-play-btn');
        allBtns.forEach(b => b.textContent = '▶️');
    }

    // 创建新播放器
    const url = URL.createObjectURL(task.file);
    batchAudioPlayer = new Audio(url);
    currentPlayingIndex = idx;

    btn.textContent = '⏸️';

    batchAudioPlayer.play().catch(err => {
        showToast('播放失败: ' + err.message, 'error');
        btn.textContent = '▶️';
    });

    batchAudioPlayer.onended = () => {
        btn.textContent = '▶️';
        currentPlayingIndex = -1;
        URL.revokeObjectURL(url);
    };

    batchAudioPlayer.onerror = () => {
        btn.textContent = '▶️';
        currentPlayingIndex = -1;
        showToast('音频加载失败', 'error');
    };
}

// 渲染批量任务列表
function renderSubtitleBatchList() {
    const list = document.getElementById('subtitle-batch-list');
    const countSpan = document.getElementById('subtitle-batch-count');
    if (!list) return;

    countSpan.textContent = `${subtitleBatchTasks.length} 个任务`;

    if (subtitleBatchTasks.length === 0) {
        list.innerHTML = '<div style="text-align: center; padding: 20px; color: var(--text-secondary);">暂无任务，点击"批量添加音频"添加文件</div>';
        return;
    }

    list.innerHTML = '';

    subtitleBatchTasks.forEach((task, idx) => {
        const item = document.createElement('div');
        item.className = 'subtitle-batch-item';
        item.style.cssText = 'background: var(--bg-secondary); border-radius: 6px; margin-bottom: 8px; overflow: hidden;';

        // 头部（可折叠）
        const header = document.createElement('div');
        header.style.cssText = 'display: flex; align-items: center; gap: 8px; padding: 10px 12px; cursor: pointer; flex-wrap: wrap;';
        header.onclick = () => {
            const body = item.querySelector('.batch-item-body');
            body.classList.toggle('hidden');
            arrow.textContent = body.classList.contains('hidden') ? '▶' : '▼';
        };

        const arrow = document.createElement('span');
        arrow.textContent = '▶';
        arrow.style.cssText = 'font-size: 10px; color: var(--text-secondary);';

        const indexSpan = document.createElement('span');
        indexSpan.textContent = `${idx + 1}.`;
        indexSpan.style.cssText = 'font-weight: 500; color: var(--text-primary); min-width: 24px;';

        const fileName = document.createElement('span');
        fileName.textContent = task.fileName;
        fileName.style.cssText = 'color: var(--text-primary); font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 200px;';

        // 时长显示
        const durationSpan = document.createElement('span');
        durationSpan.style.cssText = 'font-size: 11px; color: var(--text-secondary); min-width: 40px;';
        durationSpan.textContent = formatDuration(task.duration);

        // 文案预览（显示前 20 字）
        const previewSpan = document.createElement('span');
        previewSpan.style.cssText = 'flex: 1; font-size: 11px; color: #888; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;';
        const srcPreview = task.sourceText.trim().substring(0, 15) || '--';
        const transPreview = task.translateText.trim().substring(0, 15) || '--';
        previewSpan.textContent = `原: ${srcPreview}${task.sourceText.length > 15 ? '...' : ''} | 译: ${transPreview}${task.translateText.length > 15 ? '...' : ''}`;

        const statusSpan = document.createElement('span');
        statusSpan.className = 'batch-item-status';
        statusSpan.style.cssText = 'font-size: 11px; padding: 2px 6px; border-radius: 4px;';
        const hasSource = task.sourceText.trim().length > 0;
        const hasTrans = task.translateText.trim().length > 0;
        if (hasSource && hasTrans) {
            statusSpan.textContent = '✅ 就绪';
            statusSpan.style.background = 'rgba(0,255,0,0.2)';
            statusSpan.style.color = '#51cf66';
        } else {
            statusSpan.textContent = '⚠️ 缺字幕';
            statusSpan.style.background = 'rgba(255,165,0,0.2)';
            statusSpan.style.color = '#ffa500';
        }

        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'btn btn-secondary';
        deleteBtn.style.cssText = 'padding: 2px 8px; font-size: 11px;';
        deleteBtn.textContent = '✕';
        deleteBtn.onclick = (e) => {
            e.stopPropagation();
            subtitleBatchTasks.splice(idx, 1);
            renderSubtitleBatchList();
        };

        // 播放按钮
        const playBtn = document.createElement('button');
        playBtn.className = 'btn btn-secondary subtitle-play-btn';
        playBtn.style.cssText = 'padding: 2px 8px; font-size: 11px;';
        playBtn.textContent = '▶️';
        playBtn.title = '试听音频';
        playBtn.onclick = (e) => {
            e.stopPropagation();
            playBatchAudio(idx, playBtn);
        };

        header.appendChild(arrow);
        header.appendChild(indexSpan);
        header.appendChild(fileName);
        header.appendChild(durationSpan);
        header.appendChild(playBtn);
        header.appendChild(previewSpan);
        header.appendChild(statusSpan);
        header.appendChild(deleteBtn);

        // 内容（可折叠）
        const body = document.createElement('div');
        body.className = 'batch-item-body hidden';
        body.style.cssText = 'padding: 0 12px 12px 12px;';

        const sourceLabel = document.createElement('label');
        sourceLabel.textContent = '原文本:';
        sourceLabel.style.cssText = 'display: block; font-size: 12px; color: var(--text-secondary); margin-bottom: 4px;';

        const sourceTextarea = document.createElement('textarea');
        sourceTextarea.className = 'textarea batch-source-text';
        sourceTextarea.style.cssText = 'width: 100%; margin-bottom: 8px;';
        sourceTextarea.rows = 3;
        sourceTextarea.placeholder = '粘贴原文本...';
        sourceTextarea.value = task.sourceText;
        sourceTextarea.oninput = () => {
            subtitleBatchTasks[idx].sourceText = sourceTextarea.value;
            renderSubtitleBatchList(); // 更新状态
        };

        const transLabel = document.createElement('label');
        transLabel.textContent = '译文本:';
        transLabel.style.cssText = 'display: block; font-size: 12px; color: var(--text-secondary); margin-bottom: 4px;';

        const transTextarea = document.createElement('textarea');
        transTextarea.className = 'textarea batch-trans-text';
        transTextarea.style.cssText = 'width: 100%;';
        transTextarea.rows = 3;
        transTextarea.placeholder = '粘贴译文本...';
        transTextarea.value = task.translateText;
        transTextarea.oninput = () => {
            subtitleBatchTasks[idx].translateText = transTextarea.value;
            renderSubtitleBatchList();
        };

        body.appendChild(sourceLabel);
        body.appendChild(sourceTextarea);
        body.appendChild(transLabel);
        body.appendChild(transTextarea);

        item.appendChild(header);
        item.appendChild(body);
        list.appendChild(item);
    });

}

// 批量粘贴字幕文本
async function batchPasteSubtitleText(type) {
    try {
        const clipboardItems = await navigator.clipboard.read();
        let texts = [];
        let isTwoColumn = false;

        // 辅助函数：提取单元格文本，保留换行
        function getCellText(cell) {
            // 将 <br> 转换为换行符
            let html = cell.innerHTML;
            html = html.replace(/<br\s*\/?>/gi, '\n');
            // 创建临时元素获取纯文本
            const temp = document.createElement('div');
            temp.innerHTML = html;
            return temp.textContent.trim();
        }

        for (const item of clipboardItems) {
            // 优先解析 HTML（Google 表格格式）
            if (item.types.includes('text/html')) {
                const blob = await item.getType('text/html');
                const html = await blob.text();
                console.log('解析 HTML:', html.substring(0, 500));

                const parser = new DOMParser();
                const doc = parser.parseFromString(html, 'text/html');
                const rows = doc.querySelectorAll('tr');

                if (rows.length > 0) {
                    // 表格格式
                    rows.forEach(row => {
                        const cells = row.querySelectorAll('td, th');
                        if (cells.length >= 2) {
                            // 两列：原文 + 译文
                            isTwoColumn = true;
                            texts.push({
                                source: getCellText(cells[0]),
                                translate: getCellText(cells[1])
                            });
                        } else if (cells.length === 1) {
                            texts.push(getCellText(cells[0]));
                        }
                    });
                } else {
                    // 无表格，尝试解析单元格
                    const cells = doc.querySelectorAll('td, th');
                    cells.forEach(cell => {
                        const text = getCellText(cell);
                        if (text) texts.push(text);
                    });
                }
            }

            // 如果 HTML 没解析到内容，用纯文本
            if (texts.length === 0 && item.types.includes('text/plain')) {
                const blob = await item.getType('text/plain');
                const text = await blob.text();
                console.log('解析纯文本:', text.substring(0, 200));

                // 按行分割
                const lines = text.split('\n').map(t => t.trim()).filter(Boolean);
                lines.forEach(line => {
                    const parts = line.split('\t');
                    if (parts.length >= 2 && type === 'both') {
                        isTwoColumn = true;
                        texts.push({
                            source: parts[0].trim(),
                            translate: parts[1].trim()
                        });
                    } else {
                        texts.push(line);
                    }
                });
            }
        }

        console.log('解析结果:', texts);

        if (texts.length === 0) {
            showToast('剪贴板没有内容', 'error');
            return;
        }

        // 按顺序填充到任务
        const fillCount = Math.min(texts.length, subtitleBatchTasks.length);

        if (type === 'both') {
            // 两列一起粘贴
            for (let i = 0; i < fillCount; i++) {
                if (!subtitleBatchTasks[i]) continue;
                const item = texts[i];
                if (typeof item === 'object' && item.source !== undefined) {
                    // 已解析为对象格式
                    subtitleBatchTasks[i].sourceText = item.source;
                    subtitleBatchTasks[i].translateText = item.translate || '';
                } else if (typeof item === 'string') {
                    // 字符串，尝试 tab 分割
                    const parts = item.split('\t');
                    subtitleBatchTasks[i].sourceText = parts[0].trim();
                    subtitleBatchTasks[i].translateText = parts[1]?.trim() || '';
                }
            }
            console.log('填充后任务列表:', subtitleBatchTasks);
            renderSubtitleBatchList();
            showToast(`已填充 ${fillCount} 条原文+译文`, 'success');
        } else {
            for (let i = 0; i < fillCount; i++) {
                if (!subtitleBatchTasks[i]) continue;
                const item = texts[i];
                const text = typeof item === 'object' ? (type === 'source' ? item.source : item.translate) : item;
                if (type === 'source') {
                    subtitleBatchTasks[i].sourceText = text || '';
                } else {
                    subtitleBatchTasks[i].translateText = text || '';
                }
            }
            renderSubtitleBatchList();
            showToast(`已填充 ${fillCount} 条${type === 'source' ? '原文' : '译文'}`, 'success');
        }
    } catch (error) {
        showToast('粘贴失败: ' + error.message, 'error');
    }
}

// 清空批量列表
function clearSubtitleBatchList() {
    subtitleBatchTasks = [];
    renderSubtitleBatchList();
    showToast('已清空', 'info');
}

// 批量生成字幕
let isSubtitleBatchProcessing = false;

async function startBatchGeneration() {
    // 防止重复点击
    if (isSubtitleBatchProcessing) {
        showToast('正在处理中，请稍候', 'info');
        return;
    }

    console.log('批量任务列表:', subtitleBatchTasks);

    if (subtitleBatchTasks.length === 0) {
        showToast('请先添加任务', 'error');
        return;
    }

    // 只需要原文即可（译文可选）
    const readyTasks = subtitleBatchTasks.filter(t => t.sourceText && t.sourceText.trim());
    console.log('就绪任务:', readyTasks.length);

    if (readyTasks.length === 0) {
        showToast('没有就绪的任务（需要原文）', 'error');
        return;
    }

    const language = document.getElementById('language')?.value || '英语';
    const cutLength = parseFloat(document.getElementById('cut-length')?.value) || 5.0;
    const seamless = document.getElementById('seamless')?.checked || false;
    const exportFcpxml = document.getElementById('export-fcpxml')?.checked || false;
    const sourceUp = document.getElementById('source-up')?.checked || false;
    const mergeSrt = document.getElementById('merge-srt')?.checked || false;

    const gladiaKeysText = document.getElementById('gladia-keys')?.value || '';
    const gladiaKeys = gladiaKeysText.split('\n').filter(k => k.trim());

    // 并行数 = Key 数量（至少1个）
    const concurrency = Math.max(gladiaKeys.length, 1);
    console.log(`并行数: ${concurrency}, Key 数量: ${gladiaKeys.length}`);

    const generateBtn = document.getElementById('generate-btn');
    generateBtn.disabled = true;
    generateBtn.textContent = '⏳ 批量处理中...';

    let successCount = 0;
    let failCount = 0;
    let processedCount = 0;

    // 收集需要处理的任务
    const readyTaskIndices = [];
    for (let i = 0; i < subtitleBatchTasks.length; i++) {
        const task = subtitleBatchTasks[i];
        if (task.sourceText && task.sourceText.trim()) {
            readyTaskIndices.push(i);
        }
    }

    const totalTasks = readyTaskIndices.length;

    // 处理单个任务
    async function processTask(taskIndex, keyIndex) {
        const task = subtitleBatchTasks[taskIndex];
        const keyToUse = gladiaKeys.length > 0 ? [gladiaKeys[keyIndex % gladiaKeys.length]] : [];

        updateStatus(`处理中 ${processedCount + 1}/${totalTasks}: ${task.fileName}`, 'processing');

        // 创建 FormData 上传文件
        const formData = new FormData();
        formData.append('audio_file', task.file);
        formData.append('source_text', task.sourceText);
        formData.append('translate_text', task.translateText || '');
        formData.append('language', language);
        formData.append('audio_cut_length', cutLength);
        formData.append('gladia_keys', JSON.stringify(keyToUse));
        formData.append('gen_merge_srt', mergeSrt);
        formData.append('source_up_order', sourceUp);
        formData.append('export_fcpxml', exportFcpxml);
        formData.append('seamless_fcpxml', seamless);

        try {
            const response = await fetch(`${API_BASE}/subtitle/generate-with-file`, {
                method: 'POST',
                body: formData
            });

            processedCount++;

            if (response.ok) {
                successCount++;
                subtitleBatchTasks[taskIndex].status = 'success';
                const result = await response.json();
                subtitleBatchTasks[taskIndex].files = result.files || [];
                // 更新任务状态
                const items = document.querySelectorAll('.subtitle-batch-item');
                if (items[taskIndex]) {
                    const status = items[taskIndex].querySelector('.batch-item-status');
                    if (status) {
                        status.textContent = '✅ 完成';
                        status.style.background = 'rgba(0,255,0,0.2)';
                        status.style.color = '#51cf66';
                    }
                    // 移除重试按钮
                    const retryBtn = items[taskIndex].querySelector('.subtitle-retry-btn');
                    if (retryBtn) retryBtn.remove();
                }
                return { success: true, taskIndex };
            } else {
                failCount++;
                subtitleBatchTasks[taskIndex].status = 'failed';
                const error = await response.json();
                subtitleBatchTasks[taskIndex].error = error.error || '未知错误';
                const items = document.querySelectorAll('.subtitle-batch-item');
                if (items[taskIndex]) {
                    const status = items[taskIndex].querySelector('.batch-item-status');
                    if (status) {
                        status.textContent = '❌ 失败';
                        status.style.background = 'rgba(255,0,0,0.2)';
                        status.style.color = '#ff6b6b';
                    }
                    addSubtitleRetryButton(items[taskIndex], taskIndex);
                }
                return { success: false, taskIndex };
            }
        } catch (error) {
            processedCount++;
            failCount++;
            subtitleBatchTasks[taskIndex].status = 'failed';
            subtitleBatchTasks[taskIndex].error = error.message;
            console.error(`任务 ${taskIndex + 1} 失败:`, error);
            const items = document.querySelectorAll('.subtitle-batch-item');
            if (items[taskIndex]) {
                const status = items[taskIndex].querySelector('.batch-item-status');
                if (status) {
                    status.textContent = '❌ 失败';
                    status.style.background = 'rgba(255,0,0,0.2)';
                    status.style.color = '#ff6b6b';
                }
                addSubtitleRetryButton(items[taskIndex], taskIndex);
            }
            return { success: false, taskIndex };
        }
    }

    // 并行执行（每个 Key 处理一个任务）
    let taskQueue = [...readyTaskIndices];
    const runningTasks = [];

    async function runParallel() {
        while (taskQueue.length > 0 || runningTasks.length > 0) {
            // 启动新任务直到达到并行数
            while (runningTasks.length < concurrency && taskQueue.length > 0) {
                const taskIndex = taskQueue.shift();
                const keyIndex = runningTasks.length;
                const promise = processTask(taskIndex, keyIndex).then(result => {
                    // 从运行队列移除
                    const idx = runningTasks.indexOf(promise);
                    if (idx > -1) runningTasks.splice(idx, 1);
                    return result;
                });
                runningTasks.push(promise);
            }

            // 等待任意一个任务完成
            if (runningTasks.length > 0) {
                await Promise.race(runningTasks);
            }

            updateStatus(`处理中 ${processedCount}/${totalTasks}`, 'processing');
        }
    }

    await runParallel();

    generateBtn.disabled = false;
    generateBtn.textContent = '🚀 生成字幕';

    if (failCount === 0) {
        updateStatus(`批量完成: ${successCount} 个成功`, 'success');
        showToast(`批量完成: ${successCount} 个成功`, 'success');
    } else {
        updateStatus(`批量完成: ${successCount} 成功, ${failCount} 失败`, 'warning');
        showToast(`批量完成: ${successCount} 成功, ${failCount} 失败（可重试）`, 'warning');
        showSubtitleRetryAllButton();
    }

    // 显示结果和下载按钮
    if (successCount > 0) {
        showSubtitleResultsPanel();
    }

    isSubtitleBatchProcessing = false;
}

// 显示结果面板
function showSubtitleResultsPanel() {
    const section = document.getElementById('subtitle-batch-section');
    if (!section) return;

    // 移除旧的结果面板
    const oldPanel = document.getElementById('subtitle-results-panel');
    if (oldPanel) oldPanel.remove();

    // 收集所有成功的文件
    const allFiles = [];
    subtitleBatchTasks.forEach(task => {
        if (task.status === 'success' && task.files) {
            allFiles.push(...task.files);
        }
    });

    if (allFiles.length === 0) return;

    const panel = document.createElement('div');
    panel.id = 'subtitle-results-panel';
    panel.style.cssText = 'margin-top: 16px; padding: 16px; background: var(--bg-secondary); border-radius: 8px;';

    panel.innerHTML = `
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
            <h4 style="margin: 0; color: var(--text-primary);">📁 生成结果 (${allFiles.length} 个文件)</h4>
            <button id="download-all-subtitles-btn" class="btn btn-primary" style="padding: 8px 16px;">
                📦 下载全部 (ZIP)
            </button>
        </div>
        <div id="subtitle-file-list" style="max-height: 200px; overflow-y: auto;"></div>
    `;

    section.appendChild(panel);

    // 渲染文件列表
    const fileList = document.getElementById('subtitle-file-list');
    allFiles.forEach(filePath => {
        const fileName = filePath.split('/').pop();
        const item = document.createElement('div');
        item.style.cssText = 'display: flex; align-items: center; padding: 6px 8px; background: var(--bg-tertiary); border-radius: 4px; margin-bottom: 4px;';
        item.innerHTML = `
            <span style="flex: 1; font-size: 12px; color: var(--text-secondary);">${fileName}</span>
        `;
        fileList.appendChild(item);
    });

    // 下载按钮事件
    document.getElementById('download-all-subtitles-btn').onclick = async () => {
        try {
            showToast('正在打包...', 'info');
            const response = await fetch(`${API_BASE}/subtitle/download-zip`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ files: allFiles })
            });

            if (response.ok) {
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `subtitles_${Date.now()}.zip`;
                a.click();
                URL.revokeObjectURL(url);
                showToast('下载完成', 'success');
            } else {
                showToast('下载失败', 'error');
            }
        } catch (e) {
            showToast('下载失败: ' + e.message, 'error');
        }
    };
}

// 添加单个重试按钮
function addSubtitleRetryButton(item, index) {
    if (item.querySelector('.subtitle-retry-btn')) return;

    const header = item.querySelector('div');
    const retryBtn = document.createElement('button');
    retryBtn.className = 'btn btn-secondary subtitle-retry-btn';
    retryBtn.style.cssText = 'padding: 2px 8px; font-size: 11px; margin-left: 4px;';
    retryBtn.textContent = '🔄 重试';
    retryBtn.onclick = (e) => {
        e.stopPropagation();
        retrySingleSubtitleTask(index);
    };
    header.appendChild(retryBtn);
}

// 显示"重试所有失败"按钮
function showSubtitleRetryAllButton() {
    const section = document.getElementById('subtitle-batch-section');
    if (!section) return;

    const oldBtn = document.getElementById('subtitle-retry-all-btn');
    if (oldBtn) oldBtn.remove();

    const btn = document.createElement('button');
    btn.id = 'subtitle-retry-all-btn';
    btn.className = 'btn btn-primary';
    btn.style.cssText = 'margin-top: 12px; width: 100%;';
    btn.textContent = '🔄 重试所有失败项';
    btn.onclick = retryAllSubtitleTasks;

    const list = document.getElementById('subtitle-batch-list');
    if (list) {
        list.parentNode.insertBefore(btn, list.nextSibling);
    }
}

// 重试单个任务
async function retrySingleSubtitleTask(index) {
    const task = subtitleBatchTasks[index];
    if (!task) return;

    const items = document.querySelectorAll('.subtitle-batch-item');
    const item = items[index];
    if (item) {
        const status = item.querySelector('.batch-item-status');
        if (status) {
            status.textContent = '⏳ 重试中...';
            status.style.background = 'rgba(255,165,0,0.2)';
            status.style.color = '#ffa500';
        }
    }

    const language = document.getElementById('language')?.value || '英语';
    const cutLength = parseFloat(document.getElementById('cut-length')?.value) || 5.0;
    const seamless = document.getElementById('seamless')?.checked || false;
    const exportFcpxml = document.getElementById('export-fcpxml')?.checked || false;
    const sourceUp = document.getElementById('source-up')?.checked || false;
    const mergeSrt = document.getElementById('merge-srt')?.checked || false;
    const gladiaKeysText = document.getElementById('gladia-keys')?.value || '';
    const gladiaKeys = gladiaKeysText.split('\n').filter(k => k.trim());

    const formData = new FormData();
    formData.append('audio_file', task.file);
    formData.append('source_text', task.sourceText);
    formData.append('translate_text', task.translateText);
    formData.append('language', language);
    formData.append('audio_cut_length', cutLength);
    formData.append('gladia_keys', JSON.stringify(gladiaKeys));
    formData.append('gen_merge_srt', mergeSrt);
    formData.append('source_up_order', sourceUp);
    formData.append('export_fcpxml', exportFcpxml);
    formData.append('seamless_fcpxml', seamless);

    try {
        const response = await fetch(`${API_BASE}/subtitle/generate-with-file`, {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            task.status = 'success';
            if (item) {
                const status = item.querySelector('.batch-item-status');
                if (status) {
                    status.textContent = '✅ 完成';
                    status.style.background = 'rgba(0,255,0,0.2)';
                    status.style.color = '#51cf66';
                }
                const retryBtn = item.querySelector('.subtitle-retry-btn');
                if (retryBtn) retryBtn.remove();
            }
            showToast('重试成功', 'success');

            // 检查是否还有失败项
            const hasFailed = subtitleBatchTasks.some(t => t.status === 'failed');
            if (!hasFailed) {
                const retryAllBtn = document.getElementById('subtitle-retry-all-btn');
                if (retryAllBtn) retryAllBtn.remove();
            }
        } else {
            const error = await response.json();
            task.error = error.error || '未知错误';
            if (item) {
                const status = item.querySelector('.batch-item-status');
                if (status) {
                    status.textContent = '❌ 失败';
                    status.style.background = 'rgba(255,0,0,0.2)';
                    status.style.color = '#ff6b6b';
                }
            }
            showToast('重试失败: ' + (error.error || '未知错误'), 'error');
        }
    } catch (error) {
        task.error = error.message;
        if (item) {
            const status = item.querySelector('.batch-item-status');
            if (status) {
                status.textContent = '❌ 失败';
            }
        }
        showToast('重试失败: ' + error.message, 'error');
    }
}

// 重试所有失败任务
async function retryAllSubtitleTasks() {
    const failedIndexes = subtitleBatchTasks
        .map((t, i) => t.status === 'failed' ? i : -1)
        .filter(i => i >= 0);

    if (failedIndexes.length === 0) {
        showToast('没有失败项需要重试', 'info');
        return;
    }

    showToast(`正在重试 ${failedIndexes.length} 个失败项...`, 'info');

    for (const idx of failedIndexes) {
        await retrySingleSubtitleTask(idx);
    }
}

// ==================== 字幕对齐功能 ====================

async function startGeneration() {
    // 检查是否批量模式
    const batchMode = document.getElementById('subtitle-batch-mode')?.checked;

    if (batchMode) {
        await startBatchGeneration();
        return;
    }

    const audioPath = currentAudioPath;
    const sourceText = document.getElementById('source-text').value;
    const translateText = document.getElementById('translate-text').value;
    const language = document.getElementById('language').value;
    const cutLength = parseFloat(document.getElementById('cut-length').value);

    if (!audioPath) {
        showToast('请先选择音视频文件', 'error');
        return;
    }

    if (!sourceText) {
        showToast('请输入原文本', 'error');
        return;
    }

    const seamless = document.getElementById('seamless').checked;
    const exportFcpxml = document.getElementById('export-fcpxml').checked;
    const sourceUp = document.getElementById('source-up').checked;
    const mergeSrt = document.getElementById('merge-srt').checked;

    const gladiaKeysText = document.getElementById('gladia-keys').value;
    const gladiaKeys = gladiaKeysText.split('\n').filter(k => k.trim());

    const requestData = {
        audio_path: audioPath,
        source_text: sourceText,
        translate_text: translateText,
        language: language,
        audio_cut_length: cutLength,
        gladia_keys: gladiaKeys,
        gen_merge_srt: mergeSrt,
        source_up_order: sourceUp,
        export_fcpxml: exportFcpxml,
        seamless_fcpxml: seamless
    };

    try {
        updateStatus('开始处理...', 'processing');
        document.getElementById('progress-bar').classList.remove('hidden');
        setIndeterminateProgress('progress-bar', true);
        document.getElementById('generate-btn').disabled = true;

        const response = await fetch(`${API_BASE}/subtitle/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || '请求失败');
        }

        showToast('开始处理...', 'info');
        pollStatus();

    } catch (error) {
        updateStatus('错误: ' + error.message, 'error');
        showToast('错误: ' + error.message, 'error');
        document.getElementById('progress-bar').classList.add('hidden');
        setIndeterminateProgress('progress-bar', false);
        document.getElementById('generate-btn').disabled = false;
    }
}

async function pollStatus() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        const status = await response.json();

        if (status.is_processing) {
            updateStatus(status.progress || '处理中...', 'processing');
            setTimeout(pollStatus, 1000);
        } else {
            document.getElementById('progress-bar').classList.add('hidden');
            setIndeterminateProgress('progress-bar', false);
            document.getElementById('generate-btn').disabled = false;

            if (status.error) {
                updateStatus('错误: ' + status.error, 'error');
                showToast('处理失败', 'error');
            } else if (status.result) {
                updateStatus('完成！', 'success');
                showToast('字幕生成完成！', 'success', 5000);
            }
        }
    } catch (error) {
        setTimeout(pollStatus, 2000);
    }
}

// ==================== SRT 工具功能 ====================

async function adjustSrt() {
    if (!currentSrtSrcPath) {
        showToast('请先选择源 SRT 文件', 'error');
        return;
    }

    const intervalTime = parseFloat(document.getElementById('interval-time').value);
    const charTime = parseFloat(document.getElementById('char-time').value);
    const minChar = parseInt(document.getElementById('min-char').value);
    const scale = parseFloat(document.getElementById('scale').value);
    const ignoreChars = document.getElementById('ignore-chars').value;

    try {
        const response = await fetch(`${API_BASE}/srt/adjust`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                src_path: currentSrtSrcPath,
                interval_time: intervalTime,
                char_time: charTime,
                min_char_count: minChar,
                scale: scale,
                ignore: ignoreChars
            })
        });

        const result = await response.json();

        if (response.ok) {
            showToast('调整完成！', 'success');
            updateStatus('输出: ' + result.output_path, 'success');
        } else {
            showToast('错误: ' + result.error, 'error');
        }
    } catch (error) {
        showToast('请求失败: ' + error.message, 'error');
    }
}

async function computeCharTime() {
    if (!currentSrtRefPath) {
        showToast('请先选择参考 SRT 文件', 'error');
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/srt/compute-char-time`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ref_path: currentSrtRefPath,
                interval_time: parseFloat(document.getElementById('interval-time').value)
            })
        });

        const result = await response.json();

        if (response.ok) {
            document.getElementById('char-time').value = result.char_time.toFixed(4);
            showToast('字符时间已计算', 'success');
        } else {
            showToast('错误: ' + result.error, 'error');
        }
    } catch (error) {
        showToast('请求失败: ' + error.message, 'error');
    }
}

async function generateSeamlessSrt() {
    if (!currentSeamlessSrtPath) {
        showToast('请先选择 SRT 文件', 'error');
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/srt/seamless`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                src_path: currentSeamlessSrtPath
            })
        });

        const result = await response.json();

        if (response.ok) {
            showToast('生成完成！', 'success');
        } else {
            showToast('错误: ' + result.error, 'error');
        }
    } catch (error) {
        showToast('请求失败: ' + error.message, 'error');
    }
}

// ==================== 媒体转换功能 ====================

async function startMediaConvert() {
    if (currentMediaFileInfos.length === 0) {
        showToast('请先选择要转换的文件', 'error');
        return;
    }

    const outputPath = document.getElementById('media-output-path').value;
    const statusEl = document.getElementById('media-status');

    // 先上传文件到后端（浏览器无法获取本地文件路径）
    statusEl.textContent = '正在上传文件...';
    const uploadedPaths = [];
    const pathMapping = {};  // 原路径 -> 上传后路径的映射

    for (let i = 0; i < currentMediaFileInfos.length; i++) {
        const fileInfo = currentMediaFileInfos[i];
        if (fileInfo.file) {
            // 需要上传
            const formData = new FormData();
            formData.append('file', fileInfo.file);

            try {
                const resp = await fetch(`${API_BASE}/file/upload`, {
                    method: 'POST',
                    body: formData
                });
                const result = await resp.json();
                if (result.success) {
                    uploadedPaths.push(result.path);
                    // 保存映射：原路径 -> 上传后路径
                    pathMapping[fileInfo.path] = result.path;
                    // 同时更新 fileInfo，后续使用
                    fileInfo.uploadedPath = result.path;
                } else {
                    showToast(`上传失败: ${result.error}`, 'error');
                    statusEl.textContent = '上传失败';
                    return;
                }
            } catch (err) {
                showToast(`上传失败: ${err.message}`, 'error');
                statusEl.textContent = '上传失败';
                return;
            }
        } else if (fileInfo.path && fileInfo.path !== fileInfo.name) {
            // 已有完整路径（Electron 环境）
            uploadedPaths.push(fileInfo.path);
            pathMapping[fileInfo.path] = fileInfo.path;
            fileInfo.uploadedPath = fileInfo.path;
        }
    }

    if (uploadedPaths.length === 0) {
        showToast('没有有效的文件路径', 'error');
        return;
    }

    // 确定当前激活的子标签页
    const activeSubtab = document.querySelector('#media-panel .subtab-content.active');
    const subtabId = activeSubtab?.id || '';

    let payload = {
        files: uploadedPaths,
        output_dir: outputPath
    };

    // 根据子标签页构建请求参数
    if (subtabId === 'media-logo-subtab') {
        // Logo 叠加模式
        const logoPreset = document.querySelector('input[name="logo-preset"]:checked')?.value;

        if (logoPreset === 'custom') {
            // 自定义 Logo
            const customLogoPath = document.getElementById('custom-logo-file')?.dataset?.filePath;
            if (!customLogoPath) {
                showToast('请选择自定义 Logo 图片', 'error');
                return;
            }
            payload.mode = 'custom_logo';
            payload.custom_logo = {
                path: customLogoPath,
                x: parseInt(document.getElementById('logo-pos-x').value) || 590,
                y: parseInt(document.getElementById('logo-pos-y').value) || 1810,
                width: parseInt(document.getElementById('logo-width').value) || 400,
                height: parseInt(document.getElementById('logo-height').value) || 90
            };
        } else {
            payload.mode = logoPreset || 'hailuo';
            payload.logo_override = getLogoOverrideFromInputs();
        }

    } else if (subtabId === 'media-watermark-subtab') {
        // 文字水印模式
        payload.mode = 'watermark';

        const text = document.getElementById('watermark-text').value || 'AI Generated';
        const fontFamily = document.getElementById('watermark-font').value || 'Arial';
        const fontSize = parseInt(document.getElementById('watermark-fontsize').value) || 24;
        const color = document.getElementById('watermark-color').value || '#ffffff';
        const opacity = parseFloat(document.getElementById('watermark-opacity').value) || 1;
        const hasStroke = document.getElementById('watermark-stroke').checked;
        const strokeColor = document.getElementById('watermark-stroke-color').value || '#000000';
        const strokeWidth = parseInt(document.getElementById('watermark-stroke-width').value) || 2;
        const hasShadow = document.getElementById('watermark-shadow').checked;
        const position = document.getElementById('watermark-position').value || 'top-right';

        // 位置转换为 FFmpeg xy 表达式
        let posX = 'w-tw-10', posY = '10';
        switch (position) {
            case 'top-left': posX = '10'; posY = '10'; break;
            case 'top-right': posX = 'w-tw-10'; posY = '10'; break;
            case 'bottom-left': posX = '10'; posY = 'h-th-10'; break;
            case 'bottom-right': posX = 'w-tw-10'; posY = 'h-th-10'; break;
            case 'center': posX = '(w-tw)/2'; posY = '(h-th)/2'; break;
            case 'custom':
                posX = document.getElementById('watermark-pos-x').value || 'w-tw-10';
                posY = document.getElementById('watermark-pos-y').value || '10';
                break;
        }

        payload.watermark = {
            text: text,
            font: fontFamily,
            font_size: fontSize,
            color: color,
            opacity: opacity,
            stroke: hasStroke,
            stroke_color: strokeColor,
            stroke_width: strokeWidth,
            shadow: hasShadow,
            x: posX,
            y: posY
        };

    } else if (subtabId === 'media-format-subtab') {
        // 格式转换模式
        const formatMode = document.querySelector('input[name="format-mode"]:checked')?.value || 'h264';
        payload.mode = formatMode;

        if (formatMode === 'audio_split') {
            const exportMp3 = document.getElementById('export-split-mp3').checked;
            const exportMp4 = document.getElementById('export-split-mp4').checked;

            if (!exportMp3 && !exportMp4) {
                showToast('请至少选择一种导出格式', 'error');
                return;
            }

            const cutPointsMap = {};
            for (let i = 0; i < currentMediaFileInfos.length; i++) {
                const file = currentMediaFileInfos[i];
                const input = document.getElementById(`audio-cut-points-${i}`);
                const value = input ? input.value.trim() : '';

                // 允许不填写裁切点（直接转换整个文件）
                // 使用上传后的路径作为 key
                const serverPath = file.uploadedPath || file.path;
                if (value) {
                    cutPointsMap[serverPath] = value;
                    currentAudioCutPoints[file.path] = value;
                }
            }

            payload.cut_points_map = cutPointsMap;
            payload.export_mp3 = exportMp3;
            payload.export_mp4 = exportMp4;
        }
    } else {
        // 默认：使用第一个子标签页的 Logo 模式
        payload.mode = document.querySelector('input[name="logo-preset"]:checked')?.value || 'hailuo';
    }

    try {
        updateStatus('开始转换...', 'processing', 'media-status');
        document.getElementById('media-progress').classList.remove('hidden');
        setIndeterminateProgress('media-progress', true);

        const response = await fetch(`${API_BASE}/media/convert`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const result = await response.json();

        document.getElementById('media-progress').classList.add('hidden');
        setIndeterminateProgress('media-progress', false);

        if (response.ok) {
            updateStatus('转换完成！', 'success', 'media-status');
            showToast(result.message, 'success');

            // 显示下载链接（带时长信息）
            if (result.files && result.files.length > 0) {
                showConvertedFilesDownload(result.files, result.files_info);
            }
        } else {
            updateStatus('错误: ' + result.error, 'error', 'media-status');
            showToast('错误: ' + result.error, 'error');
        }
    } catch (error) {
        document.getElementById('media-progress').classList.add('hidden');
        setIndeterminateProgress('media-progress', false);
        updateStatus('请求失败', 'error', 'media-status');
        showToast('请求失败: ' + error.message, 'error');
    }
}

function showConvertedFilesDownload(files, filesInfo) {
    // 在状态区域下方显示下载链接
    const statusSection = document.querySelector('#media-panel .status-section');
    if (!statusSection) return;

    // 移除旧的下载区域
    const oldDownloadArea = document.getElementById('media-download-area');
    if (oldDownloadArea) oldDownloadArea.remove();

    const downloadArea = document.createElement('div');
    downloadArea.id = 'media-download-area';
    downloadArea.style.cssText = 'margin-top: 12px; padding: 12px; background: var(--bg-tertiary); border-radius: 8px;';

    // 标题行和下载全部按钮
    const header = document.createElement('div');
    header.style.cssText = 'display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;';

    const title = document.createElement('div');
    title.style.cssText = 'font-weight: 500; color: var(--text-primary);';
    title.textContent = `✅ 已生成 ${files.length} 个文件:`;

    const downloadAllBtn = document.createElement('button');
    downloadAllBtn.className = 'btn btn-primary';
    downloadAllBtn.style.cssText = 'padding: 4px 12px; font-size: 12px;';
    downloadAllBtn.textContent = '📦 下载全部';
    downloadAllBtn.onclick = () => downloadAllFiles(files);

    header.appendChild(title);
    header.appendChild(downloadAllBtn);
    downloadArea.appendChild(header);

    const list = document.createElement('div');
    list.style.cssText = 'display: flex; flex-direction: column; gap: 4px; max-height: 150px; overflow-y: auto;';

    // 创建路径到时长的映射
    const durationMap = {};
    if (filesInfo) {
        filesInfo.forEach(info => {
            durationMap[info.path] = info.duration;
        });
    }

    files.forEach(filePath => {
        const filename = filePath.split('/').pop();
        // 去掉 UUID 前缀
        let displayName = filename;
        if (filename.includes('_') && filename.split('_')[0].length === 8) {
            displayName = filename.split('_').slice(1).join('_');
        }

        // 获取时长
        const duration = durationMap[filePath];
        const durationStr = duration ? ` (${formatDuration(duration)})` : '';

        const link = document.createElement('a');
        link.href = `${API_BASE}/file/download?path=${encodeURIComponent(filePath)}`;
        link.textContent = `📥 ${displayName}${durationStr}`;
        link.style.cssText = 'color: var(--accent); text-decoration: none; font-size: 13px;';
        link.download = displayName;
        list.appendChild(link);
    });

    downloadArea.appendChild(list);
    statusSection.appendChild(downloadArea);
}

function formatDuration(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.round((seconds % 1) * 10);
    return `${mins}:${secs.toString().padStart(2, '0')}.${ms}`;
}

async function downloadAllFiles(files) {
    showToast('正在打包 ZIP...', 'info');

    try {
        const response = await fetch(`${API_BASE}/file/download-zip`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ files })
        });

        if (!response.ok) {
            throw new Error('打包失败');
        }

        // 获取 blob 并触发下载
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'converted_files.zip';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        showToast('ZIP 打包下载完成', 'success');
    } catch (error) {
        showToast('下载失败: ' + error.message, 'error');
    }
}

async function selectMediaOutputDir() {
    try {
        const dir = await window.electronAPI.selectDirectory();
        if (dir) {
            document.getElementById('media-output-path').value = dir;
        }
    } catch (error) {
        console.error('选择目录失败:', error);
        showToast('选择目录失败', 'error');
    }
}

// ==================== ElevenLabs 功能 ====================

async function saveElevenLabsKey() {
    const rawKeys = document.getElementById('elevenlabs-api-keys').value;
    const apiKeys = rawKeys.split(/[\s,;]+/).map(k => k.trim()).filter(Boolean);

    try {
        const response = await fetch(`${API_BASE}/settings/elevenlabs`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ api_keys: apiKeys })
        });

        if (response.ok) {
            showToast('API Keys 已保存', 'success');
            loadVoices();
            loadQuota();
        }
    } catch (error) {
        showToast('保存失败: ' + error.message, 'error');
    }
}

async function loadVoices() {
    updateElevenLabsStatus('连接中...');

    try {
        const response = await fetch(`${API_BASE}/elevenlabs/voices`);
        const data = await response.json();

        const select = document.getElementById('voice-select');
        select.innerHTML = '';
        voiceCache.clear();

        if (data.voices && data.voices.length > 0) {
            data.voices.forEach(voice => {
                const option = document.createElement('option');
                option.value = voice.voice_id;
                option.textContent = voice.name;
                option.dataset.previewUrl = voice.preview_url || '';
                select.appendChild(option);

                if (voice.voice_id) {
                    voiceCache.set(voice.voice_id, voice.voice_id);
                }
                if (voice.name) {
                    const cleanName = voice.name.replace(/^\[[^\]]+\]\s*/, '');
                    voiceCache.set(cleanName.toLowerCase(), voice.voice_id);
                }
            });
            updateElevenLabsStatus(`已加载 ${data.voices.length} 个语音`);
            showToast(`已加载 ${data.voices.length} 个语音`, 'success');
        } else {
            select.innerHTML = '<option value="">无可用语音</option>';
            updateElevenLabsStatus('无可用语音');
        }

        syncBatchVoiceOptions();

        // 同时加载额度
        loadQuota();
    } catch (error) {
        console.error('加载语音失败:', error);
        updateElevenLabsStatus('加载失败');
    }
}

async function searchVoices() {
    const searchTerm = document.getElementById('voice-search-input').value.trim();

    if (!searchTerm) {
        showToast('请输入搜索关键词', 'error');
        return;
    }

    updateElevenLabsStatus(`搜索 "${searchTerm}"...`);

    try {
        const response = await fetch(`${API_BASE}/elevenlabs/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ search_term: searchTerm })
        });

        const data = await response.json();
        const select = document.getElementById('voice-select');

        console.log('搜索结果:', data);
        console.log('voice-select 元素:', select);

        if (data.voices && data.voices.length > 0) {
            // 添加搜索结果到下拉框（添加标记）
            let addedCount = 0;
            data.voices.forEach(voice => {
                // 检查是否已存在
                let exists = false;
                for (let i = 0; i < select.options.length; i++) {
                    if (select.options[i].value === voice.voice_id) {
                        exists = true;
                        break;
                    }
                }

                if (!exists && voice.voice_id) {
                    const option = document.createElement('option');
                    option.value = voice.voice_id;
                    option.textContent = `[搜索] ${voice.name}`;
                    option.dataset.previewUrl = voice.preview_url || '';
                    select.appendChild(option);
                    addedCount++;

                    if (voice.voice_id) {
                        voiceCache.set(voice.voice_id, voice.voice_id);
                    }
                    if (voice.name) {
                        const cleanName = voice.name.replace(/^\[[^\]]+\]\s*/, '');
                        voiceCache.set(cleanName.toLowerCase(), voice.voice_id);
                    }
                }
            });

            console.log(`添加了 ${addedCount} 个声音到下拉框`);
            console.log('下拉框当前选项数:', select.options.length);

            // 显示搜索结果列表
            const resultsDiv = document.getElementById('voice-search-results');
            resultsDiv.innerHTML = '';
            resultsDiv.classList.remove('hidden');

            data.voices.forEach((voice, idx) => {
                const item = document.createElement('div');
                item.style.cssText = 'padding: 8px 12px; cursor: pointer; border-radius: 4px; display: flex; justify-content: space-between; align-items: center;';
                item.onmouseenter = () => item.style.background = 'rgba(255,255,255,0.1)';
                item.onmouseleave = () => item.style.background = '';

                const nameSpan = document.createElement('span');
                nameSpan.textContent = `${idx + 1}. ${voice.name}`;
                nameSpan.style.cssText = 'flex: 1; color: var(--text-primary);';

                const selectBtn = document.createElement('button');
                selectBtn.className = 'btn btn-primary';
                selectBtn.style.cssText = 'padding: 4px 10px; font-size: 12px;';
                selectBtn.textContent = '选择';
                selectBtn.onclick = (e) => {
                    e.stopPropagation();
                    // 设置下拉框选中值
                    select.value = voice.voice_id;
                    syncBatchVoiceOptions();
                    showToast(`已选择: ${voice.name}`, 'success');
                    resultsDiv.classList.add('hidden');
                };

                item.appendChild(nameSpan);
                item.appendChild(selectBtn);
                resultsDiv.appendChild(item);
            });

            // 选中第一个搜索结果
            if (data.voices.length > 0) {
                select.value = data.voices[0].voice_id;
            }

            syncBatchVoiceOptions();

            updateElevenLabsStatus(`找到 ${data.voices.length} 个结果`);
            showToast(`找到 ${data.voices.length} 个声音，请从列表中选择`, 'success');
        } else {
            const resultsDiv = document.getElementById('voice-search-results');
            resultsDiv.innerHTML = '<div style="padding: 12px; color: var(--text-secondary); text-align: center;">没有找到匹配的声音</div>';
            resultsDiv.classList.remove('hidden');
            updateElevenLabsStatus('没有找到匹配的声音');
            showToast('没有找到匹配的声音', 'info');
        }
    } catch (error) {
        console.error('搜索失败:', error);
        updateElevenLabsStatus('搜索失败');
        showToast('搜索失败: ' + error.message, 'error');
    }
}

function updateQuotaSummary(quotas) {
    const quotaBar = document.getElementById('quota-bar-inner');
    const quotaText = document.getElementById('quota-text');
    const quotaMeta = document.getElementById('quota-meta');

    if (!quotaBar || !quotaText) return;

    if (!Array.isArray(quotas) || quotas.length === 0) {
        quotaBar.style.width = '0%';
        quotaText.textContent = 'N/A';
        if (quotaMeta) {
            quotaMeta.textContent = '未配置 API Key';
        }
        return;
    }

    let enabledCount = 0;
    let disabledCount = 0;
    let availableCount = 0;
    let errorCount = 0;
    let usageTotal = 0;
    let limitTotal = 0;

    quotas.forEach((quota) => {
        const enabled = quota && quota.enabled !== false;
        if (enabled) {
            enabledCount += 1;
        } else {
            disabledCount += 1;
        }

        if (quota && quota.error) {
            errorCount += 1;
            return;
        }

        const usage = typeof quota.usage === 'number' ? quota.usage : null;
        const limit = typeof quota.limit === 'number' ? quota.limit : null;

        if (enabled && usage !== null && limit !== null && limit > 0) {
            usageTotal += usage;
            limitTotal += limit;
            const remaining = typeof quota.remaining === 'number' ? quota.remaining : (limit - usage);
            if (remaining > 0) {
                availableCount += 1;
            }
        }
    });

    if (limitTotal > 0) {
        const percent = Math.round((usageTotal / limitTotal) * 100);
        quotaBar.style.width = `${percent}%`;
        quotaText.textContent = `总计 ${usageTotal.toLocaleString()} / ${limitTotal.toLocaleString()} (${percent}%)`;

        if (percent > 90) {
            quotaBar.style.background = '#ff4757';
        } else {
            quotaBar.style.background = 'linear-gradient(135deg, #00d9a5, #00b4d8)';
        }

        if (quotaMeta) {
            const parts = [
                `停用 ${disabledCount}`,
                `有额度 ${availableCount}`
            ];
            if (errorCount > 0) {
                parts.push(`异常 ${errorCount}`);
            }
            quotaMeta.textContent = parts.join(' | ');
        }
    } else {
        quotaBar.style.width = '0%';
        quotaText.textContent = 'N/A';
        if (quotaMeta) {
            const parts = [];
            if (enabledCount > 0) parts.push(`启用 ${enabledCount}`);
            if (disabledCount > 0) parts.push(`停用 ${disabledCount}`);
            if (errorCount > 0) parts.push(`异常 ${errorCount}`);
            quotaMeta.textContent = parts.length ? parts.join(' | ') : '无可用额度';
        }
    }
}

async function loadQuota() {
    try {
        const response = await fetch(`${API_BASE}/elevenlabs/all-quotas`);
        const data = await response.json();
        updateQuotaSummary(data.keys || []);
    } catch (error) {
        console.error('加载额度失败:', error);
    }
}

// 加载所有 API Key 的额度和管理界面
async function loadAllQuotas() {
    const container = document.getElementById('all-keys-quota');
    const list = document.getElementById('all-keys-list');

    if (!container || !list) return;

    container.classList.remove('hidden');
    list.innerHTML = '<div style="text-align: center; color: var(--text-secondary);">加载中...</div>';

    try {
        // 同时获取 key 列表和额度
        const [keysResponse, quotasResponse] = await Promise.all([
            fetch(`${API_BASE}/settings/elevenlabs/keys`),
            fetch(`${API_BASE}/elevenlabs/all-quotas`)
        ]);

        const keysData = await keysResponse.json();
        const quotasData = await quotasResponse.json();

        const keys = keysData.keys || [];
        const quotas = quotasData.keys || [];

        updateQuotaSummary(quotas);

        // 创建额度映射
        const quotaMap = {};
        quotas.forEach(q => {
            quotaMap[q.key_prefix] = q;
        });

        if (keys.length > 0) {
            list.innerHTML = '';

            // 排序：启用的在前，停用的在后
            const sortedKeys = keys.map((k, i) => ({ ...k, originalIndex: i }));
            sortedKeys.sort((a, b) => {
                const aEnabled = a.enabled !== false;
                const bEnabled = b.enabled !== false;
                if (aEnabled && !bEnabled) return -1;
                if (!aEnabled && bEnabled) return 1;
                return 0;
            });

            sortedKeys.forEach((keyItem, displayIdx) => {
                const idx = keyItem.originalIndex;
                const keyStr = keyItem.key || '';
                const enabled = keyItem.enabled !== false;
                const keyPrefix = keyStr.slice(0, 8) + '...' + keyStr.slice(-4);
                const quota = quotas[idx] || {};

                // 判断颜色：停用=红色，有额度=绿色，无额度=默认
                let rowBg = 'transparent';
                if (!enabled) {
                    rowBg = 'rgba(255, 107, 107, 0.15)';  // 红色背景
                } else if (quota.remaining && quota.remaining > 200) {
                    rowBg = 'rgba(81, 207, 102, 0.1)';  // 绿色背景
                }

                const item = document.createElement('div');
                item.style.cssText = `display: flex; align-items: center; gap: 8px; padding: 8px; margin-bottom: 4px; border-radius: 6px; background: ${rowBg}; opacity: ${enabled ? 1 : 0.7};`;
                item.dataset.index = idx;

                // 排序按钮
                const orderBtns = document.createElement('div');
                orderBtns.style.cssText = 'display: flex; flex-direction: column; gap: 2px;';

                const upBtn = document.createElement('button');
                upBtn.textContent = '▲';
                upBtn.style.cssText = 'padding: 0 4px; font-size: 10px; cursor: pointer; background: none; border: 1px solid rgba(255,255,255,0.2); border-radius: 2px; color: var(--text-secondary);';
                upBtn.onclick = () => moveKey(idx, idx - 1);
                upBtn.disabled = idx === 0;

                const downBtn = document.createElement('button');
                downBtn.textContent = '▼';
                downBtn.style.cssText = 'padding: 0 4px; font-size: 10px; cursor: pointer; background: none; border: 1px solid rgba(255,255,255,0.2); border-radius: 2px; color: var(--text-secondary);';
                downBtn.onclick = () => moveKey(idx, idx + 1);
                downBtn.disabled = idx === keys.length - 1;

                orderBtns.appendChild(upBtn);
                orderBtns.appendChild(downBtn);

                // Key 标签
                const label = document.createElement('span');
                label.style.cssText = 'min-width: 120px; font-size: 12px; color: var(--text-secondary);';
                label.textContent = `${idx + 1}. ${keyPrefix}`;
                if (!enabled) label.textContent += ' (已停用)';

                // 额度条
                const bar = document.createElement('div');
                bar.style.cssText = 'flex: 1; height: 6px; background: rgba(255,255,255,0.1); border-radius: 3px; overflow: hidden;';

                if (quota.percent !== undefined) {
                    const barInner = document.createElement('div');
                    const color = quota.percent > 90 ? '#ff4757' : (quota.percent > 70 ? '#ffa502' : '#2ed573');
                    barInner.style.cssText = `width: ${quota.percent}%; height: 100%; background: ${color};`;
                    bar.appendChild(barInner);
                }

                // 额度文字
                const text = document.createElement('span');
                text.style.cssText = 'min-width: 100px; font-size: 11px; color: var(--text-primary); text-align: right;';
                if (quota.error) {
                    text.textContent = `❌ 错误`;
                    text.style.color = '#ff6b6b';
                } else if (quota.remaining !== undefined) {
                    text.textContent = `剩余: ${quota.remaining.toLocaleString()}`;
                } else {
                    text.textContent = '--';
                }

                // 操作按钮
                const actions = document.createElement('div');
                actions.style.cssText = 'display: flex; gap: 4px;';

                const toggleBtn = document.createElement('button');
                toggleBtn.className = 'btn btn-secondary';
                if (enabled) {
                    toggleBtn.style.cssText = 'padding: 2px 6px; font-size: 10px;';
                    toggleBtn.textContent = '⏸ 停用';
                } else {
                    toggleBtn.style.cssText = 'padding: 2px 6px; font-size: 10px; background: #51cf66; color: #fff;';
                    toggleBtn.textContent = '▶ 启用';
                }
                toggleBtn.onclick = () => toggleKey(idx);

                const deleteBtn = document.createElement('button');
                deleteBtn.className = 'btn btn-secondary';
                deleteBtn.style.cssText = 'padding: 2px 6px; font-size: 10px; color: #ff6b6b;';
                deleteBtn.textContent = '🗑 删除';
                deleteBtn.onclick = () => deleteKey(idx);

                actions.appendChild(toggleBtn);
                actions.appendChild(deleteBtn);

                item.appendChild(orderBtns);
                item.appendChild(label);
                item.appendChild(bar);
                item.appendChild(text);
                item.appendChild(actions);
                list.appendChild(item);
            });
        } else {
            list.innerHTML = '<div style="text-align: center; color: var(--text-secondary);">没有配置 API Key</div>';
        }
    } catch (error) {
        list.innerHTML = `<div style="text-align: center; color: #ff6b6b;">加载失败: ${error.message}</div>`;
    }
}

// 切换 Key 启用/停用
async function toggleKey(index) {
    try {
        const response = await fetch(`${API_BASE}/settings/elevenlabs/keys`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'toggle', index })
        });
        const result = await response.json();
        if (response.ok) {
            showToast(result.enabled ? 'Key 已启用' : 'Key 已停用', 'success');
            loadAllQuotas();
        } else {
            showToast('操作失败: ' + result.error, 'error');
        }
    } catch (error) {
        showToast('请求失败: ' + error.message, 'error');
    }
}

// 删除 Key
async function deleteKey(index) {
    if (!confirm('确定要删除这个 API Key 吗？')) return;

    try {
        const response = await fetch(`${API_BASE}/settings/elevenlabs/keys`, {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ index })
        });
        const result = await response.json();
        if (response.ok) {
            showToast('Key 已删除', 'success');
            loadAllQuotas();
            loadSettings(true);
        } else {
            showToast('删除失败: ' + result.error, 'error');
        }
    } catch (error) {
        showToast('请求失败: ' + error.message, 'error');
    }
}

// 移动 Key 顺序
async function moveKey(fromIndex, toIndex) {
    try {
        const response = await fetch(`${API_BASE}/settings/elevenlabs/keys`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'move', from: fromIndex, to: toIndex })
        });
        const result = await response.json();
        if (response.ok) {
            loadAllQuotas();
        } else {
            showToast('移动失败: ' + result.error, 'error');
        }
    } catch (error) {
        showToast('请求失败: ' + error.message, 'error');
    }
}

async function previewVoice() {
    const select = document.getElementById('voice-select');
    const selectedOption = select.options[select.selectedIndex];

    if (!selectedOption || !selectedOption.value) {
        showToast('请先选择一个语音', 'error');
        return;
    }

    const previewUrl = selectedOption.dataset.previewUrl;

    if (!previewUrl) {
        showToast('该声音没有提供预览样本', 'info');
        return;
    }

    updateElevenLabsStatus('正在试听...');
    audioPlayer.src = previewUrl;
    audioPlayer.play();
    document.getElementById('btn-play').disabled = false;
    document.getElementById('btn-play').textContent = '⏸ 暂停';
}

async function generateTTS() {
    const text = document.getElementById('tts-text')?.value?.trim();
    const voiceId = document.getElementById('voice-select')?.value;
    const modelId = document.getElementById('model-select')?.value || 'eleven_v3';
    const savePath = document.getElementById('tts-save-path')?.value?.trim() || '';

    if (!text) {
        showToast('请输入要转换的文本', 'error');
        return;
    }

    if (!voiceId) {
        showToast('请先选择一个语音', 'error');
        return;
    }

    updateElevenLabsStatus('生成中...');

    try {
        const response = await fetch(`${API_BASE}/elevenlabs/tts`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: text,
                voice_id: voiceId,
                model_id: modelId,
                stability: parseInt(document.getElementById('tts-stability')?.value || 50) / 100,
                save_path: savePath
            })
        });

        const result = await response.json();

        if (response.ok) {
            updateElevenLabsStatus('生成成功');
            showToast('语音生成成功！', 'success');

            // 加载生成的音频
            currentAudioPath_elevenlabs = result.file_path;
            audioPlayer.src = `file://${result.file_path}`;
            document.getElementById('btn-play').disabled = false;
            document.getElementById('seek-slider').disabled = false;

            // 刷新额度
            loadQuota();

            // 自动更新保存路径
            document.getElementById('tts-save-path').value = '';
        } else {
            updateElevenLabsStatus('生成失败');
            showToast('错误: ' + result.error, 'error');
        }
    } catch (error) {
        updateElevenLabsStatus('生成失败');
        showToast('请求失败: ' + error.message, 'error');
    }
}

function copyVoiceOptions(sourceSelect, targetSelect, preferredValue = '') {
    if (!targetSelect) return;

    const currentValue = preferredValue || targetSelect.value;
    targetSelect.innerHTML = '';

    if (!sourceSelect || sourceSelect.options.length === 0) {
        const option = document.createElement('option');
        option.value = '';
        option.textContent = '请先刷新语音...';
        targetSelect.appendChild(option);
        return;
    }

    Array.from(sourceSelect.options).forEach(option => {
        const cloned = option.cloneNode(true);
        targetSelect.appendChild(cloned);
    });

    if (currentValue) {
        targetSelect.value = currentValue;
    }

    if (!targetSelect.value && targetSelect.options.length > 0) {
        targetSelect.selectedIndex = 0;
    }
}

function syncBatchVoiceOptions() {
    const sourceSelect = document.getElementById('voice-select');
    if (!sourceSelect) return;

    const globalSelect = document.getElementById('tts-batch-voice');
    const globalFallback = (globalSelect && globalSelect.value) || sourceSelect.value;

    if (globalSelect) {
        copyVoiceOptions(sourceSelect, globalSelect, globalFallback);
    }

    const rowSelects = document.querySelectorAll('.batch-voice-select');
    rowSelects.forEach(select => {
        const fallback = select.value || (globalSelect ? globalSelect.value : sourceSelect.value);
        copyVoiceOptions(sourceSelect, select, fallback);
    });

    updateBatchVoiceMode();
}

function applyBatchVoiceToRows(voiceId) {
    if (!voiceId) return;
    const rows = document.querySelectorAll('.batch-row');
    rows.forEach(row => {
        const select = row.querySelector('.batch-voice-select');
        if (select) {
            select.value = voiceId;
        }
    });
}

function updateBatchVoiceMode() {
    const useSameCheckbox = document.getElementById('tts-batch-use-same');
    const globalSelect = document.getElementById('tts-batch-voice');
    if (!useSameCheckbox || !globalSelect) return;

    const useSame = useSameCheckbox.checked;
    globalSelect.disabled = !useSame;

    const globalVoice = globalSelect.value || document.getElementById('voice-select')?.value || '';
    const rows = document.querySelectorAll('.batch-row');

    rows.forEach(row => {
        const select = row.querySelector('.batch-voice-select');
        if (!select) return;

        if (useSame) {
            if (row.dataset.prevVoice === undefined) {
                row.dataset.prevVoice = select.value;
            }
            if (globalVoice) {
                select.value = globalVoice;
            }
        } else if (row.dataset.prevVoice !== undefined) {
            select.value = row.dataset.prevVoice;
            delete row.dataset.prevVoice;
        }

        select.disabled = useSame;
    });
}

function addBatchRow(initialText = '', initialVoiceId = '') {
    const list = document.getElementById('tts-batch-list');
    if (!list) return;

    const row = document.createElement('div');
    row.className = 'batch-row';

    const voiceSelect = document.createElement('select');
    voiceSelect.className = 'select batch-voice-select';
    voiceSelect.dataset.initialVoiceId = initialVoiceId;  // 保存初始 Voice ID

    const textArea = document.createElement('textarea');
    textArea.className = 'textarea batch-text';
    textArea.rows = 3;
    textArea.placeholder = '输入文本...';
    textArea.value = initialText;

    const removeButton = document.createElement('button');
    removeButton.type = 'button';
    removeButton.className = 'btn btn-secondary batch-remove';
    removeButton.textContent = '删除';
    removeButton.addEventListener('click', () => {
        row.remove();
        if (list.children.length === 0) {
            addBatchRow();
        } else {
            updateBatchVoiceMode();
        }
    });

    row.appendChild(voiceSelect);
    row.appendChild(textArea);
    row.appendChild(removeButton);
    list.appendChild(row);

    syncBatchVoiceOptions();
    updateBatchVoiceMode();

    // 如果有初始 Voice ID，设置选中
    if (initialVoiceId) {
        setTimeout(() => {
            // 尝试选中对应的 voice
            const options = voiceSelect.querySelectorAll('option');
            for (const opt of options) {
                if (opt.value === initialVoiceId) {
                    voiceSelect.value = initialVoiceId;
                    break;
                }
            }
        }, 100);
    }
}

function clearBatchRows() {
    const list = document.getElementById('tts-batch-list');
    if (!list) return;
    list.innerHTML = '';
    showToast('已清空', 'info');
}

// 从剪贴板批量粘贴（支持 Google 表格/Excel）
// 格式：文案 | Voice ID（可选）
async function batchPasteFromClipboard() {
    try {
        const clipboardItems = await navigator.clipboard.read();
        let rows = [];  // 存储 {text, voiceId} 对象

        for (const item of clipboardItems) {
            console.log('剪贴板类型:', item.types);

            // 尝试读取 HTML 格式（表格）- 按行解析
            if (item.types.includes('text/html')) {
                const blob = await item.getType('text/html');
                const html = await blob.text();
                console.log('HTML 内容:', html.substring(0, 500));

                const parser = new DOMParser();
                const doc = parser.parseFromString(html, 'text/html');
                const tableRows = doc.querySelectorAll('tr');

                if (tableRows.length > 0) {
                    tableRows.forEach(tr => {
                        const cells = tr.querySelectorAll('td, th');
                        if (cells.length >= 1) {
                            const text = cells[0]?.textContent.trim() || '';
                            const voiceId = cells[1]?.textContent.trim() || '';
                            if (text) {
                                rows.push({ text, voiceId: isVoiceId(voiceId) ? voiceId : '' });
                            }
                        }
                    });
                }
            }

            // 如果没有 HTML 或没有提取到内容，尝试纯文本
            if (rows.length === 0 && item.types.includes('text/plain')) {
                const blob = await item.getType('text/plain');
                const text = await blob.text();
                console.log('纯文本内容:', text.substring(0, 500));

                // Google 表格用 \n 分隔行，\t 分隔列
                const lines = text.split('\n');
                lines.forEach(line => {
                    if (!line.trim()) return;
                    const cells = line.split('\t');
                    const textContent = cells[0]?.trim() || '';
                    const voiceId = cells[1]?.trim() || '';
                    if (textContent) {
                        rows.push({ text: textContent, voiceId: isVoiceId(voiceId) ? voiceId : '' });
                    }
                });
            }
        }

        console.log('解析到的任务:', rows);

        if (rows.length === 0) {
            showToast('剪贴板没有有效内容', 'error');
            return;
        }

        // 清空现有内容
        const list = document.getElementById('tts-batch-list');
        if (!list) return;
        list.innerHTML = '';

        // 统计有多少条指定了 Voice ID
        let withVoiceId = 0;

        // 添加新行
        rows.forEach(row => {
            addBatchRow(row.text, row.voiceId);
            if (row.voiceId) withVoiceId++;
        });

        let msg = `已添加 ${rows.length} 条文案`;
        if (withVoiceId > 0) {
            msg += `，其中 ${withVoiceId} 条指定了 Voice ID`;
        }
        showToast(msg, 'success');
    } catch (error) {
        console.error('粘贴失败:', error);
        showToast('粘贴失败: ' + error.message, 'error');
    }
}

// 判断是否是有效的 Voice ID（ElevenLabs Voice ID 通常是 21 位字符）
function isVoiceId(str) {
    if (!str) return false;
    // ElevenLabs Voice ID 格式：21位字母数字组合
    // 例如：JBFqnCBsd6RMkjVDRZzb
    return /^[a-zA-Z0-9]{10,30}$/.test(str);
}

function initBatchTTS() {
    const list = document.getElementById('tts-batch-list');
    if (!list || list.dataset.initialized === 'true') return;
    list.dataset.initialized = 'true';

    const addButton = document.getElementById('tts-batch-add');
    const clearButton = document.getElementById('tts-batch-clear');
    const generateButton = document.getElementById('tts-batch-generate');
    const useSameCheckbox = document.getElementById('tts-batch-use-same');
    const globalSelect = document.getElementById('tts-batch-voice');

    if (addButton) {
        addButton.addEventListener('click', addBatchRow);
    }

    const pasteButton = document.getElementById('tts-batch-paste');
    if (pasteButton) {
        pasteButton.addEventListener('click', batchPasteFromClipboard);
    }

    if (clearButton) {
        clearButton.addEventListener('click', clearBatchRows);
    }
    if (generateButton) {
        generateButton.addEventListener('click', generateTTSBatch);
    }

    if (useSameCheckbox) {
        useSameCheckbox.addEventListener('change', () => {
            updateBatchVoiceMode();
        });
    }

    if (globalSelect) {
        globalSelect.addEventListener('change', () => {
            if (useSameCheckbox && useSameCheckbox.checked) {
                applyBatchVoiceToRows(globalSelect.value);
            }
        });
    }

    if (list.children.length === 0) {
        addBatchRow();
    } else {
        syncBatchVoiceOptions();
    }
}

async function generateTTSBatch() {
    const list = document.getElementById('tts-batch-list');
    const rows = list ? Array.from(list.querySelectorAll('.batch-row')) : [];
    const generateBtn = document.getElementById('tts-batch-generate');

    if (rows.length === 0) {
        showToast('请先添加文本', 'error');
        return;
    }

    const useSame = document.getElementById('tts-batch-use-same')?.checked;
    const globalVoice = document.getElementById('tts-batch-voice')?.value;
    const modelId = document.getElementById('model-select')?.value || 'eleven_v3';

    if (useSame && !globalVoice) {
        showToast('请选择语音', 'error');
        return;
    }

    // 收集任务
    const tasks = [];
    for (let i = 0; i < rows.length; i++) {
        const row = rows[i];
        const text = row.querySelector('.batch-text')?.value.trim();
        if (!text) continue;

        const voiceSelect = row.querySelector('.batch-voice-select');
        const voiceId = useSame ? globalVoice : voiceSelect?.value;

        if (!voiceId) {
            showToast(`第 ${i + 1} 条未选择语音`, 'error');
            return;
        }

        tasks.push({
            rowIndex: i,
            row: row,
            text: text,
            voice_id: voiceId,
            model_id: modelId,
            seq_num: tasks.length + 1  // 序号
        });
    }

    if (tasks.length === 0) {
        showToast('请先输入要生成的文本', 'error');
        return;
    }

    // 获取启用的 Key 数量用于并行
    let enabledKeyCount = 1;
    try {
        const keysResponse = await fetch(`${API_BASE}/settings/elevenlabs/keys`);
        const keysData = await keysResponse.json();
        enabledKeyCount = (keysData.keys || []).filter(k => k.enabled !== false).length || 1;
    } catch (e) {
        console.log('获取 Key 数量失败，使用默认并行数 1');
    }

    const concurrency = Math.min(enabledKeyCount, tasks.length);
    console.log(`ElevenLabs 并行数: ${concurrency}, 启用 Key 数: ${enabledKeyCount}`);

    // 更新按钮状态
    const originalText = generateBtn.textContent;
    generateBtn.textContent = '⏳ 生成中...';
    generateBtn.disabled = true;
    generateBtn.style.opacity = '0.6';

    let successCount = 0;
    let failCount = 0;
    let processedCount = 0;
    const totalTasks = tasks.length;

    updateElevenLabsStatus(`批量生成中 (0/${totalTasks})，并行: ${concurrency}...`);

    // 处理单个任务
    async function processTask(task, keyIndex) {
        const { row, text, voice_id, model_id, seq_num, rowIndex } = task;

        // 更新行状态
        let statusSpan = row.querySelector('.batch-status');
        if (!statusSpan) {
            statusSpan = document.createElement('span');
            statusSpan.className = 'batch-status';
            statusSpan.style.cssText = 'font-size: 12px; margin-left: 8px; padding: 2px 6px; border-radius: 4px;';
            row.appendChild(statusSpan);
        }
        statusSpan.textContent = '⏳ 生成中...';
        statusSpan.style.background = 'rgba(255,165,0,0.2)';
        statusSpan.style.color = '#ffa500';

        try {
            const enableCircuitBreaker = document.getElementById('tts-circuit-breaker')?.checked || false;
            const response = await fetch(`${API_BASE}/elevenlabs/tts-batch`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    items: [{
                        text,
                        voice_id,
                        model_id,
                        seq_num,
                        key_index: keyIndex  // 指定使用哪个 Key
                    }],
                    default_model_id: model_id,
                    output_format: 'mp3_44100_128',
                    enable_circuit_breaker: enableCircuitBreaker
                })
            });

            const result = await response.json();
            const r = result.results?.[0];
            processedCount++;

            if (r && !r.error) {
                successCount++;
                statusSpan.textContent = '✅ 成功';
                statusSpan.style.background = 'rgba(0,255,0,0.2)';
                statusSpan.style.color = '#51cf66';
                row.dataset.failed = 'false';
                row.dataset.filePath = r.file_path || '';

                // 移除重试按钮
                const retryBtn = row.querySelector('.batch-retry');
                if (retryBtn) retryBtn.remove();

                return { success: true, file_path: r.file_path };
            } else {
                failCount++;
                statusSpan.textContent = `❌ ${(r?.error || '未知错误').substring(0, 20)}`;
                statusSpan.style.background = 'rgba(255,0,0,0.2)';
                statusSpan.style.color = '#ff6b6b';
                row.dataset.failed = 'true';
                row.dataset.error = r?.error || '未知错误';

                // 添加重试按钮
                if (!row.querySelector('.batch-retry')) {
                    const retryBtn = document.createElement('button');
                    retryBtn.className = 'btn btn-secondary batch-retry';
                    retryBtn.style.cssText = 'padding: 4px 8px; font-size: 11px; margin-left: 4px;';
                    retryBtn.textContent = '🔄 重试';
                    retryBtn.onclick = () => retrySingleBatch(row);
                    row.appendChild(retryBtn);
                }

                return { success: false };
            }
        } catch (error) {
            processedCount++;
            failCount++;
            statusSpan.textContent = `❌ ${error.message.substring(0, 20)}`;
            statusSpan.style.background = 'rgba(255,0,0,0.2)';
            statusSpan.style.color = '#ff6b6b';
            row.dataset.failed = 'true';
            row.dataset.error = error.message;
            return { success: false };
        }
    }

    // 并行执行
    const taskQueue = [...tasks];
    const runningTasks = [];
    const successResults = [];

    async function runParallel() {
        while (taskQueue.length > 0 || runningTasks.length > 0) {
            // 启动新任务
            while (runningTasks.length < concurrency && taskQueue.length > 0) {
                const task = taskQueue.shift();
                const keyIndex = runningTasks.length;
                const promise = processTask(task, keyIndex).then(result => {
                    const idx = runningTasks.indexOf(promise);
                    if (idx > -1) runningTasks.splice(idx, 1);
                    if (result.success && result.file_path) {
                        successResults.push(result);
                    }
                    return result;
                });
                runningTasks.push(promise);
            }

            if (runningTasks.length > 0) {
                await Promise.race(runningTasks);
            }

            updateElevenLabsStatus(`批量生成中 (${processedCount}/${totalTasks})...`);
        }
    }

    await runParallel();

    // 完成
    generateBtn.textContent = originalText;
    generateBtn.disabled = false;
    generateBtn.style.opacity = '1';

    loadQuota();

    // 自动下载成功的文件
    if (successResults.length > 0) {
        showToast(`正在下载 ${successResults.length} 个文件...`, 'info');
        for (const r of successResults) {
            const filename = r.file_path.split('/').pop();
            const link = document.createElement('a');
            link.href = `${API_BASE}/file/download?path=${encodeURIComponent(r.file_path)}`;
            link.download = filename;
            link.click();
            await new Promise(resolve => setTimeout(resolve, 500));
        }
    }

    if (failCount > 0) {
        updateElevenLabsStatus(`完成: ${successCount} 成功, ${failCount} 失败`);
        showToast(`成功 ${successCount} 个，失败 ${failCount} 个（可点击重试）`, 'warning');
        showRetryAllFailedButton();
    } else {
        updateElevenLabsStatus(`批量完成: ${successCount} 个成功`);
        showToast(`全部成功: ${successCount} 个`, 'success');
    }
}

// 显示"重试所有失败"按钮
function showRetryAllFailedButton() {
    const container = document.querySelector('#tts-batch-list');
    if (!container) return;

    // 移除旧的重试按钮
    const oldBtn = document.getElementById('retry-all-failed-btn');
    if (oldBtn) oldBtn.remove();

    const btn = document.createElement('button');
    btn.id = 'retry-all-failed-btn';
    btn.className = 'btn btn-primary';
    btn.style.cssText = 'margin-top: 12px; width: 100%;';
    btn.textContent = '🔄 重试所有失败项';
    btn.onclick = retryAllFailed;

    container.parentNode.insertBefore(btn, container.nextSibling);
}

// 重试单个失败项
async function retrySingleBatch(row) {
    const text = row.querySelector('.batch-text')?.value?.trim();
    const voiceSelect = row.querySelector('.batch-voice-select');
    const useSame = document.getElementById('tts-batch-use-same')?.checked;
    const globalVoice = document.getElementById('tts-batch-voice')?.value;
    const modelId = document.getElementById('model-select')?.value || 'eleven_v3';
    const voiceId = useSame ? globalVoice : voiceSelect?.value;

    if (!text || !voiceId) {
        showToast('缺少文本或语音', 'error');
        return;
    }

    const statusSpan = row.querySelector('.batch-status');
    if (statusSpan) {
        statusSpan.textContent = '⏳ 重试中...';
        statusSpan.style.background = 'rgba(255,165,0,0.2)';
        statusSpan.style.color = '#ffa500';
    }

    try {
        const response = await fetch(`${API_BASE}/elevenlabs/tts-batch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                items: [{ text, voice_id: voiceId, model_id: modelId }],
                default_model_id: modelId,
                output_format: 'mp3_44100_128'
            })
        });

        const result = await response.json();
        const r = result.results?.[0];

        if (r && !r.error) {
            statusSpan.textContent = '✅ 成功';
            statusSpan.style.background = 'rgba(0,255,0,0.2)';
            statusSpan.style.color = '#51cf66';
            row.dataset.failed = 'false';

            // 移除重试按钮
            const retryBtn = row.querySelector('.batch-retry');
            if (retryBtn) retryBtn.remove();

            // 下载文件
            if (r.file_path) {
                const filename = r.file_path.split('/').pop();
                const link = document.createElement('a');
                link.href = `${API_BASE}/file/download?path=${encodeURIComponent(r.file_path)}`;
                link.download = filename;
                link.click();
            }

            showToast('重试成功', 'success');
            loadQuota();

            // 检查是否还有失败项
            const failedRows = document.querySelectorAll('.batch-row[data-failed="true"]');
            if (failedRows.length === 0) {
                const retryAllBtn = document.getElementById('retry-all-failed-btn');
                if (retryAllBtn) retryAllBtn.remove();
            }
        } else {
            statusSpan.textContent = `❌ ${(r?.error || '未知错误').substring(0, 30)}...`;
            statusSpan.style.background = 'rgba(255,0,0,0.2)';
            statusSpan.style.color = '#ff6b6b';
            showToast('重试失败: ' + (r?.error || '未知错误'), 'error');
        }
    } catch (error) {
        statusSpan.textContent = '❌ 请求失败';
        showToast('重试失败: ' + error.message, 'error');
    }
}

// 重试所有失败项
async function retryAllFailed() {
    const failedRows = document.querySelectorAll('.batch-row[data-failed="true"]');
    if (failedRows.length === 0) {
        showToast('没有失败项需要重试', 'info');
        return;
    }

    showToast(`正在重试 ${failedRows.length} 个失败项...`, 'info');

    for (const row of failedRows) {
        await retrySingleBatch(row);
        await new Promise(resolve => setTimeout(resolve, 1500)); // 间隔 1.5 秒
    }

    loadQuota();
}

async function generateSFX() {
    const prompt = document.getElementById('sfx-prompt').value.trim();
    const duration = parseInt(document.getElementById('sfx-duration').value);
    const savePath = document.getElementById('sfx-save-path').value.trim();

    if (!prompt) {
        showToast('请输入音效描述', 'error');
        return;
    }

    updateElevenLabsStatus('生成音效中...');

    try {
        const response = await fetch(`${API_BASE}/elevenlabs/sfx`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: prompt,
                duration: duration,
                save_path: savePath
            })
        });

        const result = await response.json();

        if (response.ok) {
            updateElevenLabsStatus('音效生成成功');
            showToast('音效生成成功！', 'success');

            // 加载生成的音频
            currentAudioPath_elevenlabs = result.file_path;
            audioPlayer.src = `file://${result.file_path}`;
            document.getElementById('btn-play').disabled = false;
            document.getElementById('seek-slider').disabled = false;

            // 刷新额度
            loadQuota();

            // 自动更新保存路径
            document.getElementById('sfx-save-path').value = '';
        } else {
            updateElevenLabsStatus('生成失败');
            showToast('错误: ' + result.error, 'error');
        }
    } catch (error) {
        updateElevenLabsStatus('生成失败');
        showToast('请求失败: ' + error.message, 'error');
    }
}

function browseTtsSavePath() {
    const path = prompt('请输入 TTS 保存路径 (留空使用默认):');
    if (path !== null) {
        document.getElementById('tts-save-path').value = path;
    }
}

function browseSfxSavePath() {
    const path = prompt('请输入 SFX 保存路径 (留空使用默认):');
    if (path !== null) {
        document.getElementById('sfx-save-path').value = path;
    }
}

function updateElevenLabsStatus(text) {
    const statusEl = document.getElementById('elevenlabs-status');
    if (statusEl) {
        statusEl.textContent = text;
    }
}

// ==================== 视频下载功能 ====================

// 视频下载状态
let videoListData = [];
let isDownloading = false;

async function analyzeVideoUrl() {
    const url = document.getElementById('video-url').value.trim();

    if (!url) {
        showToast('请输入视频链接', 'error');
        return;
    }

    const btnAnalyze = document.getElementById('btn-analyze');
    btnAnalyze.disabled = true;
    btnAnalyze.textContent = '解析中...';
    updateStatus('正在解析链接信息...', 'processing', 'download-status');

    // 重置列表
    videoListData = [];
    document.getElementById('video-table-body').innerHTML = '';
    document.getElementById('video-list-section').style.display = 'none';

    try {
        const response = await fetch(`${API_BASE}/video/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url })
        });

        const data = await response.json();

        if (response.ok) {
            // 处理播放列表或单个视频
            const entries = data.entries || [data];
            videoListData = entries;

            displayVideoList(entries);
            document.getElementById('video-list-section').style.display = 'block';
            document.getElementById('video-count').textContent = `共 ${entries.length} 个视频`;

            updateStatus('解析完成', 'success', 'download-status');
            showToast(`解析完成，共 ${entries.length} 个视频`, 'success');
        } else {
            updateStatus('错误: ' + data.error, 'error', 'download-status');
            showToast('错误: ' + data.error, 'error');
        }
    } catch (error) {
        updateStatus('请求失败: ' + error.message, 'error', 'download-status');
        showToast('请求失败', 'error');
    } finally {
        btnAnalyze.disabled = false;
        btnAnalyze.textContent = '🔍 解析链接';
    }
}

function displayVideoList(entries) {
    const tbody = document.getElementById('video-table-body');
    tbody.innerHTML = '';

    entries.forEach((entry, index) => {
        const tr = document.createElement('tr');
        tr.style.borderBottom = '1px solid var(--border-color)';

        // 复选框
        const tdCheck = document.createElement('td');
        tdCheck.style.padding = '8px';
        tdCheck.style.textAlign = 'center';
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = true;
        checkbox.dataset.index = index;
        checkbox.className = 'video-checkbox';
        tdCheck.appendChild(checkbox);
        tr.appendChild(tdCheck);

        // 标题
        const tdTitle = document.createElement('td');
        tdTitle.style.padding = '8px';
        tdTitle.textContent = truncateText(entry.title || 'Unknown', 50);
        tdTitle.title = entry.title || '';
        tr.appendChild(tdTitle);

        // 时长
        const tdDuration = document.createElement('td');
        tdDuration.style.padding = '8px';
        tdDuration.style.textAlign = 'center';
        const dur = entry.duration;
        tdDuration.textContent = dur ? `${Math.floor(dur / 60)}:${(dur % 60).toString().padStart(2, '0')}` : '--:--';
        tr.appendChild(tdDuration);

        // 状态
        const tdStatus = document.createElement('td');
        tdStatus.style.padding = '8px';
        tdStatus.style.textAlign = 'center';
        tdStatus.id = `video-status-${index}`;
        tdStatus.textContent = '待下载';
        tr.appendChild(tdStatus);

        tbody.appendChild(tr);
    });
}

function truncateText(text, maxLen) {
    if (!text) return '';
    return text.length <= maxLen ? text : text.substring(0, maxLen - 1) + '…';
}

function toggleSelectAllVideos() {
    const selectAll = document.getElementById('select-all-videos').checked;
    document.querySelectorAll('.video-checkbox').forEach(cb => {
        cb.checked = selectAll;
    });
}

function toggleAudioOnly() {
    const audioOnly = document.getElementById('audio-only').checked;
    const formatSelect = document.getElementById('video-format');
    const qualitySelect = document.getElementById('video-quality');
    const subtitleCheckbox = document.getElementById('download-subtitle');

    formatSelect.innerHTML = '';

    if (audioOnly) {
        formatSelect.innerHTML = `
            <option value="mp3">mp3</option>
            <option value="m4a">m4a</option>
            <option value="wav">wav</option>
        `;
        qualitySelect.disabled = true;
        subtitleCheckbox.disabled = true;
    } else {
        formatSelect.innerHTML = `
            <option value="mp4">mp4</option>
            <option value="mkv">mkv</option>
            <option value="webm">webm</option>
        `;
        qualitySelect.disabled = false;
        subtitleCheckbox.disabled = false;
    }
}

function toggleVideoDownload() {
    if (isDownloading) {
        // TODO: 实现停止下载
        showToast('正在停止下载...', 'info');
        isDownloading = false;
        document.getElementById('btn-download').textContent = '⬇️ 开始下载';
        document.getElementById('btn-download').classList.remove('btn-danger');
    } else {
        startVideoDownload();
    }
}

async function startVideoDownload() {
    // 获取选中的视频
    const selectedVideos = [];
    document.querySelectorAll('.video-checkbox:checked').forEach(cb => {
        const index = parseInt(cb.dataset.index);
        if (videoListData[index]) {
            selectedVideos.push({
                url: videoListData[index].webpage_url || videoListData[index].url,
                title: videoListData[index].title,
                ui_index: index
            });
        }
    });

    if (selectedVideos.length === 0) {
        showToast('请至少选择一个视频', 'error');
        return;
    }

    const downloadDir = document.getElementById('download-dir').value.trim();
    const format = document.getElementById('video-format').value;
    const quality = document.getElementById('video-quality').value;
    const audioOnly = document.getElementById('audio-only').checked;
    const downloadSubtitle = document.getElementById('download-subtitle').checked;
    const subtitleLang = document.getElementById('subtitle-lang').value;
    const threads = parseInt(document.getElementById('download-threads').value) || 4;

    isDownloading = true;
    document.getElementById('btn-download').textContent = '⏹ 停止下载';
    document.getElementById('btn-download').classList.add('btn-danger');
    setIndeterminateProgress('download-progress', true);

    // 重置状态
    selectedVideos.forEach(v => {
        document.getElementById(`video-status-${v.ui_index}`).textContent = '准备中...';
    });

    try {
        updateStatus('下载中...', 'processing', 'download-status');

        const response = await fetch(`${API_BASE}/video/download-batch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                items: selectedVideos,
                options: {
                    audio_only: audioOnly,
                    ext: format,
                    quality: quality,
                    subtitles: downloadSubtitle,
                    sub_lang: subtitleLang,
                    concurrency: threads
                },
                output_dir: downloadDir || ''
            })
        });

        const result = await response.json();

        if (response.ok) {
            updateStatus('下载完成！', 'success', 'download-status');
            showToast('下载完成！', 'success');
            document.getElementById('download-progress-inner').style.width = '100%';

            // 更新每个视频状态
            selectedVideos.forEach(v => {
                document.getElementById(`video-status-${v.ui_index}`).textContent = '完成';
            });
        } else {
            updateStatus('错误: ' + result.error, 'error', 'download-status');
            showToast('错误: ' + result.error, 'error');
        }
    } catch (error) {
        updateStatus('请求失败: ' + error.message, 'error', 'download-status');
        showToast('请求失败', 'error');
    } finally {
        setIndeterminateProgress('download-progress', false);
        isDownloading = false;
        document.getElementById('btn-download').textContent = '⬇️ 开始下载';
        document.getElementById('btn-download').classList.remove('btn-danger');
    }
}

function selectDownloadDir() {
    const dir = prompt('请输入下载目录路径:');
    if (dir) {
        document.getElementById('download-dir').value = dir;
    }
}

// ==================== 设置功能 ====================

async function saveGladiaKeys() {
    const keysText = document.getElementById('gladia-keys').value;
    const keys = keysText.split('\n').filter(k => k.trim());

    try {
        const response = await fetch(`${API_BASE}/settings/gladia-keys`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ keys })
        });

        if (response.ok) {
            showToast('Gladia Keys 已保存！', 'success');
        }
    } catch (error) {
        showToast('保存失败: ' + error.message, 'error');
    }
}

async function saveReplaceRules() {
    const language = document.getElementById('replace-language').value;
    const rules = document.getElementById('replace-rules').value;

    try {
        const response = await fetch(`${API_BASE}/settings/replace-rules`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ language, rules })
        });

        if (response.ok) {
            showToast('替换规则已保存！', 'success');
        }
    } catch (error) {
        showToast('保存失败: ' + error.message, 'error');
    }
}

function showGladiaKeysModal() {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    document.querySelector('[data-tab="settings"]').classList.add('active');
    document.getElementById('settings-panel').classList.add('active');
}

function showTransReplaceModal() {
    showGladiaKeysModal();
}

// ==================== 字幕断行功能 ====================

// 初始化字幕断行滑杆
document.addEventListener('DOMContentLoaded', () => {
    const maxCharsSlider = document.getElementById('subtitle-max-chars');
    if (maxCharsSlider) {
        maxCharsSlider.addEventListener('input', (e) => {
            const value = e.target.value;
            document.getElementById('max-chars-label').textContent = `${value} 字符/行`;

            // 如果有文本，实时重新断行
            const text = document.getElementById('tts-text').value.trim();
            if (text) {
                const cleanText = stripEmotionTags(text);
                doAutoBreak(cleanText, parseInt(value), false);
            }
        });
    }
});

// 去除情绪标签
function stripEmotionTags(text) {
    // 匹配常见的情绪标签格式: [tag], <tag>, (tag)
    let result = text
        .replace(/\[[a-zA-Z_]+\]/g, '')  // [sad], [happy]
        .replace(/<[a-zA-Z_]+>/g, '')    // <sad>, <happy>
        .replace(/\([a-zA-Z_]+\)/g, ''); // (whisper)
    // 清理多余空格
    return result.replace(/\s+/g, ' ').trim();
}

// 自动断行按钮点击
function autoBreakSubtitle() {
    const text = document.getElementById('tts-text').value.trim();
    if (!text) {
        showToast('请先在上方输入要转换的文本', 'error');
        return;
    }

    const cleanText = stripEmotionTags(text);
    const maxChars = parseInt(document.getElementById('subtitle-max-chars').value);
    doAutoBreak(cleanText, maxChars, true);
}

// 执行自动断行核心逻辑
function doAutoBreak(text, maxChars, showMessage = true) {
    // 句末标点符号（强制断行）
    const sentenceEnders = ['.', '!', '?', '。', '！', '？', '；'];
    // 次级断点（超长时可断）
    const softBreaks = [',', '，', ':', '：', ';', ' '];
    // 孤立词阈值
    const orphanThreshold = 8;

    const lines = [];
    let currentLine = '';
    let lastSoftBreak = -1;

    let i = 0;
    while (i < text.length) {
        const char = text[i];
        currentLine += char;

        // 记录次级断点位置
        if (softBreaks.includes(char)) {
            lastSoftBreak = currentLine.length;
        }

        // 检测是否是句末标点
        if (sentenceEnders.includes(char)) {
            // 跳过连续的标点（如 "..." 或 "!?"）
            while (i + 1 < text.length && sentenceEnders.includes(text[i + 1])) {
                i++;
                currentLine += text[i];
            }

            // 跳过引号等收尾标点
            while (i + 1 < text.length && ['"', '"', "'", "'"].includes(text[i + 1])) {
                i++;
                currentLine += text[i];
            }

            lines.push(currentLine.trim());
            currentLine = '';
            lastSoftBreak = -1;
        }
        // 如果行太长，在次级断点处断开
        else if (currentLine.length >= maxChars) {
            if (lastSoftBreak > 10) {
                // 在最后一个次级断点处断开
                const lineToAdd = currentLine.substring(0, lastSoftBreak).trim();
                const remaining = currentLine.substring(lastSoftBreak).trimStart();

                lines.push(lineToAdd);
                currentLine = remaining;
                lastSoftBreak = -1;
            }
            // 如果没有合适的断点，继续累积
        }

        i++;
    }

    // 处理最后剩余的文本
    if (currentLine.trim()) {
        lines.push(currentLine.trim());
    }

    // 智能处理：合并孤立的短片段
    const mergedLines = mergeOrphanWords(lines, orphanThreshold);

    // 设置到字幕输入框
    const result = mergedLines.join('\n');
    document.getElementById('subtitle-text').value = result;

    // 提示（只有手动点击按钮时才显示）
    if (showMessage) {
        updateElevenLabsStatus(`已自动断行为 ${mergedLines.length} 条字幕（每行≤${maxChars}字符）`);
        showToast(`已断行为 ${mergedLines.length} 条字幕`, 'success');
    }
}

// 合并孤立的短片段到前一行
function mergeOrphanWords(lines, threshold) {
    if (lines.length <= 1) return lines;

    const result = [];

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];

        // 检查是否是孤立短片段（开头不是大写字母或中文，且很短）
        if (i > 0 && line.length < threshold) {
            const words = line.split(/\s+/);
            const firstWord = words[0] || '';

            // 检查是否是句子的开头
            const isSentenceStart = (
                line.length > 0 && (
                    /[A-Z]/.test(line[0]) || // 大写字母开头
                    /[\u4e00-\u9fff]/.test(line[0]) || // 中文字符
                    ['I', 'A', 'The', 'An', 'He', 'She', 'It', 'We', 'They', 'You', 'My', 'Your', 'Our', 'His', 'Her'].includes(firstWord)
                )
            );

            if (!isSentenceStart) {
                // 不是句子开头的短片段，合并到前一行
                if (result.length > 0) {
                    result[result.length - 1] = result[result.length - 1] + ' ' + line;
                    continue;
                }
            }
        }

        result.push(line);
    }

    return result;
}

// ==================== 智能分割功能 ====================
let smartSplitSegments = [];
let smartSplitTargetFile = null;

// 初始化智能分割事件
document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('smart-split-analyze-btn');

    if (analyzeBtn) {
        analyzeBtn.onclick = analyzeSmartSplit;
    }
});

// 更新分析按钮状态
function updateSmartSplitButtonState() {
    const btn = document.getElementById('smart-split-analyze-btn');

    if (btn) {
        btn.disabled = currentMediaFileInfos.length === 0;
    }
}

// 分析智能分割点（批量分析所有文件）
async function analyzeSmartSplit() {
    if (currentMediaFileInfos.length === 0) {
        showToast('请先添加音频文件', 'warning');
        return;
    }

    const maxDuration = parseInt(document.getElementById('smart-split-max-duration')?.value) || 29;
    const btn = document.getElementById('smart-split-analyze-btn');
    const preview = document.getElementById('smart-split-preview');

    btn.disabled = true;

    const total = currentMediaFileInfos.length;
    let success = 0;
    let allResults = [];  // 存储所有文件的分析结果

    try {
        for (let i = 0; i < currentMediaFileInfos.length; i++) {
            const fileInfo = currentMediaFileInfos[i];
            if (!fileInfo.file) continue;

            btn.textContent = `⏳ 分析中 (${i + 1}/${total})...`;

            try {
                const formData = new FormData();
                formData.append('audio_file', fileInfo.file);
                formData.append('max_duration', maxDuration.toString());

                const response = await fetch(`${API_BASE}/audio/smart-split-analyze`, {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (response.ok && data.segments) {
                    // 直接应用到对应文件的裁切点输入框
                    const cutPoints = data.segments.slice(1).map(seg => formatTimeAudio(seg.start));
                    const cutPointsStr = cutPoints.join(', ');

                    const input = document.getElementById(`audio-cut-points-${i}`);
                    if (input) {
                        input.value = cutPointsStr;
                        currentAudioCutPoints[fileInfo.path] = cutPointsStr;
                    }

                    // 更新卡片状态
                    const statusEl = document.getElementById(`audio-card-status-${i}`);
                    if (statusEl) {
                        statusEl.textContent = `${data.segments.length} 段`;
                        statusEl.style.background = 'rgba(81, 207, 102, 0.2)';
                        statusEl.style.color = '#51cf66';
                    }

                    // 重绘波形显示分割点
                    const canvas = document.getElementById(`audio-waveform-${i}`);
                    const cardData = window.audioCardData?.[i];
                    if (canvas && cardData) {
                        const cutTimes = data.segments.slice(1).map(seg => seg.start);
                        drawWaveform(canvas, cardData.peaks, cutTimes, data.total_duration);
                    }

                    allResults.push({
                        index: i,
                        name: fileInfo.name,
                        segments: data.segments.length,
                        duration: data.total_duration,
                        cutTimes: data.segments.slice(1).map(seg => seg.start)
                    });

                    success++;
                } else {
                    // 更新卡片状态为失败
                    const statusEl = document.getElementById(`audio-card-status-${i}`);
                    if (statusEl) {
                        statusEl.textContent = '分析失败';
                        statusEl.style.background = 'rgba(255, 107, 107, 0.2)';
                        statusEl.style.color = '#ff6b6b';
                    }
                }
            } catch (err) {
                console.error(`分析失败 ${fileInfo.name}:`, err);
                const statusEl = document.getElementById(`audio-card-status-${i}`);
                if (statusEl) {
                    statusEl.textContent = '出错';
                    statusEl.style.background = 'rgba(255, 107, 107, 0.2)';
                    statusEl.style.color = '#ff6b6b';
                }
            }
        }

        // 显示总结
        if (success > 0) {
            const totalSegments = allResults.reduce((sum, r) => sum + r.segments, 0);
            showToast(`批量分析完成: ${success}/${total} 个文件，共 ${totalSegments} 个分割点`, 'success');

            // 更新工具栏状态文本
            const statusEl = document.getElementById('smart-split-status');
            if (statusEl) {
                statusEl.textContent = `✅ 已分析 ${success} 个文件`;
                statusEl.style.color = '#51cf66';
            }
        } else {
            showToast('分析失败，没有成功处理的文件', 'error');
            const statusEl = document.getElementById('smart-split-status');
            if (statusEl) {
                statusEl.textContent = '❌ 分析失败';
                statusEl.style.color = '#ff6b6b';
            }
        }

    } catch (error) {
        showToast('分析失败: ' + error.message, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = '🔍 批量分析分割点';
    }
}

// 渲染分割点列表
function renderSmartSplitSegments() {
    const container = document.getElementById('smart-split-segments');
    if (!container) return;

    container.innerHTML = smartSplitSegments.map((seg, idx) => `
        <div class="smart-split-segment" style="display: flex; align-items: center; gap: 8px; padding: 6px; margin-bottom: 4px; background: rgba(255,255,255,0.05); border-radius: 4px;">
            <span style="font-size: 12px; color: var(--text-muted); min-width: 24px;">#${seg.index}</span>
            <input type="text" class="smart-split-start" data-idx="${idx}" value="${formatTimeAudio(seg.start)}" 
                style="width: 70px; padding: 2px 6px; font-size: 11px; border-radius: 3px; border: 1px solid rgba(255,255,255,0.1); background: var(--bg-tertiary); color: var(--text-primary);">
            <span style="color: var(--text-muted);">-</span>
            <input type="text" class="smart-split-end" data-idx="${idx}" value="${formatTimeAudio(seg.end)}"
                style="width: 70px; padding: 2px 6px; font-size: 11px; border-radius: 3px; border: 1px solid rgba(255,255,255,0.1); background: var(--bg-tertiary); color: var(--text-primary);">
            <span style="font-size: 11px; color: var(--text-muted);">(${seg.duration.toFixed(1)}s)</span>
            <button class="btn btn-secondary" onclick="deleteSmartSplitSegment(${idx})" style="padding: 2px 6px; font-size: 10px; color: #ff6b6b;">✕</button>
        </div>
    `).join('');

    // 添加输入事件监听
    container.querySelectorAll('.smart-split-start, .smart-split-end').forEach(input => {
        input.onchange = updateSmartSplitFromInput;
    });
}

// 从输入更新分割点
function updateSmartSplitFromInput(e) {
    const idx = parseInt(e.target.dataset.idx);
    const isStart = e.target.classList.contains('smart-split-start');
    const timeValue = parseTimeInput(e.target.value);

    if (idx >= 0 && idx < smartSplitSegments.length) {
        if (isStart) {
            smartSplitSegments[idx].start = timeValue;
        } else {
            smartSplitSegments[idx].end = timeValue;
        }
        smartSplitSegments[idx].duration = smartSplitSegments[idx].end - smartSplitSegments[idx].start;
    }
}

// 解析时间输入 (mm:ss.s 或 s.ss)
function parseTimeInput(str) {
    str = str.trim();
    if (str.includes(':')) {
        const parts = str.split(':');
        const mins = parseInt(parts[0]) || 0;
        const secs = parseFloat(parts[1]) || 0;
        return mins * 60 + secs;
    } else {
        return parseFloat(str) || 0;
    }
}

// 删除分割点
function deleteSmartSplitSegment(idx) {
    if (idx >= 0 && idx < smartSplitSegments.length) {
        // 如果删除中间的，合并到前一个
        if (idx > 0 && idx < smartSplitSegments.length - 1) {
            smartSplitSegments[idx - 1].end = smartSplitSegments[idx].end;
            smartSplitSegments[idx - 1].duration = smartSplitSegments[idx - 1].end - smartSplitSegments[idx - 1].start;
        }
        smartSplitSegments.splice(idx, 1);
        // 重新编号
        smartSplitSegments.forEach((seg, i) => seg.index = i + 1);
        renderSmartSplitSegments();
    }
}

// 添加分割点
function addSmartSplitPoint() {
    if (smartSplitSegments.length === 0) {
        showToast('请先分析分割点', 'warning');
        return;
    }

    const lastSeg = smartSplitSegments[smartSplitSegments.length - 1];
    const midPoint = (lastSeg.start + lastSeg.end) / 2;

    // 在最后一个片段中间添加分割点
    const newSeg = {
        index: smartSplitSegments.length + 1,
        start: midPoint,
        end: lastSeg.end,
        duration: lastSeg.end - midPoint
    };

    lastSeg.end = midPoint;
    lastSeg.duration = lastSeg.end - lastSeg.start;

    smartSplitSegments.push(newSeg);
    renderSmartSplitSegments();
}

// 应用智能分割点到裁切输入框
function applySmartSplitPoints() {
    if (!smartSplitTargetFile || smartSplitSegments.length === 0) {
        showToast('没有可应用的分割点', 'warning');
        return;
    }

    // 生成裁切点时间（只需要起始时间，不包括第一个0）
    const cutPoints = smartSplitSegments.slice(1).map(seg => formatTimeAudio(seg.start));
    const cutPointsStr = cutPoints.join(', ');

    // 找到对应文件的输入框
    const fileIdx = currentMediaFileInfos.findIndex(f => f.path === smartSplitTargetFile);
    if (fileIdx === -1) {
        showToast('找不到对应文件', 'error');
        return;
    }

    const input = document.getElementById(`audio-cut-points-${fileIdx}`);
    if (input) {
        input.value = cutPointsStr;
        currentAudioCutPoints[smartSplitTargetFile] = cutPointsStr;
        showToast(`已应用 ${smartSplitSegments.length} 个分割片段`, 'success');
    }
}

// ==================== 场景检测模块（批量） ====================

let sceneFiles = [];         // [{path, name}]
let sceneResults = {};       // { filePath: { data, segments } }
let sceneOutputDir = '';

// 初始化场景检测
document.addEventListener('DOMContentLoaded', () => {
    const sceneInput = document.getElementById('scene-video-input');
    if (sceneInput) {
        sceneInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                const newFiles = Array.from(e.target.files).map(f => ({
                    path: f.path || f.name,
                    name: f.name
                }));
                // 合并去重
                newFiles.forEach(nf => {
                    if (!sceneFiles.find(sf => sf.path === nf.path)) {
                        sceneFiles.push(nf);
                    }
                });
                updateSceneFileDisplay();
                renderSceneFileCards();
                showToast(`已添加 ${newFiles.length} 个文件，共 ${sceneFiles.length} 个`, 'success');
            }
        });
    }

    // 拖拽支持
    const dropZone = document.getElementById('scene-drop-zone');
    if (dropZone) {
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.style.borderColor = 'var(--accent)';
            dropZone.style.background = 'rgba(102, 126, 234, 0.05)';
        });
        dropZone.addEventListener('dragleave', () => {
            dropZone.style.borderColor = 'var(--border-color)';
            dropZone.style.background = '';
        });
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.style.borderColor = 'var(--border-color)';
            dropZone.style.background = '';
            const videoExts = ['.mp4', '.mov', '.mkv', '.avi', '.wmv', '.flv', '.webm', '.m4v'];
            const files = Array.from(e.dataTransfer.files).filter(f =>
                videoExts.some(ext => f.name.toLowerCase().endsWith(ext))
            );
            if (files.length > 0) {
                files.forEach(f => {
                    const info = { path: f.path || f.name, name: f.name };
                    if (!sceneFiles.find(sf => sf.path === info.path)) {
                        sceneFiles.push(info);
                    }
                });
                updateSceneFileDisplay();
                renderSceneFileCards();
                showToast(`已添加 ${files.length} 个文件`, 'success');
            }
        });
    }
});

function updateSceneFileDisplay() {
    const pathEl = document.getElementById('scene-video-path');
    if (sceneFiles.length === 0) {
        pathEl.value = '';
    } else if (sceneFiles.length === 1) {
        pathEl.value = sceneFiles[0].name;
    } else {
        pathEl.value = `${sceneFiles.length} 个视频文件`;
    }
}

function clearSceneFiles() {
    sceneFiles = [];
    sceneResults = {};
    sceneOutputDir = '';
    updateSceneFileDisplay();
    renderSceneFileCards();
    document.getElementById('scene-export-status').classList.add('hidden');
    document.getElementById('scene-export-all-btn').style.display = 'none';
    document.getElementById('scene-detect-status').textContent = '就绪';
    document.getElementById('scene-detect-status').style.color = '';
}

function renderSceneFileCards() {
    const container = document.getElementById('scene-file-cards');
    container.innerHTML = '';

    if (sceneFiles.length === 0) {
        container.innerHTML = '<p class="hint">请先选择视频文件。</p>';
        return;
    }

    sceneFiles.forEach((file, idx) => {
        const result = sceneResults[file.path];
        const card = document.createElement('div');
        card.className = 'scene-file-card';
        card.dataset.idx = idx;
        card.style.cssText = 'background: var(--bg-tertiary); border-radius: 8px; padding: 12px; border: 1px solid rgba(255,255,255,0.05);';

        // ---- 卡片头部 ----
        const header = document.createElement('div');
        header.style.cssText = 'display: flex; align-items: center; gap: 8px;';

        // 文件名
        const nameEl = document.createElement('div');
        nameEl.style.cssText = 'flex: 1; font-size: 13px; color: var(--text-primary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;';
        nameEl.textContent = `🎬 ${file.name}`;
        nameEl.title = file.path;

        // 状态标签
        const statusTag = document.createElement('span');
        statusTag.id = `scene-status-${idx}`;
        statusTag.style.cssText = 'font-size: 11px; padding: 2px 8px; border-radius: 3px;';
        if (result) {
            statusTag.textContent = `✅ ${result.scene_points.length} 个切换点`;
            statusTag.style.background = 'rgba(0, 217, 165, 0.15)';
            statusTag.style.color = '#00d9a5';
        } else {
            statusTag.textContent = '待检测';
            statusTag.style.background = 'rgba(128,128,128,0.2)';
            statusTag.style.color = 'var(--text-muted)';
        }

        // 单个文件检测按钮
        const detectBtn = document.createElement('button');
        detectBtn.className = 'btn btn-secondary';
        detectBtn.style.cssText = 'padding: 4px 10px; font-size: 11px;';
        detectBtn.textContent = result ? '🔄 重新检测' : '🔍 检测';
        detectBtn.onclick = () => detectSingleFile(idx);

        // 导出按钮（检测完成后显示）
        const exportBtn = document.createElement('button');
        exportBtn.className = 'btn btn-primary';
        exportBtn.style.cssText = 'padding: 4px 10px; font-size: 11px;';
        exportBtn.textContent = '📦 导出';
        exportBtn.style.display = result ? '' : 'none';
        exportBtn.id = `scene-export-btn-${idx}`;
        exportBtn.onclick = () => exportSingleFile(idx);

        // 裁切按钮
        const trimBtn = document.createElement('button');
        trimBtn.className = 'btn btn-secondary';
        trimBtn.style.cssText = 'padding: 4px 10px; font-size: 11px;';
        trimBtn.textContent = '✂️ 裁切';
        trimBtn.title = '打开手动裁切工具';
        trimBtn.onclick = () => openTrimModal(file.path, file.name);

        // 删除按钮
        const removeBtn = document.createElement('button');
        removeBtn.className = 'btn btn-secondary';
        removeBtn.style.cssText = 'padding: 4px 8px; font-size: 11px; color: var(--error);';
        removeBtn.textContent = '✕';
        removeBtn.title = '移除此文件';
        removeBtn.onclick = () => {
            delete sceneResults[sceneFiles[idx].path];
            sceneFiles.splice(idx, 1);
            updateSceneFileDisplay();
            renderSceneFileCards();
            updateSceneExportAllBtn();
        };

        header.appendChild(nameEl);
        header.appendChild(statusTag);
        header.appendChild(detectBtn);
        header.appendChild(exportBtn);
        header.appendChild(trimBtn);
        header.appendChild(removeBtn);
        card.appendChild(header);

        // ---- 视频信息 + 片段列表（检测完成后展示）----
        if (result) {
            // 视频信息
            const infoRow = document.createElement('div');
            infoRow.style.cssText = 'display: flex; gap: 12px; font-size: 11px; color: var(--text-muted); margin-top: 8px; padding: 6px 8px; background: rgba(255,255,255,0.03); border-radius: 4px;';
            infoRow.innerHTML = `
                <span>📐 ${result.resolution || '-'}</span>
                <span>🖼️ ${result.fps} FPS</span>
                <span>⏱️ ${formatTimeAudio(result.duration)}</span>
                <span>✂️ ${result.segments.length} 片段</span>
            `;
            card.appendChild(infoRow);

            // 展开/收起按钮
            const toggleBtn = document.createElement('button');
            toggleBtn.className = 'btn btn-secondary';
            toggleBtn.style.cssText = 'padding: 2px 10px; font-size: 11px; margin-top: 8px; width: 100%;';
            toggleBtn.textContent = '▼ 展开片段列表';
            const segListContainer = document.createElement('div');
            segListContainer.style.cssText = 'display: none; margin-top: 8px; max-height: 300px; overflow-y: auto;';
            toggleBtn.onclick = () => {
                const hidden = segListContainer.style.display === 'none';
                segListContainer.style.display = hidden ? 'flex' : 'none';
                segListContainer.style.flexDirection = 'column';
                segListContainer.style.gap = '4px';
                toggleBtn.textContent = hidden ? '▲ 收起片段列表' : '▼ 展开片段列表';
            };
            card.appendChild(toggleBtn);

            // 片段列表
            const maxDur = Math.max(...result.segments.map(s => s.duration));
            result.segments.forEach((seg, sIdx) => {
                const row = document.createElement('div');
                row.style.cssText = 'display: flex; align-items: center; gap: 8px; padding: 5px 8px; background: var(--bg-secondary); border-radius: 4px; font-size: 12px;';

                const cb = document.createElement('input');
                cb.type = 'checkbox';
                cb.checked = true;
                cb.className = `scene-cb-${idx}`;
                cb.dataset.segIndex = sIdx;

                const num = document.createElement('span');
                num.style.cssText = 'min-width: 28px; font-weight: 600; color: var(--accent);';
                num.textContent = `#${seg.index}`;

                const time = document.createElement('span');
                time.style.cssText = 'flex: 1; font-family: monospace; color: var(--text-primary);';
                time.textContent = `${seg.start_str} → ${seg.end_str}`;

                const barC = document.createElement('div');
                barC.style.cssText = 'width: 60px; height: 4px; background: rgba(255,255,255,0.1); border-radius: 2px; overflow: hidden;';
                const bar = document.createElement('div');
                bar.style.cssText = `width: ${Math.max(2, (seg.duration / maxDur) * 100)}%; height: 100%; background: linear-gradient(90deg, #667eea, #764ba2); border-radius: 2px;`;
                barC.appendChild(bar);

                const dur = document.createElement('span');
                dur.style.cssText = 'min-width: 55px; color: var(--text-muted); text-align: right;';
                dur.textContent = seg.duration_str;

                row.appendChild(cb);
                row.appendChild(num);
                row.appendChild(time);
                row.appendChild(barC);
                row.appendChild(dur);
                segListContainer.appendChild(row);
            });

            card.appendChild(segListContainer);
        }

        container.appendChild(card);
    });
}

// 检测单个文件
async function detectSingleFile(idx) {
    const file = sceneFiles[idx];
    if (!file) return;

    const statusTag = document.getElementById(`scene-status-${idx}`);
    if (statusTag) {
        statusTag.textContent = '⏳ 分析中...';
        statusTag.style.background = 'rgba(102, 126, 234, 0.15)';
        statusTag.style.color = 'var(--accent)';
    }

    const threshold = parseFloat(document.getElementById('scene-threshold').value);
    const minInterval = parseFloat(document.getElementById('scene-min-interval').value);

    try {
        const response = await fetch(`${API_BASE}/media/scene-detect`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                file_path: file.path,
                threshold: threshold,
                min_interval: minInterval
            })
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.error || '检测失败');

        sceneResults[file.path] = data;
        renderSceneFileCards();
        updateSceneExportAllBtn();
        showToast(`${file.name}: ${data.message}`, 'success');

    } catch (error) {
        if (statusTag) {
            statusTag.textContent = `❌ ${error.message}`;
            statusTag.style.background = 'rgba(255, 71, 87, 0.15)';
            statusTag.style.color = '#ff4757';
        }
        showToast(`${file.name}: ${error.message}`, 'error');
    }
}

// 批量检测全部
async function startSceneDetectAll() {
    if (sceneFiles.length === 0) {
        showToast('请先选择视频文件', 'error');
        return;
    }

    const btn = document.getElementById('scene-detect-btn');
    const statusEl = document.getElementById('scene-detect-status');

    btn.disabled = true;
    btn.textContent = '⏳ 批量分析中...';

    let successCount = 0;
    let failCount = 0;

    for (let i = 0; i < sceneFiles.length; i++) {
        statusEl.textContent = `正在分析 (${i + 1}/${sceneFiles.length}): ${sceneFiles[i].name}`;
        statusEl.style.color = 'var(--accent)';
        await detectSingleFile(i);

        if (sceneResults[sceneFiles[i].path]) {
            successCount++;
        } else {
            failCount++;
        }
    }

    btn.disabled = false;
    btn.textContent = '🔍 批量场景检测';
    const msg = `批量检测完成: ${successCount} 成功${failCount > 0 ? `, ${failCount} 失败` : ''}`;
    statusEl.textContent = msg;
    statusEl.style.color = failCount > 0 ? 'var(--warning)' : 'var(--success)';
    showToast(msg, successCount > 0 ? 'success' : 'error');
}

// 导出单个文件的选中片段
async function exportSingleFile(idx) {
    const file = sceneFiles[idx];
    const result = sceneResults[file.path];
    if (!result) return;

    // 收集选中的片段
    const checkboxes = document.querySelectorAll(`.scene-cb-${idx}`);
    const selectedSegments = [];
    checkboxes.forEach(cb => {
        const sIdx = parseInt(cb.dataset.segIndex);
        if (cb.checked && result.segments[sIdx]) {
            selectedSegments.push(result.segments[sIdx]);
        }
    });

    // 如果没有勾选（列表未展开），默认导出全部
    if (selectedSegments.length === 0 && checkboxes.length === 0) {
        selectedSegments.push(...result.segments);
    }

    if (selectedSegments.length === 0) {
        showToast('请至少选择一个片段', 'error');
        return;
    }

    const outputDir = document.getElementById('media-output-path').value || '';
    const statusEl = document.getElementById('scene-export-text');
    const exportSection = document.getElementById('scene-export-status');

    exportSection.classList.remove('hidden');
    statusEl.textContent = `正在导出 ${file.name} 的 ${selectedSegments.length} 个片段...`;

    try {
        const response = await fetch(`${API_BASE}/media/scene-split`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                file_path: file.path,
                segments: selectedSegments,
                output_dir: outputDir
            })
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.error || '导出失败');

        sceneOutputDir = data.output_dir || '';
        statusEl.textContent = data.message;
        showToast(data.message, 'success');

    } catch (error) {
        statusEl.textContent = `导出失败: ${error.message}`;
        showToast(`导出失败: ${error.message}`, 'error');
    }
}

// 批量导出全部文件
async function exportAllScenes() {
    const filesToExport = sceneFiles.filter(f => sceneResults[f.path]);
    if (filesToExport.length === 0) {
        showToast('没有已检测的文件可导出', 'error');
        return;
    }

    const outputDir = document.getElementById('media-output-path').value || '';
    const statusEl = document.getElementById('scene-export-text');
    const progressEl = document.getElementById('scene-export-progress');
    const exportSection = document.getElementById('scene-export-status');

    exportSection.classList.remove('hidden');
    let totalExported = 0;

    for (let i = 0; i < filesToExport.length; i++) {
        const file = filesToExport[i];
        const result = sceneResults[file.path];
        statusEl.textContent = `正在导出 (${i + 1}/${filesToExport.length}): ${file.name}...`;
        progressEl.querySelector('.progress-bar-inner').style.width = `${((i) / filesToExport.length) * 100}%`;

        try {
            const response = await fetch(`${API_BASE}/media/scene-split`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    file_path: file.path,
                    segments: result.segments,
                    output_dir: outputDir
                })
            });

            const data = await response.json();
            if (response.ok) {
                totalExported += data.files?.length || 0;
                sceneOutputDir = data.output_dir || sceneOutputDir;
            }
        } catch (error) {
            console.error(`导出 ${file.name} 失败:`, error);
        }
    }

    progressEl.querySelector('.progress-bar-inner').style.width = '100%';
    statusEl.textContent = `批量导出完成: 共导出 ${totalExported} 个片段`;
    showToast(`批量导出完成: ${totalExported} 个片段`, 'success');
}

function updateSceneExportAllBtn() {
    const btn = document.getElementById('scene-export-all-btn');
    const hasResults = sceneFiles.some(f => sceneResults[f.path]);
    if (btn) btn.style.display = hasResults ? '' : 'none';
}

async function openSceneOutputDir() {
    let dir = sceneOutputDir;
    if (!dir && sceneFiles.length > 0) {
        const p = sceneFiles[0].path;
        dir = p.substring(0, Math.max(p.lastIndexOf('/'), p.lastIndexOf('\\')));
    }
    if (!dir) {
        showToast('没有输出目录', 'error');
        return;
    }

    try {
        await fetch(`${API_BASE}/open-folder`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: dir })
        });
    } catch (error) {
        showToast('打开目录失败', 'error');
    }
}

// ==================== 手动裁切弹窗模块 ====================

let trimState = {
    filePath: '',
    fileName: '',
    duration: 0,
    inTime: 0,
    outTime: 0,
    peaks: [],
    scenePoints: [],   // 场景切割点时间戳
    isPlaying: false,
    animFrameId: null,
    dragging: null,  // 'in' | 'out' | null
    // 缩放状态
    zoom: 1,          // 1 = 全览，10 = 只看 1/10 时长
    viewStart: 0,     // 可见时间窗口起点
    viewEnd: 0        // 可见时间窗口终点
};

async function openTrimModal(filePath, fileName) {
    trimState.filePath = filePath;
    trimState.fileName = fileName;
    trimState.inTime = 0;
    trimState.outTime = 0;
    trimState.isPlaying = false;
    trimState.dragging = null;

    // 从场景检测结果读取切割点
    const fileResult = sceneResults[filePath];
    if (fileResult && fileResult.scene_points) {
        trimState.scenePoints = fileResult.scene_points.map(p => p.time || p);
    } else {
        trimState.scenePoints = [];
    }

    document.getElementById('trim-file-name').textContent = fileName;
    document.getElementById('trim-export-status').textContent = '';

    // 加载视频
    const video = document.getElementById('trim-video-player');
    const videoUrl = `${API_BASE}/file/proxy?path=${encodeURIComponent(filePath)}`;
    video.src = videoUrl;
    video.currentTime = 0;

    video.onloadedmetadata = () => {
        trimState.duration = video.duration;
        trimState.outTime = video.duration;
        trimState.zoom = 1;
        trimState.viewStart = 0;
        trimState.viewEnd = video.duration;
        updateTrimUI();
        updateTrimTimeRuler();
    };

    video.ontimeupdate = () => {
        updateTrimPlayhead();
        document.getElementById('trim-current-time').textContent = formatTrimTime(video.currentTime);
    };

    video.onended = () => {
        trimState.isPlaying = false;
        document.getElementById('trim-play-btn').textContent = '▶ 播放';
    };

    // 显示弹窗
    document.getElementById('trim-modal').style.display = 'flex';

    // 加载波形
    loadTrimWaveform(filePath);

    // 设置事件监听
    setupTrimDragHandles();
    setupTrimTimelineClick();
    setupTrimZoom();
}

function closeTrimModal() {
    const video = document.getElementById('trim-video-player');
    video.pause();
    video.src = '';
    trimState.isPlaying = false;
    if (trimState.animFrameId) {
        cancelAnimationFrame(trimState.animFrameId);
        trimState.animFrameId = null;
    }
    document.getElementById('trim-modal').style.display = 'none';
}

async function loadTrimWaveform(filePath) {
    const canvas = document.getElementById('trim-waveform-canvas');
    const ctx = canvas.getContext('2d');
    const container = document.getElementById('trim-timeline-container');

    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;

    // 显示加载中
    ctx.fillStyle = 'rgba(255,255,255,0.1)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('⏳ 加载波形中...', canvas.width / 2, canvas.height / 2);

    try {
        const response = await fetch(`${API_BASE}/media/waveform`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ file_path: filePath, num_peaks: Math.min(600, container.clientWidth) })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error);

        trimState.peaks = data.peaks || [];
        if (data.duration && data.duration > 0) {
            trimState.duration = data.duration;
            trimState.outTime = data.duration;
        }
        trimState.viewEnd = trimState.duration;
        drawTrimWaveform();
        updateTrimUI();
        updateTrimTimeRuler();
    } catch (error) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = 'rgba(255,255,255,0.1)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#ff4757';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`波形加载失败: ${error.message}`, canvas.width / 2, canvas.height / 2);
    }
}

// 将时间转换为可见区域内的百分比 (0-100)
function timeToViewPct(t) {
    const vd = trimState.viewEnd - trimState.viewStart;
    if (vd <= 0) return 0;
    return ((t - trimState.viewStart) / vd) * 100;
}

// 将可见区域内的百分比转换为时间
function viewPctToTime(pct) {
    const vd = trimState.viewEnd - trimState.viewStart;
    return trimState.viewStart + (pct / 100) * vd;
}

function drawTrimWaveform() {
    const canvas = document.getElementById('trim-waveform-canvas');
    const ctx = canvas.getContext('2d');
    const container = document.getElementById('trim-timeline-container');
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    const w = canvas.width;
    const h = canvas.height;
    const peaks = trimState.peaks;

    ctx.clearRect(0, 0, w, h);

    // 背景
    ctx.fillStyle = 'rgba(30, 32, 40, 0.9)';
    ctx.fillRect(0, 0, w, h);

    if (!peaks.length || trimState.duration <= 0) return;

    const vStart = trimState.viewStart;
    const vEnd = trimState.viewEnd;
    const vDur = vEnd - vStart;
    const mid = h / 2;

    // 根据可见时间窗口绘制波形
    const totalPeaks = peaks.length;
    const startIdx = Math.floor((vStart / trimState.duration) * totalPeaks);
    const endIdx = Math.ceil((vEnd / trimState.duration) * totalPeaks);
    const visiblePeaks = endIdx - startIdx;
    const barWidth = w / Math.max(visiblePeaks, 1);

    for (let i = startIdx; i < endIdx && i < totalPeaks; i++) {
        const barH = peaks[i] * mid * 0.9;
        const x = (i - startIdx) * barWidth;

        const ratio = i / totalPeaks;
        const r = Math.round(46 + ratio * 56);
        const g = Math.round(213 - ratio * 138);
        const b = Math.round(115 + ratio * 47);

        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.8)`;
        ctx.fillRect(x, mid - barH, Math.max(barWidth - 0.3, 1), barH * 2);
    }

    // 中线
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, mid);
    ctx.lineTo(w, mid);
    ctx.stroke();

    // ====== 绘制场景切割点标记 ======
    if (trimState.scenePoints.length > 0) {
        ctx.save();
        trimState.scenePoints.forEach((t, idx) => {
            if (t < vStart || t > vEnd) return; // 只绘制可见范围内的
            const x = ((t - vStart) / vDur) * w;

            // 黄色竖线
            ctx.strokeStyle = 'rgba(255, 215, 0, 0.9)';
            ctx.lineWidth = 2;
            ctx.setLineDash([4, 2]);
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, h);
            ctx.stroke();
            ctx.setLineDash([]);

            // 顶部三角标记
            ctx.fillStyle = '#ffd700';
            ctx.beginPath();
            ctx.moveTo(x - 5, 0);
            ctx.lineTo(x + 5, 0);
            ctx.lineTo(x, 10);
            ctx.closePath();
            ctx.fill();

            // 切割点编号 + 时间
            ctx.fillStyle = 'rgba(255, 215, 0, 0.95)';
            ctx.font = 'bold 10px sans-serif';
            ctx.textAlign = 'center';
            const yPos = (idx % 2 === 0) ? 22 : h - 4;
            ctx.fillText(`#${idx + 1} ${formatTrimTime(t)}`, x, yPos);
        });
        ctx.restore();
    }

    // ====== 缩放指示器 ======
    if (trimState.zoom > 1.05) {
        ctx.save();
        ctx.fillStyle = 'rgba(255,255,255,0.6)';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(`🔍 ${trimState.zoom.toFixed(1)}x  [滚轮缩放 / 拖动平移 / 双击复位]`, w - 6, h - 6);
        ctx.restore();
    }
}

function updateTrimUI() {
    const dur = trimState.duration;
    if (dur <= 0) return;

    // 使用可见窗口百分比计算 handle 位置
    const inPct = timeToViewPct(trimState.inTime);
    const outPct = timeToViewPct(trimState.outTime);

    // IN/OUT handle 位置（限制在 0-100 范围内）
    const clampIn = Math.max(-2, Math.min(102, inPct));
    const clampOut = Math.max(-2, Math.min(102, outPct));
    document.getElementById('trim-handle-in').style.left = `${clampIn}%`;
    document.getElementById('trim-handle-out').style.left = `${clampOut}%`;

    // 遮罩
    document.getElementById('trim-mask-left').style.width = `${Math.max(0, clampIn)}%`;
    document.getElementById('trim-mask-right').style.left = `${Math.min(100, clampOut)}%`;
    document.getElementById('trim-mask-right').style.width = `${Math.max(0, 100 - clampOut)}%`;

    // 时间输入框
    document.getElementById('trim-in-time').value = formatTrimTime(trimState.inTime);
    document.getElementById('trim-out-time').value = formatTrimTime(trimState.outTime);
    document.getElementById('trim-total-time').textContent = formatTrimTime(dur);

    // 选区时长
    const selDur = trimState.outTime - trimState.inTime;
    document.getElementById('trim-selection-duration').textContent = formatTrimTime(Math.max(0, selDur));
}

function updateTrimPlayhead() {
    const video = document.getElementById('trim-video-player');
    const dur = trimState.duration;
    if (dur <= 0) return;
    const pct = timeToViewPct(video.currentTime);
    document.getElementById('trim-playhead').style.left = `${Math.max(-1, Math.min(101, pct))}%`;
}

function updateTrimTimeRuler() {
    const ruler = document.getElementById('trim-time-ruler');
    const vStart = trimState.viewStart;
    const vEnd = trimState.viewEnd;
    const vDur = vEnd - vStart;
    const numMarks = 10;
    ruler.innerHTML = '';
    for (let i = 0; i <= numMarks; i++) {
        const t = vStart + (vDur / numMarks) * i;
        const span = document.createElement('span');
        span.textContent = formatTrimTime(t);
        ruler.appendChild(span);
    }
}

function formatTrimTime(s) {
    if (!s || s < 0) s = 0;
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m.toString().padStart(2, '0')}:${sec.toFixed(3).padStart(6, '0')}`;
}

function parseTrimTime(str) {
    const parts = str.trim().split(':');
    if (parts.length === 2) {
        return parseFloat(parts[0]) * 60 + parseFloat(parts[1]);
    } else if (parts.length === 3) {
        return parseFloat(parts[0]) * 3600 + parseFloat(parts[1]) * 60 + parseFloat(parts[2]);
    }
    return parseFloat(str) || 0;
}

// ---- 播放控制 ----
function toggleTrimPlay() {
    const video = document.getElementById('trim-video-player');
    if (video.paused) {
        // 如果播放头超出OUT点，从IN点开始
        if (video.currentTime >= trimState.outTime - 0.05) {
            video.currentTime = trimState.inTime;
        }
        video.play();
        trimState.isPlaying = true;
        document.getElementById('trim-play-btn').textContent = '⏸ 暂停';
        monitorTrimPlayback();
    } else {
        video.pause();
        trimState.isPlaying = false;
        document.getElementById('trim-play-btn').textContent = '▶ 播放';
    }
}

function monitorTrimPlayback() {
    const video = document.getElementById('trim-video-player');
    if (!trimState.isPlaying) return;
    // 到达OUT点自动暂停
    if (video.currentTime >= trimState.outTime - 0.03) {
        video.pause();
        video.currentTime = trimState.outTime;
        trimState.isPlaying = false;
        document.getElementById('trim-play-btn').textContent = '▶ 播放';
        return;
    }
    // 缩放时自动跟随播放头
    if (trimState.zoom > 1.05) {
        const ct = video.currentTime;
        const viewDur = trimState.viewEnd - trimState.viewStart;
        const margin = viewDur * 0.15;
        if (ct > trimState.viewEnd - margin || ct < trimState.viewStart + margin) {
            let newStart = ct - viewDur * 0.3;
            newStart = Math.max(0, Math.min(trimState.duration - viewDur, newStart));
            trimState.viewStart = newStart;
            trimState.viewEnd = newStart + viewDur;
            drawTrimWaveform();
            updateTrimUI();
            updateTrimTimeRuler();
        }
    }
    requestAnimationFrame(monitorTrimPlayback);
}

function trimJumpToIn() {
    document.getElementById('trim-video-player').currentTime = trimState.inTime;
}

function trimJumpToOut() {
    document.getElementById('trim-video-player').currentTime = Math.max(0, trimState.outTime - 0.1);
}

function setTrimSpeed() {
    const speed = parseFloat(document.getElementById('trim-speed').value);
    document.getElementById('trim-video-player').playbackRate = speed;
}

function setTrimInAtCurrent() {
    const t = document.getElementById('trim-video-player').currentTime;
    trimState.inTime = Math.min(t, trimState.outTime - 0.1);
    updateTrimUI();
}

function setTrimOutAtCurrent() {
    const t = document.getElementById('trim-video-player').currentTime;
    trimState.outTime = Math.max(t, trimState.inTime + 0.1);
    updateTrimUI();
}

function onTrimTimeInputChange(which) {
    if (which === 'in') {
        const t = parseTrimTime(document.getElementById('trim-in-time').value);
        trimState.inTime = Math.max(0, Math.min(t, trimState.outTime - 0.1));
    } else {
        const t = parseTrimTime(document.getElementById('trim-out-time').value);
        trimState.outTime = Math.min(trimState.duration, Math.max(t, trimState.inTime + 0.1));
    }
    updateTrimUI();
}

// ---- IN/OUT 手柄 + 播放头拖动 ----
function setupTrimDragHandles() {
    const container = document.getElementById('trim-timeline-container');
    const handleIn = document.getElementById('trim-handle-in');
    const handleOut = document.getElementById('trim-handle-out');
    const playhead = document.getElementById('trim-playhead');

    const startDrag = (which) => (e) => {
        e.preventDefault();
        e.stopPropagation();
        trimState.dragging = which;
        document.addEventListener('mousemove', onDragMove);
        document.addEventListener('mouseup', onDragEnd);
    };

    handleIn.addEventListener('mousedown', startDrag('in'));
    handleOut.addEventListener('mousedown', startDrag('out'));
    playhead.addEventListener('mousedown', startDrag('playhead'));

    function onDragMove(e) {
        if (!trimState.dragging) return;
        const rect = container.getBoundingClientRect();
        const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        const t = viewPctToTime(pct * 100);

        if (trimState.dragging === 'in') {
            trimState.inTime = Math.max(0, Math.min(t, trimState.outTime - 0.1));
        } else if (trimState.dragging === 'out') {
            trimState.outTime = Math.min(trimState.duration, Math.max(t, trimState.inTime + 0.1));
        } else if (trimState.dragging === 'playhead') {
            // 拖动播放头 = 实时 scrub
            document.getElementById('trim-video-player').currentTime = Math.max(0, Math.min(trimState.duration, t));
            updateTrimPlayhead();
            return;
        }

        updateTrimUI();
        document.getElementById('trim-video-player').currentTime = t;
    }

    function onDragEnd() {
        trimState.dragging = null;
        document.removeEventListener('mousemove', onDragMove);
        document.removeEventListener('mouseup', onDragEnd);
    }
}

// ---- 时间轴点击跳转 ----
function setupTrimTimelineClick() {
    const container = document.getElementById('trim-timeline-container');
    // 点击跳转（通过 mousedown/up 距离判断，避免和拖动平移冲突）
    let clickStartX = 0;
    let clickStartY = 0;
    container.addEventListener('mousedown', (e) => {
        if (e.target.closest('.trim-handle')) return;
        clickStartX = e.clientX;
        clickStartY = e.clientY;
    });
    container.addEventListener('mouseup', (e) => {
        if (trimState.dragging) return;
        if (trimState._panning) return; // 刚拖动完不触发click
        if (e.target.closest('.trim-handle')) return;
        const dx = Math.abs(e.clientX - clickStartX);
        const dy = Math.abs(e.clientY - clickStartY);
        if (dx > 4 || dy > 4) return; // 移动超过4px认为是拖动而不是点击
        const rect = container.getBoundingClientRect();
        const pct = (e.clientX - rect.left) / rect.width;
        const t = viewPctToTime(pct * 100);
        document.getElementById('trim-video-player').currentTime = t;
    });
}

// ---- 导出裁切 ----
async function executeTrim() {
    const statusEl = document.getElementById('trim-export-status');
    const inT = trimState.inTime;
    const outT = trimState.outTime;
    const precise = document.getElementById('trim-precise-mode')?.checked ?? true;

    if (outT - inT < 0.1) {
        showToast('选区时长太短', 'error');
        return;
    }

    const modeText = precise ? '精确模式（重编码，可能较慢）' : '快速模式';
    statusEl.textContent = `⏳ 正在裁切（${modeText}）...`;
    statusEl.style.color = 'var(--accent)';

    try {
        const outputDir = document.getElementById('media-output-path')?.value || '';
        const response = await fetch(`${API_BASE}/media/trim`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                file_path: trimState.filePath,
                start: inT,
                end: outT,
                output_dir: outputDir,
                precise: precise
            })
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.error || '裁切失败');

        statusEl.textContent = `✅ ${data.message} (${data.mode || ''})`;
        statusEl.style.color = 'var(--success)';
        showToast(data.message, 'success');
    } catch (error) {
        statusEl.textContent = `❌ ${error.message}`;
        statusEl.style.color = 'var(--error)';
        showToast(`裁切失败: ${error.message}`, 'error');
    }
}

// ---- 预览选区：从IN播放到OUT自动停止 ----
function previewTrimSelection() {
    const video = document.getElementById('trim-video-player');
    video.currentTime = trimState.inTime;
    video.play();
    trimState.isPlaying = true;
    document.getElementById('trim-play-btn').textContent = '⏸ 暂停';
    monitorTrimPlayback();
}

// ---- 逐帧步进 ----
function trimStepFrame(direction) {
    const video = document.getElementById('trim-video-player');
    video.pause();
    trimState.isPlaying = false;
    document.getElementById('trim-play-btn').textContent = '▶ 播放';

    // 估算帧时长（默认30fps）
    // 如果有检测结果则使用实际fps
    let fps = 30;
    const fileResult = sceneResults[trimState.filePath];
    if (fileResult && fileResult.fps) {
        fps = fileResult.fps;
    }
    const frameDuration = 1 / fps;
    video.currentTime = Math.max(0, Math.min(trimState.duration, video.currentTime + direction * frameDuration));
}

// ---- 波形缩放 + 拖动平移 ----
function setupTrimZoom() {
    const container = document.getElementById('trim-timeline-container');
    trimState._panning = false;
    trimState._panStartX = 0;
    trimState._panStartViewStart = 0;

    // 滚轮 = 缩放（以鼠标位置为中心）
    container.addEventListener('wheel', (e) => {
        e.preventDefault();
        const rect = container.getBoundingClientRect();
        const mousePct = (e.clientX - rect.left) / rect.width;
        const mouseTime = viewPctToTime(mousePct * 100);

        const factor = e.deltaY > 0 ? 0.85 : 1.2;
        trimState.zoom = Math.max(1, Math.min(100, trimState.zoom * factor));

        const newViewDur = trimState.duration / trimState.zoom;
        let newStart = mouseTime - mousePct * newViewDur;
        newStart = Math.max(0, Math.min(trimState.duration - newViewDur, newStart));
        trimState.viewStart = newStart;
        trimState.viewEnd = Math.min(trimState.duration, newStart + newViewDur);

        drawTrimWaveform();
        updateTrimUI();
        updateTrimPlayhead();
        updateTrimTimeRuler();
    }, { passive: false });

    // 拖动平移（缩放后拖动超过4px才开始平移，单击仍可定位）
    container.addEventListener('mousedown', (e) => {
        if (trimState.zoom <= 1.05) return;
        if (e.target.closest('.trim-handle')) return;
        if (trimState.dragging) return;

        // 不立刻进入平移，先记录起点
        trimState._panning = false;
        trimState._panStartX = e.clientX;
        trimState._panStartViewStart = trimState.viewStart;
        let panActivated = false;

        const onPanMove = (ev) => {
            const dx = ev.clientX - trimState._panStartX;
            // 移动超过4px才激活平移
            if (!panActivated && Math.abs(dx) > 4) {
                panActivated = true;
                trimState._panning = true;
                container.style.cursor = 'grabbing';
            }
            if (!panActivated) return;

            const rect = container.getBoundingClientRect();
            const viewDur = trimState.viewEnd - trimState.viewStart;
            const timeDelta = -(dx / rect.width) * viewDur;
            let newStart = trimState._panStartViewStart + timeDelta;
            newStart = Math.max(0, Math.min(trimState.duration - viewDur, newStart));
            trimState.viewStart = newStart;
            trimState.viewEnd = newStart + viewDur;

            drawTrimWaveform();
            updateTrimUI();
            updateTrimPlayhead();
            updateTrimTimeRuler();
        };

        const onPanEnd = () => {
            container.style.cursor = 'pointer';
            if (panActivated) {
                // 延迟重置平移标志，避免触发click
                setTimeout(() => { trimState._panning = false; }, 50);
            }
            document.removeEventListener('mousemove', onPanMove);
            document.removeEventListener('mouseup', onPanEnd);
        };

        document.addEventListener('mousemove', onPanMove);
        document.addEventListener('mouseup', onPanEnd);
    });

    // 双击复位缩放
    container.addEventListener('dblclick', (e) => {
        if (e.target.closest('.trim-handle')) return;
        trimState.zoom = 1;
        trimState.viewStart = 0;
        trimState.viewEnd = trimState.duration;
        drawTrimWaveform();
        updateTrimUI();
        updateTrimPlayhead();
        updateTrimTimeRuler();
    });
}

function trimZoomIn() {
    const center = (trimState.viewStart + trimState.viewEnd) / 2;
    trimState.zoom = Math.min(100, trimState.zoom * 1.5);
    const newViewDur = trimState.duration / trimState.zoom;
    trimState.viewStart = Math.max(0, center - newViewDur / 2);
    trimState.viewEnd = Math.min(trimState.duration, trimState.viewStart + newViewDur);
    drawTrimWaveform();
    updateTrimUI();
    updateTrimPlayhead();
    updateTrimTimeRuler();
}

function trimZoomOut() {
    const center = (trimState.viewStart + trimState.viewEnd) / 2;
    trimState.zoom = Math.max(1, trimState.zoom / 1.5);
    const newViewDur = trimState.duration / trimState.zoom;
    trimState.viewStart = Math.max(0, center - newViewDur / 2);
    trimState.viewEnd = Math.min(trimState.duration, trimState.viewStart + newViewDur);
    if (trimState.zoom <= 1.01) {
        trimState.viewStart = 0;
        trimState.viewEnd = trimState.duration;
    }
    drawTrimWaveform();
    updateTrimUI();
    updateTrimPlayhead();
    updateTrimTimeRuler();
}

function trimZoomReset() {
    trimState.zoom = 1;
    trimState.viewStart = 0;
    trimState.viewEnd = trimState.duration;
    drawTrimWaveform();
    updateTrimUI();
    updateTrimPlayhead();
    updateTrimTimeRuler();
}

// 缩放到当前播放头位置
function trimZoomToPlayhead() {
    const ct = document.getElementById('trim-video-player').currentTime;
    trimState.zoom = Math.min(100, trimState.zoom * 2);
    const newViewDur = trimState.duration / trimState.zoom;
    trimState.viewStart = Math.max(0, ct - newViewDur / 2);
    trimState.viewEnd = Math.min(trimState.duration, trimState.viewStart + newViewDur);
    drawTrimWaveform();
    updateTrimUI();
    updateTrimPlayhead();
    updateTrimTimeRuler();
}

// ESC 关闭裁切弹窗 + 快捷键
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && document.getElementById('trim-modal').style.display === 'flex') {
        closeTrimModal();
    }
    if (document.getElementById('trim-modal').style.display === 'flex') {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
        if (e.key === 'ArrowLeft') {
            e.preventDefault();
            trimStepFrame(-1);
        } else if (e.key === 'ArrowRight') {
            e.preventDefault();
            trimStepFrame(1);
        } else if (e.key === ' ') {
            e.preventDefault();
            toggleTrimPlay();
        } else if (e.key === 'i' || e.key === 'I') {
            setTrimInAtCurrent();
        } else if (e.key === 'o' || e.key === 'O') {
            setTrimOutAtCurrent();
        } else if (e.key === '=' || e.key === '+') {
            e.preventDefault();
            trimZoomIn();
        } else if (e.key === '-') {
            e.preventDefault();
            trimZoomOut();
        } else if (e.key === '0') {
            e.preventDefault();
            trimZoomReset();
        }
    }
});

