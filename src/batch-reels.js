/**
 * batch-reels.js — 批量Reels模块主逻辑
 * 
 * 完整移植自 AutoSub_v8 SubtitleStylePanel + FrameRenderer
 * 
 * 功能：
 * - 任务管理 (添加视频+SRT、自动配对、拖拽)
 * - 实时 Canvas 字幕预览 (含动画)
 * - 样式参数双向绑定 (所有 AutoSub 参数)
 * - 预设管理 (保存/加载/删除/导入/导出)
 * - 批量导出 (通过 IPC 调用 FFmpeg)
 */

// ═══════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════

const _reelsState = {
    tasks: [],
    selectedIdx: -1,
    renderer: null,
    previewRAF: null,
    previewFadeVideo: null,
    previewFadeVideoSrc: '',
    isExporting: false,
    lastExportOutputPath: '',
    pendingFiles: { backgrounds: [], audios: [], srts: [], txts: [] },
    backgroundLibrary: [],
    // Overlay interaction state
    overlaySelectedId: null,
    overlayDrag: null,        // { ovId, startX, startY, origX, origY, handle: null|'tl'|'tr'|'bl'|'br'|... }
    // Mock play state for items without media
    mockPlaying: false,
    mockPausedTime: 0,
    mockStartTime: 0,
    // AI watermarks
    watermarks: [],
};
window._reelsState = _reelsState;

const REELS_DEFAULT_PRESET_KEY = 'reels_default_preset_name';
const REELS_WATERMARK_STORAGE_KEY = 'reels_watermarks';
const REELS_BACKGROUND_EXTS = new Set(['mp4', 'mov', 'mkv', 'avi', 'wmv', 'flv', 'webm', 'jpg', 'jpeg', 'png', 'webp']);
const REELS_AUDIO_EXTS = new Set(['mp3', 'wav', 'm4a', 'aac', 'flac', 'ogg']);
const REELS_TXT_EXTS = new Set(['txt']);
const REELS_MATCH_STOPWORDS = new Set([
    'srt', 'sub', 'subtitle', 'source', 'src', 'audio', 'voice', 'vo',
    'en', 'cn', 'zh', 'ja', 'jp', 'ko', 'kr', 'es', 'de', 'fr', 'pt', 'it', 'ru', 'ar',
    '720p', '1080p', '4k', 'hd', 'fhd', 'mp4', 'mp3', 'wav', 'aac', 'h264', 'h265', 'hevc', 'x264', 'x265',
    'trim', 'cut', 'clip', 'final', 'v1', 'v2', 'v3', 'edit', 'edited', 'render', 'copy', 'out', 'output'
]);

const REELS_FONT_PRESETS = {
    bebashook: {
        label: 'Bebas Neue 标题粗体',
        font_family: 'Bebas Neue',
        font_weight: 800,
        fontsize: 86,
        bold: true,
        italic: false,
        letter_spacing: 1,
    },
    oswald_clean: {
        label: 'Oswald 干净信息流',
        font_family: 'Oswald',
        font_weight: 700,
        fontsize: 74,
        bold: true,
        italic: false,
        letter_spacing: 0,
    },
    montserrat_modern: {
        label: 'Montserrat 现代通用',
        font_family: 'Montserrat',
        font_weight: 700,
        fontsize: 72,
        bold: true,
        italic: false,
        letter_spacing: 0,
    },
    playfair_story: {
        label: 'Playfair Display 叙事感',
        font_family: 'Playfair Display',
        font_weight: 700,
        fontsize: 70,
        bold: true,
        italic: false,
        letter_spacing: 0,
    },
    noto_sans_cn: {
        label: 'Noto Sans SC 中文清晰',
        font_family: 'Noto Sans SC',
        font_weight: 700,
        fontsize: 70,
        bold: true,
        italic: false,
        letter_spacing: 0,
    },
    noto_serif_cn: {
        label: 'Noto Serif SC 中文衬线',
        font_family: 'Noto Serif SC',
        font_weight: 700,
        fontsize: 68,
        bold: true,
        italic: false,
        letter_spacing: 0,
    },
};

let _reelsHotkeyBound = false;

// ═══════════════════════════════════════════════════════
// Initialization
// ═══════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    _initReelsModule();
});

function _initReelsModule() {
    const canvas = document.getElementById('reels-preview-canvas');
    if (canvas) {
        canvas.width = 1080;
        canvas.height = 1920;
        _reelsState.renderer = new ReelsCanvasRenderer(canvas);
    }

    const videoInput = document.getElementById('reels-video-input');
    const audioInput = document.getElementById('reels-audio-input');
    const srtInput = document.getElementById('reels-srt-input');
    const txtInput = document.getElementById('reels-txt-input');
    const folderInput = document.getElementById('reels-folder-input');
    if (videoInput) videoInput.addEventListener('change', _onVideoFilesSelected);
    if (audioInput) audioInput.addEventListener('change', _onAudioFilesSelected);
    if (srtInput) srtInput.addEventListener('change', _onSrtFilesSelected);
    if (txtInput) txtInput.addEventListener('change', _onTxtFilesSelected);
    if (folderInput) folderInput.addEventListener('change', _onFolderFilesSelected);

    const taskList = document.getElementById('reels-task-list');
    if (taskList) {
        taskList.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
            taskList.style.borderColor = 'var(--accent-color)';
            taskList.style.backgroundColor = 'rgba(233, 69, 96, 0.10)';
            taskList.style.boxShadow = '0 0 0 2px rgba(233, 69, 96, 0.22) inset';
        });
        taskList.addEventListener('dragleave', () => {
            taskList.style.borderColor = '';
            taskList.style.backgroundColor = '';
            taskList.style.boxShadow = '';
        });
        taskList.addEventListener('drop', _onTaskListDrop);
    }

    const seekBar = document.getElementById('reels-preview-seek');
    if (seekBar) seekBar.addEventListener('input', _onSeek);
    const voiceVolumeEl = document.getElementById('reels-voice-volume');
    const bgVolumeEl = document.getElementById('reels-bg-volume');
    const bindMix = (el) => {
        if (!el) return;
        el.addEventListener('input', _applyPreviewAudioMix);
        el.addEventListener('change', _applyPreviewAudioMix);
    };
    bindMix(voiceVolumeEl);
    bindMix(bgVolumeEl);

    // ── 混响 / 立体声控件 ──
    const reverbIds = ['reels-reverb-enabled', 'reels-reverb-preset', 'reels-reverb-mix', 'reels-stereo-width', 'reels-audio-fx-target'];
    for (const rid of reverbIds) {
        const el = document.getElementById(rid);
        if (el) {
            el.addEventListener('change', _setupPreviewReverb);
            el.addEventListener('input', _setupPreviewReverb);
        }
    }

    const video = document.getElementById('reels-preview-video');
    if (video) {
        video.addEventListener('timeupdate', _onVideoTimeUpdate);
        video.addEventListener('loadedmetadata', _onVideoLoaded);
    }
    const audio = document.getElementById('reels-preview-audio');
    if (audio) {
        audio.addEventListener('timeupdate', _onAudioTimeUpdate);
        audio.addEventListener('loadedmetadata', _onAudioLoaded);
        audio.addEventListener('ended', () => {
            const video = document.getElementById('reels-preview-video');
            if (video) video.pause();
            const fadeVideo = _reelsState.previewFadeVideo;
            if (fadeVideo) fadeVideo.pause();
            // 同步暂停 BGM
            const bgmAudio = _reelsState._bgmAudioEl;
            if (bgmAudio) bgmAudio.pause();
            const btn = document.getElementById('reels-preview-play');
            if (btn) btn.textContent = '▶️';
        });
    }

    // ═══ 创建 BGM 音频元素（隐藏） ═══
    if (!_reelsState._bgmAudioEl) {
        const bgmEl = document.createElement('audio');
        bgmEl.id = 'reels-preview-bgm';
        bgmEl.style.display = 'none';
        bgmEl.loop = true;  // BGM 循环播放
        document.body.appendChild(bgmEl);
        _reelsState._bgmAudioEl = bgmEl;
    }

    _reelsRefreshPresetList();
    _reelsApplyDefaultPreset();
    _initReelsFontPresetUI();

    // ═══ 字体管理器初始化 ═══
    _initFontManager();

    // ═══ NLE UI 组件初始化 ═══

    // 时间线编辑器
    const tlContainer = document.getElementById('reels-timeline-container');
    if (tlContainer && typeof ReelsTimelineEditor !== 'undefined') {
        _reelsState.timelineEditor = new ReelsTimelineEditor(tlContainer);
        _reelsState.timelineEditor.onSeek = (t) => {
            const video = document.getElementById('reels-preview-video');
            const audio = document.getElementById('reels-preview-audio');
            const task = _getSelectedTask();
            if (task && task.audioPath && audio && audio.duration) {
                audio.currentTime = Math.max(0, Math.min(t, audio.duration));
            }
            if (video && video.duration) {
                video.currentTime = (task && task.audioPath) ? (t % video.duration) : Math.max(0, Math.min(t, video.duration));
            }
            _reelsState.timelineEditor.setPlayhead(t);
            _updatePreviewTimeUI(t, _getPreviewDuration());
        };
        _reelsState.timelineEditor.onClipSelect = (ti, ci, clip) => {
            console.log('[Timeline] Selected clip', ti, ci, clip);
        };
        // 加载默认空轨道
        _reelsState.timelineEditor.setTracks([
            { type: 'video', name: '视频', clips: [], locked: false, visible: true, domain: 'visual' },
            { type: 'subs', name: '字幕', clips: [], locked: false, visible: true, domain: 'visual' },
            { type: 'text', name: '文本覆层', clips: [], locked: false, visible: true, domain: 'visual' },
            { type: 'image', name: '图片覆层', clips: [], locked: false, visible: true, domain: 'visual' },
            { type: 'audio', name: '音频', clips: [], locked: false, visible: true, domain: 'audio' },
        ]);
    }

    // 覆层面板
    const ovPanelRoot = document.getElementById('reels-overlay-panel-root');
    if (ovPanelRoot && typeof ReelsOverlayPanel !== 'undefined') {
        // 创建轻量画布代理，让覆层面板可以管理覆层
        if (!_reelsState.overlayProxy) {
            const ReelsOverlayMod = window.ReelsOverlay;
            const mgr = ReelsOverlayMod ? new ReelsOverlayMod.OverlayManager() : { overlays: [], addOverlay(o) { this.overlays.push(o); return o; }, removeOverlay(id) { this.overlays = this.overlays.filter(o => o.id !== id); }, getOverlay(id) { return this.overlays.find(o => o.id === id) || null; } };
            _reelsState.overlayProxy = {
                overlayMgr: mgr,
                addOverlay(ov) { mgr.addOverlay(ov); },
                removeOverlay(id) { mgr.removeOverlay(id); },
                getSelected() { return null; },
                render() { /* rAF loop handles rendering */ },
                // 回调占位
                onSelect: null,
                onDeselect: null,
                onOverlayChange: null,
            };
        }
        _reelsState.overlayPanel = new ReelsOverlayPanel(ovPanelRoot, _reelsState.overlayProxy);
    }

    reelsUpdatePreview();
    _bindReelsHotkeys();

    // ═══ 覆层预览交互 ═══
    _initOverlayCanvasInteraction();

    // ═══ 预览窗口缩放/平移初始化 ═══
    _initPreviewZoomPan();
    _initReelsExportDefaults();
    _reelsUpdateLastOutputUI('');
    _reelsUpdateExportProgressUI(0, 0);
    _reelsUpdateLastErrorUI('');

    // ═══ 面板拖拽调整宽度 ═══
    _initReelsColumnResize();
}

function _initReelsColumnResize() {
    const handles = document.querySelectorAll('.reels-resize-handle');
    if (!handles.length) return;

    // Restore saved widths
    const saved = localStorage.getItem('reels-col-widths');
    if (saved) {
        try {
            const widths = JSON.parse(saved);
            for (const [id, w] of Object.entries(widths)) {
                const el = document.getElementById(id);
                if (el && el.id !== 'reels-col-preview') {
                    el.style.width = w + 'px';
                    el.style.flex = 'none';
                }
            }
        } catch (e) { }
    }

    handles.forEach(handle => {
        handle.addEventListener('mousedown', (e) => {
            e.preventDefault();
            const leftId = handle.dataset.left;
            const rightId = handle.dataset.right;
            const leftEl = document.getElementById(leftId);
            const rightEl = document.getElementById(rightId);
            if (!leftEl || !rightEl) return;

            handle.classList.add('active');
            const startX = e.clientX;
            const leftW0 = leftEl.getBoundingClientRect().width;
            const rightW0 = rightEl.getBoundingClientRect().width;
            const leftMin = parseInt(getComputedStyle(leftEl).minWidth) || 100;
            const rightMin = parseInt(getComputedStyle(rightEl).minWidth) || 100;

            // Prevent text selection during drag
            const overlay = document.createElement('div');
            overlay.style.cssText = 'position:fixed;inset:0;z-index:99999;cursor:col-resize;';
            document.body.appendChild(overlay);

            const onMove = (ev) => {
                const dx = ev.clientX - startX;
                const newLeft = Math.max(leftMin, leftW0 + dx);
                const newRight = Math.max(rightMin, rightW0 - dx);
                // Only apply if both panels stay above minimum
                if (newLeft >= leftMin && newRight >= rightMin) {
                    leftEl.style.width = newLeft + 'px';
                    leftEl.style.flex = 'none';
                    // For the preview (flex:1) column, set flex instead
                    if (rightId === 'reels-col-preview') {
                        rightEl.style.flex = '1';
                    } else {
                        rightEl.style.width = newRight + 'px';
                        rightEl.style.flex = 'none';
                    }
                    if (leftId === 'reels-col-preview') {
                        leftEl.style.flex = '1';
                    }
                }
            };

            const onUp = () => {
                handle.classList.remove('active');
                overlay.remove();
                document.removeEventListener('mousemove', onMove);
                document.removeEventListener('mouseup', onUp);

                // Save column widths
                const cols = {};
                ['reels-col-tasks', 'reels-col-subtitle', 'reels-col-overlay'].forEach(id => {
                    const el = document.getElementById(id);
                    if (el) cols[id] = Math.round(el.getBoundingClientRect().width);
                });
                localStorage.setItem('reels-col-widths', JSON.stringify(cols));
            };

            document.addEventListener('mousemove', onMove);
            document.addEventListener('mouseup', onUp);
        });
    });
}

async function _getSystemDownloadsPath() {
    try {
        if (window.electronAPI && typeof window.electronAPI.getDownloadsPath === 'function') {
            const p = await window.electronAPI.getDownloadsPath();
            if (p) return p;
        }
    } catch (e) { }
    return '~/Downloads';
}

async function _initReelsExportDefaults() {
    const outputEl = document.getElementById('reels-output-dir');
    if (!outputEl || outputEl.value) return;
    outputEl.value = await _getSystemDownloadsPath();
}

function _bindReelsHotkeys() {
    if (_reelsHotkeyBound) return;
    _reelsHotkeyBound = true;
    document.addEventListener('keydown', (e) => {
        if (e.code !== 'Space') return;
        const tag = (e.target && e.target.tagName) ? e.target.tagName.toLowerCase() : '';
        if (tag === 'input' || tag === 'textarea' || (e.target && e.target.isContentEditable)) return;
        const panel = document.getElementById('batch-reels-panel');
        if (!panel || !panel.classList.contains('active')) return;
        e.preventDefault();
        reelsTogglePlay();
    });

    // Delete key removes selected overlay
    document.addEventListener('keydown', (e) => {
        if (e.code !== 'Delete' && e.code !== 'Backspace') return;
        const tag = (e.target && e.target.tagName) ? e.target.tagName.toLowerCase() : '';
        if (tag === 'input' || tag === 'textarea' || (e.target && e.target.isContentEditable)) return;
        if (!_isReelsPanelActive()) return;
        if (!_reelsState.overlaySelectedId) return;
        e.preventDefault();
        const proxy = _reelsState.overlayProxy;
        if (proxy) {
            proxy.removeOverlay(_reelsState.overlaySelectedId);
            _reelsState.overlaySelectedId = null;
            if (_reelsState.overlayPanel) {
                _reelsState.overlayPanel.deselectOverlay();
                _reelsState.overlayPanel._refreshList();
            }
        }
    });
}

// ═══════════════════════════════════════════════════════
// Overlay canvas interaction (drag, select, resize)
// ═══════════════════════════════════════════════════════

const _OV_HANDLE_SIZE = 12; // px in canvas coordinates

function _initOverlayCanvasInteraction() {
    const canvas = document.getElementById('reels-preview-canvas');
    if (!canvas) return;

    canvas.addEventListener('mousedown', _ovOnMouseDown);
    canvas.addEventListener('mousemove', _ovOnMouseMove);
    canvas.addEventListener('mouseup', _ovOnMouseUp);
    canvas.addEventListener('mouseleave', _ovOnMouseUp);
}

/** Convert client (screen) coordinates → canvas logical coordinates */
function _clientToCanvas(clientX, clientY) {
    const canvas = document.getElementById('reels-preview-canvas');
    if (!canvas) return { x: 0, y: 0 };

    const rect = canvas.getBoundingClientRect();
    // The canvas is 1080x1920 logical, but displayed at rect.width x rect.height
    // Plus there's zoom/pan on the container
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    return {
        x: (clientX - rect.left) * scaleX,
        y: (clientY - rect.top) * scaleY,
    };
}

/** Get the bounding box of an overlay in canvas coords */
function _ovGetBounds(ov) {
    const x = parseFloat(ov.x || 0);
    const w = parseFloat(ov.w || 100);
    let y, h;
    if (ov.type === 'textcard' && ov._renderedY != null) {
        y = ov._renderedY;
        h = ov._renderedH || 100;
    } else {
        y = parseFloat(ov.y || 0);
        h = parseFloat(ov.h || 100);
    }
    return { x, y, w, h };
}

/** Check if point hits one of the 8 resize handles. Returns handle name or null */
function _ovHitHandle(mx, my, bounds) {
    const hs = _OV_HANDLE_SIZE;
    const { x, y, w, h } = bounds;
    const handles = {
        'tl': { cx: x, cy: y },
        'tc': { cx: x + w / 2, cy: y },
        'tr': { cx: x + w, cy: y },
        'ml': { cx: x, cy: y + h / 2 },
        'mr': { cx: x + w, cy: y + h / 2 },
        'bl': { cx: x, cy: y + h },
        'bc': { cx: x + w / 2, cy: y + h },
        'br': { cx: x + w, cy: y + h },
    };
    for (const [name, pos] of Object.entries(handles)) {
        if (Math.abs(mx - pos.cx) <= hs && Math.abs(my - pos.cy) <= hs) {
            return name;
        }
    }
    return null;
}

function _ovOnMouseDown(e) {
    if (e.button !== 0) return; // left click only
    const proxy = _reelsState.overlayProxy;
    if (!proxy || !proxy.overlayMgr) return;

    const { x: mx, y: my } = _clientToCanvas(e.clientX, e.clientY);
    const overlays = proxy.overlayMgr.overlays || [];

    // 1. If already selected, check if clicking a resize handle
    if (_reelsState.overlaySelectedId) {
        const selOv = overlays.find(o => o.id === _reelsState.overlaySelectedId);
        if (selOv) {
            const bounds = _ovGetBounds(selOv);
            const handle = _ovHitHandle(mx, my, bounds);
            if (handle) {
                _reelsState.overlayDrag = {
                    ovId: selOv.id,
                    startX: mx, startY: my,
                    origX: selOv.x, origY: selOv.y, origW: selOv.w, origH: selOv.h || selOv._renderedH || 100,
                    handle,
                };
                e.stopPropagation();
                return;
            }
        }
    }

    // 2. Hit test all overlays (reverse z-order: topmost first)
    let hit = null;
    for (let i = overlays.length - 1; i >= 0; i--) {
        const ov = overlays[i];
        const bounds = _ovGetBounds(ov);
        if (mx >= bounds.x && mx <= bounds.x + bounds.w && my >= bounds.y && my <= bounds.y + bounds.h) {
            hit = ov;
            break;
        }
    }

    if (hit) {
        _reelsState.overlaySelectedId = hit.id;
        _reelsState.overlayDrag = {
            ovId: hit.id,
            startX: mx, startY: my,
            origX: hit.x, origY: hit.y, origW: hit.w, origH: hit.h || hit._renderedH || 100,
            handle: null, // move mode
        };
        // Sync with overlay panel
        if (_reelsState.overlayPanel) {
            _reelsState.overlayPanel.selectOverlay(hit);
        }
        e.stopPropagation();
    } else {
        // Deselect
        _reelsState.overlaySelectedId = null;
        _reelsState.overlayDrag = null;
        if (_reelsState.overlayPanel) {
            _reelsState.overlayPanel.deselectOverlay();
        }
    }
}

function _ovOnMouseMove(e) {
    const drag = _reelsState.overlayDrag;
    if (!drag) {
        // Update cursor based on hover
        _ovUpdateCursor(e);
        return;
    }
    if (e.buttons === 0) { // mouse released outside
        _reelsState.overlayDrag = null;
        return;
    }

    const proxy = _reelsState.overlayProxy;
    if (!proxy) return;
    const ov = (proxy.overlayMgr.overlays || []).find(o => o.id === drag.ovId);
    if (!ov) return;

    const { x: mx, y: my } = _clientToCanvas(e.clientX, e.clientY);
    const dx = mx - drag.startX;
    const dy = my - drag.startY;

    if (!drag.handle) {
        // Move
        ov.x = drag.origX + dx;
        ov.y = drag.origY + dy;
        if (ov.auto_center_v) ov.auto_center_v = false; // disable auto-center when manually moved
    } else {
        // Resize via handle
        _ovApplyResize(ov, drag, dx, dy);
    }

    // Sync panel
    if (_reelsState.overlayPanel && _reelsState.overlayPanel._selectedOv?.id === ov.id) {
        _reelsState.overlayPanel._syncFromOverlay(ov);
    }
}

function _ovOnMouseUp(e) {
    _reelsState.overlayDrag = null;
}

function _ovApplyResize(ov, drag, dx, dy) {
    const h = drag.handle;
    let x = drag.origX, y = drag.origY, w = drag.origW, ht = drag.origH;

    // Horizontal
    if (h.includes('l')) { x += dx; w -= dx; }
    if (h.includes('r')) { w += dx; }
    // Vertical
    if (h.includes('t')) { y += dy; ht -= dy; }
    if (h.includes('b')) { ht += dy; }

    // Enforce minimums
    if (w < 50) { w = 50; if (h.includes('l')) x = drag.origX + drag.origW - 50; }
    if (ht < 30) { ht = 30; if (h.includes('t')) y = drag.origY + drag.origH - 30; }

    ov.x = x;
    ov.y = y;
    ov.w = w;
    if (ov.type !== 'textcard' || !ov.auto_fit) {
        ov.h = ht;
    }
    if (ov.auto_center_v) ov.auto_center_v = false;
}

function _ovUpdateCursor(e) {
    const canvas = document.getElementById('reels-preview-canvas');
    if (!canvas || !_reelsState.overlaySelectedId) return;

    const proxy = _reelsState.overlayProxy;
    if (!proxy) return;
    const ov = (proxy.overlayMgr.overlays || []).find(o => o.id === _reelsState.overlaySelectedId);
    if (!ov) return;

    const { x: mx, y: my } = _clientToCanvas(e.clientX, e.clientY);
    const bounds = _ovGetBounds(ov);
    const handle = _ovHitHandle(mx, my, bounds);

    const cursors = {
        'tl': 'nw-resize', 'tr': 'ne-resize', 'bl': 'sw-resize', 'br': 'se-resize',
        'tc': 'n-resize', 'bc': 's-resize', 'ml': 'w-resize', 'mr': 'e-resize',
    };

    if (handle && cursors[handle]) {
        canvas.style.cursor = cursors[handle];
    } else if (mx >= bounds.x && mx <= bounds.x + bounds.w && my >= bounds.y && my <= bounds.y + bounds.h) {
        canvas.style.cursor = 'move';
    } else {
        canvas.style.cursor = '';
    }
}

/** Draw selection frame + 8 resize handles around the selected overlay */
function _drawOverlaySelectionUI(ctx, canvasW, canvasH) {
    if (!_reelsState.overlaySelectedId) return;
    const proxy = _reelsState.overlayProxy;
    if (!proxy) return;
    const ov = (proxy.overlayMgr.overlays || []).find(o => o.id === _reelsState.overlaySelectedId);
    if (!ov) return;

    const bounds = _ovGetBounds(ov);
    const { x, y, w, h } = bounds;
    const hs = _OV_HANDLE_SIZE;

    ctx.save();
    // Dashed selection border
    ctx.strokeStyle = '#00D4FF';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);

    // 8 resize handles
    ctx.fillStyle = '#FFFFFF';
    ctx.strokeStyle = '#00D4FF';
    ctx.lineWidth = 2;
    const handles = [
        [x, y], [x + w / 2, y], [x + w, y],
        [x, y + h / 2], [x + w, y + h / 2],
        [x, y + h], [x + w / 2, y + h], [x + w, y + h],
    ];
    for (const [hx, hy] of handles) {
        ctx.fillRect(hx - hs / 2, hy - hs / 2, hs, hs);
        ctx.strokeRect(hx - hs / 2, hy - hs / 2, hs, hs);
    }
    ctx.restore();
}

// ═══════════════════════════════════════════════════════
// Preview viewport zoom / pan
// ═══════════════════════════════════════════════════════

const _previewView = { scale: 1, panX: 0, panY: 0, dragging: false, lastX: 0, lastY: 0 };
let _reelsFitTimer = null;

function _isReelsPanelActive() {
    const panel = document.getElementById('batch-reels-panel');
    return !!(panel && panel.classList.contains('active'));
}

function _fitPreviewWhenReady(retry = 0) {
    const viewport = document.getElementById('reels-preview-viewport');
    const container = document.getElementById('reels-preview-container');
    if (!viewport || !container) return;
    const vpRect = viewport.getBoundingClientRect();
    if (vpRect.width > 20 && vpRect.height > 20 && container.offsetWidth > 20 && container.offsetHeight > 20) {
        reelsPreviewZoom('fit');
        return;
    }
    if (retry >= 12) return;
    if (_reelsFitTimer) clearTimeout(_reelsFitTimer);
    _reelsFitTimer = setTimeout(() => _fitPreviewWhenReady(retry + 1), 80);
}

function _initPreviewZoomPan() {
    const viewport = document.getElementById('reels-preview-viewport');
    if (!viewport) return;

    // 滚轮缩放
    viewport.addEventListener('wheel', (e) => {
        e.preventDefault();
        const factor = e.deltaY > 0 ? 0.9 : 1.1;
        const newScale = Math.max(0.1, Math.min(5, _previewView.scale * factor));

        // 以鼠标位置为中心缩放
        const rect = viewport.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        const ratio = newScale / _previewView.scale;
        _previewView.panX = mx - ratio * (mx - _previewView.panX);
        _previewView.panY = my - ratio * (my - _previewView.panY);
        _previewView.scale = newScale;

        _applyPreviewTransform();
    }, { passive: false });

    // 拖拽平移 — 只在没有命中覆层时启用，或按住空格键强制平移
    viewport.addEventListener('mousedown', (e) => {
        if (e.button !== 0) return;

        // Check if the mouse is over the canvas and hits an overlay
        const canvas = document.getElementById('reels-preview-canvas');
        if (canvas && _reelsState.overlayProxy && _reelsState.overlayProxy.overlayMgr) {
            const rect = canvas.getBoundingClientRect();
            if (e.clientX >= rect.left && e.clientX <= rect.right &&
                e.clientY >= rect.top && e.clientY <= rect.bottom) {
                // Convert to canvas coords and check for overlay hit
                const scaleX = canvas.width / rect.width;
                const scaleY = canvas.height / rect.height;
                const mx = (e.clientX - rect.left) * scaleX;
                const my = (e.clientY - rect.top) * scaleY;
                const overlays = _reelsState.overlayProxy.overlayMgr.overlays || [];
                for (let i = overlays.length - 1; i >= 0; i--) {
                    const ov = overlays[i];
                    const bounds = _ovGetBounds(ov);
                    if (mx >= bounds.x && mx <= bounds.x + bounds.w &&
                        my >= bounds.y && my <= bounds.y + bounds.h) {
                        // Hit an overlay — let the overlay interaction handle this
                        return;
                    }
                    // Also check if hitting a resize handle of selected overlay
                    if (_reelsState.overlaySelectedId && ov.id === _reelsState.overlaySelectedId) {
                        const handle = _ovHitHandle(mx, my, bounds);
                        if (handle) return; // Let resize handle work
                    }
                }
            }
        }

        _previewView.dragging = true;
        _previewView.lastX = e.clientX;
        _previewView.lastY = e.clientY;
        viewport.style.cursor = 'grabbing';
    });
    window.addEventListener('mousemove', (e) => {
        if (!_previewView.dragging) return;
        _previewView.panX += e.clientX - _previewView.lastX;
        _previewView.panY += e.clientY - _previewView.lastY;
        _previewView.lastX = e.clientX;
        _previewView.lastY = e.clientY;
        _applyPreviewTransform();
    });
    window.addEventListener('mouseup', () => {
        if (_previewView.dragging) {
            _previewView.dragging = false;
            const vp = document.getElementById('reels-preview-viewport');
            if (vp) vp.style.cursor = 'grab';
        }
    });

    // 初始适应（面板可能初始隐藏，需等待真实尺寸）
    setTimeout(() => _fitPreviewWhenReady(), 100);
}

function _applyPreviewTransform() {
    const el = document.getElementById('reels-preview-transform');
    if (!el) return;
    el.style.transform = `translate(${_previewView.panX}px, ${_previewView.panY}px) scale(${_previewView.scale})`;

    const label = document.getElementById('reels-preview-zoom-label');
    if (label) label.textContent = `${Math.round(_previewView.scale * 100)}%`;
}

function reelsPreviewZoom(action) {
    const viewport = document.getElementById('reels-preview-viewport');
    const container = document.getElementById('reels-preview-container');
    if (!viewport || !container) return;

    const vpRect = viewport.getBoundingClientRect();

    if (action === 'fit') {
        // 适应窗口：使 9:16 内容完整填入视口
        const containerW = container.offsetWidth;
        const containerH = container.offsetHeight;
        if (vpRect.width <= 0 || vpRect.height <= 0 || containerW <= 0 || containerH <= 0) return;
        const scaleW = vpRect.width / containerW;
        const scaleH = vpRect.height / containerH;
        _previewView.scale = Math.min(scaleW, scaleH) * 0.95; // 留 5% 边距
        // 居中
        _previewView.panX = (vpRect.width - containerW * _previewView.scale) / 2;
        _previewView.panY = (vpRect.height - containerH * _previewView.scale) / 2;
    } else if (action === 'reset') {
        _previewView.scale = 1;
        const containerW = container.offsetWidth;
        const containerH = container.offsetHeight;
        _previewView.panX = (vpRect.width - containerW) / 2;
        _previewView.panY = (vpRect.height - containerH) / 2;
    } else if (action === 'in') {
        const newScale = Math.min(5, _previewView.scale * 1.25);
        const cx = vpRect.width / 2;
        const cy = vpRect.height / 2;
        const ratio = newScale / _previewView.scale;
        _previewView.panX = cx - ratio * (cx - _previewView.panX);
        _previewView.panY = cy - ratio * (cy - _previewView.panY);
        _previewView.scale = newScale;
    } else if (action === 'out') {
        const newScale = Math.max(0.1, _previewView.scale * 0.8);
        const cx = vpRect.width / 2;
        const cy = vpRect.height / 2;
        const ratio = newScale / _previewView.scale;
        _previewView.panX = cx - ratio * (cx - _previewView.panX);
        _previewView.panY = cy - ratio * (cy - _previewView.panY);
        _previewView.scale = newScale;
    }

    _applyPreviewTransform();
}

async function _initFontManager() {
    if (typeof getFontManager !== 'function') {
        console.warn('[Reels] FontManager not loaded');
        return;
    }
    const fm = getFontManager();
    await fm.register();
    fm.refreshFontSelect('reels-font-family', _reelsState.renderer ? 'Arial' : undefined);
    fm.refreshFontSelect('rop-font', 'Arial');
    fm.refreshFontSelect('rop-title-font', 'Crimson Pro');
    fm.refreshFontSelect('rop-body-font', 'Arial');
    try { await fm.loadGoogleFont('Crimson Pro'); } catch (_) { }
    reelsRefreshSubtitleWeightOptions();
    console.log(`[Reels] FontManager ready — ${fm.getAllFonts().length} fonts available`);
}

function _initReelsFontPresetUI() {
    const select = document.getElementById('reels-font-preset');
    if (!select) return;
    const current = select.value;
    select.innerHTML = '<option value="">-- 字体预设 --</option>';
    for (const [key, preset] of Object.entries(REELS_FONT_PRESETS)) {
        const opt = document.createElement('option');
        opt.value = key;
        opt.textContent = preset.label;
        select.appendChild(opt);
    }
    if (current && REELS_FONT_PRESETS[current]) select.value = current;
}

function reelsRefreshSubtitleWeightOptions() {
    const familyEl = document.getElementById('reels-font-family');
    const weightEl = document.getElementById('reels-font-weight');
    if (!familyEl || !weightEl) return;

    const currentWeight = String(weightEl.value || '700');
    const fallback = [
        { value: '100', label: 'Thin' },
        { value: '200', label: 'ExtraLight' },
        { value: '300', label: 'Light' },
        { value: '400', label: 'Regular' },
        { value: '500', label: 'Medium' },
        { value: '600', label: 'SemiBold' },
        { value: '700', label: 'Bold' },
        { value: '800', label: 'ExtraBold' },
        { value: '900', label: 'Black' },
    ];
    let entries = fallback;

    if (typeof getFontManager === 'function') {
        const fm = getFontManager();
        if (fm && typeof fm.getFontWeightEntries === 'function') {
            const preferStyle = document.getElementById('reels-italic')?.checked ? 'italic' : 'normal';
            const list = fm.getFontWeightEntries(familyEl.value, preferStyle);
            if (Array.isArray(list) && list.length > 0) {
                entries = list.map(item => {
                    const value = String(item.value || '400');
                    const label = String(item.label || value);
                    return { value, label };
                });
            }
        } else if (fm && typeof fm.getFontWeightOptions === 'function') {
            const list = fm.getFontWeightOptions(familyEl.value);
            if (Array.isArray(list) && list.length > 0) {
                entries = list.map(v => ({ value: String(v), label: String(v) }));
            }
        }
    }

    const weights = entries.map(e => e.value);
    weightEl.innerHTML = entries.map(e => `<option value="${e.value}">${e.label}</option>`).join('');
    if (weights.includes(currentWeight)) {
        weightEl.value = currentWeight;
    } else if (weights.includes('700')) {
        weightEl.value = '700';
    } else {
        weightEl.value = weights[weights.length - 1] || '700';
    }
    reelsSyncWeightToBold();
}

async function reelsOnSubtitleFontFamilyChange() {
    const familyEl = document.getElementById('reels-font-family');
    if (!familyEl) return;
    if (typeof getFontManager === 'function') {
        try {
            const fm = getFontManager();
            await fm.loadGoogleFont(familyEl.value);
        } catch (_) { }
    }
    reelsRefreshSubtitleWeightOptions();
    reelsUpdatePreview();
}

function reelsSyncBoldToWeight() {
    const boldEl = document.getElementById('reels-bold');
    const weightEl = document.getElementById('reels-font-weight');
    if (!boldEl || !weightEl) return;
    const next = boldEl.checked ? '700' : '400';
    const opts = Array.from(weightEl.options || []).map(o => o.value);
    if (opts.includes(next)) {
        weightEl.value = next;
    } else if (boldEl.checked) {
        const high = opts.filter(v => parseInt(v, 10) >= 600);
        if (high.length > 0) weightEl.value = high[Math.min(1, high.length - 1)];
    } else {
        const low = opts.filter(v => parseInt(v, 10) < 600);
        if (low.length > 0) weightEl.value = low[Math.max(0, low.length - 2)];
    }
}

function reelsSyncWeightToBold() {
    const boldEl = document.getElementById('reels-bold');
    const weightEl = document.getElementById('reels-font-weight');
    if (!boldEl || !weightEl) return;
    const w = parseInt(weightEl.value || '700', 10);
    boldEl.checked = Number.isFinite(w) ? w >= 600 : true;
}

function reelsUploadFont() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.ttf,.otf,.woff,.woff2';
    input.onchange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const fm = getFontManager();
        const familyName = await fm.uploadFont(file);
        if (familyName) {
            fm.refreshFontSelect('reels-font-family', familyName);
            fm.refreshFontSelect('rop-font', familyName);
            fm.refreshFontSelect('rop-title-font', familyName);
            fm.refreshFontSelect('rop-body-font', familyName);
            const select = document.getElementById('reels-font-family');
            if (select) select.value = familyName;
            reelsRefreshSubtitleWeightOptions();
            reelsUpdatePreview();
            console.log(`[Reels] Custom font uploaded: ${familyName}`);
        } else {
            alert('字体加载失败，请确认文件格式正确');
        }
    };
    input.click();
}

async function reelsApplyFontPreset() {
    const select = document.getElementById('reels-font-preset');
    if (!select || !select.value) {
        alert('请先选择一个字体预设');
        return;
    }
    const preset = REELS_FONT_PRESETS[select.value];
    if (!preset) return;

    const set = (id, val) => {
        const el = document.getElementById(id);
        if (!el) return;
        el.value = String(val);
    };
    const setChk = (id, val) => {
        const el = document.getElementById(id);
        if (!el) return;
        el.checked = !!val;
    };

    if (typeof getFontManager === 'function') {
        const fm = getFontManager();
        await fm.loadGoogleFont(preset.font_family);
        fm.refreshFontSelect('reels-font-family', preset.font_family);
    }

    set('reels-font-family', preset.font_family);
    set('reels-font-weight', preset.font_weight || (preset.bold ? 700 : 400));
    set('reels-fontsize', preset.fontsize);
    set('reels-fontsize-range', preset.fontsize);
    setChk('reels-bold', preset.bold);
    setChk('reels-italic', preset.italic);
    set('reels-letter-spacing', preset.letter_spacing);
    reelsRefreshSubtitleWeightOptions();
    reelsSyncWeightToBold();

    reelsUpdatePreview();
}

// ═══════════════════════════════════════════════════════
// Style: read all params from UI → style object
// ═══════════════════════════════════════════════════════

function _readStyleFromUI() {
    const get = (id) => document.getElementById(id);
    const val = (id) => get(id) ? get(id).value : '';
    const num = (id, def) => { const v = parseFloat(val(id)); return isNaN(v) ? def : v; };
    const chk = (id) => get(id) ? get(id).checked : false;

    // Update labels
    const swLabel = get('reels-stroke-width-label');
    if (swLabel) swLabel.textContent = val('reels-stroke-width');
    const sbLabel = get('reels-shadow-blur-label');
    if (sbLabel) sbLabel.textContent = val('reels-shadow-blur');
    const pyLabel = get('reels-pos-y-label');
    if (pyLabel) pyLabel.textContent = val('reels-pos-y') + '%';
    const wwLabel = get('reels-wrap-width-label');
    if (wwLabel) wwLabel.textContent = val('reels-wrap-width') + '%';

    return {
        // Font
        font_family: val('reels-font-family') || 'Arial',
        font_weight: num('reels-font-weight', chk('reels-bold') ? 700 : 400),
        fontsize: num('reels-fontsize', 74),
        bold: num('reels-font-weight', chk('reels-bold') ? 700 : 400) >= 600,
        italic: chk('reels-italic'),
        letter_spacing: num('reels-letter-spacing', 0),

        // Colors
        color_text: val('reels-color-text') || '#FFFFFF',
        color_high: val('reels-color-high') || '#FFD700',

        // Stroke
        use_stroke: chk('reels-use-stroke'),
        color_outline: val('reels-stroke-color') || '#000000',
        border_width: num('reels-stroke-width', 3),
        opacity_outline: 255,

        // Multi-layer stroke expand
        stroke_expand_enabled: chk('reels-stroke-expand'),
        stroke_expand_layers: num('reels-se-layers', 3),
        stroke_expand_step: num('reels-se-step', 4),
        stroke_expand_feather: num('reels-se-feather', 8),
        stroke_expand_colors: val('reels-se-colors') || '#FF0000,#00FF00,#0000FF',

        // Shadow
        shadow_blur: chk('reels-shadow') ? num('reels-shadow-blur', 4) : 0,
        color_shadow: val('reels-shadow-color') || '#000000',
        opacity_shadow: chk('reels-shadow') ? 150 : 0,

        // Box
        use_box: chk('reels-use-box'),
        color_bg: val('reels-box-color') || '#000000',
        opacity_bg: num('reels-box-opacity', 150),
        box_radius: num('reels-box-radius', 8),
        box_blur: num('reels-box-blur', 0),
        box_padding_x: 12,
        box_padding_y: 8,

        // Box gradient
        bg_gradient_enabled: chk('reels-bg-gradient'),
        bg_gradient_type: val('reels-bg-gradient-type') || 'linear_h',
        bg_gradient_colors: val('reels-bg-gradient-colors') || '#e0c3fc,#8ec5fc',
        bg_gradient_highlight: chk('reels-bg-gradient-hl'),

        // Box color transition
        box_transition_enabled: chk('reels-box-transition'),
        box_transition_color_to: val('reels-box-transition-color') || '#FF6600',

        // Dynamic box
        dynamic_box: chk('reels-dynamic-box'),
        color_high_bg: val('reels-high-bg-color') || '#FFD700',
        opacity_high_bg: 200,
        dyn_box_anim: chk('reels-dyn-anim'),
        dyn_box_anim_overshoot: 1.3,
        dyn_box_anim_duration: 0.15,
        dynamic_radius: num('reels-dyn-radius', 6),
        high_padding: num('reels-high-padding', 4),
        high_offset_y: 0,

        // Underline
        use_underline: chk('reels-use-underline'),
        color_underline: val('reels-underline-color') || '#FFD700',

        // Position / layout
        pos_x: 0.5,
        pos_y: num('reels-pos-y', 50) / 100,
        wrap_width_percent: num('reels-wrap-width', 90),
        line_spacing: num('reels-line-spacing', 4),
        rotation: num('reels-rotation', 0),
        opacity_text_global: 255,

        // Global mask
        global_mask_enabled: chk('reels-global-mask'),
        global_mask_color: val('reels-global-mask-color') || '#000000',
        global_mask_opacity: num('reels-global-mask-opacity', 128) / 255,

        // Advanced textbox
        advanced_textbox_enabled: chk('reels-adv-textbox'),
        advanced_textbox_align: val('reels-adv-textbox-align') || 'center',
        advanced_textbox_x: num('reels-adv-x', 200),
        advanced_textbox_y: num('reels-adv-y', 1400),
        advanced_textbox_w: num('reels-adv-w', 680),
        advanced_textbox_h: num('reels-adv-h', 280),
        adv_bg_enabled: chk('reels-adv-bg'),
        adv_bg_color: val('reels-adv-bg-color') || '#000000',
        adv_bg_opacity: num('reels-adv-bg-opacity', 150),
        adv_bg_radius: num('reels-adv-bg-radius', 8),

        // Animation
        anim_in_type: val('reels-anim-in') || 'fade',
        anim_in_duration: num('reels-anim-in-dur', 0.3),
        anim_in_easing: 'ease_out',
        anim_out_type: val('reels-anim-out') || 'fade',
        anim_out_duration: num('reels-anim-out-dur', 0.25),
        anim_out_easing: 'ease_in_out',

        // Animation params
        floating_amplitude: num('reels-float-amp', 8),
        floating_period: num('reels-float-period', 2.0),
        char_bounce_height: num('reels-bounce-height', 20),
        char_bounce_stagger: 0.05,
        metronome_bpm: num('reels-metro-bpm', 120),
        letter_jump_scale: num('reels-jump-scale', 1.5),
        letter_jump_duration: 0.2,
        flash_color: val('reels-flash-color') || '#FFFFFF',
        flash_duration: 0.1,
        bullet_stagger: 0.15,
        holy_glow_color: val('reels-glow-color') || '#FFFFAA',
        holy_glow_radius: num('reels-glow-radius', 6),
        holy_glow_period: 3.0,
        blur_sharp_max: num('reels-blur-max', 20),
        blur_sharp_clear_frac: 0.4,

        // Typewriter
        tw_revealed_color: '#FFFFFF',
        tw_revealed_stroke_color: '#000000',
        tw_unrevealed_color: '#808080',
        tw_unrevealed_stroke_color: '#404040',
        tw_unrevealed_opacity: 100,

        // Metronome
        metro_read_color: '#FFFFFF',
        metro_read_stroke_color: '#000000',
        metro_unread_color: '#808080',
        metro_unread_stroke_color: '#404040',
        metro_unread_opacity: 100,
    };
}

function _writeStyleToUI(style) {
    const set = (id, val) => { const el = document.getElementById(id); if (el) el.value = val; };
    const setChk = (id, val) => { const el = document.getElementById(id); if (el) el.checked = !!val; };

    set('reels-font-family', style.font_family || 'Arial');
    // 如果字体是 Google Font，按需加载
    if (typeof getFontManager === 'function') {
        const fm = getFontManager();
        fm.loadGoogleFont(style.font_family || 'Arial');
    }
    set('reels-fontsize', style.fontsize || 74);
    set('reels-fontsize-range', style.fontsize || 74);
    const weight = Math.max(100, Math.min(900, parseInt(style.font_weight || ((style.bold !== false) ? 700 : 400), 10) || 700));
    set('reels-font-weight', String(weight));
    setChk('reels-bold', weight >= 600);
    setChk('reels-italic', style.italic);
    set('reels-letter-spacing', style.letter_spacing || 0);
    reelsRefreshSubtitleWeightOptions();
    set('reels-color-text', style.color_text || '#FFFFFF');
    set('reels-color-high', style.color_high || '#FFD700');
    setChk('reels-use-stroke', style.use_stroke !== false);
    set('reels-stroke-color', style.color_outline || '#000000');
    set('reels-stroke-width', style.border_width || 3);
    setChk('reels-stroke-expand', style.stroke_expand_enabled);
    set('reels-se-layers', style.stroke_expand_layers || 3);
    set('reels-se-step', style.stroke_expand_step || 4);
    set('reels-se-feather', style.stroke_expand_feather || 8);
    set('reels-se-colors', typeof style.stroke_expand_colors === 'string' ? style.stroke_expand_colors : '#FF0000,#00FF00,#0000FF');
    setChk('reels-shadow', (style.shadow_blur || 0) > 0);
    set('reels-shadow-color', style.color_shadow || '#000000');
    set('reels-shadow-blur', style.shadow_blur || 4);
    setChk('reels-use-box', style.use_box);
    set('reels-box-color', style.color_bg || '#000000');
    set('reels-box-opacity', style.opacity_bg || 150);
    set('reels-box-radius', style.box_radius || 8);
    set('reels-box-blur', style.box_blur || 0);
    setChk('reels-bg-gradient', style.bg_gradient_enabled);
    set('reels-bg-gradient-type', style.bg_gradient_type || 'linear_h');
    set('reels-bg-gradient-colors', typeof style.bg_gradient_colors === 'string' ? style.bg_gradient_colors : '#e0c3fc,#8ec5fc');
    setChk('reels-bg-gradient-hl', style.bg_gradient_highlight);
    setChk('reels-box-transition', style.box_transition_enabled);
    set('reels-box-transition-color', style.box_transition_color_to || '#FF6600');
    setChk('reels-dynamic-box', style.dynamic_box);
    set('reels-high-bg-color', style.color_high_bg || '#FFD700');
    setChk('reels-dyn-anim', style.dyn_box_anim);
    set('reels-high-padding', style.high_padding || 4);
    set('reels-dyn-radius', style.dynamic_radius || 6);
    setChk('reels-use-underline', style.use_underline);
    set('reels-underline-color', style.color_underline || '#FFD700');
    set('reels-pos-y', Math.round((style.pos_y || 0.5) * 100));
    set('reels-wrap-width', style.wrap_width_percent || 90);
    set('reels-line-spacing', style.line_spacing || 4);
    set('reels-rotation', style.rotation || 0);
    setChk('reels-adv-textbox', style.advanced_textbox_enabled);
    set('reels-adv-textbox-align', style.advanced_textbox_align || 'center');

    // Global mask
    setChk('reels-global-mask', style.global_mask_enabled);
    set('reels-global-mask-color', style.global_mask_color || '#000000');
    set('reels-global-mask-opacity', Math.round((style.global_mask_opacity ?? 0.5) * 255));

    set('reels-adv-x', style.advanced_textbox_x || 200);
    set('reels-adv-y', style.advanced_textbox_y || 1400);
    set('reels-adv-w', style.advanced_textbox_w || 680);
    set('reels-adv-h', style.advanced_textbox_h || 280);
    setChk('reels-adv-bg', style.adv_bg_enabled);
    set('reels-adv-bg-color', style.adv_bg_color || '#000000');
    set('reels-adv-bg-opacity', style.adv_bg_opacity || 150);
    set('reels-adv-bg-radius', style.adv_bg_radius || 8);
    set('reels-anim-in', style.anim_in_type || 'fade');
    set('reels-anim-in-dur', style.anim_in_duration || 0.3);
    set('reels-anim-out', style.anim_out_type || 'fade');
    set('reels-anim-out-dur', style.anim_out_duration || 0.25);
    set('reels-float-amp', style.floating_amplitude || 8);
    set('reels-float-period', style.floating_period || 2);
    set('reels-bounce-height', style.char_bounce_height || 20);
    set('reels-metro-bpm', style.metronome_bpm || 120);
    set('reels-jump-scale', style.letter_jump_scale || 1.5);
    set('reels-flash-color', style.flash_color || '#FFFFFF');
    set('reels-glow-color', style.holy_glow_color || '#FFFFAA');
    set('reels-glow-radius', style.holy_glow_radius || 6);
    set('reels-blur-max', style.blur_sharp_max || 20);
}

// ═══════════════════════════════════════════════════════
// Preview rendering loop
// ═══════════════════════════════════════════════════════

function reelsUpdatePreview() {
    const renderer = _reelsState.renderer;
    if (!renderer) return;

    const style = _readStyleFromUI();
    const previewText = (document.getElementById('reels-preview-text') || {}).value || 'Hello World 这是一个测试字幕';
    const canvas = renderer.canvas;
    const ctx = renderer.ctx;
    const w = canvas.width;
    const h = canvas.height;

    const placeholder = document.getElementById('reels-preview-placeholder');
    if (placeholder) placeholder.style.display = 'none';

    renderer.clear();
    _syncBackgroundVideoToMaster();

    const video = document.getElementById('reels-preview-video');
    // Check for image background
    const bgImg = _reelsState._previewBgImage;
    const hasBgImg = bgImg && bgImg.complete && bgImg.naturalWidth > 0;

    // Debug: log video state periodically
    if (video && video.src && !_reelsState._lastVideoDebugTime) _reelsState._lastVideoDebugTime = 0;
    if (video && video.src && (Date.now() - (_reelsState._lastVideoDebugTime || 0)) > 3000) {
        console.log('[Preview] video.src:', video.src?.substring(0, 80), 'readyState:', video.readyState, 'videoWidth:', video.videoWidth, 'error:', video.error?.message);
        _reelsState._lastVideoDebugTime = Date.now();
    }

    const _selectedTask = _getSelectedTask();
    const _bgScalePct = _selectedTask ? (_selectedTask.bgScale || 100) : 100;

    if (video && video.src && video.readyState >= 1 && video.videoWidth > 0) {
        _drawVideoCover(ctx, video, w, h, _bgScalePct);
        const fadeFrame = _calcPreviewLoopFadeFrame();
        if (fadeFrame && fadeFrame.video && fadeFrame.video.readyState >= 2) {
            ctx.save();
            ctx.globalAlpha = fadeFrame.alpha;
            _drawVideoCover(ctx, fadeFrame.video, w, h, _bgScalePct);
            ctx.restore();
        }

        // Draw global mask if enabled
        if (style.global_mask_enabled) {
            ctx.save();
            ctx.globalAlpha = style.global_mask_opacity ?? 0.5;
            ctx.fillStyle = style.global_mask_color || '#000000';
            ctx.fillRect(0, 0, w, h);
            ctx.restore();
        }
    } else if (hasBgImg) {
        // Draw image background using cover mode
        _drawVideoCover(ctx, bgImg, w, h, _bgScalePct);

        // Draw global mask if enabled
        if (style.global_mask_enabled) {
            ctx.save();
            ctx.globalAlpha = style.global_mask_opacity ?? 0.5;
            ctx.fillStyle = style.global_mask_color || '#000000';
            ctx.fillRect(0, 0, w, h);
            ctx.restore();
        }
    } else {
        const grad = ctx.createLinearGradient(0, 0, 0, h);
        grad.addColorStop(0, '#1a1a2e');
        grad.addColorStop(0.5, '#16213e');
        grad.addColorStop(1, '#0f3460');
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, w, h);

        // Draw global mask over placeholder background as well
        if (style.global_mask_enabled) {
            ctx.save();
            ctx.globalAlpha = style.global_mask_opacity ?? 0.5;
            ctx.fillStyle = style.global_mask_color || '#000000';
            ctx.fillRect(0, 0, w, h);
            ctx.restore();
        }
    }

    // Calculate max overlay end time for cycle period
    let maxOverlayEnd = 0;
    if (_reelsState.overlayProxy && _reelsState.overlayProxy.overlayMgr) {
        const overlays = _reelsState.overlayProxy.overlayMgr.overlays || [];
        for (const ov of overlays) {
            const end = parseFloat(ov.end || 0);
            if (end > maxOverlayEnd) maxOverlayEnd = end;
        }
    }

    let cycleTime = _getPreviewCurrentTime();
    if (!(cycleTime > 0)) {
        // 检查是否有媒体正在播放
        const video = document.getElementById('reels-preview-video');
        const audio = document.getElementById('reels-preview-audio');
        const isMediaPlaying = (video && !video.paused) || (audio && !audio.paused);

        if (isMediaPlaying) {
            // 媒体正在播放但 currentTime 尚为0，等下一帧
            cycleTime = 0;
        } else {
            // 没有媒体在播放 → 静止在 time=0 (不再自动循环)
            cycleTime = 0;
        }
    }

    const demoWords = previewText.split(/\s+/).filter(Boolean);
    const wordCount = demoWords.length || 1;
    const totalDur = Math.max(3, wordCount * 0.6);

    const wordsInfo = demoWords.map((w, i) => ({
        word: w,
        start: (totalDur * i / wordCount),
        end: (totalDur * (i + 1) / wordCount),
    }));

    const segment = {
        text: previewText,
        start: 0,
        end: totalDur,
        words: wordsInfo,
    };

    // If an actual task and segment exists, try to sync it.
    // For now, render exactly what the user inputs as a test segment if no timeline clip matches.
    // A more sophisticated system will find the correct segment based on video.currentTime
    let activeSegment = segment;
    if (_reelsState.selectedIdx !== -1 && _reelsState.tasks[_reelsState.selectedIdx]) {
        const task = _reelsState.tasks[_reelsState.selectedIdx];
        const segs = task.segments || [];
        // Find segment
        const s = segs.find(s => cycleTime >= s.start && cycleTime <= s.end);
        if (s) {
            activeSegment = s;
        } else {
            // Not speaking, don't show test text
            activeSegment = null;
        }
    }

    const subtitleToggle = document.getElementById('reels-subtitle-toggle');
    const showSubtitle = !subtitleToggle || subtitleToggle.checked;

    if (activeSegment && showSubtitle) {
        renderer.renderSubtitle(style, activeSegment, cycleTime, w, h);
    }

    // ── 渲染覆层 (文案卡片等) ──
    if (_reelsState.overlayProxy && _reelsState.overlayProxy.overlayMgr) {
        const ovMgr = _reelsState.overlayProxy.overlayMgr;
        const overlays = ovMgr.overlays || [];
        if (overlays.length > 0 && window.ReelsOverlay) {
            for (const ov of overlays) {
                // "显示终点" 模式：滚动字幕用终点时间渲染
                let ovTime = cycleTime;
                if (_reelsState._scrollPreviewEnd && ov.type === 'scroll') {
                    ovTime = parseFloat(ov.end || 10); // 用 end 时间，确保在时间范围内
                }
                ReelsOverlay.drawOverlay(ctx, ov, ovTime, w, h);
            }
        }
        // ── 选中框 + 拖拽手柄 ──
        _drawOverlaySelectionUI(ctx, w, h);
    }

    // ── AI 水印 ──
    _drawWatermarks(ctx, w, h);

    // ── 更新时间显示 (覆层预览时间) ──
    const dDur = _getPreviewDuration();
    const cTime = _getPreviewCurrentTime();

    // 如果没有真实的媒体元素，渲染循环必须主动驱动时间轴和 UI 的更新 (没有 timeupdate 事件)
    if (!_getPreviewMasterElement()) {
        _updatePreviewTimeUI(cTime, dDur);
    } else if (!cTime) {
        _updatePreviewTimeUI(0, dDur);
    }

    // 检查是否到达终点以自动停止
    if (dDur > 0 && cTime >= dDur) {
        // Reached the end
        const video = document.getElementById('reels-preview-video');
        const audio = document.getElementById('reels-preview-audio');
        const fadeVideo = _reelsState.previewFadeVideo;
        const btn = document.getElementById('reels-preview-play');
        
        // Only force pause if it wasn't already manually paused to avoid spamming
        const isPlaying = (audio && !audio.paused) || (video && !video.paused) || _reelsState.mockPlaying;
        if (isPlaying) {
            if (audio) audio.pause();
            if (video) video.pause();
            if (fadeVideo) fadeVideo.pause();
            if (_reelsState._bgmAudioEl) _reelsState._bgmAudioEl.pause();
            
            _reelsState.mockPlaying = false;
            _reelsState.mockPausedTime = dDur; // Ensure UI stays at the end
            if (btn) btn.textContent = '▶️';
            // 确保进度条刚好停在满格位置
            _updatePreviewTimeUI(dDur, dDur);
        }
    }

    if (_reelsState.previewRAF) cancelAnimationFrame(_reelsState.previewRAF);
    const panel = document.getElementById('batch-reels-panel');
    if (panel && (panel.classList.contains('active') || panel.style.display !== 'none')) {
        _reelsState.previewRAF = requestAnimationFrame(() => reelsUpdatePreview());
    }
}

/**
 * 绘制 AI 水印 (预览 + 导出共用)
 */
function _drawWatermarks(ctx, canvasW, canvasH) {
    const watermarks = _reelsState.watermarks || [];
    for (const wm of watermarks) {
        if (!wm.enabled || !wm.text) continue;
        const fontSize = wm.fontSize || 20;
        const padH = Math.round(fontSize * 0.5);
        const padV = Math.round(fontSize * 0.35);

        ctx.save();
        ctx.font = `${fontSize}px Arial, sans-serif`;
        const textW = ctx.measureText(wm.text).width;
        const boxW = textW + padH * 2;
        const boxH = fontSize + padV * 2;

        // 计算位置
        const margin = 16;
        let bx, by;
        switch (wm.position || 'top-right') {
            case 'top-left': bx = margin; by = margin; break;
            case 'top-center': bx = (canvasW - boxW) / 2; by = margin; break;
            case 'top-right': bx = canvasW - boxW - margin; by = margin; break;
            case 'center-left': bx = margin; by = (canvasH - boxH) / 2; break;
            case 'center': bx = (canvasW - boxW) / 2; by = (canvasH - boxH) / 2; break;
            case 'center-right': bx = canvasW - boxW - margin; by = (canvasH - boxH) / 2; break;
            case 'bottom-left': bx = margin; by = canvasH - boxH - margin; break;
            case 'bottom-center': bx = (canvasW - boxW) / 2; by = canvasH - boxH - margin; break;
            case 'bottom-right': bx = canvasW - boxW - margin; by = canvasH - boxH - margin; break;
            default: bx = canvasW - boxW - margin; by = margin; break;
        }

        // 半透明背景
        ctx.globalAlpha = wm.bgOpacity ?? 0.5;
        ctx.fillStyle = wm.bgColor || '#000000';
        const r = Math.round(fontSize * 0.2);
        ctx.beginPath();
        ctx.roundRect(bx, by, boxW, boxH, r);
        ctx.fill();

        // 文字
        ctx.globalAlpha = wm.textOpacity ?? 1.0;
        ctx.fillStyle = wm.color || '#FFFFFF';
        ctx.textBaseline = 'middle';
        ctx.fillText(wm.text, bx + padH, by + boxH / 2);
        ctx.restore();
    }
}

const REELS_DEFAULT_WATERMARK = [
    {
        text: 'AI Generated', fontSize: 25, color: '#FFFFFF', textOpacity: 0.8,
        bgColor: '#000000', bgOpacity: 0.5, position: 'top-right', enabled: true
    },
    {
        text: 'Attribution to11.ai', fontSize: 20, color: '#FFFFFF', textOpacity: 1.0,
        bgColor: '#000000', bgOpacity: 0.5, position: 'bottom-left', enabled: true
    },
];

function _reelsSaveWatermarks() {
    try {
        localStorage.setItem(REELS_WATERMARK_STORAGE_KEY, JSON.stringify(_reelsState.watermarks));
    } catch (e) { /* quota exceeded etc */ }
}

function _reelsLoadWatermarks() {
    try {
        const saved = localStorage.getItem(REELS_WATERMARK_STORAGE_KEY);
        if (saved) {
            const parsed = JSON.parse(saved);
            if (Array.isArray(parsed)) {
                _reelsState.watermarks = parsed;
                return;
            }
        }
    } catch (e) { /* parse error */ }
    // No saved data — use default
    _reelsState.watermarks = JSON.parse(JSON.stringify(REELS_DEFAULT_WATERMARK));
}

function reelsAddWatermark() {
    _reelsState.watermarks.push({
        text: 'Attribution to11.ai', fontSize: 20, color: '#FFFFFF', textOpacity: 1.0,
        bgColor: '#000000', bgOpacity: 0.5, position: 'bottom-left', enabled: true,
    });
    _reelsRefreshWatermarkUI();
    _reelsSaveWatermarks();
}

function reelsRemoveWatermark(idx) {
    _reelsState.watermarks.splice(idx, 1);
    _reelsRefreshWatermarkUI();
    _reelsSaveWatermarks();
}

function _reelsSyncWatermarkFromUI() {
    const list = document.getElementById('reels-watermark-list');
    if (!list) return;
    const rows = list.querySelectorAll('.wm-row');
    rows.forEach((row, i) => {
        const wm = _reelsState.watermarks[i];
        if (!wm) return;
        wm.text = row.querySelector('.wm-text')?.value || '';
        wm.fontSize = parseInt(row.querySelector('.wm-fontsize')?.value) || 20;
        wm.color = row.querySelector('.wm-color')?.value || '#FFFFFF';
        wm.bgColor = row.querySelector('.wm-bgcolor')?.value || '#000000';
        wm.bgOpacity = parseFloat(row.querySelector('.wm-bgopacity')?.value) / 100 || 0.5;
        wm.textOpacity = parseFloat(row.querySelector('.wm-textopacity')?.value) / 100 || 1.0;
        wm.position = row.querySelector('.wm-position')?.value || 'top-right';
        wm.enabled = row.querySelector('.wm-enabled')?.checked ?? true;
    });
    _reelsSaveWatermarks();
}

function _reelsRefreshWatermarkUI() {
    const list = document.getElementById('reels-watermark-list');
    const countEl = document.getElementById('reels-wm-count');
    if (!list) return;
    const wms = _reelsState.watermarks;
    if (countEl) countEl.textContent = `${wms.length} 个`;
    const posOptions = [
        ['top-left', '左上'], ['top-center', '上中'], ['top-right', '右上'],
        ['center-left', '左中'], ['center', '居中'], ['center-right', '右中'],
        ['bottom-left', '左下'], ['bottom-center', '下中'], ['bottom-right', '右下'],
    ].map(([v, l]) => `<option value="${v}">${l}</option>`).join('');

    list.innerHTML = wms.map((wm, i) => `
        <div class="wm-row" style="display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:6px;padding:6px;background:var(--bg-tertiary);border-radius:6px;">
            <label style="display:flex;align-items:center;gap:3px;"><input type="checkbox" class="wm-enabled" ${wm.enabled ? 'checked' : ''} onchange="_reelsSyncWatermarkFromUI()"> 启用</label>
            <input class="wm-text input" value="${(wm.text || '').replace(/"/g, '&quot;')}" style="width:150px;font-size:11px;padding:4px 6px;" oninput="_reelsSyncWatermarkFromUI()">
            <label style="display:flex;align-items:center;gap:2px;">字号:<input class="wm-fontsize input input-small" type="number" value="${wm.fontSize || 20}" min="8" max="80" style="width:48px;font-size:11px;padding:3px;" oninput="_reelsSyncWatermarkFromUI()"></label>
            <label style="display:flex;align-items:center;gap:2px;">字色:<input class="wm-color" type="color" value="${wm.color || '#FFFFFF'}" style="width:24px;height:20px;border:none;cursor:pointer;" oninput="_reelsSyncWatermarkFromUI()"></label>
            <label style="display:flex;align-items:center;gap:2px;">字透明:<input class="wm-textopacity input input-small" type="number" value="${Math.round((wm.textOpacity ?? 1.0) * 100)}" min="0" max="100" style="width:42px;font-size:11px;padding:3px;" oninput="_reelsSyncWatermarkFromUI()">%</label>
            <label style="display:flex;align-items:center;gap:2px;">底色:<input class="wm-bgcolor" type="color" value="${wm.bgColor || '#000000'}" style="width:24px;height:20px;border:none;cursor:pointer;" oninput="_reelsSyncWatermarkFromUI()"></label>
            <label style="display:flex;align-items:center;gap:2px;">底透明:<input class="wm-bgopacity input input-small" type="number" value="${Math.round((wm.bgOpacity ?? 0.5) * 100)}" min="0" max="100" style="width:42px;font-size:11px;padding:3px;" oninput="_reelsSyncWatermarkFromUI()">%</label>
            <select class="wm-position select" style="width:70px;font-size:11px;padding:3px;" onchange="_reelsSyncWatermarkFromUI()">${posOptions.replace(`value="${wm.position || 'top-right'}"`, `value="${wm.position || 'top-right'}" selected`)}</select>
            <button class="btn btn-secondary" style="font-size:10px;padding:2px 6px;color:#ff6b6b;" onclick="reelsRemoveWatermark(${i})">✕</button>
        </div>
    `).join('');
}

// 初始化水印 — 从 localStorage 恢复
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => {
        _reelsLoadWatermarks();
        _reelsRefreshWatermarkUI();
    }, 500);
});

function _drawVideoCover(ctx, videoEl, targetW, targetH, scalePct) {
    if (!ctx || !videoEl || !(targetW > 0) || !(targetH > 0)) return;
    const srcW = videoEl.videoWidth || videoEl.naturalWidth || targetW;
    const srcH = videoEl.videoHeight || videoEl.naturalHeight || targetH;
    if (!(srcW > 0) || !(srcH > 0)) {
        ctx.drawImage(videoEl, 0, 0, targetW, targetH);
        return;
    }
    const userScale = (scalePct || 100) / 100;
    const scale = Math.max(targetW / srcW, targetH / srcH) * userScale;
    const drawW = srcW * scale;
    const drawH = srcH * scale;
    const drawX = (targetW - drawW) / 2;
    const drawY = (targetH - drawH) / 2;
    ctx.drawImage(videoEl, drawX, drawY, drawW, drawH);
}

// ═══════════════════════════════════════════════════════
// File / Task management
// ═══════════════════════════════════════════════════════

function _normalizeBaseName(name) {
    return String(name || '').replace(/\.[^.]+$/, '').trim().toLowerCase();
}

function _fileExt(name) {
    const m = String(name || '').toLowerCase().match(/\.([a-z0-9]+)$/);
    return m ? m[1] : '';
}

function _isImagePath(filePath) {
    const ext = _fileExt(filePath || '');
    return ext === 'jpg' || ext === 'jpeg' || ext === 'png' || ext === 'webp';
}

function _getSelectedTask() {
    if (_reelsState.selectedIdx < 0) return null;
    return _reelsState.tasks[_reelsState.selectedIdx] || null;
}

function _toPlayablePath(filePath, srcUrl = null) {
    if (srcUrl) return srcUrl;
    if (!filePath) return '';
    if (window.electronAPI && typeof window.electronAPI.toFileUrl === 'function') {
        const u = window.electronAPI.toFileUrl(filePath);
        if (u) return u;
    }
    return filePath.startsWith('/') ? `file://${filePath}` : filePath;
}

function _getPreviewMasterElement() {
    const task = _getSelectedTask();
    if (!task) return null;
    const audio = document.getElementById('reels-preview-audio');
    if (task.audioPath && audio && audio.src && audio.readyState >= 1) return audio;
    const video = document.getElementById('reels-preview-video');
    const isVideo = task.bgPath && !_isImagePath(task.bgPath);
    if (isVideo && video && video.src && video.readyState >= 1) return video;
    const bgm = _reelsState._bgmAudioEl;
    if (task.bgmPath && bgm && bgm.src && bgm.readyState >= 1) return bgm;
    return null;
}

function _getPreviewCurrentTime() {
    const master = _getPreviewMasterElement();
    if (master) {
        const t = master.currentTime || 0;
        // 变速时 currentTime 要按比例缩放以匹配 _getPreviewDuration 的缩放后时长
        const task = _getSelectedTask();
        const aDurScale = (task && task.audioDurScale) ? (task.audioDurScale / 100) : 1;
        return t * aDurScale;
    } else {
        if (_reelsState.mockPlaying) {
            return Math.max(0, (performance.now() / 1000) - (_reelsState.mockStartTime || 0));
        }
        return _reelsState.mockPausedTime || 0;
    }
}

function _getPreviewDuration() {
    const task = _getSelectedTask();
    const audio = document.getElementById('reels-preview-audio');
    const video = document.getElementById('reels-preview-video');
    const subDur = task && task.segments && task.segments.length > 0
        ? (task.segments[task.segments.length - 1].end || 0)
        : 0;
    const aDur = audio && isFinite(audio.duration) ? (audio.duration || 0) : 0;
    const vDur = video && isFinite(video.duration) ? (video.duration || 0) : 0;

    // 音频变速：audioDurScale=150% → 实际播放时长 = 原时长 × 1.5
    const aDurScale = (task && task.audioDurScale) ? (task.audioDurScale / 100) : 1;
    const scaledADur = aDur * aDurScale;

    // 自定义时长优先
    if (task && task.customDuration && task.customDuration > 0) {
        return task.customDuration;
    }
    // 有音频时以变速后的音频时长为准（背景自动循环）
    if (scaledADur > 0) {
        return Math.max(scaledADur, subDur * aDurScale);
    }
    // 无音频时以视频时长为准，若仍无时长则推算虚拟进度
    const baseDur = Math.max(vDur, subDur, 0);
    if (baseDur <= 0 && !_getPreviewMasterElement()) {
        let maxOverlayEnd = 0;
        if (_reelsState.overlayProxy && _reelsState.overlayProxy.overlayMgr) {
            for (const ov of (_reelsState.overlayProxy.overlayMgr.overlays || [])) {
                if (parseFloat(ov.end || 0) > maxOverlayEnd) maxOverlayEnd = parseFloat(ov.end || 0);
            }
        }
        const demoWords = ((document.getElementById('reels-preview-text') || {}).value || '').split(/\s+/).filter(Boolean);
        const totalDur = Math.max(3, (demoWords.length || 1) * 0.6);
        return maxOverlayEnd > 0 ? maxOverlayEnd + 0.5 : totalDur;
    }
    return baseDur;
}

function _getPreviewLoopFadeConfig() {
    const loopFadeEl = document.getElementById('reels-loop-fade');
    const loopFadeDurEl = document.getElementById('reels-loop-fade-dur');
    const enabled = loopFadeEl ? loopFadeEl.checked : true;
    let duration = parseFloat(loopFadeDurEl ? loopFadeDurEl.value : '1');
    if (!Number.isFinite(duration) || duration <= 0) duration = 1.0;
    duration = Math.max(0.1, Math.min(3, duration));
    return { enabled, duration };
}

function _getPreviewAudioMixConfig() {
    let voiceVolume = parseFloat((document.getElementById('reels-voice-volume') || {}).value || '100');
    let bgVolume = parseFloat((document.getElementById('reels-bg-volume') || {}).value || '5');
    if (!Number.isFinite(voiceVolume)) voiceVolume = 100;
    if (!Number.isFinite(bgVolume)) bgVolume = 10;
    voiceVolume = Math.max(0, Math.min(200, voiceVolume));
    bgVolume = Math.max(0, Math.min(200, bgVolume));
    return { voiceGain: voiceVolume / 100, bgGain: bgVolume / 100 };
}

function _applyPreviewAudioMix() {
    const task = _getSelectedTask();
    const audio = document.getElementById('reels-preview-audio');
    const video = document.getElementById('reels-preview-video');
    const cfg = _getPreviewAudioMixConfig();
    const hasVoice = !!(task && task.audioPath && audio && audio.src);

    if (audio) {
        audio.volume = hasVoice ? cfg.voiceGain : 1.0;
        audio.muted = hasVoice ? (cfg.voiceGain <= 0.0001) : false;
    }
    if (video) {
        if (hasVoice) {
            video.volume = cfg.bgGain;
            video.muted = cfg.bgGain <= 0.0001;
        } else {
            // 无配音时预览以背景原声为主，和旧导出行为一致。
            video.volume = 1.0;
            video.muted = false;
        }
    }

    // ── BGM 音量 ──
    const bgmAudio = _reelsState._bgmAudioEl;
    if (bgmAudio) {
        if (task && task.bgmPath && bgmAudio.src) {
            const bgmVol = (task.bgmVolume != null ? task.bgmVolume : 10) / 100;
            bgmAudio.volume = Math.max(0, Math.min(1, bgmVol));
            bgmAudio.muted = bgmVol <= 0.001;
        } else {
            bgmAudio.volume = 0;
            bgmAudio.muted = true;
        }
    }
}

// ═══════════════════════════════════════════════════════
// 预览音频混响 + 立体声增强 (Web Audio API)
// ═══════════════════════════════════════════════════════

const _REVERB_PRESETS = {
    room:   { decay: 0.8, duration: 0.6, density: 3000, lpFreq: 8000 },
    hall:   { decay: 2.0, duration: 1.5, density: 5000, lpFreq: 6000 },
    church: { decay: 4.0, duration: 3.0, density: 8000, lpFreq: 4000 },
    plate:  { decay: 1.2, duration: 1.0, density: 6000, lpFreq: 10000 },
    echo:   { decay: 1.5, duration: 0.8, density: 1500, lpFreq: 5000 },
};

// 确定性伪随机数生成器（mulberry32），保证相同preset的IR在预览和导出中一致
function _mulberry32(seed) {
    return function() {
        seed |= 0; seed = seed + 0x6D2B79F5 | 0;
        let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}

// 将 preset 名字转为固定种子
function _presetSeed(preset) {
    let h = 0x811c9dc5;
    for (let i = 0; i < preset.length; i++) {
        h ^= preset.charCodeAt(i);
        h = Math.imul(h, 0x01000193);
    }
    return h >>> 0;
}

function _generateImpulseResponse(ctx, preset) {
    const p = _REVERB_PRESETS[preset] || _REVERB_PRESETS.hall;
    const presetKey = preset || 'hall';
    const sampleRate = ctx.sampleRate;
    const length = Math.ceil(sampleRate * p.duration);
    const buffer = ctx.createBuffer(2, length, sampleRate);
    for (let ch = 0; ch < 2; ch++) {
        // 每个声道用不同的种子，但同一 preset 始终相同
        const rng = _mulberry32(_presetSeed(presetKey) + ch * 0xDEAD);
        const data = buffer.getChannelData(ch);
        for (let i = 0; i < length; i++) {
            const t = i / sampleRate;
            const envelope = Math.exp(-t / (p.decay * 0.3));
            data[i] = (rng() * 2 - 1) * envelope;
        }
    }
    return buffer;
}

function _setupPreviewReverb() {
    const audio = document.getElementById('reels-preview-audio');
    const video = document.getElementById('reels-preview-video');
    const bgm = _reelsState._bgmAudioEl;
    if (!audio && !video && !bgm) return;

    const enabled = document.getElementById('reels-reverb-enabled')?.checked || false;
    const targetFx = document.getElementById('reels-audio-fx-target')?.value || 'all';
    const stereoWidth = (parseFloat(document.getElementById('reels-stereo-width')?.value) || 100) / 100;
    const mix = (parseFloat(document.getElementById('reels-reverb-mix')?.value) || 30) / 100;
    const needsFx = enabled || (stereoWidth > 1.05);

    // Initialize AudioContext and mediaSources Map if not present
    if (!_reelsState._audioCtx) {
        try {
            _reelsState._audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            _reelsState._mediaSources = new Map();
        } catch (e) {
            console.warn('[Reverb] Web Audio not supported', e);
            return;
        }
    }
    if (!_reelsState._mediaSources) _reelsState._mediaSources = new Map();
    const ctx = _reelsState._audioCtx;

    // Attach MediaElementSource for any new elements
    const els = [audio, video, bgm].filter(Boolean);
    for (const el of els) {
        if (!_reelsState._mediaSources.has(el)) {
            try {
                const source = ctx.createMediaElementSource(el);
                _reelsState._mediaSources.set(el, source);
            } catch (e) {
                console.warn('[Reverb] Failed to create source for', el, e);
            }
        }
    }

    // Disconnect everything fully before rewiring
    for (const source of _reelsState._mediaSources.values()) {
        try { source.disconnect(); } catch (e) { }
    }
    if (_reelsState._reverbGainWet) { try { _reelsState._reverbGainWet.disconnect(); } catch(e){} }
    if (_reelsState._reverbGainDry) { try { _reelsState._reverbGainDry.disconnect(); } catch(e){} }
    if (_reelsState._convolver) { try { _reelsState._convolver.disconnect(); } catch(e){} }
    if (_reelsState._stereoDelay) {
        try {
            _reelsState._stereoDelay.masterGain.disconnect();
            _reelsState._stereoDelay.splitter.disconnect();
            _reelsState._stereoDelay.delayL.disconnect();
            _reelsState._stereoDelay.delayR.disconnect();
            _reelsState._stereoDelay.merger.disconnect();
        } catch (e) {}
    }

    // Determine target element
    let targetEl = null;
    let targetSource = null;
    
    if (needsFx) {
        if ((targetFx === 'voice' || targetFx === 'all') && audio?.src) targetEl = audio;
        else if ((targetFx === 'bg' || targetFx === 'all') && video?.src) targetEl = video;
        else if ((targetFx === 'bgm' || targetFx === 'all') && bgm?.src) targetEl = bgm;
        // Fallback cascade
        if (!targetEl) {
            if (audio?.src) targetEl = audio;
            else if (video?.src) targetEl = video;
            else if (bgm?.src) targetEl = bgm;
        }
        if (targetEl) targetSource = _reelsState._mediaSources.get(targetEl);
    }

    // Connect non-targets directly to destination
    for (const [el, source] of _reelsState._mediaSources.entries()) {
        if (source !== targetSource) {
            source.connect(ctx.destination);
        }
    }

    // If no FX or no target, clean up and exit
    if (!needsFx || !targetSource) {
        if (targetSource) targetSource.connect(ctx.destination);
        return;
    }

    // --- Build FX Chain for target ---
    const preset = document.getElementById('reels-reverb-preset')?.value || 'hall';

    // Dry Gain
    const dryGain = ctx.createGain();
    dryGain.gain.value = enabled ? (1 - mix * 0.5) : 1.0; 

    // Wet Gain (Reverb)
    let convolver = null;
    let wetGain = null;
    if (enabled) {
        convolver = ctx.createConvolver();
        convolver.buffer = _generateImpulseResponse(ctx, preset);
        wetGain = ctx.createGain();
        wetGain.gain.value = mix;
    }

    const masterGain = ctx.createGain();
    masterGain.gain.value = 1.0;

    targetSource.connect(dryGain);
    dryGain.connect(masterGain);

    if (enabled) {
        targetSource.connect(convolver);
        convolver.connect(wetGain);
        wetGain.connect(masterGain);
    }

    // Stereo Expansion
    if (stereoWidth > 1.05) {
        const merger = ctx.createChannelMerger(2);
        const splitter = ctx.createChannelSplitter(2);
        const delayL = ctx.createDelay(0.05);
        const delayR = ctx.createDelay(0.05);
        const widthFactor = Math.max(0, (stereoWidth - 1)) * 0.015; 
        delayL.delayTime.value = widthFactor * 0.3;
        delayR.delayTime.value = widthFactor * 0.7;

        masterGain.connect(splitter);
        splitter.connect(delayL, 0);
        splitter.connect(delayR, 1);
        delayL.connect(merger, 0, 0);
        delayR.connect(merger, 0, 1);
        merger.connect(ctx.destination);

        _reelsState._stereoDelay = { delayL, delayR, splitter, merger, masterGain };
    } else {
        masterGain.connect(ctx.destination);
        _reelsState._stereoDelay = null;
    }

    // Save refs
    _reelsState._convolver = convolver;
    _reelsState._reverbGainWet = wetGain;
    _reelsState._reverbGainDry = dryGain;
}

function _getReverbConfig() {
    return {
        enabled: document.getElementById('reels-reverb-enabled')?.checked || false,
        preset: document.getElementById('reels-reverb-preset')?.value || 'hall',
        mix: parseFloat(document.getElementById('reels-reverb-mix')?.value || '30'),
        stereoWidth: parseFloat(document.getElementById('reels-stereo-width')?.value || '100'),
        audioFxTarget: document.getElementById('reels-audio-fx-target')?.value || 'all',
    };
}

function _resetPreviewFadeVideo() {
    const fadeVideo = _reelsState.previewFadeVideo;
    if (!fadeVideo) return;
    fadeVideo.pause();
    fadeVideo.removeAttribute('src');
    _reelsState.previewFadeVideoSrc = '';
}

function _ensurePreviewFadeVideo(mainVideo) {
    if (!mainVideo || !mainVideo.src) return null;
    if (!_reelsState.previewFadeVideo) {
        const fadeVideo = document.createElement('video');
        fadeVideo.id = 'reels-preview-video-fade';
        fadeVideo.muted = true;
        fadeVideo.loop = true;
        fadeVideo.playsInline = true;
        fadeVideo.preload = 'auto';
        fadeVideo.style.display = 'none';
        const host = document.getElementById('reels-preview-container') || document.body;
        host.appendChild(fadeVideo);
        _reelsState.previewFadeVideo = fadeVideo;
    }

    const fadeVideo = _reelsState.previewFadeVideo;
    if (_reelsState.previewFadeVideoSrc !== mainVideo.src) {
        fadeVideo.pause();
        fadeVideo.src = mainVideo.src;
        _reelsState.previewFadeVideoSrc = mainVideo.src;
    }
    return fadeVideo;
}

function _calcPreviewLoopFadeFrame() {
    const task = _getSelectedTask();
    const video = document.getElementById('reels-preview-video');
    const audio = document.getElementById('reels-preview-audio');
    if (!task || !video || !video.src || video.readyState < 2) return null;
    if (!task.audioPath || _isImagePath(task.bgPath || task.videoPath)) return null;

    const cfg = _getPreviewLoopFadeConfig();
    if (!cfg.enabled || !(video.duration > 0)) return null;

    const fadeDur = Math.min(cfg.duration, Math.max(0.1, video.duration * 0.45));
    if (!(video.duration > fadeDur + 0.05)) return null;

    const masterTime = _getPreviewCurrentTime();
    if (!Number.isFinite(masterTime) || masterTime < 0) return null;

    const loopTime = ((masterTime % video.duration) + video.duration) % video.duration;
    const remain = video.duration - loopTime;
    if (!(remain < fadeDur)) return null;

    const fadeVideo = _ensurePreviewFadeVideo(video);
    if (!fadeVideo) return null;

    const target = (loopTime + fadeDur) % video.duration;
    if (Math.abs((fadeVideo.currentTime || 0) - target) > 0.08) {
        try { fadeVideo.currentTime = target; } catch (e) { }
    }
    if (audio && !audio.paused && fadeVideo.paused) {
        fadeVideo.play().catch(() => { });
    }

    const alpha = Math.max(0, Math.min(1, (fadeDur - remain) / fadeDur));
    if (!(alpha > 0.001)) return null;
    return { video: fadeVideo, alpha };
}

function _syncBackgroundVideoToMaster() {
    const task = _getSelectedTask();
    const video = document.getElementById('reels-preview-video');
    const audio = document.getElementById('reels-preview-audio');
    if (!task || !video || !video.src || video.readyState < 1 || _isImagePath(task.bgPath || task.videoPath)) return;
    const masterTime = _getPreviewCurrentTime();
    if (!isFinite(masterTime) || masterTime < 0) return;
    if (video.duration > 0) {
        const target = masterTime % video.duration;
        if (Math.abs((video.currentTime || 0) - target) > 0.25) {
            try { video.currentTime = target; } catch (e) { }
        }
        // 某些容器在 ended 后会暂停，手动拉起继续播，保证背景持续循环。
        if (audio && !audio.paused && video.paused) {
            video.play().catch(() => { });
        }

        const cfg = _getPreviewLoopFadeConfig();
        if (task.audioPath && cfg.enabled && video.duration > cfg.duration + 0.05) {
            const fadeVideo = _ensurePreviewFadeVideo(video);
            if (fadeVideo) {
                const fadeDur = Math.min(cfg.duration, Math.max(0.1, video.duration * 0.45));
                const fadeTarget = (target + fadeDur) % video.duration;
                if (Math.abs((fadeVideo.currentTime || 0) - fadeTarget) > 0.2) {
                    try { fadeVideo.currentTime = fadeTarget; } catch (e) { }
                }
                if (audio && !audio.paused && fadeVideo.paused) {
                    fadeVideo.play().catch(() => { });
                }
            }
        } else if (_reelsState.previewFadeVideo) {
            _reelsState.previewFadeVideo.pause();
        }
    }
}

function _updatePreviewTimeUI(currentTime, duration) {
    const seekBar = document.getElementById('reels-preview-seek');
    const timeLabel = document.getElementById('reels-preview-time');
    if (seekBar && duration > 0) seekBar.value = Math.max(0, Math.min(100, (currentTime / duration) * 100));
    if (timeLabel) {
        const fmt = (s) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
        timeLabel.textContent = `${fmt(currentTime || 0)}/${fmt(duration || 0)}`;
    }
}

function _updateTimelineForTask(task) {
    if (!_reelsState.timelineEditor || !task) return;
    const editor = _reelsState.timelineEditor;
    if (task.segments && task.segments.length > 0) editor.loadSubtitleTrack(task.segments);
    else editor.loadSubtitleTrack([]);

    const audio = document.getElementById('reels-preview-audio');
    const video = document.getElementById('reels-preview-video');
    const aDur = audio && isFinite(audio.duration) ? (audio.duration || 0) : 0;
    const vDur = video && isFinite(video.duration) ? (video.duration || 0) : 0;
    const subDur = task.segments && task.segments.length > 0
        ? (task.segments[task.segments.length - 1].end || 0)
        : 0;
    const totalDur = Math.max(aDur, vDur, subDur, 1);
    editor.loadAudioTrack(aDur, task.audioPath ? '人声' : '音频');
    const bgTrackDur = task.audioPath ? totalDur : vDur;
    editor.loadBackgroundTrack(bgTrackDur, task.audioPath ? '背景(循环)' : '背景');
    editor.setDuration(totalDur);
}

function _buildAudioSubtitleMatchKey(name) {
    const normalized = _normalizeBaseName(name).replace(/[\u2013\u2014]/g, '-');
    const tokens = normalized
        .split(/[^a-z0-9]+/)
        .filter(Boolean)
        .filter(t => !REELS_MATCH_STOPWORDS.has(t));
    return tokens.join('_') || normalized;
}
window._buildAudioSubtitleMatchKey = _buildAudioSubtitleMatchKey;

function _inferTaskBaseName(task) {
    const src = task.baseName || task.fileName || task.audioPath || task.bgPath || task.videoPath || task.srtPath || '';
    const fileName = String(src).split(/[\\/]/).pop();
    return _normalizeBaseName(fileName);
}

function _getOrCreateTaskByBase(baseName, fallbackName = '') {
    const normalized = _normalizeBaseName(baseName || fallbackName);
    let task = _reelsState.tasks.find(t => _inferTaskBaseName(t) === normalized);
    if (task) return task;

    task = {
        baseName: normalized,
        fileName: fallbackName || `${normalized || 'reel'}.mp4`,
        bgPath: null,
        bgSrcUrl: null,
        audioPath: null,
        srtPath: null,
        segments: [],
        // 兼容旧字段
        videoPath: null,
        srcUrl: null,
    };
    _reelsState.tasks.push(task);
    return task;
}

function _buildFileInfo(file) {
    const name = file.name || '';
    let filePath = name;
    
    // 1. 尝试 Electron API（最可靠）
    if (window.electronAPI && window.electronAPI.getFilePath) {
        try {
            const p = window.electronAPI.getFilePath(file);
            if (p) filePath = p;
            console.log(`[_buildFileInfo] electronAPI.getFilePath("${name}") → "${p}"`);
        } catch (e) {
            console.warn(`[_buildFileInfo] electronAPI.getFilePath error:`, e);
        }
    }
    
    // 2. 回退: file.path（旧 Electron / contextIsolation:false）
    if (filePath === name && file.path) {
        filePath = file.path;
        console.log(`[_buildFileInfo] fallback to file.path: "${filePath}"`);
    }
    
    // 3. 最终回退: 仅文件名
    if (filePath === name) {
        console.warn(`[_buildFileInfo] ⚠️ 无法获取完整路径，仅文件名: "${name}"`);
    }
    
    return {
        name,
        path: filePath,
        baseName: _normalizeBaseName(name),
        matchKey: _buildAudioSubtitleMatchKey(name),
    };
}

function _pushPendingUnique(list, item) {
    const key = item.path || item.name;
    const exists = list.some(x => (x.path || x.name) === key);
    if (!exists) list.push(item);
}

function _queueBackgroundFile(file) {
    const info = _buildFileInfo(file);
    let srcUrl = null;
    try { srcUrl = URL.createObjectURL(file); } catch (e) { }
    info.srcUrl = srcUrl;
    _pushPendingUnique(_reelsState.pendingFiles.backgrounds, info);
}

function _upsertBackgroundLibrary(bg) {
    if (!bg || !bg.path) return;
    const idx = _reelsState.backgroundLibrary.findIndex(x => x.path === bg.path);
    if (idx >= 0) {
        _reelsState.backgroundLibrary[idx] = { ..._reelsState.backgroundLibrary[idx], ...bg };
    } else {
        _reelsState.backgroundLibrary.push({ ...bg });
    }
}

function _queueAudioFile(file) {
    const info = _buildFileInfo(file);
    _pushPendingUnique(_reelsState.pendingFiles.audios, info);
}

function _queueSrtFile(file) {
    const info = _buildFileInfo(file);
    const reader = new FileReader();
    reader.onload = (ev) => {
        info.content = ev.target.result;
        _pushPendingUnique(_reelsState.pendingFiles.srts, info);
        reelsAutoMatchFiles();
    };
    reader.readAsText(file);
}

function _queueTxtFile(file) {
    const info = _buildFileInfo(file);
    const reader = new FileReader();
    reader.onload = (ev) => {
        info.content = ev.target.result;
        _pushPendingUnique(_reelsState.pendingFiles.txts, info);
        reelsAutoMatchFiles();
    };
    reader.readAsText(file);
}

function _onTxtFilesSelected(e) {
    const files = Array.from(e.target.files || []);
    for (const f of files) _queueTxtFile(f);
    e.target.value = '';
}

// 手动输入字幕弹窗
async function reelsManualSubtitleInput() {
    const result = await _showTextareaDialog(
        '✏️ 手动输入字幕文本',
        '每行 = 一条字幕段落（已断行的文本）\n支持多行，每行将作为独立字幕条目。',
        ''
    );
    if (!result || !result.trim()) return;

    // 视为一条 TXT 输入
    const info = {
        name: '_manual_input.txt',
        path: '_manual_input.txt',
        baseName: '_manual_input',
        matchKey: '',
        content: result,
    };
    _pushPendingUnique(_reelsState.pendingFiles.txts, info);
    reelsAutoMatchFiles();
}

// 通用 textarea 弹窗
function _showTextareaDialog(title, placeholder, defaultVal) {
    return new Promise(resolve => {
        const overlay = document.createElement('div');
        overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.55);z-index:99999;display:flex;align-items:center;justify-content:center;';
        const box = document.createElement('div');
        box.style.cssText = 'background:var(--bg-secondary,#1e1e2e);border-radius:12px;padding:24px;width:520px;max-width:90vw;box-shadow:0 12px 40px rgba(0,0,0,0.5);';
        box.innerHTML = `
            <h3 style="margin:0 0 12px;font-size:16px;">${title}</h3>
            <textarea id="_reels_textarea_dlg" rows="10"
                style="width:100%;padding:10px;border-radius:8px;border:1px solid var(--border-color,#444);
                       background:var(--bg-tertiary,#2a2a3e);color:var(--text-primary,#eee);
                       font-size:13px;resize:vertical;font-family:inherit;"
                placeholder="${placeholder}">${defaultVal || ''}</textarea>
            <div style="display:flex;gap:8px;justify-content:flex-end;margin-top:12px;">
                <button class="btn btn-secondary" id="_reels_textarea_cancel">取消</button>
                <button class="btn btn-primary" id="_reels_textarea_ok">确认</button>
            </div>`;
        overlay.appendChild(box);
        document.body.appendChild(overlay);

        const ta = box.querySelector('#_reels_textarea_dlg');
        const close = (val) => { document.body.removeChild(overlay); resolve(val); };
        box.querySelector('#_reels_textarea_cancel').onclick = () => close(null);
        box.querySelector('#_reels_textarea_ok').onclick = () => close(ta.value);
        overlay.onclick = (e) => { if (e.target === overlay) close(null); };
        setTimeout(() => ta.focus(), 50);
    });
}

// ═══════════════════════════════════════════════════════
// Subtitle alignment (call existing subtitle/generate API)
// ═══════════════════════════════════════════════════════

async function _reelsAlignSubtitles(task) {
    const txtContent = task.txtContent || task.manualText || '';
    if (!txtContent.trim()) throw new Error('没有字幕文本');

    // 确定音频源路径
    const workMode = _getWorkMode();
    let audioPath;
    if (workMode === 'voiced_bg') {
        audioPath = task.bgPath || task.videoPath;
    } else {
        audioPath = task.audioPath;
    }
    if (!audioPath) throw new Error('没有音频文件可用于对齐');

    // 调用现有的 subtitle/generate API
    const language = document.getElementById('reels-align-lang')?.value || '英语';
    // 输出目录 = 音频/视频文件所在目录（SRT 保存到文件旁边）
    const audioDir = audioPath.replace(/[\\/][^\\/]+$/, '');
    const response = await apiFetch(`${API_BASE}/subtitle/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            audio_path: audioPath,
            source_text: txtContent,
            language: language,
            audio_cut_length: 5.0,
            output_dir: audioDir,
        }),
    });

    if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || '字幕对齐失败');
    }

    const data = await response.json();

    // 生成的 SRT 文件路径
    if (data.files && data.files.length > 0) {
        // 找到 source.srt 文件
        const srtFile = data.files.find(f => f.endsWith('_source.srt')) || data.files[0];
        task.srtPath = srtFile;

        // 读取 SRT 文件并解析为 segments
        if (window.electronAPI && window.electronAPI.readFileText) {
            const srtContent = await window.electronAPI.readFileText(srtFile);
            const rawSegs = parseSRT(srtContent).map(seg => ({ ...seg, _timeUnit: 'sec' }));
            task.segments = window.ReelsSubtitleProcessor
                ? ReelsSubtitleProcessor.srtToSegmentsWithWords(rawSegs)
                : rawSegs;
        }
    }

    task.aligned = true;
    return task;
}

// 对齐所有未对齐的 TXT 任务
async function reelsAlignAllTasks() {
    const workMode = _getWorkMode();
    if (workMode === 'srt') return; // SRT 模式不需要对齐

    const tasksToAlign = _reelsState.tasks.filter(t =>
        t.txtContent && !t.aligned && (t.segments || []).length === 0
    );
    if (tasksToAlign.length === 0) {
        alert('没有需要对齐的任务');
        return;
    }

    const statusEl = document.getElementById('reels-export-status');
    let ok = 0, fail = 0;

    for (let i = 0; i < tasksToAlign.length; i++) {
        const task = tasksToAlign[i];
        if (statusEl) statusEl.textContent = `对齐中 ${i + 1}/${tasksToAlign.length}: ${task.fileName}`;
        try {
            await _reelsAlignSubtitles(task);
            ok++;
        } catch (err) {
            console.error('[Reels] Align failed:', task.fileName, err);
            fail++;
        }
    }

    _renderTaskList();
    if (statusEl) {
        statusEl.textContent = fail > 0
            ? `⚠️ 对齐完成 ${ok}/${tasksToAlign.length}，失败 ${fail}`
            : `✅ 对齐完成 (${ok}个任务)`;
    }
    if (fail === 0 && ok > 0) {
        alert(`字幕对齐完成 ${ok} 个任务`);
    }
}

// ═══════════════════════════════════════════════════════
// Work mode switching
// ═══════════════════════════════════════════════════════

function reelsOnWorkModeChange() {
    const mode = _getWorkMode();
    const audioBtn = document.getElementById('reels-audio-btn');
    const srtBtn = document.getElementById('reels-srt-btn');
    const txtBtn = document.getElementById('reels-txt-btn');
    const manualBtn = document.getElementById('reels-manual-btn');
    const alignBtn = document.getElementById('reels-align-btn');
    const alignLang = document.getElementById('reels-align-lang');
    const bgLabel = document.getElementById('reels-bg-btn');

    if (mode === 'srt') {
        // 人声Reels: 背景 + 配音 + SRT
        if (audioBtn) audioBtn.style.display = '';
        if (srtBtn) srtBtn.style.display = '';
        if (txtBtn) txtBtn.style.display = 'none';
        if (manualBtn) manualBtn.style.display = 'none';
        if (alignBtn) alignBtn.style.display = 'none';
        if (alignLang) alignLang.style.display = 'none';
        if (bgLabel) bgLabel.textContent = '📁 背景';
    } else if (mode === 'dubbed_text') {
        // 配音+文本: 背景 + 配音 + TXT
        if (audioBtn) audioBtn.style.display = '';
        if (srtBtn) srtBtn.style.display = 'none';
        if (txtBtn) txtBtn.style.display = '';
        if (manualBtn) manualBtn.style.display = '';
        if (alignBtn) alignBtn.style.display = '';
        if (alignLang) alignLang.style.display = '';
        if (bgLabel) bgLabel.textContent = '📁 背景';
    } else if (mode === 'voiced_bg') {
        // 带声视频: 带声视频 + TXT
        if (audioBtn) audioBtn.style.display = 'none';
        if (srtBtn) srtBtn.style.display = 'none';
        if (txtBtn) txtBtn.style.display = '';
        if (manualBtn) manualBtn.style.display = '';
        if (alignBtn) alignBtn.style.display = '';
        if (alignLang) alignLang.style.display = '';
        if (bgLabel) bgLabel.textContent = '📁 带声视频';
    }
}

function _queueMixedFiles(files) {
    const workMode = _getWorkMode();
    for (const file of files) {
        const ext = _fileExt(file.name || '');
        if (ext === 'srt' && workMode === 'srt') {
            _queueSrtFile(file);
        } else if (ext === 'txt' && workMode !== 'srt') {
            _queueTxtFile(file);
        } else if (ext === 'srt') {
            _queueSrtFile(file);
        } else if (REELS_AUDIO_EXTS.has(ext)) {
            _queueAudioFile(file);
        } else if (REELS_BACKGROUND_EXTS.has(ext)) {
            _queueBackgroundFile(file);
        }
    }
}

function _onVideoFilesSelected(e) {
    const files = Array.from(e.target.files || []);
    for (const f of files) _queueBackgroundFile(f);
    reelsAutoMatchFiles();
    e.target.value = '';
}

function _onAudioFilesSelected(e) {
    const files = Array.from(e.target.files || []);
    for (const f of files) _queueAudioFile(f);
    reelsAutoMatchFiles();
    e.target.value = '';
}

function _onSrtFilesSelected(e) {
    const files = Array.from(e.target.files || []);
    for (const f of files) _queueSrtFile(f);
    e.target.value = '';
}

function _onFolderFilesSelected(e) {
    const files = Array.from(e.target.files || []);
    _queueMixedFiles(files);
    reelsAutoMatchFiles();
    e.target.value = '';
}

function _onTaskListDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.style.borderColor = '';
    e.currentTarget.style.backgroundColor = '';
    e.currentTarget.style.boxShadow = '';
    const files = Array.from(e.dataTransfer.files || []);
    _queueMixedFiles(files);
    reelsAutoMatchFiles();
}

function _getMatchMode() {
    const el = document.getElementById('reels-match-mode');
    return el ? el.value : 'free';
}

function _getWorkMode() {
    const el = document.getElementById('reels-work-mode');
    return el ? el.value : 'srt';
}

function _getBgAssignMode() {
    const el = document.getElementById('reels-bg-assign-mode');
    return el ? el.value : 'cycle';
}

function _applyFreeBackgroundAssignment() {
    const library = _reelsState.backgroundLibrary || [];
    if (library.length === 0) return;

    const assignMode = _getBgAssignMode();
    // free 模式下，TXT/手动文本任务也需要拿到背景，便于点击预览
    const targetTasks = _reelsState.tasks.filter(t =>
        t.audioPath || t.srtPath || t.txtContent || t.manualText
    );
    if (targetTasks.length === 0) return;

    if (assignMode === 'single') {
        const firstBg = library[0];
        for (const task of targetTasks) {
            task.bgPath = firstBg.path;
            task.bgSrcUrl = firstBg.srcUrl || null;
            task.videoPath = firstBg.path;
            task.srcUrl = firstBg.srcUrl || null;
        }
        return;
    }

    for (let i = 0; i < targetTasks.length; i++) {
        const bg = library[i % library.length];
        targetTasks[i].bgPath = bg.path;
        targetTasks[i].bgSrcUrl = bg.srcUrl || null;
        targetTasks[i].videoPath = bg.path;
        targetTasks[i].srcUrl = bg.srcUrl || null;
    }
}

function _ensurePreviewTaskForBackgroundOnlyInFreeMode() {
    if (_getMatchMode() !== 'free') return;
    if (_reelsState.tasks.length > 0) return;
    const firstBg = (_reelsState.backgroundLibrary || [])[0];
    if (!firstBg || !firstBg.path) return;

    const task = _getOrCreateTaskByBase(firstBg.baseName || firstBg.name, firstBg.name || 'background.mp4');
    task.baseName = firstBg.baseName || _normalizeBaseName(firstBg.name || firstBg.path);
    task.fileName = firstBg.name || String(firstBg.path).split(/[\\/]/).pop() || 'background.mp4';
    task.bgPath = firstBg.path;
    task.bgSrcUrl = firstBg.srcUrl || null;
    task.videoPath = firstBg.path;
    task.srcUrl = firstBg.srcUrl || null;
}

function _ensureBackgroundLibraryFromTasks() {
    for (const task of _reelsState.tasks) {
        const bgPath = task.bgPath || task.videoPath;
        if (!bgPath) continue;
        _upsertBackgroundLibrary({
            path: bgPath,
            name: String(bgPath).split(/[\\/]/).pop(),
            baseName: _normalizeBaseName(bgPath),
            srcUrl: task.bgSrcUrl || task.srcUrl || null,
        });
    }
}

function _getOrCreateFreeTaskForAudio(audio) {
    const base = _normalizeBaseName(audio.baseName || audio.name);
    const key = audio.matchKey || _buildAudioSubtitleMatchKey(base);
    let task = _reelsState.tasks.find(t => !t.audioPath && t.matchKey === key);
    if (!task) task = _getOrCreateTaskByBase(base, audio.name);
    task.matchKey = key;
    return task;
}

function _getOrCreateFreeTaskForSrt(srt) {
    const base = _normalizeBaseName(srt.baseName || srt.name);
    const key = srt.matchKey || _buildAudioSubtitleMatchKey(base);
    let task = _reelsState.tasks.find(t => !t.srtPath && t.matchKey === key);
    if (!task) task = _getOrCreateTaskByBase(base, srt.name);
    task.matchKey = key;
    return task;
}

function _pruneFreeBgOnlyTasks() {
    _reelsState.tasks = _reelsState.tasks.filter(t => {
        const hasBg = !!(t.bgPath || t.videoPath);
        const hasAudio = !!t.audioPath;
        const hasSrt = !!t.srtPath;
        const hasTxt = !!t.txtContent;
        if (hasBg && !hasAudio && !hasSrt && !hasTxt) return false;
        return true;
    });
}

function reelsAutoMatchFiles() {
    const backgrounds = _reelsState.pendingFiles.backgrounds.splice(0);
    const audios = _reelsState.pendingFiles.audios.splice(0);
    const srts = _reelsState.pendingFiles.srts.splice(0);
    const txts = _reelsState.pendingFiles.txts.splice(0);
    const matchMode = _getMatchMode();

    for (const bg of backgrounds) {
        _upsertBackgroundLibrary(bg);
        if (matchMode !== 'strict') continue;
        const task = _getOrCreateTaskByBase(bg.baseName, bg.name);
        task.baseName = bg.baseName;
        task.bgPath = bg.path;
        task.bgSrcUrl = bg.srcUrl || null;
        // 兼容旧导出逻辑字段
        task.videoPath = bg.path;
        task.srcUrl = bg.srcUrl || null;
        if (!task.fileName) task.fileName = bg.name;
    }

    for (const audio of audios) {
        const task = matchMode === 'free'
            ? _getOrCreateFreeTaskForAudio(audio)
            : _getOrCreateTaskByBase(audio.baseName, audio.name);
        task.baseName = audio.baseName;
        task.audioPath = audio.path;
        if (matchMode === 'free') {
            task.fileName = audio.name;
        } else if (!task.fileName) {
            task.fileName = audio.name;
        }
    }

    for (const srt of srts) {
        const task = matchMode === 'free'
            ? _getOrCreateFreeTaskForSrt(srt)
            : _getOrCreateTaskByBase(srt.baseName, srt.name);
        const rawSegs = parseSRT(srt.content || '').map(seg => ({
            ...seg,
            _timeUnit: 'sec',
        }));
        const segments = window.ReelsSubtitleProcessor
            ? ReelsSubtitleProcessor.srtToSegmentsWithWords(rawSegs)
            : rawSegs;
        task.baseName = srt.baseName;
        task.srtPath = srt.path;
        task.segments = segments;
        if (!task.fileName) task.fileName = srt.name.replace(/\.srt$/i, '.mp4');
    }

    // TXT 文件处理（模式 A/B）
    for (const txt of txts) {
        const task = matchMode === 'free'
            ? _getOrCreateFreeTaskForSrt(txt) // 复用 free 匹配逻辑
            : _getOrCreateTaskByBase(txt.baseName, txt.name);
        task.baseName = txt.baseName;
        task.txtPath = txt.path;
        task.txtContent = txt.content;
        task.aligned = false;
        // 暂不设置 segments，等待对齐后填入
        if (!task.fileName) task.fileName = txt.name.replace(/\.txt$/i, '.mp4');
    }

    if (matchMode === 'free') {
        _pruneFreeBgOnlyTasks();
        _ensureBackgroundLibraryFromTasks();
        _applyFreeBackgroundAssignment();
        _ensurePreviewTaskForBackgroundOnlyInFreeMode();
    }

    for (const task of _reelsState.tasks) {
        if (!task.baseName) task.baseName = _inferTaskBaseName(task);
        if (!task.fileName) {
            const src = task.audioPath || task.bgPath || task.videoPath || task.srtPath || task.txtPath || '';
            const name = src ? src.split(/[\\/]/).pop() : `${task.baseName || 'reel'}.mp4`;
            task.fileName = name;
        }
    }

    if (_reelsState.selectedIdx >= _reelsState.tasks.length) {
        _reelsState.selectedIdx = _reelsState.tasks.length - 1;
    }
    _renderTaskList();
    if (_reelsState.selectedIdx < 0 && _reelsState.tasks.length > 0) {
        reelsSelectTask(0);
    }
}

function reelsClearTasks() {
    _reelsState.tasks = [];
    _reelsState.selectedIdx = -1;
    _reelsState.pendingFiles = { backgrounds: [], audios: [], srts: [], txts: [] };
    _reelsState.backgroundLibrary = [];

    // Clear overlay manager and panel
    if (_reelsState.overlayProxy && _reelsState.overlayProxy.overlayMgr) {
        _reelsState.overlayProxy.overlayMgr.overlays = [];
    }
    if (_reelsState.overlayPanel) {
        _reelsState.overlayPanel.deselectOverlay();
        _reelsState.overlayPanel._refreshList();
    }

    // Clear video/audio preview
    _reelsState._previewBgImage = null;
    const video = document.getElementById('reels-preview-video');
    const audio = document.getElementById('reels-preview-audio');
    const placeholder = document.getElementById('reels-preview-placeholder');
    if (video) {
        video.pause();
        video.removeAttribute('src');
        video.style.display = 'none';
    }
    if (audio) {
        audio.pause();
        audio.removeAttribute('src');
    }
    if (placeholder) {
        placeholder.style.display = 'flex';
        placeholder.textContent = '选择视频任务后可实时预览字幕效果';
    }
    _resetPreviewFadeVideo();

    _renderTaskList();
}

function _renderTaskList() {
    const container = document.getElementById('reels-task-list');
    const countEl = document.getElementById('reels-task-count');
    const countPanelEl = document.getElementById('reels-task-count-panel');
    if (!container) return;

    const tasks = _reelsState.tasks;
    const workMode = _getWorkMode();
    if (countEl) countEl.textContent = `${tasks.length} 个任务`;
    if (countPanelEl) countPanelEl.textContent = tasks.length > 0 ? `${tasks.length}` : '0';

    if (tasks.length === 0) {
        const hint = workMode === 'srt'
            ? '添加背景素材 + 配音 + SRT，支持拖拽和文件夹导入；同名自动配对。'
            : workMode === 'dubbed_text'
                ? '添加背景素材 + 配音 + TXT（或手动输入），然后点击「🔗 对齐」生成字幕时间轴。'
                : '添加带声视频 + TXT（或手动输入），然后点击「🔗 对齐」生成字幕时间轴。';
        container.innerHTML = `<p class="hint" style="font-size:11px;">${hint}</p>`;
        return;
    }

    container.innerHTML = tasks.map((task, i) => {
        const selected = i === _reelsState.selectedIdx;
        const hasBg = !!(task.bgPath || task.videoPath);
        const hasAudio = !!task.audioPath;
        const hasSrt = !!task.srtPath && (task.segments || []).length > 0;
        const hasTxt = !!task.txtContent;

        let statusParts;
        if (workMode === 'voiced_bg') {
            statusParts = [
                hasBg ? '<span style="color:#4ecdc4;">BG</span>' : '<span style="color:#ff6b6b;">BG</span>',
            ];
            if (hasSrt) {
                statusParts.push(`<span style="color:#4ecdc4;">SRT</span>`);
            } else if (hasTxt) {
                statusParts.push(`<span style="color:#ffa502;">TXT</span>`);
            } else {
                statusParts.push('<span style="color:#ff6b6b;">TXT</span>');
            }
        } else if (workMode === 'dubbed_text') {
            statusParts = [
                hasBg ? '<span style="color:#4ecdc4;">BG</span>' : '<span style="color:#ff6b6b;">BG</span>',
                hasAudio ? '<span style="color:#4ecdc4;">VO</span>' : '<span style="color:#ff6b6b;">VO</span>',
            ];
            if (hasSrt) {
                statusParts.push(`<span style="color:#4ecdc4;">SRT</span>`);
            } else if (hasTxt) {
                statusParts.push(`<span style="color:#ffa502;">TXT</span>`);
            } else {
                statusParts.push('<span style="color:#ff6b6b;">TXT</span>');
            }
        } else {
            statusParts = [
                hasBg ? '<span style="color:#4ecdc4;">BG</span>' : '<span style="color:#ff6b6b;">BG</span>',
                hasAudio ? '<span style="color:#4ecdc4;">VO</span>' : '<span style="color:#ff6b6b;">VO</span>',
                hasSrt ? `<span style="color:#4ecdc4;">SRT</span>` : '<span style="color:#ff6b6b;">SRT</span>',
            ];
        }
        const statusText = statusParts.join(' ');
        // Shorten filename for compact display
        const baseName = task.fileName.replace(/\.[^.]+$/, '');
        const shortName = baseName.length > 18 ? baseName.substring(0, 16) + '…' : baseName;

        // 覆层内容预览
        let ovPreview = '';
        if (task.overlays && task.overlays.length > 0) {
            const ov0 = task.overlays[0];
            let ovTitle = '', ovBody = '';
            if (ov0.type === 'scroll') {
                ovTitle = (ov0.scroll_title || '').trim();
                ovBody = (ov0.content || '').trim().replace(/\n/g, ' ');
            } else if (ov0.type === 'textcard') {
                ovTitle = (ov0.title_text || '').trim();
                ovBody = (ov0.body_text || '').trim().replace(/\n/g, ' ');
            }
            if (ovTitle || ovBody) {
                const icon = ov0.type === 'scroll' ? '🔄' : '📝';
                const tSnip = ovTitle.length > 12 ? ovTitle.substring(0, 10) + '…' : ovTitle;
                const bSnip = ovBody.length > 20 ? ovBody.substring(0, 18) + '…' : ovBody;
                const parts = [];
                if (tSnip) parts.push(`<b>${tSnip}</b>`);
                if (bSnip) parts.push(bSnip);
                ovPreview = `<div style="font-size:10px;color:#8899aa;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:160px;" title="${ovTitle}\n${ovBody}">${icon} ${parts.join(' | ')}</div>`;
            }
        }

        return `
            <div class="reels-task-item ${selected ? 'reels-task-selected' : ''}"
                 onclick="reelsSelectTask(${i})"
                 title="${task.fileName}"
                 style="display:flex; align-items:center; gap:4px; padding:5px 6px; margin-bottom:2px;
                        border-radius:5px; cursor:pointer; transition:background .12s;
                        background: ${selected ? 'rgba(0,212,255,0.15)' : 'transparent'};
                        border-left: 3px solid ${selected ? '#00D4FF' : 'transparent'};
                        ${selected ? 'box-shadow: inset 0 0 0 1px rgba(0,212,255,0.3);' : ''}">
                <span style="font-size:12px; font-weight:${selected ? '600' : '400'}; color:${selected ? '#fff' : 'var(--text-primary)'}; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; min-width:60px; max-width:120px;">${shortName}</span>
                ${ovPreview}
                <span style="font-size:10px; white-space:nowrap; opacity:0.8; margin-left:auto;">${statusText}</span>
                <button class="btn" style="padding:1px 4px; font-size:10px; opacity:0.5; border:none; background:transparent; color:var(--text-secondary);" onclick="event.stopPropagation(); reelsRemoveTask(${i})" title="删除">✕</button>
            </div>
        `;
    }).join('');

    // Auto-scroll selected task into view
    const selectedEl = container.querySelector('.reels-task-selected');
    if (selectedEl) selectedEl.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}

function reelsSelectTask(idx) {
    // ── 保存当前任务的覆层 ──
    const prevTask = _reelsState.tasks[_reelsState.selectedIdx];
    if (prevTask && _reelsState.overlayProxy && _reelsState.overlayProxy.overlayMgr) {
        prevTask.overlays = [...(_reelsState.overlayProxy.overlayMgr.overlays || [])];
    }

    _reelsState.selectedIdx = idx;
    _renderTaskList();
    const task = _reelsState.tasks[idx];
    if (!task) return;

    // ── 加载新任务的覆层 ──
    if (_reelsState.overlayProxy && _reelsState.overlayProxy.overlayMgr) {
        const mgr = _reelsState.overlayProxy.overlayMgr;
        mgr.overlays = task.overlays ? [...task.overlays] : [];
        // 刷新覆层面板
        if (_reelsState.overlayPanel) {
            _reelsState.overlayPanel.deselectOverlay();
            _reelsState.overlayPanel._refreshList();
        }
    }

    const video = document.getElementById('reels-preview-video');
    const audio = document.getElementById('reels-preview-audio');
    const playBtn = document.getElementById('reels-preview-play');
    const placeholder = document.getElementById('reels-preview-placeholder');
    const bgPath = task.bgPath || task.videoPath;
    const bgSrc = task.bgSrcUrl || task.srcUrl;
    const workMode = _getWorkMode();
    // In voiced_bg mode, the background video IS the audio source
    const voicePath = task.audioPath || (workMode === 'voiced_bg' ? bgPath : '') || '';

    if (audio) {
        audio.pause();
        if (voicePath) {
            const audioUrl = _toPlayablePath(voicePath, null);
            if (audio.src !== audioUrl) audio.src = audioUrl;
        } else {
            audio.removeAttribute('src');
        }
        // 应用音频变速预览：audioDurScale=150% → playbackRate=0.667（减速）
        const aDurScale = task.audioDurScale || 100;
        audio.playbackRate = (aDurScale !== 100) ? (100 / aDurScale) : 1.0;
        audio.preservesPitch = true; // 变速不变调
    }

    // ── 加载 BGM ──
    const bgmAudio = _reelsState._bgmAudioEl;
    if (bgmAudio) {
        bgmAudio.pause();
        if (task.bgmPath) {
            const bgmUrl = _toPlayablePath(task.bgmPath, null);
            if (bgmAudio.src !== bgmUrl) bgmAudio.src = bgmUrl;
        } else {
            bgmAudio.removeAttribute('src');
        }
    }
    _applyPreviewAudioMix();

    if (video && bgPath) {
        if (_isImagePath(bgPath)) {
            video.pause();
            video.removeAttribute('src');
            _resetPreviewFadeVideo();
            video.style.display = 'none';
            // Load image background for canvas rendering
            const imgUrl = _toPlayablePath(bgPath, bgSrc);
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = () => { _reelsState._previewBgImage = img; };
            img.src = imgUrl;
            _reelsState._previewBgImage = img;
            if (placeholder) {
                placeholder.style.display = 'none';
            }
        } else {
            _reelsState._previewBgImage = null; // Clear image bg
            const filePath = _toPlayablePath(bgPath, bgSrc);
            // 总是重新设置 src（避免 URL 规范化导致比较失误）
            video.pause();
            video.src = filePath;
            video.load();
            video.loop = !!voicePath;
            // 强制加载第一帧 — seek 到 0.01s 触发帧数据加载
            video.addEventListener('loadeddata', function _onLoaded() {
                video.removeEventListener('loadeddata', _onLoaded);
                console.log('[Preview] Video loadeddata, readyState:', video.readyState);
            }, { once: true });
            try { video.currentTime = 0.01; } catch (e) { }
            const fadeVideo = _ensurePreviewFadeVideo(video);
            if (fadeVideo) {
                fadeVideo.pause();
                try { fadeVideo.currentTime = 0; } catch (e) { }
            }
            video.style.display = 'block';
            if (placeholder) {
                placeholder.style.display = 'none';
                placeholder.textContent = '选择视频任务后可实时预览字幕效果';
            }
        }
    } else if (video) {
        _reelsState._previewBgImage = null; // Clear image bg
        video.pause();
        video.removeAttribute('src');
        _resetPreviewFadeVideo();
        video.style.display = 'none';
        if (placeholder) {
            placeholder.style.display = 'flex';
            placeholder.textContent = '当前任务没有背景素材，预览将显示纯色底。';
        }
    }

    const previewText = document.getElementById('reels-preview-text');
    if (previewText && task.segments.length > 0) {
        previewText.value = task.segments[0].text;
    }

    _updateTimelineForTask(task);
    _reelsState.mockPlaying = false;
    _reelsState.mockPausedTime = 0;
    _updatePreviewTimeUI(0, _getPreviewDuration());
    if (playBtn) playBtn.textContent = '▶️';
}

function reelsRemoveTask(idx) {
    if (idx < 0 || idx >= _reelsState.tasks.length) return;
    const prevSelectedIdx = _reelsState.selectedIdx;
    _reelsState.tasks.splice(idx, 1);

    if (_reelsState.tasks.length === 0) {
        _reelsState.selectedIdx = -1;
        _renderTaskList();

        _reelsState._previewBgImage = null;
        const video = document.getElementById('reels-preview-video');
        const audio = document.getElementById('reels-preview-audio');
        const playBtn = document.getElementById('reels-preview-play');
        const placeholder = document.getElementById('reels-preview-placeholder');
        if (video) {
            video.pause();
            video.removeAttribute('src');
            video.style.display = 'none';
        }
        if (audio) {
            audio.pause();
            audio.removeAttribute('src');
        }
        _resetPreviewFadeVideo();
        if (placeholder) {
            placeholder.style.display = 'flex';
            placeholder.textContent = '选择任务以预览';
        }
        if (playBtn) playBtn.textContent = '▶️';
        _updatePreviewTimeUI(0, 0);
        return;
    }

    // 维护删除后的选中索引
    let nextSelectedIdx = prevSelectedIdx;
    if (prevSelectedIdx === idx) {
        nextSelectedIdx = Math.min(idx, _reelsState.tasks.length - 1);
    } else if (prevSelectedIdx > idx) {
        nextSelectedIdx = prevSelectedIdx - 1;
    }
    _reelsState.selectedIdx = Math.max(0, Math.min(nextSelectedIdx, _reelsState.tasks.length - 1));

    // 统一走选择逻辑，确保预览背景/音频/时间线同步
    reelsSelectTask(_reelsState.selectedIdx);
}

// ═══════════════════════════════════════════════════════
// Video preview controls
function reelsTogglePlay() {
    const video = document.getElementById('reels-preview-video');
    const audio = document.getElementById('reels-preview-audio');
    const fadeVideo = _reelsState.previewFadeVideo;
    const btn = document.getElementById('reels-preview-play');
    const task = _getSelectedTask();
    _applyPreviewAudioMix();

    const hasAudio = !!(task && task.audioPath && audio && audio.src);
    const hasVideo = !!(task && task.bgPath && !_isImagePath(task.bgPath) && video && video.src);
    const master = _getPreviewMasterElement();
    const hasMedia = !!master;
    const isPlaying = hasMedia ? !master.paused : !!_reelsState.mockPlaying;

    // ── BGM 音频元素 ──
    const bgmAudio = _reelsState._bgmAudioEl;

    if (isPlaying) {
        if (master) {
            if (audio) audio.pause();
            if (video) video.pause();
            if (fadeVideo) fadeVideo.pause();
        } else {
            _reelsState.mockPlaying = false;
            _reelsState.mockPausedTime = _getPreviewCurrentTime();
        }
        if (bgmAudio) bgmAudio.pause();
        if (btn) btn.textContent = '▶️';
        return;
    }

    // 回到开头：如果当前时间已经到了或超过了总时长
    const curT = _getPreviewCurrentTime();
    const durT = _getPreviewDuration();

    if (durT > 0 && curT >= durT - 0.05) {
        if (hasAudio) audio.currentTime = 0;
        if (hasVideo) video.currentTime = 0;
        if (!hasMedia) _reelsState.mockPausedTime = 0;
    }

    if (!hasMedia) {
        _reelsState.mockPlaying = true;
        _reelsState.mockStartTime = (performance.now() / 1000) - (_reelsState.mockPausedTime || 0);
    } else {
        if (hasAudio && audio && task && task.audioPath) {
            // 应用音频变速：audioDurScale=150% → playbackRate=0.667
            const aDurScale = task.audioDurScale || 100;
            audio.playbackRate = (aDurScale !== 100) ? (100 / aDurScale) : 1.0;
            audio.preservesPitch = true;
            audio.play().catch(() => { });
        }
        if (hasVideo && video) {
            if (hasAudio && task && task.audioPath && video.duration > 0) {
                try { video.currentTime = (audio.currentTime || 0) % video.duration; } catch (e) { }
            }
            video.play().catch(() => { });
            if (fadeVideo && hasAudio && task && task.audioPath) {
                fadeVideo.play().catch(() => { });
            }
        }
    }

    // ── 同步播放 BGM ──
    if (bgmAudio && bgmAudio.src && task && task.bgmPath) {
        bgmAudio.currentTime = _getPreviewCurrentTime() || 0;
        bgmAudio.play().catch(() => { });
    }
    if (btn) btn.textContent = '⏸️';
}

function _onSeek(e) {
    const video = document.getElementById('reels-preview-video');
    const audio = document.getElementById('reels-preview-audio');
    const duration = _getPreviewDuration();
    if (!(duration > 0)) return;
    const target = (e.target.value / 100) * duration;
    const task = _getSelectedTask();
    const master = _getPreviewMasterElement();
    if (!master) {
        _reelsState.mockPausedTime = target;
        _reelsState.mockStartTime = (performance.now() / 1000) - target;
    }

    if (task && task.audioPath && audio && audio.src && isFinite(audio.duration) && audio.duration > 0) {
        audio.currentTime = Math.max(0, Math.min(target, audio.duration));
    }
    if (video && video.duration > 0) {
        video.currentTime = (task && task.audioPath) ? (target % video.duration) : Math.max(0, Math.min(target, video.duration));
        const fadeVideo = _reelsState.previewFadeVideo;
        if (fadeVideo && task && task.audioPath) {
            const cfg = _getPreviewLoopFadeConfig();
            const fadeDur = Math.min(cfg.duration, Math.max(0.1, video.duration * 0.45));
            try { fadeVideo.currentTime = (video.currentTime + fadeDur) % video.duration; } catch (e2) { }
        }
    }
    // ── 同步 BGM seek ──
    const bgmAudio = _reelsState._bgmAudioEl;
    if (bgmAudio && bgmAudio.src && bgmAudio.duration > 0) {
        bgmAudio.currentTime = target % bgmAudio.duration;
    }
    _updatePreviewTimeUI(target, duration);
    if (_reelsState.timelineEditor) {
        _reelsState.timelineEditor.setPlayhead(target);
    }
}

function _onVideoTimeUpdate() {
    const video = document.getElementById('reels-preview-video');
    if (!video) return;
    const task = _getSelectedTask();
    // 有配音时，以音频为主时钟，不用视频 timeupdate 驱动 UI
    if (task && task.audioPath) {
        _syncBackgroundVideoToMaster();
        return;
    }
    const cur = video.currentTime || 0;
    const dur = _getPreviewDuration();
    _updatePreviewTimeUI(cur, dur);
    if (_reelsState.timelineEditor) _reelsState.timelineEditor.setPlayhead(cur);
}

function _onAudioTimeUpdate() {
    const audio = document.getElementById('reels-preview-audio');
    if (!audio) return;
    const cur = audio.currentTime || 0;
    const dur = _getPreviewDuration();
    _syncBackgroundVideoToMaster();
    _updatePreviewTimeUI(cur, dur);
    if (_reelsState.timelineEditor) _reelsState.timelineEditor.setPlayhead(cur);
}

function _onVideoLoaded() {
    const video = document.getElementById('reels-preview-video');
    if (!video) return;
    const canvas = document.getElementById('reels-preview-canvas');
    if (canvas) {
        canvas.width = 1080;
        canvas.height = 1920;
    }
    _ensurePreviewFadeVideo(video);
    _applyPreviewAudioMix();
    _updateTimelineForTask(_getSelectedTask());
    _updatePreviewTimeUI(_getPreviewCurrentTime(), _getPreviewDuration());
}

function _onAudioLoaded() {
    _applyPreviewAudioMix();
    _updateTimelineForTask(_getSelectedTask());
    _updatePreviewTimeUI(_getPreviewCurrentTime(), _getPreviewDuration());
}

// ═══════════════════════════════════════════════════════
// Preset management (fully ported from AutoSub preset_manager.py)
// ═══════════════════════════════════════════════════════

function _reelsRefreshPresetList() {
    const select = document.getElementById('reels-preset-select');
    if (!select || !window.ReelsStyleEngine) return;
    const data = ReelsStyleEngine.loadSubtitlePresets();
    const defaultName = localStorage.getItem(REELS_DEFAULT_PRESET_KEY) || '';
    select.innerHTML = '<option value="">-- 选择预设 --</option>';
    for (const name of Object.keys(data.presets || {})) {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name === defaultName ? `⭐ ${name}` : name;
        select.appendChild(opt);
    }
    _reelsRefreshDefaultPresetIndicator();
}

function _reelsRefreshDefaultPresetIndicator() {
    const indicator = document.getElementById('reels-default-preset-indicator');
    if (!indicator) return;
    const defaultName = localStorage.getItem(REELS_DEFAULT_PRESET_KEY) || '';
    indicator.textContent = defaultName ? `默认模板: ${defaultName}` : '默认模板: 未设置';
}

function _reelsApplyDefaultPreset() {
    if (!window.ReelsStyleEngine) return;
    const defaultName = localStorage.getItem(REELS_DEFAULT_PRESET_KEY) || '';
    if (!defaultName) {
        _reelsRefreshDefaultPresetIndicator();
        return;
    }
    const data = ReelsStyleEngine.loadSubtitlePresets();
    if (!data.presets || !data.presets[defaultName]) {
        localStorage.removeItem(REELS_DEFAULT_PRESET_KEY);
        _reelsRefreshPresetList();
        return;
    }

    const style = ReelsStyleEngine.applySubtitlePreset(defaultName);
    _writeStyleToUI(style);
    const select = document.getElementById('reels-preset-select');
    if (select) select.value = defaultName;
    _reelsRefreshDefaultPresetIndicator();
}

function reelsSetDefaultPreset() {
    const select = document.getElementById('reels-preset-select');
    if (!select) return;
    const name = select.value;
    if (!name) {
        alert('请先选择一个预设');
        return;
    }
    localStorage.setItem(REELS_DEFAULT_PRESET_KEY, name);
    _reelsRefreshPresetList();
    alert(`已设为默认模板：${name}`);
}

/**
 * 自定义输入弹窗（替代 Electron 不支持的 prompt()）
 */
function _showInputDialog(title, placeholder) {
    return new Promise((resolve) => {
        const overlay = document.createElement('div');
        overlay.style.cssText = 'position:fixed;inset:0;z-index:99999;background:rgba(0,0,0,0.6);display:flex;align-items:center;justify-content:center;';
        const box = document.createElement('div');
        box.style.cssText = 'background:var(--bg-primary,#1e1e2e);border:1px solid var(--border-color,#444);border-radius:12px;padding:24px;min-width:340px;box-shadow:0 8px 32px rgba(0,0,0,0.5);';
        box.innerHTML = `
            <div style="font-size:15px;font-weight:600;margin-bottom:14px;color:var(--text-primary,#fff);">${title}</div>
            <input type="text" id="_input_dialog_val" placeholder="${placeholder || ''}"
                style="width:100%;box-sizing:border-box;padding:8px 12px;border-radius:6px;border:1px solid var(--border-color,#555);background:var(--bg-secondary,#2a2a3e);color:var(--text-primary,#fff);font-size:14px;outline:none;">
            <div style="display:flex;gap:10px;justify-content:flex-end;margin-top:16px;">
                <button id="_input_dialog_cancel" style="padding:6px 18px;border-radius:6px;border:1px solid var(--border-color,#555);background:transparent;color:var(--text-secondary,#aaa);cursor:pointer;font-size:13px;">取消</button>
                <button id="_input_dialog_ok" style="padding:6px 18px;border-radius:6px;border:none;background:var(--accent-primary,#5b6abf);color:#fff;cursor:pointer;font-size:13px;">确定</button>
            </div>`;
        overlay.appendChild(box);
        document.body.appendChild(overlay);

        const input = box.querySelector('#_input_dialog_val');
        const okBtn = box.querySelector('#_input_dialog_ok');
        const cancelBtn = box.querySelector('#_input_dialog_cancel');

        const close = (val) => { document.body.removeChild(overlay); resolve(val); };

        okBtn.onclick = () => close(input.value.trim() || null);
        cancelBtn.onclick = () => close(null);
        overlay.onclick = (e) => { if (e.target === overlay) close(null); };
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') close(input.value.trim() || null);
            if (e.key === 'Escape') close(null);
        });
        setTimeout(() => input.focus(), 50);
    });
}

async function reelsSavePreset() {
    console.log('[预设] 保存按钮被点击');
    try {
        const name = await _showInputDialog('保存字幕预设', '请输入预设名称');
        console.log('[预设] 用户输入名称:', name);
        if (!name) return;
        const style = _readStyleFromUI();
        if (window.ReelsStyleEngine) {
            const result = ReelsStyleEngine.saveNamedSubtitlePreset(name, style);
            if (result) {
                _reelsRefreshPresetList();
                const select = document.getElementById('reels-preset-select');
                if (select) select.value = name;
                console.log(`[预设] 保存成功: "${name}", keys: ${Object.keys(style).length}`);
            } else {
                alert(`保存失败！可能预设数量已满（${ReelsStyleEngine.MAX_PRESETS}个）或名称无效。`);
            }
        } else {
            console.error('[预设] ReelsStyleEngine 未加载！');
        }
    } catch (e) {
        console.error('[预设] 保存出错:', e);
    }
}

function reelsLoadPreset() {
    const select = document.getElementById('reels-preset-select');
    if (!select) return;
    const name = select.value;
    if (!name) { alert('请先选择一个预设'); return; }
    if (window.ReelsStyleEngine) {
        const style = ReelsStyleEngine.applySubtitlePreset(name);
        _writeStyleToUI(style);
        reelsUpdatePreview();
    }
}

function reelsDeletePreset() {
    const select = document.getElementById('reels-preset-select');
    if (!select) return;
    const name = select.value;
    if (!name) { alert('请先选择一个预设'); return; }
    if (confirm(`确定删除预设 "${name}"？`)) {
        if (window.ReelsStyleEngine) {
            ReelsStyleEngine.deleteSubtitlePreset(name);
            const defaultName = localStorage.getItem(REELS_DEFAULT_PRESET_KEY) || '';
            if (defaultName === name) {
                localStorage.removeItem(REELS_DEFAULT_PRESET_KEY);
            }
            _reelsRefreshPresetList();
        }
    }
}

function reelsExportPresets() {
    if (!window.ReelsStyleEngine) return;
    const json = ReelsStyleEngine.exportSubtitlePresets();
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'subtitle_presets.json';
    a.click();
    URL.revokeObjectURL(url);
}

function reelsImportPresets() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (ev) => {
            if (window.ReelsStyleEngine) {
                const result = ReelsStyleEngine.importSubtitlePresets(ev.target.result);
                _reelsRefreshPresetList();
                alert(`导入完成：新增 ${result.added.length} 个，冲突 ${result.conflicts.length} 个，跳过 ${result.skipped.length} 个`);
            }
        };
        reader.readAsText(file);
    };
    input.click();
}

// ═══════════════════════════════════════════════════════
// Export (FFmpeg via IPC)
// ═══════════════════════════════════════════════════════

function _reelsUpdateLastOutputUI(outputPath) {
    const outEl = document.getElementById('reels-export-last-output');
    const openBtn = document.getElementById('reels-open-last-output-btn');
    if (outEl) outEl.value = outputPath || '';
    if (openBtn) openBtn.disabled = !outputPath;
}

function _reelsUpdateLastErrorUI(message) {
    const errEl = document.getElementById('reels-export-last-error');
    if (!errEl) return;
    const text = (message && String(message).trim()) ? String(message).trim() : '无';
    errEl.textContent = text;
    errEl.style.color = text === '无' ? 'var(--text-secondary)' : '#ff8a8a';
}

function _reelsUpdateExportProgressUI(done, total) {
    const progressInner = document.getElementById('reels-export-progress-inner');
    const progressText = document.getElementById('reels-export-progress-text');
    const safeTotal = Math.max(0, total || 0);
    const safeDone = Math.max(0, Math.min(done || 0, safeTotal));
    const pct = safeTotal > 0 ? Math.round((safeDone / safeTotal) * 100) : 0;
    if (progressInner) progressInner.style.width = `${pct}%`;
    if (progressText) progressText.textContent = `${pct}% (${safeDone}/${safeTotal})`;
}

function _reelsParentDir(filePath) {
    if (!filePath || typeof filePath !== 'string') return '';
    const normalized = filePath.replace(/\\/g, '/');
    const idx = normalized.lastIndexOf('/');
    if (idx <= 0) return normalized;
    return normalized.slice(0, idx);
}

function reelsSelectOutputDir() {
    if (window.electronAPI && window.electronAPI.selectDirectory) {
        window.electronAPI.selectDirectory().then(dir => {
            if (dir) document.getElementById('reels-output-dir').value = dir;
        });
    } else {
        alert('输出目录选择需要在 Electron 环境中运行');
    }
}

async function reelsOpenOutputDir() {
    let outputDir = (document.getElementById('reels-output-dir') || {}).value || '';
    if (!outputDir) outputDir = await _getSystemDownloadsPath();
    if (!outputDir) {
        alert('暂无可打开的输出目录');
        return;
    }
    if (window.electronAPI && window.electronAPI.apiCall) {
        try {
            await window.electronAPI.apiCall('file/open-folder', { path: outputDir });
        } catch (e) {
            alert(`打开目录失败: ${e.message || e}`);
        }
        return;
    }
    alert('打开目录需要在 Electron 环境中运行');
}

async function reelsOpenLastOutputInFolder() {
    const outputPath = _reelsState.lastExportOutputPath;
    if (!outputPath) {
        alert('暂无导出文件');
        return;
    }
    const folder = _reelsParentDir(outputPath);
    if (!folder) {
        alert('无法识别输出目录');
        return;
    }
    if (window.electronAPI && window.electronAPI.apiCall) {
        try {
            await window.electronAPI.apiCall('file/open-folder', { path: folder });
        } catch (e) {
            alert(`打开目录失败: ${e.message || e}`);
        }
        return;
    }
    alert('打开目录需要在 Electron 环境中运行');
}

async function _reelsComposeViaBackend(params) {
    if (window.electronAPI && typeof window.electronAPI.reelsCompose === 'function') {
        try {
            const resp = await window.electronAPI.reelsCompose(params);
            if (!resp || resp.success === false) {
                throw new Error((resp && resp.error) || 'Reels 合成失败');
            }
            return resp;
        } catch (err) {
            const msg = err && err.message ? err.message : String(err || '');
            // 主进程未重启时，可能出现该错误；回退尝试旧通道。
            if (!msg.includes("No handler registered for 'reels-compose'")) {
                throw err;
            }
        }
    }
    if (window.electronAPI && window.electronAPI.apiCall) {
        const resp = await window.electronAPI.apiCall('media/reels-compose', params);
        if (!resp || !resp.success) {
            const errMsg = (resp && resp.error) || 'Reels 合成失败';
            if (String(errMsg).includes('未知接口: media/reels-compose')) {
                throw new Error('当前主进程版本不一致（缺少导出接口）。请先完全退出所有 VideoKit 进程，再只启动一个实例重试');
            }
            throw new Error(errMsg);
        }
        return resp;
    }
    throw new Error('缺少后端导出能力（Electron API 不可用）');
}

function reelsSelectIntro() {
    if (window.electronAPI && window.electronAPI.selectFile) {
        window.electronAPI.selectFile({ filters: [{ name: 'Video', extensions: ['mp4', 'mov', 'mkv'] }] })
            .then(path => {
                if (path) document.getElementById('reels-intro-path').value = path;
            });
    } else {
        alert('文件选择需要在 Electron 环境中运行');
    }
}

function reelsCancelExport() {
    if (_reelsState.isExporting) {
        _reelsState.isExporting = false;
        const statusEl = document.getElementById('reels-export-status');
        if (statusEl) statusEl.textContent = '⚠️ 已取消';
        const exportBtn = document.getElementById('reels-export-btn');
        if (exportBtn) exportBtn.disabled = false;
    }
}

async function reelsStartExport() {
    const workMode = _getWorkMode();

    // ── 导出前同步当前任务的覆层（用户可能删除/修改了覆层但尚未切换任务） ──
    const curTask = _reelsState.tasks[_reelsState.selectedIdx];
    if (curTask && _reelsState.overlayProxy && _reelsState.overlayProxy.overlayMgr) {
        curTask.overlays = [...(_reelsState.overlayProxy.overlayMgr.overlays || [])];
    }

    // 导出前自动对齐未对齐的 TXT 任务
    if (workMode !== 'srt') {
        const unaligned = _reelsState.tasks.filter(t =>
            t.txtContent && !t.aligned && (t.segments || []).length === 0
        );
        if (unaligned.length > 0) {
            const statusEl = document.getElementById('reels-export-status');
            if (statusEl) statusEl.textContent = `正在对齐 ${unaligned.length} 个任务...`;
            for (let i = 0; i < unaligned.length; i++) {
                const task = unaligned[i];
                if (statusEl) statusEl.textContent = `对齐中 ${i + 1}/${unaligned.length}: ${task.fileName}`;
                try { await _reelsAlignSubtitles(task); } catch (err) {
                    console.error('[Reels] Pre-export align failed:', task.fileName, err);
                }
            }
            _renderTaskList();
        }
    }

    const invalidTasks = [];
    const tasks = _reelsState.tasks.filter((t, idx) => {
        const hasSub = !!t.srtPath && (t.segments || []).length > 0;
        const bgPath = t.bgPath || t.videoPath;
        const hasBg = !!bgPath;
        const hasVoice = !!t.audioPath;
        // 有覆层（文案卡片 或 滚动字幕）则不强制要求字幕
        const ov = (t.overlays && t.overlays.length > 0) ? t.overlays[0] : null;
        const hasOverlay = ov && (ov.title_text || ov.body_text || ov.content || ov.scroll_title);

        if (workMode === 'voiced_bg') {
            // 带声视频模式：需要背景 + (字幕 或 文案卡片)
            if (!hasBg || (!hasSub && !hasOverlay)) {
                const missing = [];
                if (!hasBg) missing.push('带声视频');
                if (!hasSub && !hasOverlay) missing.push('字幕(需先对齐)');
                invalidTasks.push(`${idx + 1}. ${(t.fileName || t.baseName || '未命名任务')} 缺少: ${missing.join(' + ')}`);
                return false;
            }
            return true;
        }

        // SRT 模式和配音+文本模式
        // 有文案卡片时不要求字幕
        if ((!hasSub && !hasOverlay) || !hasBg) {
            const missing = [];
            if (!hasBg) missing.push('背景');
            if (!hasSub && !hasOverlay) missing.push(workMode === 'dubbed_text' ? '字幕(需先对齐)' : '字幕');
            invalidTasks.push(`${idx + 1}. ${(t.fileName || t.baseName || '未命名任务')} 缺少: ${missing.join(' + ')}`);
            return false;
        }
        if (hasVoice) return true;
        // 有文案卡片但无配音/字幕，也允许导出（图片背景时需要文字卡片来确定时长）
        if (hasOverlay && !hasSub && !hasVoice) {
            return true;
        }
        // 无配音时仅兼容视频背景（旧模式）；图片背景需要配音来确定时长。
        const allowNoVoice = !_isImagePath(bgPath);
        if (!allowNoVoice) {
            invalidTasks.push(`${idx + 1}. ${(t.fileName || t.baseName || '未命名任务')} 缺少: 人声音频`);
        }
        return allowNoVoice;
    });
    if (tasks.length === 0) {
        const extra = invalidTasks.length > 0 ? `\n\n任务问题:\n${invalidTasks.slice(0, 8).join('\n')}` : '';
        alert(`没有可导出的任务${extra}`);
        return;
    }

    let outputDir = document.getElementById('reels-output-dir').value;
    if (!outputDir) {
        outputDir = await _getSystemDownloadsPath();
        const outputEl = document.getElementById('reels-output-dir');
        if (outputEl) outputEl.value = outputDir || '';
    }
    if (!outputDir) { alert('请先选择输出目录'); return; }

    const quality = document.getElementById('reels-quality').value;
    const suffix = document.getElementById('reels-suffix').value || '_subtitled';
    const style = _readStyleFromUI();
    const crfMap = { high: 15, medium: 18, low: 23 };
    const crf = crfMap[quality] || 18;
    const useKaraoke = document.getElementById('reels-karaoke-hl');
    const karaokeHL = useKaraoke ? useKaraoke.checked : false;
    let voiceVolume = parseFloat((document.getElementById('reels-voice-volume') || {}).value || '100');
    let bgVolume = parseFloat((document.getElementById('reels-bg-volume') || {}).value || '5');
    if (!Number.isFinite(voiceVolume)) voiceVolume = 100;
    if (!Number.isFinite(bgVolume)) bgVolume = 5;
    voiceVolume = Math.max(0, Math.min(200, voiceVolume));
    bgVolume = Math.max(0, Math.min(200, bgVolume));
    const loopFadeEl = document.getElementById('reels-loop-fade');
    const loopFade = loopFadeEl ? loopFadeEl.checked : true;
    const loopFadeDurEl = document.getElementById('reels-loop-fade-dur');
    let loopFadeDur = parseFloat(loopFadeDurEl ? loopFadeDurEl.value : '1');
    if (!Number.isFinite(loopFadeDur) || loopFadeDur <= 0) loopFadeDur = 1.0;
    loopFadeDur = Math.max(0.1, Math.min(3, loopFadeDur));
    let customDuration = parseFloat((document.getElementById('reels-custom-duration') || {}).value || '0');
    if (!Number.isFinite(customDuration) || customDuration < 0) customDuration = 0;

    _reelsState.isExporting = true;
    const progressBar = document.getElementById('reels-export-progress');
    const statusEl = document.getElementById('reels-export-status');
    const exportBtn = document.getElementById('reels-export-btn');
    const exportBar = document.querySelector('.nle-export-bar');
    if (exportBar) exportBar.open = true;
    _reelsState.lastExportOutputPath = '';
    _reelsUpdateLastOutputUI('');
    _reelsUpdateLastErrorUI('');
    _reelsUpdateExportProgressUI(0, tasks.length);

    if (progressBar) progressBar.classList.remove('hidden');
    if (exportBtn) exportBtn.disabled = true;

    const useGPU = document.getElementById('reels-use-gpu');
    const gpuEnabled = useGPU ? useGPU.checked : false;
    const introPath = (document.getElementById('reels-intro-path') || {}).value || '';
    let failCount = 0;
    let okCount = 0;
    let canceled = false;
    const failDetails = [];
    const outputDirRaw = String(outputDir || '');
    const outputDirBase = outputDirRaw.replace(/[\\/]+$/, '') || outputDirRaw;
    const outputJoinSep = outputDirBase.includes('\\') ? '\\' : '/';

    // 自动创建带日期的子文件夹，如 "2026-03-02_批量Reels"
    const now = new Date();
    const dateStr = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
    const timeStr = `${String(now.getHours()).padStart(2, '0')}${String(now.getMinutes()).padStart(2, '0')}`;
    const subFolderName = `${dateStr}_${timeStr}_批量Reels`;
    const outputDirTrimmed = `${outputDirBase}${outputJoinSep}${subFolderName}`;

    for (let i = 0; i < tasks.length; i++) {
        if (!_reelsState.isExporting) {
            canceled = true;
            break;
        }
        const task = tasks[i];
        if (statusEl) statusEl.textContent = `导出中 ${i + 1}/${tasks.length}: ${task.fileName}`;

        try {
            const baseName = (task.fileName || `${task.baseName || 'reel'}.mp4`).replace(/\.[^.]+$/, '');
            const outputPath = `${outputDirTrimmed}${outputJoinSep}${baseName}${suffix}.mp4`;
            let bgPath = task.bgPath || task.videoPath;
            
            // ── 路径修复：如果 bgPath 仅为文件名（非绝对路径），尝试自动补全 ──
            if (bgPath && !bgPath.startsWith('/') && !/^[A-Z]:\\/i.test(bgPath)) {
                const bareFileName = bgPath.replace(/\\/g, '/').split('/').pop();
                let fixedPath = null;
                
                // 策略1: 从背景素材库中找同名文件的完整路径
                const library = _reelsState.backgroundLibrary || [];
                for (const bg of library) {
                    if (bg.path && (bg.path.startsWith('/') || /^[A-Z]:\\/i.test(bg.path))) {
                        const libName = bg.path.replace(/\\/g, '/').split('/').pop();
                        if (libName === bareFileName) { fixedPath = bg.path; break; }
                    }
                }
                
                // 策略2: 从其他任务中找同文件名的绝对路径
                if (!fixedPath) {
                    for (const t of _reelsState.tasks) {
                        const p = t.bgPath || t.videoPath;
                        if (p && (p.startsWith('/') || /^[A-Z]:\\/i.test(p))) {
                            const tName = p.replace(/\\/g, '/').split('/').pop();
                            if (tName === bareFileName) { fixedPath = p; break; }
                        }
                    }
                }
                
                // 策略3: 如果找到了任何绝对路径的任务，取其目录 + 当前文件名
                if (!fixedPath) {
                    for (const t of _reelsState.tasks) {
                        const p = t.bgPath || t.videoPath;
                        if (p && (p.startsWith('/') || /^[A-Z]:\\/i.test(p))) {
                            const dir = p.replace(/\\/g, '/').replace(/\/[^/]+$/, '');
                            fixedPath = `${dir}/${bareFileName}`;
                            break;
                        }
                    }
                }
                
                // 策略4: 从批量表格的素材文件夹 materialDir 中搜索
                if (!fixedPath && typeof _batchTableState !== 'undefined' && _batchTableState.tabs) {
                    const activeTab = _batchTableState.tabs[_batchTableState.activeIdx || 0];
                    const matDir = activeTab?.materialDir;
                    if (matDir) {
                        // 拼接 materialDir + bareFileName
                        const candidate = matDir.replace(/\\/g, '/').replace(/\/$/, '') + '/' + bareFileName;
                        fixedPath = candidate;
                        console.log(`[Reels] 策略4: 尝试素材文件夹路径: "${fixedPath}"`);
                    }
                }
                
                // 策略5: 从输出目录的父级目录搜索
                if (!fixedPath && outputDirTrimmed) {
                    const parentDir = outputDirTrimmed.replace(/\\/g, '/').replace(/\/[^/]+$/, '');
                    if (parentDir) {
                        fixedPath = parentDir + '/' + bareFileName;
                        console.log(`[Reels] 策略5: 尝试输出目录同级路径: "${fixedPath}"`);
                    }
                }
                
                if (fixedPath) {
                    console.warn(`[Reels] 自动修复 bgPath: "${bgPath}" → "${fixedPath}"`);
                    bgPath = fixedPath;
                    task.bgPath = fixedPath;
                    task.videoPath = fixedPath;
                } else {
                    console.error(`[Reels] bgPath 不是绝对路径且无法自动修复: "${bgPath}"`);
                }
            }
            
            const hasVoiceAudio = !!task.audioPath || workMode === 'voiced_bg';
            // For voiced_bg mode, use the background video's audio track as the voice source
            const voiceSource = task.audioPath || (workMode === 'voiced_bg' ? bgPath : null);
            let finalOutputPath = outputPath;

            // ── WYSIWYG 逐帧渲染导出（与预览 100% 一致）──
            const canUseWysiwyg = typeof window.reelsWysiwygExport === 'function'
                && window.electronAPI && window.electronAPI.reelsComposeWysiwyg;

            if (canUseWysiwyg) {
                // 创建离屏 canvas
                const offCanvas = document.createElement('canvas');
                offCanvas.width = 1080;
                offCanvas.height = 1920;

                const wysiwygResult = await window.reelsWysiwygExport({
                    canvas: offCanvas,
                    style,
                    segments: task.segments || [],
                    overlays: task.overlays || [],
                    backgroundPath: bgPath,
                    voicePath: voiceSource || null,
                    outputPath,
                    targetWidth: 1080,
                    targetHeight: 1920,
                    fps: 30,
                    // voiced_bg 模式: 背景音频作为主音轨，用 bgVolume 控制
                    voiceVolume: (workMode === 'voiced_bg' && !task.audioPath) ? bgVolume / 100 : voiceVolume / 100,
                    bgVolume: bgVolume / 100,
                    loopFade,
                    loopFadeDur,
                    customDuration: task.customDuration || customDuration || 0,
                    bgmPath: task.bgmPath || '',
                    bgmVolume: (task.bgmVolume != null ? task.bgmVolume : 10) / 100,
                    bgScale: task.bgScale || 100,
                    bgDurScale: task.bgDurScale || 100,
                    audioDurScale: task.audioDurScale || 100,
                    reverbEnabled: (() => { const rc = _getReverbConfig(); console.log('[Export] Reverb config:', JSON.stringify(rc)); return rc.enabled; })(),
                    reverbPreset: _getReverbConfig().preset,
                    reverbMix: _getReverbConfig().mix,
                    stereoWidth: _getReverbConfig().stereoWidth,
                    audioFxTarget: _getReverbConfig().audioFxTarget,
                    useGPU: gpuEnabled,
                    isCancelled: () => !_reelsState.isExporting,
                    onProgress: (pct) => {
                        if (statusEl) statusEl.textContent = `导出中 ${i + 1}/${tasks.length}: ${task.fileName} (${pct}%)`;
                    },
                    onLog: (msg) => console.log(`[WYSIWYG] ${task.fileName}: ${msg}`),
                });
                if (wysiwygResult && wysiwygResult.cancelled) {
                    canceled = true;
                    break;
                }
            } else if (hasVoiceAudio && voiceSource) {
                // ── 回退: ASS 字幕方式导出（需要配音）──
                const assContent = window.ReelsSubtitleProcessor
                    ? ReelsSubtitleProcessor.generateEnhancedASS(task.segments, style, {
                        karaokeHighlight: karaokeHL,
                        videoW: 1080,
                        videoH: 1920,
                    })
                    : generateASS(task.segments, style);

                const resp = await _reelsComposeViaBackend({
                    background_path: bgPath,
                    voice_path: voiceSource,
                    ass_content: assContent,
                    output_path: outputPath,
                    crf,
                    use_gpu: gpuEnabled,
                    loop_fade: loopFade,
                    loop_fade_dur: loopFadeDur,
                    voice_volume: voiceVolume / 100,
                    bg_volume: bgVolume / 100,
                    bgm_path: task.bgmPath || '',
                    bgm_volume: (task.bgmVolume != null ? task.bgmVolume : 10) / 100,
                });
            } else if (window.electronAPI && window.electronAPI.burnSubtitles) {
                const assContent = window.ReelsSubtitleProcessor
                    ? ReelsSubtitleProcessor.generateEnhancedASS(task.segments, style, {
                        karaokeHighlight: karaokeHL,
                        videoW: 1080,
                        videoH: 1920,
                    })
                    : generateASS(task.segments, style);
                await window.electronAPI.burnSubtitles({
                    videoPath: bgPath, assContent, outputPath, crf,
                    useGPU: gpuEnabled,
                });
            } else {
                console.warn('[Reels] FFmpeg IPC not available, skipping:', task.fileName);
            }

        // 拼接前置片段 (Hook -> Main)
            const finalHookPath = task.hookFile || introPath || '';
            if (finalHookPath && window.electronAPI && window.electronAPI.concatVideo) {
                const concatOutput = outputPath.replace('.mp4', '_final.mp4');
                await window.electronAPI.concatVideo({
                    introPath: finalHookPath,
                    mainPath: outputPath,
                    outputPath: concatOutput,
                    speed: task.hookSpeed || 1.0,
                    trimStart: task.hookTrimStart !== undefined ? task.hookTrimStart : null,
                    trimEnd: task.hookTrimEnd !== undefined ? task.hookTrimEnd : null,
                    transition: task.hookTransition || 'none',
                    transDuration: task.hookTransDuration || 0.5,
                    targetWidth: 1080,
                    targetHeight: 1920,
                    fps: 30
                });
                finalOutputPath = concatOutput;
            }
            okCount += 1;
            _reelsState.lastExportOutputPath = finalOutputPath;
            _reelsUpdateLastOutputUI(finalOutputPath);
        } catch (err) {
            console.error('[Reels] Export failed:', task.fileName, err);
            failCount += 1;
            const errMsg = err && err.message ? err.message : String(err || '未知错误');
            failDetails.push(`${task.fileName}: ${errMsg}`);
            if (statusEl) statusEl.textContent = `❌ 导出失败: ${task.fileName} - ${errMsg}`;
            _reelsUpdateLastErrorUI(`${task.fileName}: ${errMsg}`);
        }
        _reelsUpdateExportProgressUI(i + 1, tasks.length);
    }

    const doneCount = okCount + failCount;
    _reelsUpdateExportProgressUI(doneCount, tasks.length);
    if (statusEl) {
        if (canceled) {
            statusEl.textContent = `⚠️ 已取消 (${doneCount}/${tasks.length})`;
        } else {
            statusEl.textContent = failCount > 0
                ? `⚠️ 完成 ${okCount}/${tasks.length}，失败 ${failCount}`
                : `✅ 全部完成 (${tasks.length}个视频)`;
        }
    }
    if (!canceled && failCount > 0) {
        const shortErr = failDetails.slice(0, 5).join('\n');
        alert(`导出失败 ${failCount} 个\n输出目录: ${outputDirTrimmed}\n\n失败原因:\n${shortErr}`);
    } else if (!canceled && okCount > 0) {
        _reelsUpdateLastErrorUI('');
        const latest = _reelsState.lastExportOutputPath || `${outputDirTrimmed}${outputJoinSep}`;
        alert(`导出完成 ${okCount}/${tasks.length}\n输出目录: ${outputDirTrimmed}\n最新文件: ${latest}`);
        // 自动打开输出文件夹
        if (window.electronAPI && window.electronAPI.apiCall) {
            try { await window.electronAPI.apiCall('file/open-folder', { path: outputDirTrimmed }); } catch (e) { }
        }
    }
    if (exportBtn) exportBtn.disabled = false;
    _reelsState.isExporting = false;
}

// ═══════════════════════════════════════════════════════
// Smart Subtitle Processing (智能字幕处理)
// ═══════════════════════════════════════════════════════

/**
 * 智能重分段：按当前样式参数（字体大小、换行宽度等）重新分段所有任务的字幕。
 * 效果：自动调整每条字幕的文本量，确保不溢出预览区域。
 */
function reelsResegment() {
    if (!window.ReelsSubtitleProcessor) {
        alert('字幕处理器未加载');
        return;
    }
    const style = _readStyleFromUI();
    const videoW = 1080; // Reels 竖屏
    let totalProcessed = 0;

    for (const task of _reelsState.tasks) {
        if (!task.segments || task.segments.length === 0) continue;
        const result = ReelsSubtitleProcessor.smartSegmentation(task.segments, style, videoW);
        if (result && result.length > 0) {
            task.segments = result;
            totalProcessed++;
        }
    }
    _renderTaskList();
    if (totalProcessed > 0) {
        alert(`✅ 已智能重分段 ${totalProcessed} 个任务的字幕`);
    } else {
        alert('没有可处理的字幕（请先添加带SRT的任务）');
    }
}

/**
 * 合并短片段：合并时长过短的字幕到相邻字幕。
 */
function reelsMergeShort() {
    if (!window.ReelsSubtitleProcessor) {
        alert('字幕处理器未加载');
        return;
    }
    let totalProcessed = 0;
    for (const task of _reelsState.tasks) {
        if (!task.segments || task.segments.length === 0) continue;
        task.segments = ReelsSubtitleProcessor.mergeShortSegments(task.segments);
        totalProcessed++;
    }
    _renderTaskList();
    if (totalProcessed > 0) {
        alert(`✅ 已合并 ${totalProcessed} 个任务的短片段`);
    }
}

/**
 * 导出当前选中任务的字幕为 SRT 文件。
 */
function reelsExportSRT() {
    const task = _reelsState.tasks[_reelsState.selectedIdx];
    if (!task || !task.segments || task.segments.length === 0) {
        alert('请先选择一个带字幕的任务');
        return;
    }
    if (window.ReelsSubtitleProcessor) {
        const srtContent = ReelsSubtitleProcessor.segmentsToSRT(task.segments);
        const blob = new Blob([srtContent], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = task.fileName.replace(/\.[^.]+$/, '') + '_processed.srt';
        a.click();
        URL.revokeObjectURL(url);
    }
}

// ═══════════════════════════════════════════════════════
// Project Management
// ═══════════════════════════════════════════════════════

function reelsSaveProject() {
    if (!window.ReelsProject) { alert('项目管理模块未加载'); return; }
    const style = _readStyleFromUI();
    const exportOpts = {
        outputDir: (document.getElementById('reels-output-dir') || {}).value || '',
        quality: (document.getElementById('reels-quality') || {}).value || 'medium',
        suffix: (document.getElementById('reels-suffix') || {}).value || '_subtitled',
        voiceVolume: parseFloat((document.getElementById('reels-voice-volume') || {}).value || '100') || 100,
        bgVolume: parseFloat((document.getElementById('reels-bg-volume') || {}).value || '5') || 5,
        useGPU: (document.getElementById('reels-use-gpu') || {}).checked || false,
        loopFade: (document.getElementById('reels-loop-fade') || {}).checked !== false,
        loopFadeDuration: parseFloat((document.getElementById('reels-loop-fade-dur') || {}).value || '1') || 1,
        introPath: (document.getElementById('reels-intro-path') || {}).value || '',
        karaokeHighlight: (document.getElementById('reels-karaoke-hl') || {}).checked || false,
        reverbEnabled: (document.getElementById('reels-reverb-enabled') || {}).checked || false,
        reverbPreset: (document.getElementById('reels-reverb-preset') || {}).value || 'hall',
        reverbMix: parseFloat((document.getElementById('reels-reverb-mix') || {}).value || '30') || 30,
        stereoWidth: parseFloat((document.getElementById('reels-stereo-width') || {}).value || '100') || 100,
        audioFxTarget: (document.getElementById('reels-audio-fx-target') || {}).value || 'all',
    };
    ReelsProject.saveProject({
        tasks: _reelsState.tasks,
        style,
        exportOpts,
        selectedIdx: _reelsState.selectedIdx,
    });
}

async function reelsLoadProject() {
    if (!window.ReelsProject) { alert('项目管理模块未加载'); return; }
    const result = await ReelsProject.loadProject();
    if (!result) return;

    // 恢复任务
    _reelsState.tasks = result.tasks;
    _reelsState.selectedIdx = result.selectedIdx >= 0 ? result.selectedIdx : 0;

    // 恢复样式
    if (result.style && Object.keys(result.style).length > 0) {
        _writeStyleToUI(result.style);
    }

    // 恢复导出选项
    if (result.exportOpts) {
        const opts = result.exportOpts;
        const setVal = (id, val) => { const el = document.getElementById(id); if (el) el.value = val; };
        const setCheck = (id, val) => { const el = document.getElementById(id); if (el) el.checked = !!val; };
        if (opts.outputDir) setVal('reels-output-dir', opts.outputDir);
        if (opts.quality) setVal('reels-quality', opts.quality);
        if (opts.suffix) setVal('reels-suffix', opts.suffix);
        if (opts.voiceVolume !== undefined && opts.voiceVolume !== null) setVal('reels-voice-volume', String(opts.voiceVolume));
        if (opts.bgVolume !== undefined && opts.bgVolume !== null) setVal('reels-bg-volume', String(opts.bgVolume));
        setCheck('reels-use-gpu', opts.useGPU);
        setCheck('reels-loop-fade', opts.loopFade !== false);
        if (opts.loopFadeDuration !== undefined && opts.loopFadeDuration !== null) {
            const dur = parseFloat(opts.loopFadeDuration);
            if (Number.isFinite(dur) && dur > 0) setVal('reels-loop-fade-dur', String(dur));
        }
        if (opts.introPath) setVal('reels-intro-path', opts.introPath);
        setCheck('reels-karaoke-hl', opts.karaokeHighlight);
        setCheck('reels-reverb-enabled', opts.reverbEnabled);
        if (opts.reverbPreset) setVal('reels-reverb-preset', opts.reverbPreset);
        if (opts.reverbMix !== undefined) setVal('reels-reverb-mix', String(opts.reverbMix));
        if (opts.stereoWidth !== undefined) setVal('reels-stereo-width', String(opts.stereoWidth));
        if (opts.audioFxTarget !== undefined) setVal('reels-audio-fx-target', opts.audioFxTarget);
    }

    _renderTaskList();
    _applyPreviewAudioMix();
    reelsUpdatePreview();

    if (result.warnings && result.warnings.length > 0) {
        console.warn('[Project] Warnings:', result.warnings);
    }
    const statusEl = document.getElementById('reels-export-status');
    if (statusEl) statusEl.textContent = `✅ 已加载 ${result.tasks.length} 个任务`;
}

// ═══════════════════════════════════════════════════════
// Font Upload
// ═══════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    // 字体上传处理
    const fontInput = document.getElementById('reels-font-upload');
    if (fontInput) {
        fontInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            if (window.getFontManager) {
                const fm = getFontManager();
                const familyName = await fm.uploadFont(file);
                if (familyName) {
                    fm.refreshFontSelect('reels-font-family', familyName);
                    fm.refreshFontSelect('rop-font', familyName);
                    fm.refreshFontSelect('rop-title-font', familyName);
                    fm.refreshFontSelect('rop-body-font', familyName);
                    const familyEl = document.getElementById('reels-font-family');
                    if (familyEl) familyEl.value = familyName;
                    reelsRefreshSubtitleWeightOptions();
                    reelsUpdatePreview();
                    alert(`字体 "${familyName}" 已加载！`);
                }
            } else {
                alert('字体管理器未加载');
            }
            fontInput.value = '';
        });
    }

    // 初始化字体管理器
    if (window.getFontManager) {
        const fm = getFontManager();
        fm.register().then(() => {
            fm.refreshFontSelect('reels-font-family', 'Arial');
            fm.refreshFontSelect('rop-font', 'Arial');
            fm.refreshFontSelect('rop-title-font', 'Crimson Pro');
            fm.refreshFontSelect('rop-body-font', 'Arial');
            fm.loadGoogleFont('Crimson Pro').catch(() => { });
            reelsRefreshSubtitleWeightOptions();
        });
    }

    // 自动保存 (每 60 秒)
    setInterval(() => {
        if (_reelsState.tasks.length > 0 && window.ReelsProject) {
            const style = _readStyleFromUI();
            ReelsProject.autoSaveProject({
                tasks: _reelsState.tasks,
                style,
                selectedIdx: _reelsState.selectedIdx,
            });
        }
    }, 60000);
});

// ═══════════════════════════════════════════════════════
// Tab visibility observer
// ═══════════════════════════════════════════════════════

if (typeof MutationObserver !== 'undefined') {
    const observer = new MutationObserver(() => {
        const panel = document.getElementById('batch-reels-panel');
        if (panel && panel.classList.contains('active')) {
            _fitPreviewWhenReady();
            reelsUpdatePreview();
        } else {
            if (_reelsState.previewRAF) {
                cancelAnimationFrame(_reelsState.previewRAF);
                _reelsState.previewRAF = null;
            }
        }
    });
    setTimeout(() => {
        const panel = document.getElementById('batch-reels-panel');
        if (panel) observer.observe(panel, { attributes: true, attributeFilter: ['class'] });
    }, 500);
}
