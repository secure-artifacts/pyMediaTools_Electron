/**
 * reels-project.js — 项目管理（保存/加载）
 * 
 * 完整移植自 AutoSub_v8:
 *   - collect_project_data()  → 项目序列化
 *   - apply_project_data()    → 项目反序列化
 *   - _normalize_project_data → 版本兼容修复
 * 
 * 项目以 JSON 格式存储，包含：
 *   - 所有任务（视频路径、字幕片段、样式）
 *   - 时间线（轨道、片段、媒体源）
 *   - 导出设置
 *   - UI 状态
 */

const PROJECT_VERSION = '2.0.0';

// ═══════════════════════════════════════════════════════
// 1. Collect Project Data (序列化)
// ═══════════════════════════════════════════════════════

/**
 * 将当前工作状态序列化为可保存的 JSON 对象。
 * @param {object} state - 当前应用状态
 * @param {Array} state.tasks - 任务列表
 * @param {object} state.style - 全局字幕样式
 * @param {object} state.exportOpts - 导出选项
 * @param {number} state.selectedIdx - 当前选中任务
 * @returns {object} 项目数据
 */
function collectProjectData(state) {
    const { tasks = [], style = {}, exportOpts = {}, selectedIdx = -1 } = state;

    const tasksData = tasks.map(task => {
        const taskData = {
            baseName: task.baseName || '',
            bgPath: task.bgPath || task.videoPath || '',
            videoPath: task.videoPath || '',
            audioPath: task.audioPath || '',
            srtPath: task.srtPath || '',
            fileName: task.fileName || '',
            segments: _sanitizeSegments(task.segments || []),
            style: JSON.parse(JSON.stringify(task.style || style)),
        };

        if (task.bgSrcUrl) taskData.bgSrcUrl = task.bgSrcUrl;
        if (task.srcUrl) taskData.srcUrl = task.srcUrl;

        // 时间线序列化
        if (task.timeline) {
            if (typeof task.timeline.toJSON === 'function') {
                taskData.timeline = task.timeline.toJSON();
            } else {
                taskData.timeline = JSON.parse(JSON.stringify(task.timeline));
            }
        }

        // 覆层
        if (task.textOverlays) {
            taskData.textOverlays = JSON.parse(JSON.stringify(task.textOverlays));
        }
        if (task.imageOverlays) {
            taskData.imageOverlays = JSON.parse(JSON.stringify(task.imageOverlays));
        }

        // 前置片段
        if (task.introMedia) {
            taskData.introMedia = JSON.parse(JSON.stringify(task.introMedia));
        }

        // 音量
        taskData.srcVolume = task.srcVolume || 1.0;

        return taskData;
    });

    return {
        version: PROJECT_VERSION,
        createdAt: new Date().toISOString(),
        app: 'VideoKit',
        style: JSON.parse(JSON.stringify(style)),
        exportOpts: JSON.parse(JSON.stringify(exportOpts)),
        tasks: tasksData,
        selectedIdx: selectedIdx,
    };
}

// ═══════════════════════════════════════════════════════
// 2. Apply Project Data (反序列化)
// ═══════════════════════════════════════════════════════

/**
 * 从保存的项目数据恢复工作状态。
 * @param {object} data - 项目 JSON 数据
 * @returns {object} 恢复的状态 { tasks, style, exportOpts, selectedIdx, warnings }
 */
function applyProjectData(data) {
    if (!data || typeof data !== 'object') {
        return { tasks: [], style: {}, exportOpts: {}, selectedIdx: -1, warnings: ['无效的项目数据'] };
    }

    // 版本兼容修复
    const normalized = _normalizeProjectData(data);
    const warnings = [];

    const tasks = (normalized.tasks || []).map(t => {
        const bgPath = t.bgPath || t.videoPath || t.path || t.video || '';
        const task = {
            baseName: t.baseName || _extractFileName(bgPath).replace(/\.[^.]+$/, ''),
            bgPath: bgPath,
            videoPath: bgPath,
            bgSrcUrl: t.bgSrcUrl || t.srcUrl || null,
            srcUrl: t.srcUrl || t.bgSrcUrl || null,
            audioPath: t.audioPath || '',
            srtPath: t.srtPath || '',
            fileName: t.fileName || _extractFileName(bgPath || t.audioPath || ''),
            segments: t.segments || [],
            style: t.style || normalized.style || {},
            srcVolume: t.srcVolume || 1.0,
        };

        // 检查文件是否存在 (仅记录警告，不阻断)
        if (task.bgPath && !task.bgPath.startsWith('/') && !task.bgPath.startsWith('C:')) {
            warnings.push(`相对路径: ${task.bgPath}`);
        }

        // 时间线反序列化
        if (t.timeline && window.ReelsTimeline) {
            task.timeline = ReelsTimeline.Timeline.fromJSON(t.timeline);
        } else if (t.timeline) {
            task.timeline = t.timeline;
        }

        // 覆层
        task.textOverlays = t.textOverlays || t.text_overlays || [];
        task.imageOverlays = t.imageOverlays || t.image_overlays || [];
        task.introMedia = t.introMedia || t.intro_media || null;

        return task;
    });

    return {
        tasks,
        style: normalized.style || {},
        exportOpts: normalized.exportOpts || normalized.export_opts || {},
        selectedIdx: normalized.selectedIdx || normalized.curr_idx || 0,
        warnings,
    };
}

// ═══════════════════════════════════════════════════════
// 3. Version Normalization (版本兼容修复)
// ═══════════════════════════════════════════════════════

function _normalizeProjectData(data) {
    const version = data.version || '1.0.0';
    const result = JSON.parse(JSON.stringify(data));

    // AutoSub 旧格式兼容
    if (result.tasks) {
        for (const task of result.tasks) {
            // 旧字段映射
            if (task.path && !task.videoPath) task.videoPath = task.path;
            if (task.video && !task.videoPath) task.videoPath = task.video;
            if (!task.bgPath && task.videoPath) task.bgPath = task.videoPath;

            // 样式修复：确保关键字段存在
            if (task.style) {
                if (!task.style.font_family) task.style.font_family = 'Arial';
                if (!task.style.fontsize) task.style.fontsize = 74;
                if (task.style.color_text === undefined) task.style.color_text = '#FFFFFF';
            }

            // segments 修复：确保 start/end 是秒
            if (task.segments) {
                for (const seg of task.segments) {
                    // 毫秒 → 秒 转换 (如果值 > 1000 视为毫秒)
                    if (typeof seg.start === 'number' && seg.start > 1000) {
                        seg.start = seg.start / 1000;
                    }
                    if (typeof seg.end === 'number' && seg.end > 1000) {
                        seg.end = seg.end / 1000;
                    }
                }
            }
        }
    }

    return result;
}

// ═══════════════════════════════════════════════════════
// 4. Save & Load (文件读写)
// ═══════════════════════════════════════════════════════

/**
 * 保存项目到文件。
 * @param {object} state - 当前应用状态
 */
async function saveProject(state) {
    const projectData = collectProjectData(state);
    const json = JSON.stringify(projectData, null, 2);

    if (window.electronAPI && window.electronAPI.saveFile) {
        // Electron 环境：使用原生保存对话框
        const result = await window.electronAPI.saveFile({
            defaultPath: `reels_project_${_dateStr()}.json`,
            filters: [
                { name: 'Reels Project', extensions: ['json'] },
            ],
            content: json,
        });
        return result;
    } else {
        // 浏览器环境：下载
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `reels_project_${_dateStr()}.json`;
        a.click();
        URL.revokeObjectURL(url);
        return { success: true };
    }
}

/**
 * 从文件加载项目。
 * @returns {Promise<object>} 恢复的状态
 */
async function loadProject() {
    return new Promise((resolve) => {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.json';
        input.onchange = async (e) => {
            const file = e.target.files[0];
            if (!file) { resolve(null); return; }

            try {
                const text = await file.text();
                const data = JSON.parse(text);
                const result = applyProjectData(data);
                resolve(result);
            } catch (err) {
                console.error('[Project] Failed to load:', err);
                alert(`项目加载失败: ${err.message}`);
                resolve(null);
            }
        };
        input.click();
    });
}

/**
 * 自动保存到 localStorage。
 */
function autoSaveProject(state) {
    try {
        const projectData = collectProjectData(state);
        const json = JSON.stringify(projectData);
        localStorage.setItem('reels_autosave', json);
        localStorage.setItem('reels_autosave_time', new Date().toISOString());
    } catch (err) {
        console.warn('[Project] Auto-save failed:', err);
    }
}

/**
 * 从 localStorage 恢复自动保存。
 */
function loadAutoSave() {
    try {
        const json = localStorage.getItem('reels_autosave');
        if (!json) return null;

        const time = localStorage.getItem('reels_autosave_time');
        const data = JSON.parse(json);
        const result = applyProjectData(data);
        result.autoSaveTime = time;
        return result;
    } catch (err) {
        console.warn('[Project] Auto-load failed:', err);
        return null;
    }
}

/**
 * 清除自动保存。
 */
function clearAutoSave() {
    localStorage.removeItem('reels_autosave');
    localStorage.removeItem('reels_autosave_time');
}

// ═══════════════════════════════════════════════════════
// 5. Utilities
// ═══════════════════════════════════════════════════════

function _sanitizeSegments(segments) {
    return (segments || []).map(seg => ({
        start: seg.start,
        end: seg.end,
        text: seg.text || '',
        words: seg.words || undefined,
    })).filter(s => typeof s.start === 'number' && typeof s.end === 'number');
}

function _extractFileName(path) {
    if (!path) return '';
    return path.split('/').pop().split('\\').pop() || '';
}

function _dateStr() {
    const d = new Date();
    return `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}_${String(d.getHours()).padStart(2, '0')}${String(d.getMinutes()).padStart(2, '0')}`;
}

// ═══════════════════════════════════════════════════════
// Exports
// ═══════════════════════════════════════════════════════

const ReelsProject = {
    PROJECT_VERSION,
    collectProjectData,
    applyProjectData,
    saveProject,
    loadProject,
    autoSaveProject,
    loadAutoSave,
    clearAutoSave,
};

if (typeof window !== 'undefined') window.ReelsProject = ReelsProject;
if (typeof module !== 'undefined' && module.exports) module.exports = ReelsProject;
