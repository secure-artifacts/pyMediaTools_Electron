// ==================== 一键配音字幕 ====================

// 任务数据
let vwTasks = [];

// 刷新音色列表
async function refreshVWVoices() {
    const select = document.getElementById('vw-default-voice');
    if (!select) return;

    try {
        const response = await fetch(`${API_BASE}/elevenlabs/voices`);
        const data = await response.json();

        if (data.voices && data.voices.length > 0) {
            select.innerHTML = data.voices.map(v =>
                `<option value="${v.voice_id}">${v.name}</option>`
            ).join('');
        }
    } catch (error) {
        console.error('获取音色失败:', error);
    }
}

// 从剪贴板粘贴数据
async function vwPasteFromClipboard() {
    try {
        const clipboardItems = await navigator.clipboard.read();
        let rows = [];

        for (const item of clipboardItems) {
            // 优先解析 HTML
            if (item.types.includes('text/html')) {
                const blob = await item.getType('text/html');
                const html = await blob.text();
                const parser = new DOMParser();
                const doc = parser.parseFromString(html, 'text/html');
                const tableRows = doc.querySelectorAll('tr');

                // 辅助函数：获取单元格文本，保留换行
                const getCellTextWithBreaks = (cell) => {
                    if (!cell) return '';
                    // 把 <br> 标签替换成换行符
                    let clone = cell.cloneNode(true);
                    clone.querySelectorAll('br').forEach(br => br.replaceWith('\n'));
                    // 把 <p> 和 <div> 也替换成换行
                    clone.querySelectorAll('p, div').forEach(el => {
                        el.insertAdjacentText('beforebegin', '\n');
                    });
                    return clone.textContent.trim();
                };

                tableRows.forEach(tr => {
                    const cells = tr.querySelectorAll('td, th');
                    if (cells.length >= 2) {
                        const ttsText = getCellTextWithBreaks(cells[0]);
                        const subtitleText = getCellTextWithBreaks(cells[1]);
                        const voiceId = cells[2]?.textContent.trim() || '';
                        if (ttsText) {
                            rows.push({ ttsText, subtitleText, voiceId, split: true, exportMp4: false });
                        }
                    }
                });
            }

            // 如果 HTML 没数据，尝试纯文本
            if (rows.length === 0 && item.types.includes('text/plain')) {
                const blob = await item.getType('text/plain');
                const text = await blob.text();
                const lines = text.split('\n').filter(l => l.trim());

                lines.forEach(line => {
                    const parts = line.split('\t');
                    if (parts.length >= 2) {
                        rows.push({
                            ttsText: parts[0]?.trim() || '',
                            subtitleText: parts[1]?.trim() || '',
                            voiceId: parts[2]?.trim() || '',
                            split: true,
                            exportMp4: false
                        });
                    }
                });
            }
        }

        if (rows.length === 0) {
            showToast('未识别到有效数据', 'warning');
            return;
        }

        vwTasks = rows.map((row, idx) => ({
            id: idx,
            ...row,
            status: 'pending',
            error: null,
            audioPath: null,
            segments: null
        }));

        renderVWTasks();
        updateVWTaskCount();
        document.getElementById('vw-start-btn').disabled = false;
        showToast(`已添加 ${vwTasks.length} 条任务`, 'success');

    } catch (error) {
        showToast('粘贴失败: ' + error.message, 'error');
    }
}

// 渲染任务列表
function renderVWTasks() {
    const container = document.getElementById('vw-task-list');
    if (!container) return;

    if (vwTasks.length === 0) {
        container.innerHTML = '<p class="hint" style="text-align: center;">请从表格粘贴数据...</p>';
        return;
    }

    container.innerHTML = vwTasks.map((task, idx) => `
        <div class="vw-task-card" data-id="${task.id}" style="background: var(--bg-secondary); border-radius: 6px; padding: 10px; margin-bottom: 8px; border-left: 3px solid ${getStatusColor(task.status)};">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                <span style="font-size: 11px; color: var(--text-muted);">#${idx + 1}</span>
                <label style="display: flex; align-items: center; gap: 4px; font-size: 12px; cursor: pointer; user-select: none;" title="拆分后不能导出黑屏MP4">
                    <input type="checkbox" class="vw-split-checkbox" data-id="${task.id}" ${task.split ? 'checked' : ''} onchange="vwToggleSplit(${task.id}, this.checked)" style="cursor: pointer;">
                    <span>拆分</span>
                </label>
                <label style="display: flex; align-items: center; gap: 4px; font-size: 12px; cursor: pointer; user-select: none;" title="导出黑屏双声道MP4（会自动取消拆分）">
                    <input type="checkbox" class="vw-mp4-checkbox" data-id="${task.id}" ${task.exportMp4 ? 'checked' : ''} onchange="vwToggleMp4(${task.id}, this.checked)" style="cursor: pointer;">
                    <span>黑屏MP4</span>
                </label>
                <span class="vw-task-status" style="margin-left: auto; font-size: 11px; padding: 2px 6px; border-radius: 3px; background: ${getStatusBg(task.status)}; color: ${getStatusColor(task.status)};">
                    ${getStatusText(task.status)}
                </span>
            </div>
            <div style="font-size: 12px; color: var(--text-primary); margin-bottom: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="${escapeHtml(task.ttsText)}">
                <strong>TTS:</strong> ${escapeHtml(task.ttsText.substring(0, 80))}${task.ttsText.length > 80 ? '...' : ''}
            </div>
            <div style="font-size: 11px; color: var(--text-secondary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="${escapeHtml(task.subtitleText)}">
                <strong>字幕:</strong> ${escapeHtml(task.subtitleText.substring(0, 60).replace(/\n/g, ' | '))}${task.subtitleText.length > 60 ? '...' : ''}
            </div>
            ${task.voiceId ? `<div style="font-size: 10px; color: var(--text-muted); margin-top: 2px;">音色: ${task.voiceId}</div>` : ''}
            ${task.error ? `<div style="font-size: 10px; color: #ff6b6b; margin-top: 4px;">❌ ${escapeHtml(task.error)}</div>` : ''}
            ${task.segments ? `<div style="font-size: 10px; color: #51cf66; margin-top: 4px;">✅ ${task.segments.length} 个片段</div>` : ''}
        </div>
    `).join('');
}

function getStatusColor(status) {
    const colors = {
        pending: '#868e96',
        generating: '#ffd43b',
        splitting: '#74c0fc',
        aligning: '#b197fc',
        done: '#51cf66',
        error: '#ff6b6b'
    };
    return colors[status] || colors.pending;
}

function getStatusBg(status) {
    return getStatusColor(status) + '22';
}

function getStatusText(status) {
    const texts = {
        pending: '待处理',
        generating: '生成音频...',
        splitting: '智能拆分...',
        aligning: '对齐字幕...',
        done: '完成',
        error: '失败'
    };
    return texts[status] || '未知';
}

function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// 切换单个任务的拆分状态
function vwToggleSplit(id, checked) {
    const task = vwTasks.find(t => t.id === id);
    if (task) {
        // 如果传入了 checked 参数，使用它；否则切换
        task.split = checked !== undefined ? checked : !task.split;
        // 如果勾选拆分，自动取消黑屏MP4
        if (task.split) {
            task.exportMp4 = false;
        }
        renderVWTasks();
        updateVWTaskCount();
        updateSelectAllState();  // 更新全选状态
    }
}

// 切换单个任务的黑屏MP4状态
function vwToggleMp4(id, checked) {
    const task = vwTasks.find(t => t.id === id);
    if (task) {
        // 如果传入了 checked 参数，使用它；否则切换
        const newValue = checked !== undefined ? checked : !task.exportMp4;

        if (newValue) {
            // 如果勾选黑屏MP4，自动取消拆分
            task.split = false;
        }
        task.exportMp4 = newValue;

        renderVWTasks();  // 重新渲染以更新UI
        updateVWTaskCount();
        updateSelectAllState();  // 更新全选状态
    }
}

// 全选/取消拆分
function vwToggleAllSplit() {
    const checked = document.getElementById('vw-select-all-split').checked;
    vwTasks.forEach(t => {
        t.split = checked;
        if (checked) t.exportMp4 = false;  // 互斥：拆分时取消黑屏MP4
    });
    // 如果勾选了拆分，自动取消全选黑屏MP4
    if (checked) {
        document.getElementById('vw-select-all-mp4').checked = false;
    }
    renderVWTasks();
    updateVWTaskCount();
}

// 全选/取消黑屏MP4
function vwToggleAllMp4() {
    const checked = document.getElementById('vw-select-all-mp4').checked;

    // 如果要勾选黑屏MP4，先取消全选拆分
    if (checked) {
        document.getElementById('vw-select-all-split').checked = false;
        vwTasks.forEach(t => {
            t.split = false;
            t.exportMp4 = true;
        });
    } else {
        vwTasks.forEach(t => {
            t.exportMp4 = false;
        });
    }
    renderVWTasks();
    updateVWTaskCount();
}

// 更新全选复选框状态（根据当前任务状态）
function updateSelectAllState() {
    const allSplit = vwTasks.every(t => t.split);
    const noneSplit = vwTasks.every(t => !t.split);
    const allMp4 = vwTasks.every(t => t.exportMp4);
    const noneMp4 = vwTasks.every(t => !t.exportMp4);

    const splitCb = document.getElementById('vw-select-all-split');
    const mp4Cb = document.getElementById('vw-select-all-mp4');

    if (splitCb) {
        splitCb.checked = allSplit;
        splitCb.indeterminate = !allSplit && !noneSplit;
    }
    if (mp4Cb) {
        mp4Cb.checked = allMp4;
        mp4Cb.indeterminate = !allMp4 && !noneMp4;
    }
}

// 反选拆分
function vwInvertSplit() {
    vwTasks.forEach(t => {
        t.split = !t.split;
        if (t.split) t.exportMp4 = false;
    });
    renderVWTasks();
    updateVWTaskCount();
}

// 清空任务
function vwClearAll() {
    vwTasks = [];
    renderVWTasks();
    updateVWTaskCount();
    document.getElementById('vw-start-btn').disabled = true;
}

// 更新任务计数
function updateVWTaskCount() {
    const countEl = document.getElementById('vw-task-count');
    if (countEl) {
        const splitCount = vwTasks.filter(t => t.split).length;
        const mp4Count = vwTasks.filter(t => t.exportMp4).length;
        countEl.textContent = `共 ${vwTasks.length} 条，${splitCount} 条拆分，${mp4Count} 条黑屏MP4`;
    }
}

// 更新进度
function updateVWProgress(current, total, text) {
    const progressEl = document.getElementById('vw-progress');
    const textEl = document.getElementById('vw-progress-text');
    const percentEl = document.getElementById('vw-progress-percent');
    const barEl = document.getElementById('vw-progress-bar');

    progressEl.style.display = 'block';
    textEl.textContent = text;
    const percent = Math.round((current / total) * 100);
    percentEl.textContent = percent + '%';
    barEl.style.width = percent + '%';
}

// 开始工作流
async function startVoiceoverWorkflow() {
    if (vwTasks.length === 0) {
        showToast('请先添加任务', 'warning');
        return;
    }

    const defaultVoice = document.getElementById('vw-default-voice').value;
    const modelId = document.getElementById('vw-model')?.value || 'eleven_v3';
    const maxDuration = parseInt(document.getElementById('vw-max-duration').value) || 30;
    const outputDir = document.getElementById('vw-output-dir').value.trim();

    if (!defaultVoice) {
        showToast('请选择默认音色', 'warning');
        return;
    }

    // 输出目录可以为空，后端会使用默认的下载文件夹

    const btn = document.getElementById('vw-start-btn');
    btn.disabled = true;
    btn.textContent = '⏳ 处理中...';

    const total = vwTasks.length * 3;  // 每个任务 3 步
    let current = 0;

    try {
        for (let i = 0; i < vwTasks.length; i++) {
            const task = vwTasks[i];
            const voiceId = task.voiceId || defaultVoice;

            // Step 1: 生成音频
            task.status = 'generating';
            renderVWTasks();
            updateVWProgress(current, total, `[${i + 1}/${vwTasks.length}] 生成音频...`);

            try {
                const ttsResponse = await fetch(`${API_BASE}/elevenlabs/tts-workflow`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        text: task.ttsText,
                        voice_id: voiceId,
                        model_id: modelId,
                        task_index: i,
                        need_split: task.split,
                        max_duration: maxDuration,
                        subtitle_text: task.subtitleText,
                        export_mp4: task.exportMp4,  // 从任务读取
                        output_dir: outputDir
                    })
                });

                const ttsData = await ttsResponse.json();

                if (!ttsResponse.ok) {
                    throw new Error(ttsData.error || '生成失败');
                }

                task.audioPath = ttsData.audio_path;
                task.outputFolder = ttsData.output_folder;
                task.segments = ttsData.segments;
                task.status = 'done';

            } catch (err) {
                task.status = 'error';
                task.error = err.message;
            }

            current += 3;
            renderVWTasks();
        }

        const successCount = vwTasks.filter(t => t.status === 'done').length;
        showToast(`完成！成功 ${successCount}/${vwTasks.length} 条`, successCount === vwTasks.length ? 'success' : 'warning');

    } catch (error) {
        showToast('工作流执行失败: ' + error.message, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = '🚀 开始一键生成';
        document.getElementById('vw-progress').style.display = 'none';
    }
}

// 页面加载时刷新音色
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(refreshVWVoices, 1000);
});

// 浏览输出目录
async function vwBrowseOutputDir() {
    // 检查是否在 Electron 环境
    if (window.electronAPI && window.electronAPI.selectDirectory) {
        try {
            const dirPath = await window.electronAPI.selectDirectory();
            if (dirPath) {
                document.getElementById('vw-output-dir').value = dirPath;
            }
        } catch (err) {
            console.error('选择目录失败:', err);
            showToast('选择目录失败', 'error');
        }
    } else if (window.require) {
        // 直接使用 Electron remote
        try {
            const { dialog } = window.require('@electron/remote');
            const result = await dialog.showOpenDialog({
                properties: ['openDirectory', 'createDirectory'],
                title: '选择输出目录'
            });
            if (!result.canceled && result.filePaths.length > 0) {
                document.getElementById('vw-output-dir').value = result.filePaths[0];
            }
        } catch (err) {
            console.error('选择目录失败:', err);
            // 回退：让用户直接编辑输入框
            showToast('请直接在输入框中输入完整路径', 'info');
            document.getElementById('vw-output-dir').focus();
        }
    } else {
        // 浏览器环境
        showToast('请直接在输入框中输入完整路径', 'info');
        document.getElementById('vw-output-dir').focus();
    }
}

// 打开输出文件夹
async function vwOpenOutputDir() {
    let outputDir = document.getElementById('vw-output-dir').value.trim();

    // 如果没有指定，使用默认目录（让后端处理）
    if (!outputDir) {
        const today = new Date().toISOString().split('T')[0];  // YYYY-MM-DD
        outputDir = `~/Downloads/${today}_一键配音`;
    }

    // 调用后端 API 打开文件夹
    try {
        const response = await fetch(`${API_BASE}/open-folder`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: outputDir })
        });

        const result = await response.json();
        if (!response.ok) {
            showToast('打开失败: ' + (result.error || '未知错误'), 'error');
        }
    } catch (err) {
        console.error('打开文件夹失败:', err);
        showToast('打开失败，文件夹可能不存在', 'error');
    }
}
