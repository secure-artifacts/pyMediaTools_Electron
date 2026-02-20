const { contextBridge, ipcRenderer } = require('electron');
const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');

function resolveAssetUrl(fileName) {
    if (!fileName) return '';

    const candidates = [];
    if (typeof process.resourcesPath === 'string' && process.resourcesPath) {
        candidates.push(path.join(process.resourcesPath, 'assets', fileName));
    }
    candidates.push(path.join(__dirname, '..', 'assets', fileName));
    candidates.push(path.join(__dirname, '..', 'dist', 'assets', fileName));

    for (const p of candidates) {
        try {
            if (fs.existsSync(p)) {
                return pathToFileURL(p).toString();
            }
        } catch { }
    }
    return '';
}

function toFileUrl(filePath) {
    if (!filePath || typeof filePath !== 'string') return '';
    if (/^file:\/\//i.test(filePath)) return filePath;

    try {
        return pathToFileURL(filePath).toString();
    } catch {
        return '';
    }
}

// 暴露 API 给渲染进程
contextBridge.exposeInMainWorld('electronAPI', {
    // 平台信息
    platform: process.platform,
    resolveAssetUrl,
    toFileUrl,

    // 选择目录
    selectDirectory: () => ipcRenderer.invoke('select-directory'),

    // ==================== 统一 API 调用接口 ====================
    // 替代 fetch(`${API_BASE}/endpoint`, ...) 的调用方式
    // 用法: const result = await window.electronAPI.apiCall('elevenlabs/voices', { key_index: 0 })
    apiCall: (endpoint, data) => ipcRenderer.invoke('api-call', endpoint, data),

    // 文件上传专用接口
    // 用法: const result = await window.electronAPI.apiUpload('upload', fileArrayBuffer, fileName, { extra: 'data' })
    apiUpload: (endpoint, fileBuffer, fileName, formData) =>
        ipcRenderer.invoke('api-upload', endpoint, fileBuffer, fileName, formData),

    // Wav2Lip 进度事件监听
    onWav2LipProgress: (callback) => {
        ipcRenderer.on('wav2lip-progress', (event, data) => callback(data));
    },
});
