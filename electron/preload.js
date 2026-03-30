const { contextBridge, ipcRenderer, webUtils } = require('electron');
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
    // 获取 File 对象的本地完整路径（contextIsolation 下 File.path 不可用）
    getFilePath: (file) => {
        try {
            const p = webUtils.getPathForFile(file);
            console.log('[preload.getFilePath] success:', p);
            return p;
        } catch (e) {
            console.error('[preload.getFilePath] FAILED:', e.message, 'file:', typeof file, file?.name);
            return '';
        }
    },
    getAppVersion: () => ipcRenderer.invoke('get-app-version'),

    // 选择目录
    selectDirectory: () => ipcRenderer.invoke('select-directory'),
    scanDirectory: (dirPath) => ipcRenderer.invoke('scan-directory', dirPath),
    getDownloadsPath: () => ipcRenderer.invoke('get-downloads-path'),

    // 批量Reels - 烧录字幕
    burnSubtitles: (opts) => ipcRenderer.invoke('burn-subtitles', opts),
    reelsCompose: (opts) => ipcRenderer.invoke('reels-compose', opts),
    concatVideo: (opts) => ipcRenderer.invoke('concat-video', opts),
    reelsComposeWysiwyg: (action, data) => ipcRenderer.invoke('reels-compose-wysiwyg', action, data),
    getMediaDuration: (filePath) => ipcRenderer.invoke('get-media-duration', filePath),
    saveRenderedAudio: (wavData) => ipcRenderer.invoke('save-rendered-audio', wavData),
    readFileBuffer: (filePath) => ipcRenderer.invoke('read-file-buffer', filePath),

    // 自动更新
    checkForUpdates: () => ipcRenderer.invoke('check-for-updates'),
    downloadUpdate: () => ipcRenderer.invoke('download-update'),
    installUpdate: () => ipcRenderer.invoke('install-update'),
    getUpdateChannel: () => ipcRenderer.invoke('get-update-channel'),
    setUpdateChannel: (channel) => ipcRenderer.invoke('set-update-channel', channel),
    onUpdateStatus: (callback) => {
        const handler = (_, data) => callback(data);
        ipcRenderer.on('update-status', handler);
        return () => ipcRenderer.removeListener('update-status', handler);
    },

    // 扫描本地字体
    scanFonts: () => ipcRenderer.invoke('scan-fonts'),

    // 读取文本文件
    readFileText: (filePath) => {
        try { return fs.readFileSync(filePath, 'utf-8'); } catch { return ''; }
    },

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

    // 批量下载进度事件监听
    onBatchDownloadProgress: (callback) => {
        ipcRenderer.on('batch-download-progress', (event, data) => callback(data));
    },

    // 链接截图进度事件监听
    onUrlThumbnailProgress: (callback) => {
        ipcRenderer.on('url-thumbnail-progress', (event, data) => callback(event, data));
    },

    // 在 Finder/Explorer 中高亮显示文件
    showItemInFolder: (filePath) => {
        ipcRenderer.invoke('show-item-in-folder', filePath).catch(() => {});
    },
});
