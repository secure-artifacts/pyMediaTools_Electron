const { contextBridge, ipcRenderer } = require('electron');

// 暴露 API 给渲染进程
contextBridge.exposeInMainWorld('electronAPI', {
    // 平台信息
    platform: process.platform,
    // 选择目录
    selectDirectory: () => ipcRenderer.invoke('select-directory'),
});
