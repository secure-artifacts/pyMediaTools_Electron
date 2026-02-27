/**
 * 自动更新模块 — 基于 GitHub Releases + electron-updater
 */
const { autoUpdater } = require('electron-updater');
const { ipcMain, dialog } = require('electron');

let mainWindow = null;
let log = console.log;

function initAutoUpdater(win, logFn) {
    mainWindow = win;
    if (logFn) log = logFn;

    // 配置
    autoUpdater.autoDownload = false;      // 不自动下载，先提示用户
    autoUpdater.autoInstallOnAppQuit = true; // 退出时自动安装
    autoUpdater.allowPrerelease = false;    // 不检查预发布版本

    // ==================== 事件监听 ====================

    autoUpdater.on('checking-for-update', () => {
        log('[Updater] 正在检查更新...');
        sendToRenderer('update-status', { status: 'checking', message: '正在检查更新...' });
    });

    autoUpdater.on('update-available', (info) => {
        log(`[Updater] 发现新版本: v${info.version}`);
        sendToRenderer('update-status', {
            status: 'available',
            message: `发现新版本 v${info.version}`,
            version: info.version,
            releaseNotes: info.releaseNotes || '',
            releaseDate: info.releaseDate || '',
        });

        // 弹窗提示用户
        dialog.showMessageBox(mainWindow, {
            type: 'info',
            title: '发现新版本',
            message: `新版本 v${info.version} 可用！`,
            detail: '是否立即下载更新？',
            buttons: ['下载更新', '稍后提醒'],
            defaultId: 0,
            cancelId: 1,
        }).then(({ response }) => {
            if (response === 0) {
                autoUpdater.downloadUpdate();
            }
        });
    });

    autoUpdater.on('update-not-available', (info) => {
        log('[Updater] 当前已是最新版本');
        sendToRenderer('update-status', { status: 'up-to-date', message: '当前已是最新版本' });
    });

    autoUpdater.on('download-progress', (progress) => {
        const percent = Math.round(progress.percent);
        log(`[Updater] 下载进度: ${percent}%`);
        sendToRenderer('update-status', {
            status: 'downloading',
            message: `下载更新中... ${percent}%`,
            percent,
            bytesPerSecond: progress.bytesPerSecond,
            transferred: progress.transferred,
            total: progress.total,
        });
    });

    autoUpdater.on('update-downloaded', (info) => {
        log(`[Updater] 更新下载完成: v${info.version}`);
        sendToRenderer('update-status', {
            status: 'downloaded',
            message: `v${info.version} 已下载完成，重启即可安装`,
            version: info.version,
        });

        // 弹窗提示重启
        dialog.showMessageBox(mainWindow, {
            type: 'info',
            title: '更新已就绪',
            message: `v${info.version} 已下载完成`,
            detail: '重启应用即可完成更新。是否立即重启？',
            buttons: ['立即重启', '稍后'],
            defaultId: 0,
            cancelId: 1,
        }).then(({ response }) => {
            if (response === 0) {
                autoUpdater.quitAndInstall();
            }
        });
    });

    autoUpdater.on('error', (error) => {
        log(`[Updater] 更新错误: ${error.message}`);
        sendToRenderer('update-status', {
            status: 'error',
            message: `更新检查失败: ${error.message}`,
        });
    });

    // ==================== IPC 接口 ====================

    // 手动检查更新
    ipcMain.handle('check-for-updates', async () => {
        try {
            const result = await autoUpdater.checkForUpdates();
            return { success: true, version: result?.updateInfo?.version };
        } catch (error) {
            return { success: false, error: error.message };
        }
    });

    // 手动下载更新
    ipcMain.handle('download-update', async () => {
        try {
            await autoUpdater.downloadUpdate();
            return { success: true };
        } catch (error) {
            return { success: false, error: error.message };
        }
    });

    // 安装更新并重启
    ipcMain.handle('install-update', () => {
        autoUpdater.quitAndInstall();
    });

    // 启动后延迟 5 秒自动检查更新
    setTimeout(() => {
        log('[Updater] 自动检查更新...');
        autoUpdater.checkForUpdates().catch(err => {
            log(`[Updater] 自动检查失败: ${err.message}`);
        });
    }, 5000);

    log('[Updater] 自动更新模块已初始化');
}

function sendToRenderer(channel, data) {
    if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send(channel, data);
    }
}

module.exports = { initAutoUpdater };
