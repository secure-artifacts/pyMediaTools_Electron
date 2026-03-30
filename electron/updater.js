/**
 * 自动更新模块 — 基于 GitHub Releases + electron-updater
 * 
 * 双通道发布：
 *   - 正式版 (stable): tag 格式 v2.2.0 → GitHub Release (非 prerelease)
 *   - 测试版 (beta):   tag 格式 v2.3.0-beta.1 → GitHub Release (prerelease)
 * 
 * 用户可在设置中切换是否接收测试版更新。
 */
let autoUpdater = null;
try {
    autoUpdater = require('electron-updater').autoUpdater;
} catch (e) {
    console.warn('[Updater] electron-updater not available in dev mode:', e.message);
}
const { ipcMain, dialog } = require('electron');
const fs = require('fs');
const path = require('path');

let mainWindow = null;
let log = console.log;

// 使用 electron-store 持久化更新通道偏好
let store = null;
try {
    const Store = require('electron-store');
    store = new Store({ name: 'updater-settings' });
} catch (e) {
    // fallback: 无持久化，默认 stable
}

/**
 * 获取当前更新通道
 * @returns {'stable' | 'beta'}
 */
function getUpdateChannel() {
    if (store) {
        return store.get('updateChannel', 'stable');
    }
    return 'stable';
}

/**
 * 设置更新通道
 * @param {'stable' | 'beta'} channel
 */
function setUpdateChannel(channel) {
    const valid = ['stable', 'beta'];
    if (!valid.includes(channel)) channel = 'stable';
    if (store) {
        store.set('updateChannel', channel);
    }
    // 立即生效
    if (!autoUpdater) return;
    autoUpdater.allowPrerelease = (channel === 'beta');

    // 关键：如果当前是测试版，切回 stable 时允许降级到正式版
    const currentVersion = require('electron').app.getVersion();
    const currentIsBeta = isBetaVersion(currentVersion);
    if (channel === 'stable' && currentIsBeta) {
        autoUpdater.allowDowngrade = true;
        log(`[Updater] 当前为测试版 v${currentVersion}，已启用降级模式，可回退到正式版`);
    } else {
        autoUpdater.allowDowngrade = false;
    }

    log(`[Updater] 更新通道已切换为: ${channel} (allowPrerelease=${autoUpdater.allowPrerelease}, allowDowngrade=${autoUpdater.allowDowngrade})`);
}

/**
 * 检测当前版本是否为测试版
 */
function isBetaVersion(version) {
    return /-(beta|alpha|rc|dev)/.test(version || '');
}

function initAutoUpdater(win, logFn) {
    mainWindow = win;
    if (logFn) log = logFn;

    if (!autoUpdater) {
        log('[Updater] autoUpdater 不可用（开发模式），跳过初始化');
        return;
    }

    const appUpdateYml = path.join(process.resourcesPath || '', 'app-update.yml');
    if (!fs.existsSync(appUpdateYml)) {
        log('[Updater] 未检测到 app-update.yml，跳过自动更新初始化');
        return;
    }

    // 读取用户通道偏好
    const channel = getUpdateChannel();

    // 配置
    autoUpdater.autoDownload = false;           // 不自动下载，先提示用户
    autoUpdater.autoInstallOnAppQuit = true;    // 退出时自动安装
    autoUpdater.allowPrerelease = (channel === 'beta'); // 是否接收测试版

    // 如果当前是测试版且用户选了 stable，允许降级
    const currentVersion = require('electron').app.getVersion();
    if (channel === 'stable' && isBetaVersion(currentVersion)) {
        autoUpdater.allowDowngrade = true;
        log(`[Updater] 当前测试版 v${currentVersion}，启用降级模式`);
    }

    log(`[Updater] 初始化 — 通道: ${channel}, allowPrerelease: ${autoUpdater.allowPrerelease}, allowDowngrade: ${autoUpdater.allowDowngrade || false}`);

    // ==================== 事件监听 ====================

    autoUpdater.on('checking-for-update', () => {
        log('[Updater] 正在检查更新...');
        sendToRenderer('update-status', { status: 'checking', message: '正在检查更新...' });
    });

    autoUpdater.on('update-available', (info) => {
        const isBeta = isBetaVersion(info.version);
        const channelLabel = isBeta ? '🧪 测试版' : '✅ 正式版';
        log(`[Updater] 发现新版本: v${info.version} (${channelLabel})`);
        sendToRenderer('update-status', {
            status: 'available',
            message: `发现新版本 v${info.version} (${channelLabel})`,
            version: info.version,
            isBeta,
            releaseNotes: info.releaseNotes || '',
            releaseDate: info.releaseDate || '',
        });

        // 弹窗提示用户
        const detail = isBeta
            ? '这是一个测试版本，可能包含未完善的功能。\n是否立即下载？'
            : '是否立即下载更新？';
        dialog.showMessageBox(mainWindow, {
            type: 'info',
            title: `发现${isBeta ? '测试' : '新'}版本`,
            message: `${channelLabel} v${info.version} 可用！`,
            detail,
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
        const isBeta = isBetaVersion(info.version);
        log(`[Updater] 更新下载完成: v${info.version}`);
        sendToRenderer('update-status', {
            status: 'downloaded',
            message: `v${info.version} 已下载完成，重启即可安装`,
            version: info.version,
            isBeta,
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

    // 获取/设置更新通道
    ipcMain.handle('get-update-channel', () => {
        return {
            channel: getUpdateChannel(),
            currentVersion: require('electron').app.getVersion(),
            isBeta: isBetaVersion(require('electron').app.getVersion()),
        };
    });

    ipcMain.handle('set-update-channel', (event, channel) => {
        setUpdateChannel(channel);
        return { success: true, channel: getUpdateChannel() };
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
