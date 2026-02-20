const { app, BrowserWindow, ipcMain, dialog, powerSaveBlocker } = require('electron');
const path = require('path');
const fs = require('fs');

// Node.js API 路由器 —— 替代 Python Flask 后端
const { registerAPIHandlers } = require('./apiRouter');

let mainWindow;
let appIsReady = false;
let powerSaveId = null;
let isQuitting = false;

// 日志文件路径
const logDir = (app && app.isPackaged)
    ? path.join(app.getPath('userData'), 'logs')
    : path.join(__dirname, '..', 'logs');

// 确保日志目录存在
function ensureLogDir() {
    try {
        if (!fs.existsSync(logDir)) {
            fs.mkdirSync(logDir, { recursive: true });
        }
    } catch (e) {
        console.error('Failed to create log directory:', e);
    }
}

// 写日志到文件
function log(message) {
    const timestamp = new Date().toISOString();
    const logMessage = `[${timestamp}] ${message}`;
    console.log(logMessage);

    try {
        ensureLogDir();
        const logFile = path.join(logDir, 'app.log');
        fs.appendFileSync(logFile, logMessage + '\n');
    } catch (e) {
        // 忽略日志写入错误
    }
}

// 获取资源路径
function getResourcePath(relativePath) {
    if (app.isPackaged) {
        return path.join(process.resourcesPath, relativePath);
    }
    return path.join(__dirname, '..', relativePath);
}

// 获取 FFmpeg 路径并注入到 PATH
function setupFFmpegPath() {
    // macOS: 检查打包的 FFmpeg
    if (process.platform === 'darwin') {
        const vendorFfmpeg = path.join(getResourcePath('vendor'), 'ffmpeg');
        if (fs.existsSync(vendorFfmpeg) && fs.existsSync(path.join(vendorFfmpeg, 'ffmpeg'))) {
            log(`Using vendor FFmpeg on macOS: ${vendorFfmpeg}`);
            process.env.PATH = `${vendorFfmpeg}${path.delimiter}${process.env.PATH || ''}`;
            return;
        }

        // 回退到系统安装的 FFmpeg
        const macPaths = [
            '/opt/homebrew/bin',
            '/usr/local/bin',
            '/opt/local/bin',
        ];
        const existingPath = process.env.PATH || '';
        const additionalPaths = macPaths.filter(p => !existingPath.includes(p)).join(path.delimiter);
        if (additionalPaths) {
            process.env.PATH = `${additionalPaths}${path.delimiter}${existingPath}`;
        }
    } else if (process.platform === 'win32') {
        // Packaged: extraResources maps vendor/windows/ffmpeg → vendor/ffmpeg
        // Dev: files sit at vendor/windows/ffmpeg/bin
        const candidates = [
            path.join(getResourcePath('vendor'), 'ffmpeg', 'bin'),
            path.join(getResourcePath('vendor'), 'windows', 'ffmpeg', 'bin'),
        ];
        for (const vendorFfmpeg of candidates) {
            if (fs.existsSync(vendorFfmpeg)) {
                log(`Using vendor FFmpeg on Windows: ${vendorFfmpeg}`);
                process.env.PATH = `${vendorFfmpeg}${path.delimiter}${process.env.PATH || ''}`;
                // Set explicit paths so resolveCommand() and other services can find them
                const ffmpegExe = path.join(vendorFfmpeg, 'ffmpeg.exe');
                const ffprobeExe = path.join(vendorFfmpeg, 'ffprobe.exe');
                if (fs.existsSync(ffmpegExe)) process.env.FFMPEG_PATH = ffmpegExe;
                if (fs.existsSync(ffprobeExe)) process.env.FFPROBE_PATH = ffprobeExe;
                break;
            }
        }
    }
}

// 创建主窗口
function createWindow() {
    if (mainWindow) {
        mainWindow.focus();
        return;
    }

    if (!appIsReady) return;

    mainWindow = new BrowserWindow({
        width: 1200,
        height: 800,
        minWidth: 900,
        minHeight: 700,
        title: 'pyMediaTools',
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            sandbox: false,
            preload: path.join(__dirname, 'preload.js')
        },
        titleBarStyle: 'hiddenInset',
        trafficLightPosition: { x: 15, y: 15 },
    });

    if (!app.isPackaged) {
        mainWindow.loadURL('http://localhost:5173');
        mainWindow.webContents.openDevTools();
    } else {
        mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
    }

    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

// IPC 处理 - 基本功能
ipcMain.handle('select-directory', async () => {
    const result = await dialog.showOpenDialog(mainWindow, {
        properties: ['openDirectory', 'createDirectory'],
        title: '选择输出目录'
    });
    if (!result.canceled && result.filePaths.length > 0) {
        return result.filePaths[0];
    }
    return null;
});

// 应用启动
app.whenReady().then(async () => {
    appIsReady = true;
    log('=== App Ready (Node.js Backend) ===');

    // 设置 FFmpeg 环境
    setupFFmpegPath();
    log(`FFmpeg PATH configured`);

    // 注册 API 路由（替代 Python Flask 后端）
    registerAPIHandlers();
    log('API handlers registered - no Python backend needed');

    // 防止 macOS App Nap
    if (process.platform === 'darwin') {
        powerSaveId = powerSaveBlocker.start('prevent-app-suspension');
        log(`PowerSaveBlocker started: ${powerSaveId}`);
    }

    // 直接创建窗口（不需要等待后端启动了！）
    createWindow();
});

app.on('window-all-closed', () => {
    isQuitting = true;
    if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
    if (appIsReady && BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
});

app.on('before-quit', () => {
    isQuitting = true;
});
