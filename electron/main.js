const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

let mainWindow;
let pythonProcess;
let appIsReady = false;

// 获取资源路径 - 打包后在 resources 目录，开发时在项目根目录
function getResourcePath(relativePath) {
    if (app.isPackaged) {
        return path.join(process.resourcesPath, relativePath);
    }
    return path.join(__dirname, '..', relativePath);
}

// 获取 vendor 路径
function getVendorPath() {
    return path.join(getResourcePath('vendor'), 'windows');
}

// 获取打包的 Python 路径
function getPythonPath() {
    if (process.platform === 'win32' && app.isPackaged) {
        const vendorPython = path.join(getVendorPath(), 'python', 'python.exe');
        console.log('Checking vendor Python at:', vendorPython);
        if (fs.existsSync(vendorPython)) {
            return vendorPython;
        }
    }
    // 开发模式或 Mac
    return process.platform === 'win32' ? 'python' : 'python3';
}

// 获取 FFmpeg bin 路径
function getFfmpegBinPath() {
    if (process.platform === 'win32') {
        const vendorFfmpeg = path.join(getVendorPath(), 'ffmpeg', 'bin');
        if (fs.existsSync(vendorFfmpeg)) {
            return vendorFfmpeg;
        }
    }
    return null;
}

// 设置环境变量
function getEnv() {
    const env = { ...process.env };

    const ffmpegPath = getFfmpegBinPath();
    if (ffmpegPath) {
        env.PATH = `${ffmpegPath}${path.delimiter}${env.PATH || ''}`;
        env.FFMPEG_PATH = path.join(ffmpegPath, 'ffmpeg.exe');
        env.FFPROBE_PATH = path.join(ffmpegPath, 'ffprobe.exe');
    }

    return env;
}

// 启动 Python 后端
function startPythonBackend() {
    const pythonPath = getPythonPath();
    const scriptPath = getResourcePath('backend/server.py');
    const env = getEnv();

    console.log('=== Starting Python Backend ===');
    console.log('app.isPackaged:', app.isPackaged);
    console.log('process.resourcesPath:', process.resourcesPath);
    console.log('Python path:', pythonPath);
    console.log('Script path:', scriptPath);
    console.log('Python exists:', fs.existsSync(pythonPath));
    console.log('Script exists:', fs.existsSync(scriptPath));

    if (!fs.existsSync(scriptPath)) {
        console.error('Server script not found at:', scriptPath);
        return;
    }

    try {
        pythonProcess = spawn(pythonPath, [scriptPath], {
            stdio: ['pipe', 'pipe', 'pipe'],
            env: { ...env, PYTHONUNBUFFERED: '1' },
            cwd: path.dirname(scriptPath)
        });

        pythonProcess.stdout.on('data', (data) => {
            console.log(`Python: ${data}`);
        });

        pythonProcess.stderr.on('data', (data) => {
            console.error(`Python Error: ${data}`);
        });

        pythonProcess.on('error', (err) => {
            console.error('Failed to start Python:', err);
        });

        pythonProcess.on('close', (code) => {
            console.log(`Python exited with code ${code}`);
        });

    } catch (err) {
        console.error('Spawn error:', err);
    }
}

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

app.whenReady().then(() => {
    appIsReady = true;
    startPythonBackend();
    setTimeout(() => createWindow(), 2000);
});

app.on('window-all-closed', () => {
    if (pythonProcess) pythonProcess.kill();
    if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
    if (appIsReady && BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
});

app.on('before-quit', () => {
    if (pythonProcess) pythonProcess.kill();
});
