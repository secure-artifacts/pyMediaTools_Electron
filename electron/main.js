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

// 获取 vendor 路径 - 根据平台
function getVendorPath() {
    if (process.platform === 'win32') {
        return path.join(getResourcePath('vendor'), 'windows');
    }
    return getResourcePath('vendor');
}

// 获取打包的 Python 路径
function getPythonPath() {
    if (process.platform === 'win32') {
        // Windows: 优先使用打包的 Python
        const vendorPython = path.join(getVendorPath(), 'python', 'python.exe');
        console.log('Checking vendor Python at:', vendorPython);
        console.log('Vendor Python exists:', fs.existsSync(vendorPython));

        if (fs.existsSync(vendorPython)) {
            return vendorPython;
        }

        // 如果打包的 Python 不存在，尝试系统 Python
        console.log('Vendor Python not found, trying system python...');
        return 'python';
    }

    // macOS / Linux: 使用系统 Python
    return 'python3';
}

// 获取 FFmpeg bin 路径
function getFfmpegBinPath() {
    if (process.platform === 'win32') {
        const vendorFfmpeg = path.join(getVendorPath(), 'ffmpeg', 'bin');
        console.log('Checking FFmpeg at:', vendorFfmpeg);
        console.log('FFmpeg exists:', fs.existsSync(vendorFfmpeg));
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

    // 确保 Python 能找到正确的模块
    if (process.platform === 'win32') {
        const pythonDir = path.join(getVendorPath(), 'python');
        const sitePackages = path.join(pythonDir, 'Lib', 'site-packages');
        env.PYTHONPATH = sitePackages;
        console.log('Setting PYTHONPATH:', sitePackages);
    }

    return env;
}

// 启动 Python 后端
async function startPythonBackend() {
    const pythonPath = getPythonPath();
    const scriptPath = getResourcePath('backend/server.py');
    const env = getEnv();

    console.log('=== Starting Python Backend ===');
    console.log('Platform:', process.platform);
    console.log('app.isPackaged:', app.isPackaged);
    console.log('process.resourcesPath:', process.resourcesPath);
    console.log('Python path:', pythonPath);
    console.log('Script path:', scriptPath);
    console.log('Script exists:', fs.existsSync(scriptPath));

    // 列出 vendor 目录内容（调试用）
    const vendorPath = getVendorPath();
    console.log('Vendor path:', vendorPath);
    if (fs.existsSync(vendorPath)) {
        console.log('Vendor directory contents:', fs.readdirSync(vendorPath));
    } else {
        console.log('Vendor directory does not exist!');
    }

    if (!fs.existsSync(scriptPath)) {
        console.error('Server script not found at:', scriptPath);
        return;
    }

    // 先清理可能残留的旧进程
    await killExistingBackend();

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
            if (code !== 0 && code !== null) {
                console.error('Python backend crashed, attempting restart in 2 seconds...');
                // 如果不是正常退出，2秒后尝试重启
                setTimeout(() => {
                    if (appIsReady && !isQuitting) {
                        console.log('Restarting Python backend...');
                        startPythonBackend();
                    }
                }, 2000);
            }
        });

    } catch (err) {
        console.error('Spawn error:', err);
    }
}

// 杀死可能残留的旧后端进程
async function killExistingBackend() {
    return new Promise((resolve) => {
        const port = 5001;
        let killCmd;

        if (process.platform === 'win32') {
            // Windows: 查找并杀死占用端口的进程
            killCmd = `for /f "tokens=5" %a in ('netstat -aon ^| findstr :${port} ^| findstr LISTENING') do taskkill /F /PID %a`;
        } else {
            // macOS / Linux
            killCmd = `lsof -ti :${port} | xargs kill -9 2>/dev/null || true`;
        }

        const { exec } = require('child_process');
        exec(killCmd, (error, stdout, stderr) => {
            if (stdout) console.log('Killed existing process on port 5001:', stdout);
            if (stderr && !stderr.includes('No such process')) console.log('Kill stderr:', stderr);
            // 等待端口释放
            setTimeout(resolve, 500);
        });
    });
}

let isQuitting = false;

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
    isQuitting = true;
    killPythonProcess();
    if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
    if (appIsReady && BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
});

app.on('before-quit', () => {
    isQuitting = true;
    killPythonProcess();
});

// 强制终止 Python 进程
function killPythonProcess() {
    if (pythonProcess) {
        try {
            // 先尝试正常终止
            pythonProcess.kill('SIGTERM');
            // 1秒后强制终止
            setTimeout(() => {
                if (pythonProcess && !pythonProcess.killed) {
                    pythonProcess.kill('SIGKILL');
                }
            }, 1000);
        } catch (e) {
            console.error('Error killing Python process:', e);
        }
        pythonProcess = null;
    }
}
