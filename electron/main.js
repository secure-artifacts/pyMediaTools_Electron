const { app, BrowserWindow, ipcMain, dialog, powerSaveBlocker } = require('electron');
const path = require('path');
const { spawn, exec } = require('child_process');
const fs = require('fs');
const http = require('http');

let mainWindow;
let pythonProcess;
let appIsReady = false;
let powerSaveId = null;
let isQuitting = false;
let backendStartAttempts = 0;
const MAX_BACKEND_ATTEMPTS = 5;

// 日志文件路径
const logDir = app.isPackaged
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

// 获取 vendor 路径
function getVendorPath() {
    if (process.platform === 'win32') {
        return path.join(getResourcePath('vendor'), 'windows');
    }
    return getResourcePath('vendor');
}

// 获取 Python 路径
function getPythonPath() {
    if (process.platform === 'win32') {
        const vendorPython = path.join(getVendorPath(), 'python', 'python.exe');
        log(`Checking vendor Python at: ${vendorPython}`);
        if (fs.existsSync(vendorPython)) {
            log('Using vendor Python');
            return vendorPython;
        }
        log('Vendor Python not found, using system python');
        return 'python';
    }

    // macOS: 优先使用打包的 Python
    if (process.platform === 'darwin') {
        const vendorPython = path.join(getResourcePath('vendor'), 'python', 'bin', 'python3');
        log(`Checking macOS vendor Python at: ${vendorPython}`);
        if (fs.existsSync(vendorPython)) {
            log('Using vendor Python on macOS');
            return vendorPython;
        }
        log('Vendor Python not found on macOS, trying system Python');
    }

    // 回退: 尝试系统 Python（优先使用安装了依赖的 Python）
    const pythonPaths = [
        '/opt/homebrew/bin/python3',                                    // Homebrew (Apple Silicon)
        '/usr/local/bin/python3',                                       // Homebrew (Intel) or user-installed
        '/Library/Frameworks/Python.framework/Versions/3.13/bin/python3', // Python.org 3.13
        '/Library/Frameworks/Python.framework/Versions/3.12/bin/python3', // Python.org 3.12
        '/Library/Frameworks/Python.framework/Versions/3.11/bin/python3', // Python.org 3.11
        '/Library/Frameworks/Python.framework/Versions/3.10/bin/python3', // Python.org 3.10
        '/usr/bin/python3',                                             // System Python (no pip packages)
        'python3',
        'python'
    ];

    for (const pythonPath of pythonPaths) {
        try {
            if (pythonPath.startsWith('/') && fs.existsSync(pythonPath)) {
                log(`Found system Python at: ${pythonPath}`);
                return pythonPath;
            }
        } catch (e) {
            // 继续尝试下一个
        }
    }

    log('Using default python3');
    return 'python3';
}

// 获取 FFmpeg 路径
function getFfmpegBinPath() {
    if (process.platform === 'win32') {
        const vendorFfmpeg = path.join(getVendorPath(), 'ffmpeg', 'bin');
        if (fs.existsSync(vendorFfmpeg)) {
            return vendorFfmpeg;
        }
    }

    // macOS: 检查打包的 FFmpeg
    if (process.platform === 'darwin') {
        const vendorFfmpeg = path.join(getResourcePath('vendor'), 'ffmpeg');
        log(`Checking macOS vendor FFmpeg at: ${vendorFfmpeg}`);
        if (fs.existsSync(vendorFfmpeg) && fs.existsSync(path.join(vendorFfmpeg, 'ffmpeg'))) {
            log('Using vendor FFmpeg on macOS');
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
        // 优先使用打包的 FFmpeg
        env.PATH = `${ffmpegPath}${path.delimiter}${env.PATH || ''}`;

        if (process.platform === 'win32') {
            env.FFMPEG_PATH = path.join(ffmpegPath, 'ffmpeg.exe');
            env.FFPROBE_PATH = path.join(ffmpegPath, 'ffprobe.exe');
        } else {
            // macOS/Linux
            env.FFMPEG_PATH = path.join(ffmpegPath, 'ffmpeg');
            env.FFPROBE_PATH = path.join(ffmpegPath, 'ffprobe');
        }
        log(`Using bundled FFmpeg: ${env.FFMPEG_PATH}`);
    } else if (process.platform === 'darwin') {
        // macOS: 回退到系统安装的 FFmpeg
        const macPaths = [
            '/opt/homebrew/bin',           // Homebrew Apple Silicon
            '/usr/local/bin',              // Homebrew Intel / user-installed
            '/opt/local/bin',              // MacPorts
            '/usr/bin'
        ];
        const existingPath = env.PATH || '';
        const additionalPaths = macPaths.filter(p => !existingPath.includes(p)).join(path.delimiter);
        if (additionalPaths) {
            env.PATH = `${additionalPaths}${path.delimiter}${existingPath}`;
        }
    }

    if (process.platform === 'win32') {
        const pythonDir = path.join(getVendorPath(), 'python');
        const sitePackages = path.join(pythonDir, 'Lib', 'site-packages');
        env.PYTHONPATH = sitePackages;
    }

    return env;
}

// 检查后端是否已启动
function checkBackendHealth() {
    return new Promise((resolve) => {
        const options = {
            hostname: '127.0.0.1',
            port: 5001,
            path: '/api/health',
            method: 'GET',
            timeout: 2000
        };

        const req = http.request(options, (res) => {
            resolve(res.statusCode === 200);
        });

        req.on('error', () => resolve(false));
        req.on('timeout', () => {
            req.destroy();
            resolve(false);
        });

        req.end();
    });
}

// 等待后端启动
async function waitForBackend(maxWaitSeconds = 15) {
    log(`Waiting for backend to start (max ${maxWaitSeconds}s)...`);
    const startTime = Date.now();

    while (Date.now() - startTime < maxWaitSeconds * 1000) {
        const isHealthy = await checkBackendHealth();
        if (isHealthy) {
            log('Backend is healthy!');
            return true;
        }
        await new Promise(r => setTimeout(r, 500));
    }

    log('Backend health check timeout');
    return false;
}

// 杀死端口占用的进程
async function killExistingBackend() {
    return new Promise((resolve) => {
        const port = 5001;
        let killCmd;

        if (process.platform === 'win32') {
            killCmd = `for /f "tokens=5" %a in ('netstat -aon ^| findstr :${port} ^| findstr LISTENING') do taskkill /F /PID %a`;
        } else {
            killCmd = `lsof -ti :${port} | xargs kill -9 2>/dev/null || true`;
        }

        exec(killCmd, (error, stdout, stderr) => {
            if (stdout) log(`Killed process on port ${port}: ${stdout}`);
            setTimeout(resolve, 500);
        });
    });
}

// 启动 Python 后端
async function startPythonBackend() {
    if (isQuitting) return;

    backendStartAttempts++;
    log(`=== Starting Python Backend (Attempt ${backendStartAttempts}/${MAX_BACKEND_ATTEMPTS}) ===`);

    const pythonPath = getPythonPath();
    const scriptPath = getResourcePath('backend/server.py');
    const env = getEnv();

    log(`Platform: ${process.platform}`);
    log(`app.isPackaged: ${app.isPackaged}`);
    log(`process.resourcesPath: ${process.resourcesPath}`);
    log(`Python path: ${pythonPath}`);
    log(`Script path: ${scriptPath}`);
    log(`Script exists: ${fs.existsSync(scriptPath)}`);
    log(`Working directory: ${path.dirname(scriptPath)}`);

    // 列出 backend 目录内容
    const backendDir = path.dirname(scriptPath);
    if (fs.existsSync(backendDir)) {
        log(`Backend directory contents: ${fs.readdirSync(backendDir).join(', ')}`);
    } else {
        log('Backend directory does not exist!');
    }

    if (!fs.existsSync(scriptPath)) {
        log(`ERROR: Server script not found at: ${scriptPath}`);
        showBackendError(`后端脚本未找到: ${scriptPath}`);
        return;
    }

    // 先检查是否已有健康的后端在运行
    const alreadyHealthy = await checkBackendHealth();
    if (alreadyHealthy) {
        log('Backend already running and healthy, skipping restart');
        backendStartAttempts = 0;
        return;
    }

    // 清理旧进程
    await killExistingBackend();

    try {
        log(`Spawning: ${pythonPath} ${scriptPath}`);

        pythonProcess = spawn(pythonPath, [scriptPath], {
            stdio: ['pipe', 'pipe', 'pipe'],
            env: { ...env, PYTHONUNBUFFERED: '1' },
            cwd: backendDir,
            detached: false
        });

        pythonProcess.stdout.on('data', (data) => {
            log(`Python stdout: ${data.toString().trim()}`);
        });

        pythonProcess.stderr.on('data', (data) => {
            log(`Python stderr: ${data.toString().trim()}`);
        });

        pythonProcess.on('error', (err) => {
            log(`Python spawn error: ${err.message}`);
            handleBackendFailure(`无法启动 Python: ${err.message}`);
        });

        pythonProcess.on('close', (code, signal) => {
            log(`Python exited with code ${code}, signal ${signal}`);
            pythonProcess = null;

            if (!isQuitting && appIsReady) {
                // 非正常退出，尝试重启
                if (backendStartAttempts < MAX_BACKEND_ATTEMPTS) {
                    log('Backend crashed, restarting in 2 seconds...');
                    setTimeout(() => {
                        if (!isQuitting && appIsReady) {
                            startPythonBackend();
                        }
                    }, 2000);
                } else {
                    handleBackendFailure('后端多次启动失败，请检查 Python 环境');
                }
            }
        });

        // 等待后端启动并检查健康状态
        const isHealthy = await waitForBackend(10);
        if (isHealthy) {
            log('Backend started successfully!');
            backendStartAttempts = 0; // 重置计数器
        } else if (pythonProcess && !pythonProcess.killed) {
            log('Backend not responding, process still running...');
        }

    } catch (err) {
        log(`Spawn exception: ${err.message}`);
        handleBackendFailure(`启动后端异常: ${err.message}`);
    }
}

// 处理后端启动失败
function handleBackendFailure(message) {
    if (backendStartAttempts >= MAX_BACKEND_ATTEMPTS) {
        showBackendError(message);
    }
}

// 显示后端错误对话框
function showBackendError(message) {
    if (mainWindow) {
        dialog.showMessageBox(mainWindow, {
            type: 'error',
            title: '后端启动失败',
            message: '无法启动 Python 后端服务',
            detail: `${message}\n\n请确保已安装 Python 3 并配置好环境。\n\n日志位置: ${logDir}`,
            buttons: ['确定']
        });
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

// IPC 处理
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
    log('=== App Ready ===');

    // 防止 macOS App Nap
    if (process.platform === 'darwin') {
        powerSaveId = powerSaveBlocker.start('prevent-app-suspension');
        log(`PowerSaveBlocker started: ${powerSaveId}`);
    }

    // 启动后端
    await startPythonBackend();

    // 创建窗口（稍微延迟以确保后端有时间启动）
    setTimeout(() => createWindow(), 1000);
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
        log('Killing Python process...');
        try {
            pythonProcess.kill('SIGTERM');
            setTimeout(() => {
                if (pythonProcess && !pythonProcess.killed) {
                    pythonProcess.kill('SIGKILL');
                }
            }, 1000);
        } catch (e) {
            log(`Error killing Python: ${e.message}`);
        }
        pythonProcess = null;
    }
}
