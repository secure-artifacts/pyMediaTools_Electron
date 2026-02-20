/**
 * Wav2Lip 口型同步服务
 * 通过 Python subprocess 调用 Wav2Lip 推理
 */
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');

// Wav2Lip 目录
function getWav2LipDir() {
    // 打包环境
    if (process.resourcesPath && fs.existsSync(path.join(process.resourcesPath, 'vendor', 'wav2lip'))) {
        return path.join(process.resourcesPath, 'vendor', 'wav2lip');
    }
    // 开发环境
    return path.join(__dirname, '..', '..', 'vendor', 'wav2lip');
}

/**
 * 查找 Python3 可执行文件
 */
function findPython() {
    if (process.env.PYTHON_PATH && fs.existsSync(process.env.PYTHON_PATH)) {
        return process.env.PYTHON_PATH;
    }

    const candidates = process.platform === 'darwin'
        ? [
            '/opt/homebrew/bin/python3',
            '/usr/local/bin/python3',
            '/usr/bin/python3',
            path.join(os.homedir(), '.pyenv', 'shims', 'python3'),
        ]
        : process.platform === 'win32'
            ? ['python', 'python3', 'py']
            : ['/usr/bin/python3', '/usr/local/bin/python3', 'python3'];

    for (const p of candidates) {
        try {
            if (fs.existsSync(p)) return p;
        } catch { /* ignore */ }
    }
    return 'python3';
}

/**
 * 运行 Wav2Lip Python 脚本
 * @param {string[]} args - 命令行参数
 * @param {function} onProgress - 进度回调 (percent, message)
 * @returns {Promise<object>} 结果数据
 */
function runWav2Lip(args, onProgress) {
    const pythonCmd = findPython();
    const scriptPath = path.join(getWav2LipDir(), 'inference_api.py');

    if (!fs.existsSync(scriptPath)) {
        return Promise.reject(new Error('Wav2Lip 推理脚本不存在'));
    }

    return new Promise((resolve, reject) => {
        const proc = spawn(pythonCmd, [scriptPath, ...args], {
            env: {
                ...process.env,
                PYTHONUNBUFFERED: '1',
            },
            cwd: getWav2LipDir(),
            timeout: 600000, // 10分钟
        });

        let result = null;
        let errorMsg = '';

        proc.stdout.on('data', data => {
            const lines = data.toString().split('\n').filter(l => l.trim());
            for (const line of lines) {
                try {
                    const msg = JSON.parse(line);
                    if (msg.type === 'progress' && onProgress) {
                        onProgress(msg.percent, msg.message);
                    } else if (msg.type === 'result') {
                        result = msg;
                        delete result.type;
                    } else if (msg.type === 'error') {
                        errorMsg = msg.message;
                    }
                } catch {
                    // 非 JSON 输出，忽略
                    console.log('[Wav2Lip stdout]', line);
                }
            }
        });

        proc.stderr.on('data', data => {
            const text = data.toString().trim();
            if (text) {
                // 过滤 Python warnings
                if (!text.includes('UserWarning') && !text.includes('FutureWarning')) {
                    console.warn('[Wav2Lip stderr]', text);
                }
            }
        });

        proc.on('close', code => {
            if (errorMsg) {
                reject(new Error(errorMsg));
            } else if (code !== 0) {
                reject(new Error(`Wav2Lip 进程异常退出 (code ${code})`));
            } else if (result) {
                resolve(result);
            } else {
                reject(new Error('Wav2Lip 未返回结果'));
            }
        });

        proc.on('error', err => {
            reject(new Error(`无法启动 Python: ${err.message}。请确保已安装 Python3 和 PyTorch。`));
        });
    });
}

/**
 * 检查 Wav2Lip 环境
 */
async function checkEnvironment() {
    try {
        const result = await runWav2Lip(['check']);
        return {
            available: true,
            ...result,
        };
    } catch (e) {
        return {
            available: false,
            error: e.message,
            python_path: findPython(),
            wav2lip_dir: getWav2LipDir(),
        };
    }
}

/**
 * 执行口型同步
 * @param {object} options
 * @param {string} options.facePath - 视频/图片路径
 * @param {string} options.audioPath - 音频路径
 * @param {string} options.outputPath - 输出路径
 * @param {number[]} options.pads - 人脸 padding [top, bottom, left, right]
 * @param {number} options.resizeFactor - 缩放因子
 * @param {function} options.onProgress - 进度回调
 */
async function lipSync(options) {
    const {
        facePath,
        audioPath,
        outputPath,
        pads = [0, 10, 0, 0],
        resizeFactor = 1,
        batchSize = 32,
        onProgress,
    } = options;

    if (!facePath || !fs.existsSync(facePath)) {
        throw new Error(`视频文件不存在: ${facePath}`);
    }
    if (!audioPath || !fs.existsSync(audioPath)) {
        throw new Error(`音频文件不存在: ${audioPath}`);
    }

    // 生成默认输出路径
    let outPath = outputPath;
    if (!outPath) {
        const dir = path.dirname(facePath);
        const ext = path.extname(facePath);
        const base = path.basename(facePath, ext);
        outPath = path.join(dir, `${base}_lipsync.mp4`);
    }

    const args = [
        'run',
        '--face', facePath,
        '--audio', audioPath,
        '--output', outPath,
        '--pads', ...pads.map(String),
        '--resize_factor', String(resizeFactor),
        '--wav2lip_batch_size', String(batchSize),
    ];

    const result = await runWav2Lip(args, onProgress);
    return {
        ...result,
        output_path: outPath,
    };
}

module.exports = {
    checkEnvironment,
    lipSync,
    findPython,
    getWav2LipDir,
};
