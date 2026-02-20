/**
 * yt-dlp 视频下载服务
 * 使用 yt-dlp 命令行工具（不需要 Python yt_dlp 库）
 */
const { spawn } = require('child_process');
const path = require('path');
const os = require('os');
const fs = require('fs');

// Electron app 可能尚未初始化，安全获取 resourcesPath
function getVendorBase() {
    try {
        const { app } = require('electron');
        if (app && app.isPackaged) {
            return path.join(process.resourcesPath, 'vendor');
        }
    } catch (_) { /* 非 Electron 或未初始化 */ }
    return path.join(__dirname, '..', '..', 'vendor');
}

/**
 * 查找 yt-dlp 可执行文件路径
 * 返回 { cmd, args } — cmd 是可执行文件, args 是前置参数
 *   - 独立二进制: { cmd: '/usr/local/bin/yt-dlp', args: [] }
 *   - Python 模块: { cmd: 'python.exe', args: ['-m', 'yt_dlp'] }
 */
function resolveYtDlp() {
    // 1. 环境变量优先
    if (process.env.YTDLP_PATH && fs.existsSync(process.env.YTDLP_PATH)) {
        return { cmd: process.env.YTDLP_PATH, args: [] };
    }

    const vendorBase = getVendorBase();

    // 2. vendor Python + yt_dlp 模块（最可靠：避免 shebang 硬编码路径问题）
    if (process.platform === 'win32') {
        const pythonCandidates = [
            path.join(vendorBase, 'python', 'python.exe'),             // packaged
            path.join(vendorBase, 'windows', 'python', 'python.exe'),  // dev
        ];
        for (const pyExe of pythonCandidates) {
            if (fs.existsSync(pyExe)) {
                const sitePackages = path.join(path.dirname(pyExe), 'Lib', 'site-packages', 'yt_dlp');
                if (fs.existsSync(sitePackages)) {
                    console.log(`[yt-dlp] 使用 vendor Python 模块: ${pyExe} -m yt_dlp`);
                    return { cmd: pyExe, args: ['-m', 'yt_dlp'] };
                }
            }
        }
    } else if (process.platform === 'darwin') {
        const pythonCandidates = [
            path.join(vendorBase, 'python', 'bin', 'python3'),             // packaged
            path.join(vendorBase, 'darwin', 'python', 'bin', 'python3'),   // dev
        ];
        for (const pyExe of pythonCandidates) {
            if (fs.existsSync(pyExe)) {
                const pyDir = path.dirname(path.dirname(pyExe));
                try {
                    const libDir = fs.readdirSync(path.join(pyDir, 'lib')).find(d => d.startsWith('python3'));
                    if (libDir) {
                        const sitePackages = path.join(pyDir, 'lib', libDir, 'site-packages', 'yt_dlp');
                        if (fs.existsSync(sitePackages)) {
                            console.log(`[yt-dlp] 使用 vendor Python 模块: ${pyExe} -m yt_dlp`);
                            return { cmd: pyExe, args: ['-m', 'yt_dlp'] };
                        }
                    }
                } catch (_) { /* lib 目录不存在 */ }
            }
        }
    }

    // 3. vendor 目录中的独立 yt-dlp 二进制（回退）
    const vendorBinaryCandidates = process.platform === 'darwin'
        ? [
            path.join(vendorBase, 'python', 'bin', 'yt-dlp'),            // packaged
            path.join(vendorBase, 'darwin', 'python', 'bin', 'yt-dlp'),  // dev
        ]
        : process.platform === 'win32'
            ? [
                path.join(vendorBase, 'python', 'Scripts', 'yt-dlp.exe'),             // packaged
                path.join(vendorBase, 'windows', 'python', 'Scripts', 'yt-dlp.exe'),  // dev
            ]
            : [];

    for (const p of vendorBinaryCandidates) {
        if (fs.existsSync(p)) return { cmd: p, args: [] };
    }

    // 4. 系统常见安装路径探测
    const systemCandidates = process.platform === 'darwin'
        ? [
            '/opt/homebrew/bin/yt-dlp',
            '/usr/local/bin/yt-dlp',
            '/opt/local/bin/yt-dlp',
            path.join(os.homedir(), '.local', 'bin', 'yt-dlp'),
        ]
        : process.platform === 'win32'
            ? [
                path.join(process.env.LOCALAPPDATA || '', 'Microsoft', 'WinGet', 'Links', 'yt-dlp.exe'),
                path.join(process.env.USERPROFILE || '', 'scoop', 'shims', 'yt-dlp.exe'),
                'C:\\ProgramData\\chocolatey\\bin\\yt-dlp.exe',
            ]
            : ['/usr/bin/yt-dlp', '/usr/local/bin/yt-dlp'];

    for (const p of systemCandidates) {
        if (p && fs.existsSync(p)) return { cmd: p, args: [] };
    }

    // 5. 回退到 PATH 解析
    const fallback = process.platform === 'win32' ? 'yt-dlp.exe' : 'yt-dlp';
    return { cmd: fallback, args: [] };
}

function runYtDlp(userArgs, timeout = 600000) {
    const { cmd: ytdlpCmd, args: prefixArgs } = resolveYtDlp();
    const finalArgs = [...prefixArgs, ...userArgs];
    console.log(`[yt-dlp] 执行: ${ytdlpCmd} ${finalArgs.join(' ')}`);

    return new Promise((resolve, reject) => {
        let stdout = '';
        let stderr = '';
        let killed = false;
        const proc = spawn(ytdlpCmd, finalArgs, { timeout, env: process.env });
        proc.stdout.on('data', d => stdout += d.toString());
        proc.stderr.on('data', d => stderr += d.toString());
        proc.on('close', (code, signal) => {
            if (code === 0) {
                resolve({ stdout, stderr });
            } else if (code === null && signal) {
                reject(new Error(`yt-dlp 超时或被终止 (signal: ${signal}, timeout: ${timeout / 1000}s)。${stderr ? ' 详情: ' + stderr.slice(0, 300) : ''}`));
            } else {
                reject(new Error(`yt-dlp 错误 (code ${code}): ${stderr.slice(0, 500) || '(无详细错误信息)'}`));
            }
        });
        proc.on('error', e => {
            const installHint = process.platform === 'darwin'
                ? 'brew install yt-dlp'
                : process.platform === 'win32'
                    ? '请从 https://github.com/yt-dlp/yt-dlp/releases 下载 yt-dlp.exe 放入 vendor/windows/python/ 目录'
                    : 'sudo apt install yt-dlp';
            reject(new Error(`yt-dlp 未安装或无法找到 (${ytdlpCmd})。${installHint}`));
        });
    });
}


/** 分析视频链接 */
async function analyzeVideo(url) {
    const { stdout } = await runYtDlp([
        '--dump-json', '--flat-playlist', '--no-warnings', url
    ], 120000);

    // yt-dlp --flat-playlist 输出多行 JSON（播放列表）
    const lines = stdout.trim().split('\n').filter(l => l.trim());

    if (lines.length > 1) {
        // 播放列表
        const entries = lines.map(line => {
            try {
                const info = JSON.parse(line);
                return {
                    title: info.title || 'Unknown',
                    url: info.url || info.webpage_url || info.original_url,
                    webpage_url: info.webpage_url || info.url,
                    duration: info.duration || 0,
                    thumbnail: info.thumbnail || '',
                };
            } catch { return null; }
        }).filter(Boolean);
        return { title: '播放列表', entries };
    } else if (lines.length === 1) {
        const info = JSON.parse(lines[0]);
        if (info.entries) {
            // 播放列表 JSON
            return {
                title: info.title || '播放列表',
                entries: (info.entries || []).map(e => ({
                    title: e.title || 'Unknown',
                    url: e.url || e.webpage_url,
                    webpage_url: e.webpage_url || e.url,
                    duration: e.duration || 0,
                    thumbnail: e.thumbnail || '',
                })),
            };
        }
        // 单个视频
        return {
            title: info.title || '未知',
            url: info.webpage_url || url,
            webpage_url: info.webpage_url || url,
            duration: info.duration || 0,
            thumbnail: info.thumbnail || '',
        };
    }

    throw new Error('无法解析视频信息');
}

/** 下载单个视频 */
async function downloadVideo(url, options = {}) {
    const {
        quality = 'best',
        outputDir = path.join(os.homedir(), 'Downloads'),
        downloadSubtitle = false,
    } = options;

    fs.mkdirSync(outputDir, { recursive: true });

    const args = [
        '-o', path.join(outputDir, '%(title)s.%(ext)s'),
        '--no-warnings', '--quiet',
    ];

    switch (quality) {
        case '1080p': args.push('-f', 'bestvideo[height<=1080]+bestaudio/best'); break;
        case '720p': args.push('-f', 'bestvideo[height<=720]+bestaudio/best'); break;
        case '480p': args.push('-f', 'bestvideo[height<=480]+bestaudio/best'); break;
        default: args.push('-f', 'bestvideo+bestaudio/best');
    }

    if (downloadSubtitle) {
        args.push('--write-subs', '--sub-langs', 'en,zh-Hans');
    }

    args.push(url);
    await runYtDlp(args);

    return { message: '下载完成', output_path: outputDir };
}

/** 批量下载 */
async function downloadBatch(items, options = {}) {
    const {
        outputDir = path.join(os.homedir(), 'Downloads'),
        audioOnly = false,
        ext = 'mp4',
        quality = 'best',
        subtitles = false,
        subLang = 'en',
    } = options;

    fs.mkdirSync(outputDir, { recursive: true });

    const args = [
        '-o', path.join(outputDir, '%(title)s.%(ext)s'),
        '--no-warnings', '--quiet', '--ignore-errors',
    ];

    if (audioOnly) {
        args.push('-f', 'bestaudio/best', '-x', '--audio-format', ext, '--audio-quality', '192K');
    } else {
        if (quality === 'best') {
            args.push('-f', 'bestvideo+bestaudio/best');
        } else {
            const h = parseInt(String(quality).replace('p', '')) || 1080;
            args.push('-f', `bestvideo[height<=${h}]+bestaudio/best[height<=${h}]`);
        }

        if (['mp4', 'mkv', 'webm', 'mov'].includes(ext)) {
            args.push('--merge-output-format', ext);
        }
    }

    if (subtitles) {
        args.push('--write-subs', '--sub-langs', `${subLang},en,zh-Hans`);
    }

    const urls = items.map(i => i.url || i).filter(Boolean);
    args.push(...urls);

    await runYtDlp(args, 3600000); // 1小时超时

    return { message: `成功下载 ${urls.length} 个视频`, output_path: outputDir, count: urls.length };
}

module.exports = {
    analyzeVideo,
    downloadVideo,
    downloadBatch,
};
