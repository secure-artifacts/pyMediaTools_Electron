/**
 * yt-dlp 视频下载服务
 * 使用 yt-dlp 命令行工具（不需要 Python yt_dlp 库）
 */
const { spawn } = require('child_process');
const path = require('path');
const os = require('os');
const fs = require('fs');

/**
 * 查找 yt-dlp 可执行文件路径
 */
function resolveYtDlp() {
    // 1. 环境变量优先
    if (process.env.YTDLP_PATH && fs.existsSync(process.env.YTDLP_PATH)) {
        return process.env.YTDLP_PATH;
    }

    // 2. 常见安装路径探测
    const candidates = process.platform === 'darwin'
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

    for (const p of candidates) {
        if (p && fs.existsSync(p)) return p;
    }

    // 3. 回退到 PATH 解析
    return 'yt-dlp';
}

function runYtDlp(args, timeout = 600000) {
    const ytdlpCmd = resolveYtDlp();
    return new Promise((resolve, reject) => {
        let stdout = '';
        let stderr = '';
        const proc = spawn(ytdlpCmd, args, { timeout, env: process.env });
        proc.stdout.on('data', d => stdout += d.toString());
        proc.stderr.on('data', d => stderr += d.toString());
        proc.on('close', code => {
            if (code === 0) resolve({ stdout, stderr });
            else reject(new Error(`yt-dlp 错误 (code ${code}): ${stderr.slice(0, 500)}`));
        });
        proc.on('error', e => {
            const installHint = process.platform === 'darwin'
                ? 'brew install yt-dlp'
                : process.platform === 'win32'
                    ? 'winget install yt-dlp'
                    : 'sudo apt install yt-dlp';
            reject(new Error(`yt-dlp 未安装或无法找到 (${ytdlpCmd})。请安装: ${installHint}`));
        });
    });
}

/** 分析视频链接 */
async function analyzeVideo(url) {
    const { stdout } = await runYtDlp([
        '--dump-json', '--flat-playlist', '--no-warnings', '--quiet', url
    ], 30000);

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
