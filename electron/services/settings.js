/**
 * 设置和文件管理服务
 * 替代 Python server.py 中的设置存储和文件操作
 */
const fs = require('fs');
const path = require('path');
const os = require('os');
const { exec } = require('child_process');
const archiver = require('archiver');

// ==================== JSON 设置文件管理 ====================

/**
 * 获取可写的 backend 数据目录
 * - 开发模式：项目根目录下的 backend/
 * - 打包模式：用户数据目录下的 backend/（可写）
 */
function getBackendDir() {
    let isPackaged = false;
    try {
        const { app } = require('electron');
        isPackaged = app.isPackaged;
    } catch { }

    if (isPackaged) {
        const { app } = require('electron');
        const userDataBackend = path.join(app.getPath('userData'), 'backend');
        // 首次运行时，从打包资源中复制种子 JSON 文件
        if (!fs.existsSync(userDataBackend)) {
            fs.mkdirSync(userDataBackend, { recursive: true });
            // 尝试从 extraResources 复制初始配置
            const resourceBackend = path.join(process.resourcesPath, 'backend');
            if (fs.existsSync(resourceBackend)) {
                const jsonFiles = fs.readdirSync(resourceBackend).filter(f => f.endsWith('.json'));
                for (const f of jsonFiles) {
                    try {
                        fs.copyFileSync(path.join(resourceBackend, f), path.join(userDataBackend, f));
                    } catch { }
                }
            }
        }
        return userDataBackend;
    }
    return path.join(__dirname, '..', '..', 'backend');
}

function readJSON(filePath) {
    if (!fs.existsSync(filePath)) return null;
    try { return JSON.parse(fs.readFileSync(filePath, 'utf-8')); } catch { return null; }
}

function writeJSON(filePath, data) {
    fs.mkdirSync(path.dirname(filePath), { recursive: true });
    fs.writeFileSync(filePath, JSON.stringify(data, null, 2), 'utf-8');
}

// Gladia Keys
function getGladiaKeysPath() { return path.join(getBackendDir(), 'gladia_keys.json'); }
function loadGladiaKeys() { return readJSON(getGladiaKeysPath()) || { keys: [] }; }
function saveGladiaKeys(data) { writeJSON(getGladiaKeysPath(), data); }

// ElevenLabs Settings
function getElevenLabsSettingsPath() { return path.join(getBackendDir(), 'elevenlabs_settings.json'); }
function loadElevenLabsSettings() {
    const data = readJSON(getElevenLabsSettingsPath()) || {};
    let keys = data.api_keys || [];
    if (typeof keys === 'string') keys = [keys];
    if (keys.length === 0 && data.api_key) keys = [data.api_key];
    keys = keys.filter(k => typeof k === 'string' && k.trim());
    return {
        api_key: keys[0] || '',
        api_keys: keys,
    };
}
function saveElevenLabsSettings(data) {
    let keys = data.api_keys || [];
    if (typeof keys === 'string') keys = [keys];
    if (keys.length === 0 && data.api_key) keys = [data.api_key];
    keys = keys.filter(k => typeof k === 'string' && k.trim());
    const payload = { api_key: keys[0] || '', api_keys: keys };
    writeJSON(getElevenLabsSettingsPath(), payload);
    return payload;
}

// ElevenLabs Keys with Status
function loadElevenLabsKeysWithStatus() {
    const data = readJSON(getElevenLabsSettingsPath()) || {};
    const kws = data.keys_with_status || [];
    if (kws.length > 0) return kws;
    return (data.api_keys || []).map(k => ({ key: k, enabled: true }));
}
function saveElevenLabsKeysWithStatus(keysData) {
    const data = readJSON(getElevenLabsSettingsPath()) || {};
    data.keys_with_status = keysData;
    writeJSON(getElevenLabsSettingsPath(), data);
}

// Replace Rules
function getReplaceRulesPath() { return path.join(getBackendDir(), 'replace_rules.json'); }
function loadReplaceRules() { return readJSON(getReplaceRulesPath()) || { rules: [] }; }
function saveReplaceRules(data) { writeJSON(getReplaceRulesPath(), data); }

// ==================== 文件操作 ====================

function openFolder(folderPath) {
    let expandedPath = folderPath;
    if (expandedPath === '~') {
        expandedPath = os.homedir();
    } else if (expandedPath.startsWith('~/') || expandedPath.startsWith('~\\')) {
        expandedPath = path.join(os.homedir(), expandedPath.slice(2));
    }

    if (!fs.existsSync(expandedPath)) {
        // 尝试创建目录
        try {
            fs.mkdirSync(expandedPath, { recursive: true });
        } catch {
            throw new Error(`路径不存在且创建失败: ${expandedPath}`);
        }
    }

    const platform = process.platform;
    let cmd;
    if (platform === 'darwin') {
        cmd = `open "${expandedPath}"`;
    } else if (platform === 'win32') {
        cmd = `explorer "${expandedPath}"`;
    } else {
        cmd = `xdg-open "${expandedPath}"`;
    }

    return new Promise((resolve, reject) => {
        exec(cmd, (err) => {
            if (err) reject(new Error(`打开文件夹失败: ${err.message}`));
            else resolve({ message: '已打开' });
        });
    });
}

// Upload 目录
const UPLOAD_DIR = path.join(getBackendDir(), 'uploads');
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR, { recursive: true });

function uploadFile(buffer, filename) {
    const destPath = path.join(UPLOAD_DIR, filename);
    fs.writeFileSync(destPath, buffer);
    return { path: destPath };
}

function createZip(files, outputPath) {
    return new Promise((resolve, reject) => {
        const output = fs.createWriteStream(outputPath);
        const archive = archiver('zip', { zlib: { level: 6 } });
        output.on('close', () => resolve(outputPath));
        archive.on('error', reject);
        archive.pipe(output);
        for (const f of files) {
            if (fs.existsSync(f)) {
                archive.file(f, { name: path.basename(f) });
            }
        }
        archive.finalize();
    });
}

// ==================== 语言配置 ====================

const LANGUAGES = {
    en: { name: 'en', display: '英文', gladia: 'english' },
    ja: { name: 'ja', display: '日文', gladia: 'japanese' },
    ko: { name: 'ko', display: '韩文', gladia: 'korean' },
    es: { name: 'es', display: '西班牙文', gladia: 'spanish' },
    fr: { name: 'fr', display: '法文', gladia: 'french' },
    de: { name: 'de', display: '德文', gladia: 'german' },
    pt: { name: 'pt', display: '葡萄牙文', gladia: 'portuguese' },
    it: { name: 'it', display: '意大利文', gladia: 'italian' },
    zh: { name: 'zh', display: '中文', gladia: 'chinese' },
    ar: { name: 'ar', display: '阿拉伯文', gladia: 'arabic' },
    ru: { name: 'ru', display: '俄文', gladia: 'russian' },
};

function getLanguages() {
    return Object.entries(LANGUAGES).map(([code, lang]) => ({
        code,
        name: lang.name,
        display: lang.display,
    }));
}

module.exports = {
    getBackendDir,
    readJSON,
    writeJSON,
    loadGladiaKeys,
    saveGladiaKeys,
    loadElevenLabsSettings,
    saveElevenLabsSettings,
    loadElevenLabsKeysWithStatus,
    saveElevenLabsKeysWithStatus,
    loadReplaceRules,
    saveReplaceRules,
    openFolder,
    uploadFile,
    createZip,
    getLanguages,
    LANGUAGES,
    UPLOAD_DIR,
};
