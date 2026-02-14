/**
 * ElevenLabs TTS 服务
 * 替代 Python server.py 中所有 ElevenLabs 相关的 API 调用
 */
const fs = require('fs');
const path = require('path');
const https = require('https');

const API_BASE = 'https://api.elevenlabs.io/v1';

// ==================== HTTP 请求封装 ====================

function request(method, urlPath, apiKey, body = null, timeout = 15000) {
    return new Promise((resolve, reject) => {
        const url = new URL(API_BASE + urlPath);
        const options = {
            hostname: url.hostname,
            port: 443,
            path: url.pathname + url.search,
            method,
            headers: {
                'xi-api-key': apiKey,
                'Accept': 'application/json',
            },
            timeout,
        };
        if (body && method !== 'GET') {
            const jsonBody = JSON.stringify(body);
            options.headers['Content-Type'] = 'application/json';
            options.headers['Content-Length'] = Buffer.byteLength(jsonBody);
        }

        const req = https.request(options, (res) => {
            const chunks = [];
            res.on('data', c => chunks.push(c));
            res.on('end', () => {
                const raw = Buffer.concat(chunks);
                resolve({ status: res.statusCode, body: raw, headers: res.headers });
            });
        });
        req.on('timeout', () => { req.destroy(); reject(new Error('请求超时')); });
        req.on('error', reject);
        if (body && method !== 'GET') req.write(JSON.stringify(body));
        req.end();
    });
}

function parseJSON(buf) {
    try { return JSON.parse(buf.toString()); } catch { return null; }
}

// ==================== 设置管理 ====================

function getSettingsPath() {
    const { getBackendDir } = require('./settings');
    return path.join(getBackendDir(), 'elevenlabs_settings.json');
}

function loadSettings() {
    const p = getSettingsPath();
    if (!fs.existsSync(p)) return {};
    try { return JSON.parse(fs.readFileSync(p, 'utf-8')); } catch { return {}; }
}

function saveSettings(data) {
    const p = getSettingsPath();
    fs.mkdirSync(path.dirname(p), { recursive: true });
    fs.writeFileSync(p, JSON.stringify(data, null, 2));
}

/** 加载 API Keys，返回 key 字符串数组（启用的） */
function loadKeys(includeDisabled = false) {
    const data = loadSettings();
    const keysWithStatus = data.keys_with_status || [];

    if (keysWithStatus.length > 0) {
        if (includeDisabled) return keysWithStatus;
        return keysWithStatus
            .filter(k => k.enabled !== false)
            .map(k => (typeof k === 'string') ? k : k.key)
            .filter(k => k && k.trim());
    }

    // 兼容旧格式
    let keys = data.api_keys || [];
    if (typeof keys === 'string') keys = [keys];
    if (keys.length === 0 && data.api_key) keys = [data.api_key];
    return keys.filter(k => k && k.trim());
}

function selectKey(keys, keyIndex) {
    if (keyIndex != null && keyIndex !== '' && !isNaN(keyIndex)) {
        const idx = parseInt(keyIndex);
        if (idx >= 0 && idx < keys.length) return keys[idx];
    }
    return keys[0] || null;
}

// ==================== 错误处理 ====================

function parseElevenLabsError(status, body) {
    const json = parseJSON(body);
    let message = body.toString().slice(0, 500);
    let detailStatus = '', detailCode = '';
    if (json && json.detail) {
        if (typeof json.detail === 'object') {
            detailStatus = String(json.detail.status || '');
            detailCode = String(json.detail.code || '');
            message = json.detail.message || message;
        } else if (typeof json.detail === 'string') {
            message = json.detail;
        }
    }
    return { message, detailStatus, detailCode, httpStatus: status };
}

function setKeyEnabled(apiKey, enabled, reason = '', source = 'auto') {
    const data = loadSettings();
    const kws = data.keys_with_status || [];
    let changed = false;
    for (const entry of kws) {
        if (typeof entry === 'object' && entry.key === apiKey) {
            // 手动停用状态下，自动恢复请求应被忽略
            if (source === 'auto' && enabled && entry.manual_disabled) {
                break;
            }

            if (entry.enabled !== enabled) {
                entry.enabled = enabled;
                changed = true;
            }

            if (source === 'manual') {
                const manualDisabled = !enabled;
                if (entry.manual_disabled !== manualDisabled) {
                    entry.manual_disabled = manualDisabled;
                    changed = true;
                }
                // 手动启用时，清除自动停用标志
                if (enabled) {
                    if (entry.auto_disabled) { entry.auto_disabled = false; changed = true; }
                    if (entry.auto_disabled_reason) { entry.auto_disabled_reason = ''; changed = true; }
                }
            } else {
                // auto source
                if (!enabled) {
                    if (!entry.auto_disabled) { entry.auto_disabled = true; changed = true; }
                    if (reason && entry.auto_disabled_reason !== reason) { entry.auto_disabled_reason = reason; changed = true; }
                } else {
                    if (entry.auto_disabled) { entry.auto_disabled = false; changed = true; }
                    if (entry.auto_disabled_reason) { entry.auto_disabled_reason = ''; changed = true; }
                }
            }
            break;
        }
    }
    if (changed) {
        data.keys_with_status = kws;
        saveSettings(data);
        const action = enabled ? '启用' : '停用';
        const keyPrefix = apiKey.length >= 12 ? `${apiKey.slice(0, 8)}...${apiKey.slice(-4)}` : apiKey;
        console.log(`[ElevenLabs] 已${source === 'manual' ? '手动' : '自动'}${action} Key ${keyPrefix}${reason ? '，原因: ' + reason : ''}`);
    }
}

/**
 * 错误分类 — 返回用户友好的中文描述 + 是否可轮换 + 是否自动停用
 */
function classifyError(errInfo) {
    const merged = `${errInfo.message} ${errInfo.detailStatus} ${errInfo.detailCode}`.toLowerCase();
    const status = errInfo.httpStatus;

    // --- 额度/余量相关 → 自动停用 + 轮换下一个 Key ---
    if (merged.includes('quota_exceeded') || merged.includes('exceeded your character limit') ||
        merged.includes('character_limit_exceeded') || merged.includes('insufficient characters') ||
        merged.includes('insufficient') || merged.includes('character_limit') || merged.includes('credit')) {
        return {
            category: 'quota', retryable: true, autoDisable: true,
            userMessage: '❌ 额度不足：该 Key 字符余量已用尽，自动切换下一个 Key'
        };
    }

    // --- Key 无效/过期 → 自动停用 + 轮换 ---
    if (merged.includes('invalid api key') || merged.includes('invalid_api_key') ||
        merged.includes('unauthorized') || status === 401) {
        return {
            category: 'auth', retryable: true, autoDisable: true,
            userMessage: '🔑 Key 无效：API Key 不正确或已过期，自动切换下一个 Key'
        };
    }

    // --- IP 异常活动 → 轮换但不停用（比 403/forbidden 更具体，需优先匹配） ---
    if (merged.includes('detected_unusual_activity') || merged.includes('unusual_activity') ||
        merged.includes('unusual activity')) {
        return {
            category: 'ip_blocked', retryable: true, autoDisable: false,
            userMessage: '🛡️ IP 受限：检测到异常活动，建议稍后再试或更换网络'
        };
    }

    // --- 权限/订阅问题 → 自动停用 + 轮换 ---
    if (merged.includes('forbidden') || merged.includes('payment required') ||
        merged.includes('subscription') || merged.includes('billing') ||
        merged.includes('account_suspended') || merged.includes('account_disabled') ||
        merged.includes('plan') || merged.includes('permission') ||
        merged.includes('not available for your') || merged.includes('not allowed') ||
        status === 403 || status === 402) {
        return {
            category: 'permission', retryable: true, autoDisable: true,
            userMessage: '🚫 权限不足：当前 Key 的订阅计划不支持此功能，自动切换下一个 Key'
        };
    }

    // --- 请求频率限制 → 轮换但不停用（临时的） ---
    if (merged.includes('rate limit') || merged.includes('too many requests') || status === 429) {
        return {
            category: 'rate_limit', retryable: true, autoDisable: false,
            userMessage: '⏳ 请求过快：该 Key 触发频率限制，自动切换下一个 Key'
        };
    }

    // --- 音色不存在/不可用 → 不轮换（换 Key 也没用） ---
    if (merged.includes('voice_not_found') || merged.includes('voice not found') ||
        merged.includes('you do not have access to this voice') || merged.includes('does not have access')) {
        return {
            category: 'voice_error', retryable: false, autoDisable: false,
            userMessage: '🎤 音色错误：所选音色不存在或无权使用，请更换音色后重试'
        };
    }

    // --- 音色数量限制 → 特殊处理（自动删除旧音色） ---
    if (merged.includes('maximum amount of custom voices') || merged.includes('voice_limit') ||
        errInfo.detailStatus === 'voice_limit_reached') {
        return {
            category: 'voice_limit', retryable: false, autoDisable: false,
            userMessage: '📦 音色已满：自定义音色数量已达上限，正在尝试自动清理'
        };
    }

    // --- 模型不支持 → 不轮换 ---
    if (merged.includes('model_not_available') || merged.includes('model_not_supported') ||
        merged.includes('unsupported model') || merged.includes('feature_not_available')) {
        return {
            category: 'model_error', retryable: false, autoDisable: false,
            userMessage: '⚙️ 模型不可用：当前 Key 不支持所选模型，请更换模型'
        };
    }

    // --- 文本问题 → 不轮换 ---
    if (merged.includes('text is too long') || merged.includes('text_too_long') ||
        merged.includes('empty text') || status === 422) {
        return {
            category: 'input_error', retryable: false, autoDisable: false,
            userMessage: '📝 输入错误：文本内容不满足要求（太长、太短或格式错误）'
        };
    }

    // --- 服务器错误 → 可重试 ---
    if (status >= 500) {
        return {
            category: 'server_error', retryable: true, autoDisable: false,
            userMessage: '💥 服务器错误：ElevenLabs 服务暂时不可用，正在重试'
        };
    }

    // --- 未知错误 → 不轮换 ---
    return {
        category: 'unknown', retryable: false, autoDisable: false,
        userMessage: `❓ 未知错误 [${status}]: ${errInfo.message.slice(0, 100)}`
    };
}

// ==================== TTS 核心 ====================

async function requestTTS(apiKey, voiceId, text, modelId, stability, outputFormat, autoDeleteOnLimit = true) {
    const payload = {
        text,
        model_id: modelId,
        voice_settings: { stability, similarity_boost: 0.75 },
    };

    async function doRequest() {
        return await request('POST', `/text-to-speech/${voiceId}?output_format=${outputFormat}`, apiKey, payload, 60000);
    }

    let res = await doRequest();

    if (res.status !== 200) {
        const errInfo = parseElevenLabsError(res.status, res.body);
        const classified = classifyError(errInfo);

        // 音色数量限制 → 尝试自动删除
        if (classified.category === 'voice_limit' && autoDeleteOnLimit) {
            console.log('[TTS自动删除] 检测到音色数量限制，尝试删除最旧的音色...');
            const deleted = await deleteOldestCustomVoice(apiKey);
            if (deleted) {
                console.log(`[TTS自动删除] 已删除音色: ${deleted.name}`);
                await new Promise(r => setTimeout(r, 1000));
                const retryRes = await doRequest();
                if (retryRes.status === 200) return retryRes.body;
                const retryInfo = parseElevenLabsError(retryRes.status, retryRes.body);
                const retryClassified = classifyError(retryInfo);
                throw new Error(`${retryClassified.userMessage}（已自动删除音色「${deleted.name}」但仍失败）`);
            }
        }

        // 构建包含分类信息的错误，供 requestTTSWithRotation 读取
        const error = new Error(classified.userMessage);
        error.classified = classified;
        error.errInfo = errInfo;
        throw error;
    }

    return res.body;
}

async function requestTTSWithRotation(keys, voiceId, text, modelId, stability, outputFormat, keyIndex = null) {
    if (!keys || keys.length === 0) throw new Error('❌ 未配置 API Key，请先添加 ElevenLabs API Key');

    const preferred = keyIndex != null ? selectKey(keys, keyIndex) : null;
    const keysToTry = preferred ? [preferred, ...keys.filter(k => k !== preferred)] : [...keys];

    let lastErr = null;
    for (let i = 0; i < keysToTry.length; i++) {
        const apiKey = keysToTry[i];
        const keyLabel = `Key${i + 1}(${apiKey.slice(0, 6)}...)`;
        try {
            const audio = await requestTTS(apiKey, voiceId, text, modelId, stability, outputFormat);
            return { audio, usedKey: apiKey };
        } catch (e) {
            lastErr = e;
            const classified = e.classified || classifyError({ message: e.message, httpStatus: 0, detailStatus: '', detailCode: '' });

            // 不可轮换的错误（音色/模型/输入问题）→ 直接抛出，不试其他 Key
            if (!classified.retryable) {
                throw new Error(classified.userMessage);
            }

            // 可轮换 → 日志 + 自动停用
            const hasNext = i < keysToTry.length - 1;
            console.log(`[ElevenLabs] ${keyLabel} 失败: ${classified.userMessage}${hasNext ? '，切换下一个 Key...' : ''}`);

            if (classified.autoDisable) {
                try { setKeyEnabled(apiKey, false, classified.category); } catch { }
            }
        }
    }
    const finalClassified = lastErr?.classified;
    const summary = finalClassified ? finalClassified.userMessage : (lastErr?.message || '未知错误');
    throw new Error(`所有 Key 均尝试失败 (共 ${keysToTry.length} 个)。最后错误: ${summary}`);
}

// ==================== 音色管理 ====================

async function getVoices(apiKey) {
    const res = await request('GET', '/voices', apiKey);
    if (res.status !== 200) throw new Error(`获取音色列表失败: ${res.status}`);
    const data = parseJSON(res.body) || {};
    const voices = (data.voices || []).map(v => {
        const category = v.category || 'premade';
        const canDelete = ['cloned', 'generated', 'professional'].includes(category);
        const prefixMap = { cloned: '[克隆]', generated: '[生成]', professional: '[专业]' };
        const prefix = prefixMap[category] || '[官方]';
        return {
            voice_id: v.voice_id,
            name: `${prefix} ${v.name}`,
            preview_url: v.preview_url || '',
            can_delete: canDelete,
            category,
            created_at: v.created_date || '',
        };
    });
    return voices;
}

async function searchVoices(apiKey, searchTerm) {
    const res = await request('GET', `/shared-voices?search=${encodeURIComponent(searchTerm)}&page_size=50`, apiKey);
    if (res.status !== 200) throw new Error(`搜索失败: ${res.status}`);
    const data = parseJSON(res.body) || {};
    return (data.voices || []).map(v => ({
        voice_id: v.voice_id || v.public_owner_id,
        name: v.name,
        preview_url: v.preview_url || '',
        public_owner_id: v.public_owner_id || v.voice_id,
    }));
}

async function addVoice(apiKey, publicVoiceId, name, autoDelete = true) {
    async function tryAdd() {
        return await request('POST', `/voices/add/${publicVoiceId}`, apiKey, { new_name: name });
    }

    let res = await tryAdd();
    if (res.status === 200) {
        const data = parseJSON(res.body) || {};
        return { success: true, voice_id: data.voice_id || publicVoiceId, name };
    }

    // 检测限制错误
    const bodyStr = res.body.toString().toLowerCase();
    const isLimit = bodyStr.includes('voice_limit') || bodyStr.includes('maximum amount of custom voices');

    if (isLimit && autoDelete) {
        const deleted = await deleteOldestCustomVoice(apiKey);
        if (deleted) {
            await new Promise(r => setTimeout(r, 1000));
            const retryRes = await tryAdd();
            if (retryRes.status === 200) {
                const data = parseJSON(retryRes.body) || {};
                return {
                    success: true, voice_id: data.voice_id || publicVoiceId, name,
                    auto_deleted: deleted.name,
                    message: `已自动删除旧音色「${deleted.name}」并成功添加新音色`
                };
            }
        }
        throw new Error('voice_limit_reached: 自动删除后仍然添加失败');
    }

    if (isLimit) throw new Error('voice_limit_reached');
    throw new Error(`添加失败: ${res.body.toString().slice(0, 300)}`);
}

async function deleteVoice(apiKey, voiceId) {
    const res = await request('DELETE', `/voices/${voiceId}`, apiKey);
    if (res.status === 200) return { success: true, voice_id: voiceId };
    throw new Error(`删除失败: ${res.body.toString().slice(0, 300)}`);
}

async function deleteOldestCustomVoice(apiKey) {
    const voices = await getVoices(apiKey);
    const customVoices = voices.filter(v => v.can_delete);
    if (customVoices.length === 0) return null;

    // 按创建时间排序，删除最旧的
    customVoices.sort((a, b) => (a.created_at || '').localeCompare(b.created_at || ''));
    const oldest = customVoices[0];
    await deleteVoice(apiKey, oldest.voice_id);
    return { name: oldest.name, voice_id: oldest.voice_id };
}

async function getQuota(apiKey) {
    const res = await request('GET', '/user/subscription', apiKey);
    if (res.status !== 200) throw new Error(`API 错误: ${res.status}`);
    const data = parseJSON(res.body) || {};
    return {
        usage: data.character_count || 0,
        limit: data.character_limit || 0,
    };
}

async function getAllQuotas() {
    const keysData = loadKeys(true);
    if (!keysData || keysData.length === 0) return { keys: [], error: '未配置 API Key' };

    const results = [];
    let keysChanged = false;

    for (let i = 0; i < keysData.length; i++) {
        const entry = keysData[i];
        const key = typeof entry === 'string' ? entry : entry.key;
        let enabled = typeof entry === 'object' ? entry.enabled !== false : true;
        const manualDisabled = typeof entry === 'object' ? !!entry.manual_disabled : false;
        let autoDisabled = typeof entry === 'object' ? !!entry.auto_disabled : false;

        if (!key) continue;

        try {
            const quota = await getQuota(key);
            const remaining = quota.limit - quota.usage;

            // 自动停用余额不足 200 的 key
            if (remaining < 200 && enabled && !manualDisabled) {
                if (typeof entry === 'object') {
                    entry.enabled = false;
                    entry.auto_disabled = true;
                    entry.auto_disabled_reason = `remaining<200`;
                }
                keysChanged = true;
                enabled = false;
                autoDisabled = true;
            } else if (remaining >= 200 && !enabled && autoDisabled && !manualDisabled) {
                if (typeof entry === 'object') {
                    entry.enabled = true;
                    entry.auto_disabled = false;
                    entry.auto_disabled_reason = '';
                }
                keysChanged = true;
                enabled = true;
                autoDisabled = false;
            }

            results.push({
                index: i + 1,
                key_prefix: key.slice(0, 8) + '...' + key.slice(-4),
                usage: quota.usage, limit: quota.limit,
                remaining, percent: quota.limit > 0 ? Math.round(quota.usage / quota.limit * 1000) / 10 : 0,
                enabled, manual_disabled: manualDisabled, auto_disabled: autoDisabled,
            });
        } catch (e) {
            results.push({
                index: i + 1,
                key_prefix: key.slice(0, 8) + '...' + key.slice(-4),
                error: e.message,
                enabled, manual_disabled: manualDisabled, auto_disabled: autoDisabled,
            });
        }
    }

    if (keysChanged) {
        const data = loadSettings();
        data.keys_with_status = keysData;
        saveSettings(data);
    }

    return { keys: results };
}

// ==================== SFX 音效 ====================

async function generateSFX(apiKey, text, duration = null) {
    const payload = { text };
    if (duration) payload.duration_seconds = duration;

    const res = await request('POST', '/sound-generation', apiKey, payload, 60000);
    if (res.status !== 200) {
        const err = parseElevenLabsError(res.status, res.body);
        throw new Error(`SFX生成失败[${err.httpStatus}]: ${err.message}`);
    }
    return res.body;
}

// ==================== 构建保存路径 ====================

function buildTTSSavePath(text, outputFormat, tag, seqPrefix = '') {
    const ext = outputFormat.startsWith('mp3') ? '.mp3' : outputFormat.startsWith('pcm') ? '.wav' : '.mp3';
    const sanitized = text.replace(/[\\/:*?"<>|\x00-\x1f]/g, '_').slice(0, 60).trim();
    const filename = seqPrefix ? `${seqPrefix}-${sanitized}-${tag}${ext}` : `${sanitized}-${tag}${ext}`;
    const downloadsDir = path.join(require('os').homedir(), 'Downloads');
    return path.join(downloadsDir, filename);
}

module.exports = {
    loadKeys,
    selectKey,
    loadSettings,
    saveSettings,
    getSettingsPath,
    requestTTS,
    requestTTSWithRotation,
    getVoices,
    searchVoices,
    addVoice,
    deleteVoice,
    deleteOldestCustomVoice,
    getQuota,
    getAllQuotas,
    generateSFX,
    buildTTSSavePath,
    setKeyEnabled,
    parseElevenLabsError,
};
