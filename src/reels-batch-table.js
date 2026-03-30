/**
 * reels-batch-table.js — 批量文案卡片表格管理
 *
 * 表格驱动的批量生产面板：
 *   - 每行 = 一个 Reel 任务
 *   - 列: # | 背景素材 | 音频 | 字幕 | 标题 | 内容 | 模板
 *   - 支持从 Google Sheets 粘贴 标题+内容
 *   - 每行可独立选择卡片模板
 *   - 拖拽/批量分配素材
 */

// ═══════════════════════════════════════════════════════
// 1. Core State & Config
// ═══════════════════════════════════════════════════════

let _dragSrcIdx = null; // 拖拽排序：源行索引（模块级，供 panel drop 判断）
const _VOICE_HINT_REGEX = /(^|[\s._-])(voice|audio|dub|dubbing|narration|vo|配音|人声|旁白|解说|口播)([\s._-]|$)/i;
const _VOICE_VIDEO_EXTS = new Set(['mp4', 'mov', 'mkv', 'avi', 'wmv', 'flv', 'webm']);

function _looksLikeVoiceTrack(name = '') {
    return _VOICE_HINT_REGEX.test(name || '');
}

const _batchTableState = {
    visible: false,
    container: null,
    // ── 多标签页 ──
    tabs: [
        { id: 'tab_1', name: '默认', materialDir: '', lastRefreshTime: null, tasks: [] }
    ],
    activeTabId: 'tab_1',
    nextTabId: 2,
    // ── 视频分配模式（默认关闭） ──
    videoDropRouteEnabled: false,
    videoDropRouteMode: 'bg', // 'bg' | 'audio'
    // ── 批量选择 ──
    selectedRows: new Set(),
    actionsCollapsed: false,
};

// ── 标签页辅助 ──
function _getActiveTab() {
    return _batchTableState.tabs.find(t => t.id === _batchTableState.activeTabId) || _batchTableState.tabs[0];
}

function _syncTasksToActiveTab() {
    const tab = _getActiveTab();
    if (tab) {
        tab.tasks = _serializeTasks(window._reelsState.tasks);
        // DEBUG
        const _dbg = tab.tasks.slice(0, 3).map((t, i) => `[${i}] bgScale=${t.bgScale}`);
        console.log('[BatchTable._syncTasksToActiveTab] 保存到 tab:', _dbg.join(', '));
    }
}

function _loadTabTasks(tab) {
    window._reelsState.tasks = (tab.tasks || []).map(t => ({ ...t }));
    window._reelsState.selectedIdx = -1;
}

function _switchToTab(tabId) {
    if (tabId === _batchTableState.activeTabId) return;
    // Save current tab tasks
    _syncTasksToActiveTab();
    _batchTableState.activeTabId = tabId;
    const tab = _getActiveTab();
    _loadTabTasks(tab);
    _batchTableState.selectedRows = new Set();
    _renderBatchTable();
}

function _addTab(name) {
    const id = 'tab_' + _batchTableState.nextTabId++;
    const tab = { id, name: name || `标签${_batchTableState.tabs.length + 1}`, materialDir: '', lastRefreshTime: null, tasks: [] };
    _batchTableState.tabs.push(tab);
    _switchToTab(id);
}

function _removeTab(tabId) {
    if (_batchTableState.tabs.length <= 1) { alert('至少保留一个标签页'); return; }
    const idx = _batchTableState.tabs.findIndex(t => t.id === tabId);
    if (idx < 0) return;
    if (!confirm(`确定删除标签「${_batchTableState.tabs[idx].name}」及其所有任务？`)) return;
    _batchTableState.tabs.splice(idx, 1);
    if (_batchTableState.activeTabId === tabId) {
        _batchTableState.activeTabId = _batchTableState.tabs[Math.min(idx, _batchTableState.tabs.length - 1)].id;
        _loadTabTasks(_getActiveTab());
    }
    _renderBatchTable();
}

function _renameTab(tabId) {
    const tab = _batchTableState.tabs.find(t => t.id === tabId);
    if (!tab) return;
    const newName = prompt('输入标签名称：', tab.name);
    if (newName && newName.trim()) {
        tab.name = newName.trim();
        _renderBatchTable();
    }
}

// ═══════════════════════════════════════════════════════
// 2. Initialization
// ═══════════════════════════════════════════════════════

function _initBatchTable() {
    // 容器
    let container = document.getElementById('reels-batch-table-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'reels-batch-table-container';
        container.style.cssText = 'display:none;position:fixed;top:0;left:0;right:0;bottom:0;z-index:9000;background:rgba(0,0,0,0.85);overflow:hidden;';
        document.body.appendChild(container);
    }
    _batchTableState.container = container;
}

// ═══════════════════════════════════════════════════════
// 3. Toggle visibility
// ═══════════════════════════════════════════════════════

function reelsToggleBatchTable() {
    if (!_batchTableState.container) _initBatchTable();
    _batchTableState.visible = !_batchTableState.visible;
    if (_batchTableState.visible) {
        // DEBUG
        const _dbg = (window._reelsState?.tasks || []).slice(0, 3).map((t, i) => `[${i}]bgScale=${t.bgScale}`);
        console.log('[BatchTable.toggle] 打开表格，当前 tasks:', _dbg.join(', '));
        // Sync current tasks into active tab on open
        _syncTasksToActiveTab();
        _renderBatchTable();
        _batchTableState.container.style.display = 'flex';
    } else {
        // 关闭时先保存输入框中的值到 task
        _applyBatchTableChanges();
        console.log('[BatchTable.toggle] 关闭表格，已保存 changes');
        _batchTableState.container.style.display = 'none';
    }
}

// ═══════════════════════════════════════════════════════
// 3b. Language list for subtitle alignment
// ═══════════════════════════════════════════════════════

const REELS_LANG_STORAGE_KEY = 'reels_align_language';

const REELS_ALL_LANGUAGES = [
    // ── 常用 ──
    { name: '英语', code: 'en', en: 'English', pinned: true },
    { name: '中文', code: 'zh', en: 'Chinese', pinned: true },
    { name: '日语', code: 'ja', en: 'Japanese', pinned: true },
    { name: '韩语', code: 'ko', en: 'Korean', pinned: true },
    { name: '西班牙语', code: 'es', en: 'Spanish', pinned: true },
    { name: '法语', code: 'fr', en: 'French', pinned: true },
    { name: '德语', code: 'de', en: 'German', pinned: true },
    { name: '葡萄牙语', code: 'pt', en: 'Portuguese', pinned: true },
    { name: '俄语', code: 'ru', en: 'Russian', pinned: true },
    { name: '阿拉伯语', code: 'ar', en: 'Arabic', pinned: true },
    { name: '粤语', code: 'yue', en: 'Cantonese', pinned: true },
    // ── 亚洲 ──
    { name: '印地语', code: 'hi', en: 'Hindi' },
    { name: '泰语', code: 'th', en: 'Thai' },
    { name: '越南语', code: 'vi', en: 'Vietnamese' },
    { name: '印尼语', code: 'id', en: 'Indonesian' },
    { name: '马来语', code: 'ms', en: 'Malay' },
    { name: '泰米尔语', code: 'ta', en: 'Tamil' },
    { name: '泰卢固语', code: 'te', en: 'Telugu' },
    { name: '孟加拉语', code: 'bn', en: 'Bengali' },
    { name: '卡纳达语', code: 'kn', en: 'Kannada' },
    { name: '马拉雅拉姆语', code: 'ml', en: 'Malayalam' },
    { name: '马拉地语', code: 'mr', en: 'Marathi' },
    { name: '古吉拉特语', code: 'gu', en: 'Gujarati' },
    { name: '旁遮普语', code: 'pa', en: 'Punjabi' },
    { name: '僧伽罗语', code: 'si', en: 'Sinhala' },
    { name: '尼泊尔语', code: 'ne', en: 'Nepali' },
    { name: '乌尔都语', code: 'ur', en: 'Urdu' },
    { name: '高棉语', code: 'km', en: 'Khmer' },
    { name: '老挝语', code: 'lo', en: 'Lao' },
    { name: '蒙古语', code: 'mn', en: 'Mongolian' },
    { name: '缅甸语', code: 'my', en: 'Myanmar' },
    { name: '藏语', code: 'bo', en: 'Tibetan' },
    { name: '他加禄语', code: 'tl', en: 'Tagalog' },
    { name: '爪哇语', code: 'jw', en: 'Javanese' },
    { name: '巽他语', code: 'su', en: 'Sundanese' },
    { name: '阿萨姆语', code: 'as', en: 'Assamese' },
    // ── 欧洲 ──
    { name: '意大利语', code: 'it', en: 'Italian' },
    { name: '荷兰语', code: 'nl', en: 'Dutch' },
    { name: '波兰语', code: 'pl', en: 'Polish' },
    { name: '土耳其语', code: 'tr', en: 'Turkish' },
    { name: '瑞典语', code: 'sv', en: 'Swedish' },
    { name: '芬兰语', code: 'fi', en: 'Finnish' },
    { name: '丹麦语', code: 'da', en: 'Danish' },
    { name: '挪威语', code: 'no', en: 'Norwegian' },
    { name: '新挪威语', code: 'nn', en: 'Nynorsk' },
    { name: '捷克语', code: 'cs', en: 'Czech' },
    { name: '斯洛伐克语', code: 'sk', en: 'Slovak' },
    { name: '匈牙利语', code: 'hu', en: 'Hungarian' },
    { name: '罗马尼亚语', code: 'ro', en: 'Romanian' },
    { name: '保加利亚语', code: 'bg', en: 'Bulgarian' },
    { name: '希腊语', code: 'el', en: 'Greek' },
    { name: '乌克兰语', code: 'uk', en: 'Ukrainian' },
    { name: '白俄罗斯语', code: 'be', en: 'Belarusian' },
    { name: '克罗地亚语', code: 'hr', en: 'Croatian' },
    { name: '塞尔维亚语', code: 'sr', en: 'Serbian' },
    { name: '斯洛文尼亚语', code: 'sl', en: 'Slovenian' },
    { name: '立陶宛语', code: 'lt', en: 'Lithuanian' },
    { name: '拉脱维亚语', code: 'lv', en: 'Latvian' },
    { name: '爱沙尼亚语', code: 'et', en: 'Estonian' },
    { name: '马其顿语', code: 'mk', en: 'Macedonian' },
    { name: '波斯尼亚语', code: 'bs', en: 'Bosnian' },
    { name: '阿尔巴尼亚语', code: 'sq', en: 'Albanian' },
    { name: '冰岛语', code: 'is', en: 'Icelandic' },
    { name: '马耳他语', code: 'mt', en: 'Maltese' },
    { name: '卢森堡语', code: 'lb', en: 'Luxembourgish' },
    { name: '法罗语', code: 'fo', en: 'Faroese' },
    { name: '加泰罗尼亚语', code: 'ca', en: 'Catalan' },
    { name: '加利西亚语', code: 'gl', en: 'Galician' },
    { name: '巴斯克语', code: 'eu', en: 'Basque' },
    { name: '奥克语', code: 'oc', en: 'Occitan' },
    { name: '布列塔尼语', code: 'br', en: 'Breton' },
    { name: '威尔士语', code: 'cy', en: 'Welsh' },
    // ── 中东/中亚 ──
    { name: '波斯语', code: 'fa', en: 'Persian' },
    { name: '希伯来语', code: 'he', en: 'Hebrew' },
    { name: '亚美尼亚语', code: 'hy', en: 'Armenian' },
    { name: '格鲁吉亚语', code: 'ka', en: 'Georgian' },
    { name: '阿塞拜疆语', code: 'az', en: 'Azerbaijani' },
    { name: '哈萨克语', code: 'kk', en: 'Kazakh' },
    { name: '乌兹别克语', code: 'uz', en: 'Uzbek' },
    { name: '土库曼语', code: 'tk', en: 'Turkmen' },
    { name: '塔吉克语', code: 'tg', en: 'Tajik' },
    { name: '普什图语', code: 'ps', en: 'Pashto' },
    { name: '信德语', code: 'sd', en: 'Sindhi' },
    { name: '鞑靼语', code: 'tt', en: 'Tatar' },
    { name: '巴什基尔语', code: 'ba', en: 'Bashkir' },
    // ── 非洲 ──
    { name: '南非荷兰语', code: 'af', en: 'Afrikaans' },
    { name: '斯瓦希里语', code: 'sw', en: 'Swahili' },
    { name: '约鲁巴语', code: 'yo', en: 'Yoruba' },
    { name: '豪萨语', code: 'ha', en: 'Hausa' },
    { name: '索马里语', code: 'so', en: 'Somali' },
    { name: '绍纳语', code: 'sn', en: 'Shona' },
    { name: '阿姆哈拉语', code: 'am', en: 'Amharic' },
    { name: '林加拉语', code: 'ln', en: 'Lingala' },
    { name: '马达加斯加语', code: 'mg', en: 'Malagasy' },
    // ── 其他 ──
    { name: '拉丁语', code: 'la', en: 'Latin' },
    { name: '梵语', code: 'sa', en: 'Sanskrit' },
    { name: '毛利语', code: 'mi', en: 'Maori' },
    { name: '夏威夷语', code: 'haw', en: 'Hawaiian' },
    { name: '海地克里奥尔语', code: 'ht', en: 'Haitian Creole' },
    { name: '意第绪语', code: 'yi', en: 'Yiddish' },
];

function _initLangPicker(container) {
    const btn = container.querySelector('#rbt-lang-picker-btn');
    const dropdown = container.querySelector('#rbt-lang-dropdown');
    const searchInput = container.querySelector('#rbt-lang-search');
    const listEl = container.querySelector('#rbt-lang-list');
    const hiddenInput = container.querySelector('#rbt-align-lang');
    if (!btn || !dropdown || !listEl || !hiddenInput) return;

    // Restore saved language
    const savedLang = localStorage.getItem(REELS_LANG_STORAGE_KEY);
    if (savedLang) {
        const found = REELS_ALL_LANGUAGES.find(l => l.name === savedLang);
        if (found) {
            hiddenInput.value = found.name;
            btn.textContent = found.name + ' ▾';
        }
    }

    const renderList = (filter = '') => {
        const q = filter.toLowerCase().trim();
        const filtered = q
            ? REELS_ALL_LANGUAGES.filter(l =>
                l.name.includes(q) || l.en.toLowerCase().includes(q) || l.code.includes(q))
            : REELS_ALL_LANGUAGES;

        // Pinned first, then the rest
        const pinned = filtered.filter(l => l.pinned);
        const rest = filtered.filter(l => !l.pinned);
        const sorted = [...pinned, ...rest];

        listEl.innerHTML = sorted.map(l => `
            <div class="rbt-lang-item" data-name="${l.name}"
                 style="padding:6px 12px;cursor:pointer;font-size:12px;color:#ccc;
                        border-bottom:1px solid rgba(255,255,255,0.04);
                        display:flex;justify-content:space-between;align-items:center;
                        ${hiddenInput.value === l.name ? 'background:rgba(0,212,255,0.12);color:#00D4FF;' : ''}
                        ${l.pinned ? 'font-weight:600;' : ''}">
                <span>${l.name}</span>
                <span style="font-size:10px;color:#666;">${l.en}</span>
            </div>
        `).join('');

        if (sorted.length === 0) {
            listEl.innerHTML = '<div style="padding:12px;text-align:center;color:#555;font-size:11px;">未找到匹配语言</div>';
        }
    };

    // Toggle dropdown
    btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const isOpen = dropdown.style.display !== 'none';
        dropdown.style.display = isOpen ? 'none' : 'block';
        if (!isOpen) {
            searchInput.value = '';
            renderList();
            setTimeout(() => searchInput.focus(), 30);
        }
    });

    // Search filter
    searchInput.addEventListener('input', () => {
        renderList(searchInput.value);
    });
    searchInput.addEventListener('click', (e) => e.stopPropagation());

    // Click item
    listEl.addEventListener('click', (e) => {
        const item = e.target.closest('.rbt-lang-item');
        if (!item) return;
        const name = item.dataset.name;
        hiddenInput.value = name;
        btn.textContent = name + ' ▾';
        dropdown.style.display = 'none';
        // Persist
        localStorage.setItem(REELS_LANG_STORAGE_KEY, name);
    });

    // Hover effect
    listEl.addEventListener('mouseover', (e) => {
        const item = e.target.closest('.rbt-lang-item');
        if (item) item.style.background = 'rgba(255,255,255,0.08)';
    });
    listEl.addEventListener('mouseout', (e) => {
        const item = e.target.closest('.rbt-lang-item');
        if (item) item.style.background = hiddenInput.value === item.dataset.name ? 'rgba(0,212,255,0.12)' : '';
    });

    // Click outside to close
    document.addEventListener('click', (e) => {
        if (!container.querySelector('#rbt-lang-picker-wrap')?.contains(e.target)) {
            dropdown.style.display = 'none';
        }
    });
}

// ═══════════════════════════════════════════════════════
// 4. Render the batch table
// ═══════════════════════════════════════════════════════

function _renderBatchTable() {
    const container = _batchTableState.container;
    const state = window._reelsState;
    if (!state) return;

    // DEBUG: 追踪 bgScale 持久化
    const _dbgScales = (state.tasks || []).slice(0, 3).map((t, i) => `[${i}] bgScale=${t.bgScale} bgDur=${t.bgDurScale} audioDur=${t.audioDurScale}`);
    console.log('[BatchTable._renderBatchTable] 渲染前 task scales:', _dbgScales.join(', '));

    // 自动保存当前配置到 localStorage
    _batchAutoSave();
    const tasks = state.tasks || [];
    const activeTab = _getActiveTab();

    // 获取已保存的卡片模板列表
    const cardTemplates = _getOverlayGroupPresetList();
    const subtitlePresets = _getSubtitlePresetList();

    // 标签栏 HTML
    const tabsHtml = _batchTableState.tabs.map(tab => {
        const isActive = tab.id === _batchTableState.activeTabId;
        return `<div class="rbt-tab ${isActive ? 'rbt-tab-active' : ''}" data-tab-id="${tab.id}">
            <span class="rbt-tab-name" title="双击重命名">${_escHtml(tab.name)}</span>
            ${_batchTableState.tabs.length > 1 ? `<span class="rbt-tab-close" data-tab-id="${tab.id}" title="关闭标签">×</span>` : ''}
        </div>`;
    }).join('');

    // 素材文件夹信息
    const matDir = activeTab.materialDir || '';
    const matDirShort = matDir ? matDir.split(/[\\/]/).slice(-2).join('/') : '';
    const lastRefresh = activeTab.lastRefreshTime ? new Date(activeTab.lastRefreshTime).toLocaleTimeString() : '';

    // 批量选择子模板选项
    const batchSubOpts = subtitlePresets.map(t =>
        `<option value="${_escHtml(t)}">${_escHtml(t)}</option>`
    ).join('');
    const batchCardOpts = cardTemplates.map(t =>
        `<option value="${_escHtml(t.name)}">${_escHtml(t.name)} (${t.count}层)</option>`
    ).join('');

    container.innerHTML = `
        <div class="rbt-panel">
            <!-- ═══ 标签栏 ═══ -->
            <div class="rbt-tabbar">
                <div class="rbt-tabs-scroll">
                    ${tabsHtml}
                    <div class="rbt-tab rbt-tab-add" title="新建标签页">＋</div>
                </div>
            </div>

            <!-- ═══ 素材文件夹 + 刷新 ═══ -->
            <div class="rbt-material-bar">
                <span style="font-size:12px;color:#888;">📁 素材文件夹:</span>
                <span class="rbt-mat-path" title="${_escHtml(matDir)}">${matDirShort || '<i style="color:#555">未设置</i>'}</span>
                <button class="rbt-btn" id="rbt-select-mat-dir" style="font-size:11px;padding:3px 10px;">📂 选择</button>
                <button class="rbt-btn rbt-btn-refresh" id="rbt-refresh-mat" title="扫描文件夹并同步最新素材" ${!matDir ? 'disabled' : ''}>🔄 一键刷新</button>
                ${lastRefresh ? `<span class="rbt-refresh-time">上次刷新: ${lastRefresh}</span>` : ''}
                <span style="flex:1"></span>
                <button id="rbt-toggle-actions-btn" style="background:#222; border:1px solid #333; color:#ccc; padding:2px 8px; border-radius:4px; font-size:11px; cursor:pointer;" title="隐藏/显示操作面板">${_batchTableState.actionsCollapsed ? '🔽 展开操作面板' : '🔼 收起操作面板'}</button>
            </div>

            <div class="rbt-header" style="align-items: flex-start;">

                <div class="rbt-actions" style="display:${_batchTableState.actionsCollapsed ? 'none' : 'flex'}; flex-direction:column; gap:6px; align-items:flex-start;">
                    <!-- Line 1: 文件/素材加载 (File & Assets Load) -->
                    <div style="display:flex; flex-wrap:wrap; gap:6px; align-items:center;">
                        <button class="rbt-btn" id="rbt-upload-folder" title="选择文件夹，自动按文件类型分类+同名配对">📂 导入素材 (整个文件夹)</button>
                        <button class="rbt-btn" id="rbt-upload-bg" title="批量添加背景素材（图片/视频）">🖼 批量导入背景</button>
                        <button class="rbt-btn" id="rbt-set-bgm-btn" title="选择一篇配乐，应用到所有勾选的行">🎵 导入统一配乐</button>
                        <span class="rbt-sep"></span>
                        <button class="rbt-btn" id="rbt-upload-audio" title="批量添加人声音频">🔊 批量导入人声</button>
                        <button class="rbt-btn" id="rbt-upload-srt" title="批量加载SRT字幕文件，按顺序或同名分配到各行">📝 批量导入srt</button>
                        <button class="rbt-btn" id="rbt-upload-txt" title="批量加载人声文案TXT文件">📃 批量导入人声字幕txt</button>
                    </div>

                    <!-- Line 2: 分配规则与预设 (Allocation Rules) -->
                    <div style="display:flex; flex-wrap:wrap; gap:6px; align-items:center;">
                        <button class="rbt-btn" id="rbt-upload-hook" title="批量配置选中行的前置视频和转场">🪝 批量设置钩子视频</button>
                        <button class="rbt-btn" id="rbt-cycle-fill-btn" title="将素材池按权重交错分配到各行，自动避免相邻重复">🔄 素材循环使用设置</button>
                        <label title="开启后，拖入视频按下方模式分配" style="display:flex;align-items:center;gap:4px;font-size:11px;color:#ccc;background:#222;border:1px solid #333;border-radius:4px;padding:3px 8px;cursor:pointer;">
                            <input type="checkbox" id="rbt-video-drop-route-enabled" ${_batchTableState.videoDropRouteEnabled ? 'checked' : ''}>
                            视频分配模式
                        </label>
                        <select id="rbt-video-drop-route-mode" style="font-size:11px;padding:3px 6px;background:#222;color:#ccc;border:1px solid #333;border-radius:4px;${_batchTableState.videoDropRouteEnabled ? '' : 'opacity:.6;'}" ${_batchTableState.videoDropRouteEnabled ? '' : 'disabled'}>
                            <option value="hook" ${_batchTableState.videoDropRouteMode === 'hook' ? 'selected' : ''}>🪝 视频→前置Hook</option>
                            <option value="bg" ${_batchTableState.videoDropRouteMode === 'bg' ? 'selected' : ''}>🎬 视频→背景</option>
                            <option value="audio" ${_batchTableState.videoDropRouteMode === 'audio' ? 'selected' : ''}>🎙 视频→人声</option>
                        </select>
                    </div>

                    <!-- Line 3: 表格快捷录入 (Quick Paste & Fill) -->
                    <div style="display:flex; flex-wrap:wrap; gap:6px; align-items:center;">
                        <button class="rbt-btn" id="rbt-paste-txtcontent" title="从表格复制文案，批量粘贴到「字幕文本」列">📋 粘贴字幕文本</button>
                        <button class="rbt-btn" id="rbt-paste-btn" title="从 Google 表格粘贴标题+内容">📋 粘贴覆层文案</button>
                        <button class="rbt-btn" id="rbt-paste-scroll-btn" title="从 Google 表格粘贴滚动字幕标题+内容">🔄 粘贴滚动字幕</button>
                    </div>

                    <!-- Line 4: 对齐生成 (Alignment Automation) -->
                    <div style="display:flex; flex-wrap:wrap; gap:6px; align-items:center;">
                        <select id="rbt-align-source" style="font-size:11px;padding:3px 6px;background:#222;color:#ccc;border:1px solid #333;border-radius:4px;">
                            <option value="audio">🎙 用人声对齐</option>
                            <option value="video">🎬 用视频对齐</option>
                        </select>
                        <div style="position:relative;display:inline-block;" id="rbt-lang-picker-wrap">
                            <button id="rbt-lang-picker-btn" style="font-size:11px;padding:3px 10px;background:#222;color:#ccc;border:1px solid #333;border-radius:4px;cursor:pointer;min-width:90px;text-align:left;">英语 ▾</button>
                            <input type="hidden" id="rbt-align-lang" value="英语">
                            <div id="rbt-lang-dropdown" style="display:none;position:absolute;top:100%;left:0;z-index:9999;background:#222;border:1px solid #333;border-radius:6px;width:220px;max-height:320px;box-shadow:0 8px 24px rgba(0,0,0,0.6);overflow:hidden;">
                                <input id="rbt-lang-search" type="text" placeholder="🔍 搜索语言..." style="width:100%;box-sizing:border-box;padding:6px 10px;border:none;border-bottom:1px solid #333;background:#1a1a1a;color:#ddd;font-size:12px;outline:none;">
                                <div id="rbt-lang-list" style="max-height:270px;overflow-y:auto;"></div>
                            </div>
                        </div>
                        <button class="rbt-btn" id="rbt-align-all-btn" title="对齐所有有文案的任务，生成字幕时间轴">🔗 一键对齐字幕</button>
                        <label style="display:flex;align-items:center;gap:3px;font-size:10px;color:#888;cursor:pointer;">
                            <input type="checkbox" id="rbt-force-realign"> 强制重新对齐
                        </label>
                    </div>

                    <!-- Line 5: 工程与系统管理 (Project Management & Settings) -->
                    <div style="display:flex; flex-wrap:wrap; gap:6px; align-items:center;">
                        <button class="rbt-btn" id="rbt-add-row-btn" title="添加空行">➕ 添加行</button>
                        <button class="rbt-btn rbt-btn-danger" id="rbt-clear-btn" title="清空当前标签页所有行">🧹 清空</button>
                        <span class="rbt-sep"></span>
                        <button class="rbt-btn" id="rbt-save-config-btn" title="导出所有标签页配置为JSON工程文件">💾 保存工程</button>
                        <button class="rbt-btn" id="rbt-load-config-btn" title="从JSON工程文件恢复所有标签页">📂 加载工程</button>
                        <input type="file" id="rbt-file-config" class="rbt-hidden-input" accept=".json">
                        <span class="rbt-sep"></span>
                        <button class="rbt-btn" id="rbt-col-settings-btn" title="设置显示哪些列">⚙️ 列设置</button>
                        <button class="rbt-btn" id="rbt-close-btn" title="关闭表格">✕ 关闭界面</button>
                    </div>
                </div>
            </div>

            <!-- ═══ 批量选择操作栏 ═══ -->
            <div class="rbt-batch-actions-bar" id="rbt-batch-bar">
                <label class="rbt-batch-label"><input type="checkbox" id="rbt-select-all"> 全选</label>
                <button class="rbt-btn" id="rbt-invert-select" style="font-size:10px;padding:2px 8px;">反选</button>
                <button class="rbt-btn" id="rbt-deselect-all" style="font-size:10px;padding:2px 8px;">取消</button>
                <span class="rbt-sep"></span>
                <span style="font-size:11px;color:#888;">批量设置:</span>
                <select id="rbt-batch-sub-tpl" class="rbt-select" style="width:120px;font-size:11px;">
                    <option value="">字幕模板...</option>
                    ${batchSubOpts}
                </select>
                <button class="rbt-btn" id="rbt-apply-batch-sub" style="font-size:10px;padding:2px 8px;">应用</button>
                <select id="rbt-batch-card-tpl" class="rbt-select" style="width:140px;font-size:11px;">
                    <option value="">覆层预设...</option>
                    ${batchCardOpts}
                </select>
                <button class="rbt-btn" id="rbt-apply-batch-card" style="font-size:10px;padding:2px 8px;">应用</button>
                <span class="rbt-sep"></span>
                <button class="rbt-btn rbt-btn-danger" id="rbt-clear-overlay-btn" style="font-size:10px;padding:2px 8px;" title="清空勾选行的覆层标题/内容/结尾（未勾选则清空全部）">🧹 清空覆层</button>
                <span class="rbt-sep"></span>
                <span style="font-size:11px;color:#888;">缩放:</span>
                <label style="display:flex;align-items:center;gap:2px;font-size:10px;color:#4fc3f7;" title="批量设置选中行的背景图片缩放">🔍<input type="number" id="rbt-batch-bgscale" min="50" max="300" value="100" step="5" style="width:40px;font-size:10px;padding:1px 2px;background:#1a1a3a;color:#4fc3f7;border:1px solid #4a6a8a;border-radius:3px;text-align:center;">%</label>
                <label style="display:flex;align-items:center;gap:2px;font-size:10px;color:#81c784;" title="批量设置选中行的背景素材时长缩放">⏱🖼<input type="number" id="rbt-batch-bgdurscale" min="10" max="500" value="100" step="5" style="width:40px;font-size:10px;padding:1px 2px;background:#1a1a3a;color:#81c784;border:1px solid #4a6a8a;border-radius:3px;text-align:center;">%</label>
                <label style="display:flex;align-items:center;gap:2px;font-size:10px;color:#ffb74d;" title="批量设置选中行的人声音频素材时长缩放">⏱🔊<input type="number" id="rbt-batch-audiodurscale" min="10" max="500" value="100" step="5" style="width:40px;font-size:10px;padding:1px 2px;background:#1a1a3a;color:#ffb74d;border:1px solid #4a6a8a;border-radius:3px;text-align:center;">%</label>
                <button class="rbt-btn" id="rbt-apply-batch-scale" style="font-size:10px;padding:2px 8px;">应用缩放</button>
                <span class="rbt-sep"></span>
                <button class="rbt-btn rbt-btn-accent" id="rbt-export-selected-btn" style="font-size:10px;padding:2px 8px;display:none;" title="仅将勾选的行导出到主任务列表（表格数据不变）">📤 仅导出选中行</button>
                <span id="rbt-selected-count" style="font-size:11px;color:#00D4FF;margin-left:8px;"></span>
            </div>

            <!-- hidden file inputs -->
            <input type="file" id="rbt-file-bg" class="rbt-hidden-input" accept=".mp4,.mov,.mkv,.avi,.wmv,.flv,.webm,.jpg,.jpeg,.png,.webp" multiple>
            <input type="file" id="rbt-file-audio" class="rbt-hidden-input" accept=".mp3,.wav,.m4a,.aac,.flac,.ogg,.wma,.mp4,.mov,.mkv,.avi,.wmv,.flv,.webm" multiple>
            <input type="file" id="rbt-file-srt" class="rbt-hidden-input" accept=".srt" multiple>
            <input type="file" id="rbt-file-txt" class="rbt-hidden-input" accept=".txt" multiple>
            <input type="file" id="rbt-file-bgm" class="rbt-hidden-input" accept=".mp3,.wav,.m4a,.aac,.flac,.ogg">
            <input type="file" id="rbt-file-single" class="rbt-hidden-input" accept="*/*">
            <input type="file" id="rbt-file-folder" class="rbt-hidden-input" webkitdirectory directory multiple>
            <div class="rbt-table-wrap">
                <table class="rbt-table" id="rbt-table">
                    <thead>
                        <tr>
                            <th class="rbt-col-drag" style="width:24px;"></th>
                            <th class="rbt-col-chk" style="width:30px;text-align:center;">☐</th>
                            <th class="rbt-col-num">#</th>
                            <th class="rbt-col-hook">🪝 前置Hook</th>
                            <th class="rbt-col-bg">🖼 背景素材</th>
                            <th class="rbt-col-bgscale">🔍 背景缩放</th>
                            <th class="rbt-col-bgdurscale">⏱🖼 背景时长</th>
                            <th class="rbt-col-audio">🔊 人声音频层</th>
                            <th class="rbt-col-audiodurscale">⏱🔊 人声变速</th>
                            <th class="rbt-col-srt">📝 字幕SRT</th>
                            <th class="rbt-col-txtcontent">📃 字幕文本</th>
                            <th class="rbt-col-bgm">🎵 配乐</th>
                            <th class="rbt-col-title">📋 覆层标题</th>
                            <th class="rbt-col-body">📋 覆层内容</th>
                            <th class="rbt-col-footer">📋 覆层结尾</th>
                            <th class="rbt-col-scroll-title">🔄 滚动标题</th>
                            <th class="rbt-col-scroll-body">🔄 滚动内容</th>
                            <th class="rbt-col-dur">⏱ 时长(s)</th>
                            <th class="rbt-col-tpl">🎬 字幕模板</th>
                            <th class="rbt-col-tpl">📦 覆层预设</th>
                            <th class="rbt-col-act">操作</th>
                        </tr>
                    </thead>
                    <tbody id="rbt-tbody">
                        ${tasks.map((t, i) => _renderBatchRow(t, i, subtitlePresets, cardTemplates)).join('')}
                    </tbody>
                </table>
            </div>
            <div class="rbt-footer">
                <span id="rbt-count">${tasks.length} 个任务</span>
                <span id="rbt-align-progress" style="font-size:12px;color:#a0d0ff;margin-left:12px;"></span>
                <div class="rbt-footer-hint">提示: 先粘贴文案创建行 → 再批量添加素材自动按顺序分配 | 设置素材文件夹后可一键刷新同步最新文件</div>
                <button class="rbt-btn rbt-btn-primary" id="rbt-apply-btn">✅ 应用更改并关闭</button>
            </div>
        </div>
            <div id="rbt-hook-modal" style="display:none;position:absolute;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.8);z-index:99999;align-items:center;justify-content:center;">
                <div style="background:#1a1a2e;padding:20px;border-radius:8px;border:1px solid #333;width:400px;display:flex;flex-direction:column;gap:12px;">
                    <h3 style="margin:0;color:#fff;font-size:14px;display:flex;align-items:center;gap:6px;">🪝 前置/Hook 设置</h3>
                    
                    <div style="font-size:12px;color:#ccc;display:flex;align-items:center;gap:8px;">
                        <span>Hook:</span>
                        <input type="text" id="rbt-hook-path" style="flex:1;background:#222;color:#ccc;border:1px solid #333;padding:4px;font-size:11px;" readonly placeholder="未选择 (留空则从正片开始)">
                        <button class="rbt-btn" id="rbt-hook-select-btn" style="padding:4px 8px;">浏览素材</button>
                    </div>

                    <div id="rbt-hook-preview-container" style="display:none;width:100%;height:160px;background:#000;border-radius:4px;overflow:hidden;position:relative;border:1px solid #333;">
                        <video id="rbt-hook-preview-video" style="width:100%;height:100%;object-fit:contain;" controls preload="metadata"></video>
                        <img id="rbt-hook-preview-img" style="width:100%;height:100%;object-fit:contain;display:none;">
                    </div>

                    <div style="background:#222;padding:10px;border-radius:4px;display:flex;flex-direction:column;gap:8px;border:1px solid #333;">
                        <span style="font-size:11px;color:#888;">裁切区间 (秒，留空表示使用原文件长度)</span>
                        <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
                            <button class="rbt-btn" id="rbt-hook-mark-start" style="padding:3px 8px;font-size:11px;" title="将当前视频播放位置设为起始点">⏩ 设为起点</button>
                            <input type="number" id="rbt-hook-start" style="width:60px;background:#111;color:#fff;border:1px solid #333;padding:4px;text-align:center;" placeholder="Start" step="0.1" min="0">
                            <span style="color:#666;">至</span>
                            <input type="number" id="rbt-hook-end" style="width:60px;background:#111;color:#fff;border:1px solid #333;padding:4px;text-align:center;" placeholder="End" step="0.1" min="0">
                            <button class="rbt-btn" id="rbt-hook-mark-end" style="padding:3px 8px;font-size:11px;" title="将当前视频播放位置设为结束点">⏪ 设为终点</button>
                            <span style="flex:1;"></span>
                            <span style="font-size:11px;color:#888;">播放变速:</span>
                            <input type="number" id="rbt-hook-speed" list="rbt-hook-speed-presets" value="1" step="0.05" min="0.1" max="10" style="background:#111;color:#fff;border:1px solid #333;padding:4px;width:70px;text-align:center;">
                            <datalist id="rbt-hook-speed-presets">
                                <option value="0.25">0.25x</option>
                                <option value="0.5">0.5x</option>
                                <option value="0.75">0.75x</option>
                                <option value="1">1.0x</option>
                                <option value="1.25">1.25x</option>
                                <option value="1.5">1.5x</option>
                                <option value="2">2.0x</option>
                                <option value="3">3.0x</option>
                            </datalist>
                        </div>
                    </div>

                    <div style="background:#222;padding:10px;border-radius:4px;display:flex;flex-direction:column;gap:8px;border:1px solid #333;">
                        <span style="font-size:11px;color:#888;">正片转场 (Transition)</span>
                        <div style="display:flex;align-items:center;gap:8px;">
                            <select id="rbt-hook-transition" style="flex:1;background:#111;color:#fff;border:1px solid #333;padding:4px;font-size:11px;">
                                <option value="none" selected>⚡ 无转场 (直接硬切)</option>
                                <option value="fade">⚫ Fade 黑场淡入淡出</option>
                                <option value="fadeblack">⚫ FadeBlack 强黑场过渡</option>
                                <option value="fadewhite">⚪ FadeWhite 强白场过渡</option>
                                <option value="dissolve">✨ Dissolve 溶解叠化</option>
                                <option value="wipeleft">⬅️ Wipe Left 向左擦除</option>
                                <option value="wiperight">➡️ Wipe Right 向右擦除</option>
                                <option value="slideleft">🔙 Slide Left 向左推移</option>
                                <option value="slideright">🔜 Slide Right 向右推移</option>
                                <option value="rectcrop">🔲 RectCrop 矩形扩散</option>
                                <option value="circlecrop">⭕ CircleCrop 圆形扩散</option>
                            </select>
                            <span style="font-size:11px;color:#888;">时长(s):</span>
                            <input type="number" id="rbt-hook-trans-dur" value="0.5" step="0.1" min="0.1" max="5.0" style="width:50px;background:#111;color:#fff;border:1px solid #333;padding:4px;text-align:center;">
                        </div>
                    </div>
                    
                    <div style="display:flex;justify-content:flex-end;gap:8px;margin-top:4px;">
                        <button class="rbt-btn" id="rbt-hook-cancel" style="padding:6px 16px;">取消</button>
                        <button class="rbt-btn rbt-btn-primary" id="rbt-hook-save" style="padding:6px 16px;">保存配置</button>
                    </div>
                </div>
            </div>
        </div>
    `;

    // CSS
    _injectBatchTableCSS();

    // 事件
    _bindBatchTableEvents();

    // 应用列显示设置
    _applyColVisibility();

    // 更新批量选中计数
    _updateBatchSelectCount();
}

function _renderBatchRow(task, idx, subtitlePresets, cardTemplates) {
    // 提取覆层信息
    const ov = (task.overlays || [])[0];
    const title = ov ? (ov.title_text || '') : '';
    const body = ov ? (ov.body_text || '') : '';
    const footer = ov ? (ov.footer_text || '') : '';
    // 滚动字幕覆层信息
    const scrollOv = (task.overlays || []).find(o => o.type === 'scroll');
    const scrollTitle = scrollOv ? (scrollOv.scroll_title || '') : '';
    const scrollBody = scrollOv ? (scrollOv.content || '') : '';
    const bgName = _shortName(task.bgPath || task.videoPath || '');
    const audioName = _shortName(task.audioPath || '');
    const srtName = _shortName(task.srtPath || '');
    const txtName = _shortName(task.txtPath || '');
    const txtStatus = task.txtContent ? (task.aligned ? '✅' : '⏳') : '';
    const bgmName = _shortName(task.bgmPath || '');

    // 缩略图生成
    let bgContent = bgName || '<span class="rbt-placeholder">拖拽/双击</span>';
    const bgPath = task.bgPath || task.videoPath;
    if (bgPath) {
        const isImg = /\.(jpg|jpeg|png|webp|gif|bmp)$/i.test(bgPath);
        const urlObj = task.bgSrcUrl || (window.electronAPI && window.electronAPI.toFileUrl ? window.electronAPI.toFileUrl(bgPath) : `file://${bgPath}`);
        if (isImg) {
            bgContent = `<div style="display:flex;align-items:center;gap:6px;">
                            <img src="${_escHtml(urlObj)}" style="width:32px;height:32px;object-fit:cover;border-radius:4px;flex-shrink:0;">
                            <span class="rbt-file-name" style="flex:1;font-size:10px;word-break:break-all;" title="${_escHtml(bgPath)}">${_escHtml(bgPath)}</span>
                         </div>`;
        } else {
            bgContent = `<div style="display:flex;align-items:center;gap:6px;">
                            <video src="${_escHtml(urlObj)}#t=1" preload="metadata" style="width:32px;height:32px;object-fit:cover;border-radius:4px;flex-shrink:0;background:#000;"></video>
                            <span class="rbt-file-name" style="flex:1;font-size:10px;word-break:break-all;" title="${_escHtml(bgPath)}">${_escHtml(bgPath)}</span>
                         </div>`;
        }
    } else {
        bgContent = `<span class="rbt-file-name" title="">${bgContent}</span>`;
    }

    // Hook缩略图生成
    const hookName = _shortName(task.hookFile || '');
    let hookContent = hookName || '<span class="rbt-placeholder">拖拽/双击配置</span>';
    if (task.hookFile) {
        const hIsImg = /\.(jpg|jpeg|png|webp|gif|bmp)$/i.test(task.hookFile);
        const hUrlObj = window.electronAPI && window.electronAPI.toFileUrl ? window.electronAPI.toFileUrl(task.hookFile) : `file://${task.hookFile}`;
        hookContent = `<div class="rbt-hook-set" style="display:flex;align-items:center;gap:4px;cursor:pointer;" title="双击重新配置\n${_escHtml(task.hookFile)}">
                            ${hIsImg ? 
                                `<img src="${_escHtml(hUrlObj)}" style="width:28px;height:28px;object-fit:cover;border-radius:3px;flex-shrink:0;">` :
                                `<video src="${_escHtml(hUrlObj)}#t=0.5" preload="metadata" style="width:28px;height:28px;object-fit:cover;border-radius:3px;flex-shrink:0;background:#111;"></video>`
                            }
                            <div style="flex:1;display:flex;flex-direction:column;min-width:0;line-height:1.2;">
                              <span class="rbt-file-name" style="font-size:10px;word-break:break-all;">${_escHtml(hookName)}</span>
                              <span style="font-size:9px;color:#a0d0ff;">${task.hookTransition && task.hookTransition !== 'none' ? `✨${task.hookTransition}(${task.hookTransDuration || 0.5}s)` : '⚡硬切'} | ${task.hookSpeed ? task.hookSpeed + 'x' : '1x'}</span>
                            </div>
                       </div>`;
    } else {
        hookContent = `<div class="rbt-hook-set" style="cursor:pointer;" title="双击配置Hook属性"><span class="rbt-file-name" title="">${hookContent}</span></div>`;
    }

    const subTplOptions = subtitlePresets.map(t =>
        `<option value="${_escHtml(t)}" ${task._subtitlePreset === t ? 'selected' : ''}>${_escHtml(t)}</option>`
    ).join('');

    const cardTplOptions = cardTemplates.map(t =>
        `<option value="${_escHtml(t.name)}" ${task._overlayPresetName === t.name ? 'selected' : ''}>${_escHtml(t.name)} (${t.count}层)</option>`
    ).join('');

    return `
        <tr data-idx="${idx}" draggable="true" class="rbt-row ${idx === (window._reelsState?.selectedIdx || -1) ? 'rbt-row-selected' : ''} ${task._justRefreshed ? 'rbt-row-refreshed' : ''}">
            <td class="rbt-col-drag"><span class="rbt-drag-handle" title="拖拽调整顺序">☰</span></td>
            <td class="rbt-col-chk" style="text-align:center;"><input type="checkbox" class="rbt-row-check" data-idx="${idx}" ${_batchTableState.selectedRows.has(idx) ? 'checked' : ''}></td>
            <td class="rbt-col-num">${idx + 1}</td>
            <td class="rbt-col-hook rbt-droppable" data-field="hook">
                <div style="display:flex;align-items:center;gap:2px;">
                    <div style="flex:1;min-width:0;overflow:hidden;">${hookContent}</div>
                    ${task.hookFile ? `<button class="rbt-field-clear" data-idx="${idx}" data-field="hook" title="清除前置Hook">✕</button>` : ''}
                </div>
            </td>
            <td class="rbt-col-bg rbt-droppable" data-field="bg">
                <div style="display:flex;align-items:center;gap:2px;">
                    <div style="flex:1;min-width:0;overflow:hidden;">${bgContent}</div>
                    ${bgPath ? `<button class="rbt-field-clear" data-idx="${idx}" data-field="bg" title="清除背景">✕</button>` : ''}
                </div>
            </td>
            <td class="rbt-col-bgscale">
                <div style="display:flex;flex-direction:column;gap:2px;align-items:center;min-width:70px;">
                    <input type="range" class="rbt-bgscale-slider" data-idx="${idx}" min="50" max="300" value="${task.bgScale || 100}"
                           style="width:60px;height:14px;accent-color:#4fc3f7;" title="背景图片缩放比例">
                    <div style="display:flex;align-items:center;gap:2px;">
                        <input type="number" class="rbt-bgscale-input" data-idx="${idx}" min="50" max="300" step="5"
                               value="${task.bgScale || 100}" style="width:42px;text-align:center;font-size:10px;padding:1px 2px;background:#1a1a2e;color:#ccc;border:1px solid #333;border-radius:3px;">
                        <span style="font-size:10px;color:#666;">%</span>
                    </div>
                </div>
            </td>
            <td class="rbt-col-bgdurscale">
                <div style="display:flex;flex-direction:column;gap:2px;align-items:center;min-width:70px;">
                    <input type="range" class="rbt-bgdurscale-slider" data-idx="${idx}" min="10" max="500" value="${task.bgDurScale || 100}"
                           style="width:60px;height:14px;accent-color:#81c784;" title="背景素材时长缩放比例">
                    <div style="display:flex;align-items:center;gap:2px;">
                        <input type="number" class="rbt-bgdurscale-input" data-idx="${idx}" min="10" max="500" step="5"
                               value="${task.bgDurScale || 100}" style="width:42px;text-align:center;font-size:10px;padding:1px 2px;background:#1a1a2e;color:#ccc;border:1px solid #333;border-radius:3px;">
                        <span style="font-size:10px;color:#666;">%</span>
                    </div>
                </div>
            </td>
            <td class="rbt-col-audio rbt-droppable" data-field="audio">
                <div style="display:flex;align-items:center;gap:2px;">
                    <span class="rbt-file-name" style="flex:1" title="双击加载 | ${_escHtml(task.audioPath || '')}">${audioName || '<span class="rbt-placeholder">双击加载</span>'}</span>
                    ${audioName ? `<button class="rbt-field-clear" data-idx="${idx}" data-field="audio" title="清除人声">✕</button>` : ''}
                </div>
            </td>
            <td class="rbt-col-audiodurscale">
                <div style="display:flex;flex-direction:column;gap:2px;align-items:center;min-width:70px;">
                    <input type="range" class="rbt-audiodurscale-slider" data-idx="${idx}" min="10" max="500" value="${task.audioDurScale || 100}"
                           style="width:60px;height:14px;accent-color:#ffb74d;" title="人声变速比例">
                    <div style="display:flex;align-items:center;gap:2px;">
                        <input type="number" class="rbt-audiodurscale-input" data-idx="${idx}" min="10" max="500" step="5"
                               value="${task.audioDurScale || 100}" style="width:42px;text-align:center;font-size:10px;padding:1px 2px;background:#1a1a2e;color:#ccc;border:1px solid #333;border-radius:3px;">
                        <span style="font-size:10px;color:#666;">%</span>
                    </div>
                </div>
            </td>
            <td class="rbt-col-srt rbt-droppable" data-field="srt">
                <div style="display:flex;align-items:center;gap:2px;">
                    <span class="rbt-file-name" style="flex:1" title="双击加载 | ${_escHtml(task.srtPath || '')}">${srtName || '<span class="rbt-placeholder">双击加载</span>'}</span>
                    ${srtName ? `<button class="rbt-field-clear" data-idx="${idx}" data-field="srt" title="清除字幕">✕</button>` : ''}
                </div>
            </td>
            <td class="rbt-col-txtcontent">
                <div style="position:relative;">
                    <textarea class="rbt-textarea rbt-txtcontent-input" data-idx="${idx}" rows="2" placeholder="粘贴或输入文案..." title="双击放大编辑">${_escHtml(task.txtContent || '')}</textarea>
                    ${task.txtContent ? `<button class="rbt-field-clear" data-idx="${idx}" data-field="txt" title="清除文案" style="position:absolute;top:2px;right:2px;">✕</button>` : ''}
                </div>
                <div style="display:flex;align-items:center;gap:4px;margin-top:2px;">
                    <span style="font-size:10px;color:${task.aligned ? '#4ade80' : task.txtContent ? '#facc15' : '#666'};">${task.aligned ? '✅ 已对齐' : task.txtContent ? '⏳ 待对齐' : ''}</span>
                    ${task.srtPath ? `<span style="font-size:10px;color:#888;" title="${_escHtml(task.srtPath)}">📄 ${_shortName(task.srtPath)}</span>` : ''}
                </div>
            </td>
            <td class="rbt-col-bgm">
                <div style="display:flex;align-items:center;gap:4px;">
                    <input type="checkbox" class="rbt-bgm-check" data-idx="${idx}" title="勾选后可批量设置配乐">
                    <span class="rbt-file-name rbt-bgm-pick" data-idx="${idx}" title="双击加载 | ${_escHtml(task.bgmPath || '')}"
                          style="cursor:pointer;flex:1;">${bgmName || '<span class="rbt-placeholder">双击选择</span>'}</span>
                    ${bgmName ? `<button class="rbt-field-clear" data-idx="${idx}" data-field="bgm" title="清除配乐">✕</button>` : ''}
                </div>
                <div style="display:flex;align-items:center;gap:4px;margin-top:3px;">
                    <span style="font-size:10px;color:#888;white-space:nowrap;">🔉</span>
                    <input type="range" class="rbt-bgm-vol" data-idx="${idx}" min="0" max="100" value="${task.bgmVolume != null ? task.bgmVolume : 30}"
                           style="flex:1;height:14px;accent-color:#9b59b6;" title="配乐音量">
                    <span class="rbt-bgm-vol-label" style="font-size:10px;color:#888;min-width:28px;text-align:right;">${task.bgmVolume != null ? task.bgmVolume : 30}%</span>
                </div>
            </td>
            <td class="rbt-col-title">
                <textarea class="rbt-textarea rbt-title-input" data-idx="${idx}" rows="2" title="双击放大编辑">${_escHtml(title)}</textarea>
            </td>
            <td class="rbt-col-body">
                <textarea class="rbt-textarea rbt-body-input" data-idx="${idx}" rows="2" title="双击放大编辑">${_escHtml(body)}</textarea>
            </td>
            <td class="rbt-col-footer">
                <textarea class="rbt-textarea rbt-footer-input" data-idx="${idx}" rows="2" title="双击放大编辑">${_escHtml(footer)}</textarea>
            </td>
            <td class="rbt-col-scroll-title">
                <textarea class="rbt-textarea rbt-scroll-title-input" data-idx="${idx}" rows="2" title="双击放大编辑">${_escHtml(scrollTitle)}</textarea>
            </td>
            <td class="rbt-col-scroll-body">
                <textarea class="rbt-textarea rbt-scroll-body-input" data-idx="${idx}" rows="2" title="双击放大编辑">${_escHtml(scrollBody)}</textarea>
            </td>
            <td class="rbt-col-dur">
                <input type="number" class="rbt-textarea rbt-dur-input" data-idx="${idx}" min="0" max="600" step="0.5"
                    value="${task.customDuration ? task.customDuration : ''}" placeholder="自动" style="width:55px;text-align:center;" title="留空=自动跟随音频/视频时长，输入数字=自定义秒数">
            </td>
            <td class="rbt-col-tpl">
                <select class="rbt-select rbt-sub-tpl-select" data-idx="${idx}">
                    <option value="">默认</option>
                    ${subTplOptions}
                </select>
            </td>
            <td class="rbt-col-tpl">
                <select class="rbt-select rbt-card-tpl-select" data-idx="${idx}">
                    <option value="">无预设</option>
                    ${cardTplOptions}
                </select>
            </td>
            <td class="rbt-col-act">
                <button class="rbt-row-btn rbt-select-btn" data-idx="${idx}" title="预览此任务">👁</button>
                <button class="rbt-row-btn rbt-clone-btn" data-idx="${idx}" title="复制此行">📋</button>
                <button class="rbt-row-btn rbt-delete-btn" data-idx="${idx}" title="删除此行">🗑</button>
            </td>
        </tr>
    `;
}


// ═══════════════════════════════════════════════════════
// 5. Event Binding
// ═══════════════════════════════════════════════════════

function _bindBatchTableEvents() {
    const container = _batchTableState.container;

    // Toggle actions
    container.querySelector('#rbt-toggle-actions-btn')?.addEventListener('click', () => {
        _batchTableState.actionsCollapsed = !_batchTableState.actionsCollapsed;
        _renderBatchTable();
    });

    // Close
    container.querySelector('#rbt-close-btn')?.addEventListener('click', () => reelsToggleBatchTable());
    container.querySelector('#rbt-apply-btn')?.addEventListener('click', () => {
        _applyBatchTableChanges();
        reelsToggleBatchTable();
    });

    // ── Language searchable picker ──
    _initLangPicker(container);

    // ══ Tab bar events ══
    const tabBar = container.querySelector('.rbt-tabs-scroll');
    if (tabBar) {
        tabBar.addEventListener('click', (e) => {
            // Add new tab
            if (e.target.closest('.rbt-tab-add')) {
                _addTab();
                return;
            }
            // Close tab
            const closeEl = e.target.closest('.rbt-tab-close');
            if (closeEl) {
                e.stopPropagation();
                _removeTab(closeEl.dataset.tabId);
                return;
            }
            // Switch tab
            const tabEl = e.target.closest('.rbt-tab');
            if (tabEl && tabEl.dataset.tabId) {
                _switchToTab(tabEl.dataset.tabId);
            }
        });
        tabBar.addEventListener('dblclick', (e) => {
            const tabEl = e.target.closest('.rbt-tab');
            if (tabEl && tabEl.dataset.tabId && !tabEl.classList.contains('rbt-tab-add')) {
                _renameTab(tabEl.dataset.tabId);
            }
        });
    }

    // ══ Material folder selection & refresh ══
    container.querySelector('#rbt-select-mat-dir')?.addEventListener('click', async () => {
        if (window.electronAPI && window.electronAPI.selectDirectory) {
            const dir = await window.electronAPI.selectDirectory();
            if (dir) {
                const tab = _getActiveTab();
                tab.materialDir = dir;
                _renderBatchTable();
            }
        } else {
            alert('请在桌面应用中使用此功能');
        }
    });
    container.querySelector('#rbt-refresh-mat')?.addEventListener('click', () => {
        _refreshMaterialFolder();
    });

    // ══ Hook Batch settings ══
    container.querySelector('#rbt-upload-hook')?.addEventListener('click', () => {
        const indices = _getSelectedIndices();
        if (indices.length === 0) {
            alert('请先在左侧勾选需要批量配置前置Hook的行');
            return;
        }
        _openHookModal(-1);
    });

    const hookModal = container.querySelector('#rbt-hook-modal');
    if (hookModal) {
        container.querySelector('#rbt-hook-select-btn')?.addEventListener('click', async () => {
            if (window.electronAPI && window.electronAPI.selectFile) {
                const res = await window.electronAPI.selectFile({
                    title: '选择前置Hook短片',
                    properties: ['openFile'],
                    filters: [{name: '视频文件', extensions: ['mp4', 'mov', 'mkv', 'webm', 'avi']}]
                });
                if (res && res.length > 0) {
                    container.querySelector('#rbt-hook-path').value = res[0];
                    _updateHookPreview();
                }
            }
        });
        
        container.querySelector('#rbt-hook-start')?.addEventListener('input', () => {
            const videoPreview = container.querySelector('#rbt-hook-preview-video');
            const startInput = container.querySelector('#rbt-hook-start');
            if (videoPreview && startInput && videoPreview.readyState >= 1) {
                const startTime = parseFloat(startInput.value) || 0;
                videoPreview.currentTime = startTime;
            }
        });

        container.querySelector('#rbt-hook-mark-start')?.addEventListener('click', () => {
            const videoPreview = container.querySelector('#rbt-hook-preview-video');
            const startInput = container.querySelector('#rbt-hook-start');
            if (videoPreview && startInput && videoPreview.readyState >= 1) {
                startInput.value = Math.round(videoPreview.currentTime * 10) / 10;
            }
        });

        container.querySelector('#rbt-hook-mark-end')?.addEventListener('click', () => {
            const videoPreview = container.querySelector('#rbt-hook-preview-video');
            const endInput = container.querySelector('#rbt-hook-end');
            if (videoPreview && endInput && videoPreview.readyState >= 1) {
                endInput.value = Math.round(videoPreview.currentTime * 10) / 10;
            }
        });

        container.querySelector('#rbt-hook-cancel')?.addEventListener('click', () => {
            const videoPreview = container.querySelector('#rbt-hook-preview-video');
            if (videoPreview) {
                videoPreview.pause();
                videoPreview.removeAttribute('src');
                videoPreview.load();
            }
            hookModal.style.display = 'none';
        });
        container.querySelector('#rbt-hook-save')?.addEventListener('click', () => {
            const videoPreview = container.querySelector('#rbt-hook-preview-video');
            if (videoPreview) {
                videoPreview.pause();
                videoPreview.removeAttribute('src');
                videoPreview.load();
            }
            const idxStr = hookModal.dataset.editIdx;
            if (!idxStr) return;
            const idx = parseInt(idxStr);
            const hookFile = container.querySelector('#rbt-hook-path').value.trim();
            const startVal = container.querySelector('#rbt-hook-start').value;
            const endVal = container.querySelector('#rbt-hook-end').value;
            const hookSpeed = parseFloat(container.querySelector('#rbt-hook-speed').value) || 1;
            const hookTrans = container.querySelector('#rbt-hook-transition').value || 'none';
            const hookTransDur = parseFloat(container.querySelector('#rbt-hook-trans-dur').value) || 0.5;

            const applyToTask = (task) => {
                task.hookFile = hookFile;
                task.hookTrimStart = startVal !== '' ? parseFloat(startVal) : null;
                task.hookTrimEnd = endVal !== '' ? parseFloat(endVal) : null;
                task.hookSpeed = hookSpeed;
                task.hookTransition = hookTrans;
                task.hookTransDuration = hookTransDur;
            };

            if (idx >= 0) {
                applyToTask(window._reelsState.tasks[idx]);
            } else {
                const indices = _getSelectedIndices();
                indices.forEach(i => {
                    const task = window._reelsState.tasks[i];
                    if (task) applyToTask(task);
                });
            }
            hookModal.style.display = 'none';
            _renderBatchTable();
        });
    }

    // ══ Batch selection events ══
    container.querySelector('#rbt-select-all')?.addEventListener('change', (e) => {
        const tasks = window._reelsState.tasks || [];
        _batchTableState.selectedRows = new Set(e.target.checked ? tasks.map((_, i) => i) : []);
        container.querySelectorAll('.rbt-row-check').forEach(cb => cb.checked = e.target.checked);
        _updateBatchSelectCount();
    });
    container.querySelector('#rbt-invert-select')?.addEventListener('click', () => {
        const tasks = window._reelsState.tasks || [];
        const newSet = new Set();
        for (let i = 0; i < tasks.length; i++) {
            if (!_batchTableState.selectedRows.has(i)) newSet.add(i);
        }
        _batchTableState.selectedRows = newSet;
        container.querySelectorAll('.rbt-row-check').forEach(cb => {
            const idx = parseInt(cb.dataset.idx);
            cb.checked = newSet.has(idx);
        });
        _updateBatchSelectCount();
    });
    container.querySelector('#rbt-deselect-all')?.addEventListener('click', () => {
        _batchTableState.selectedRows = new Set();
        container.querySelectorAll('.rbt-row-check').forEach(cb => cb.checked = false);
        const selectAll = container.querySelector('#rbt-select-all');
        if (selectAll) selectAll.checked = false;
        _updateBatchSelectCount();
    });

    // ══ Batch template apply ══
    container.querySelector('#rbt-apply-batch-sub')?.addEventListener('click', () => {
        const val = container.querySelector('#rbt-batch-sub-tpl')?.value;
        if (!val) { alert('请先选择字幕模板'); return; }
        const indices = _getSelectedIndices();
        if (indices.length === 0) { alert('请先勾选需要批量设置的行'); return; }
        for (const idx of indices) {
            const task = window._reelsState.tasks[idx];
            if (task) task._subtitlePreset = val;
        }
        _renderBatchTable();
        alert(`✅ 已将字幕模板「${val}」应用到 ${indices.length} 行`);
    });
    container.querySelector('#rbt-apply-batch-card')?.addEventListener('click', () => {
        const val = container.querySelector('#rbt-batch-card-tpl')?.value;
        if (!val) { alert('请先选择覆层预设'); return; }
        const indices = _getSelectedIndices();
        if (indices.length === 0) { alert('请先勾选需要批量设置的行'); return; }
        for (const idx of indices) {
            const task = window._reelsState.tasks[idx];
            if (task) {
                task._overlayPresetName = val;
                _applyOverlayGroupPresetToTask(task, val);
            }
        }
        _renderBatchTable();
        alert(`✅ 已将覆层预设「${val}」应用到 ${indices.length} 行`);
    });

    // ══ Batch scale apply ══
    container.querySelector('#rbt-apply-batch-scale')?.addEventListener('click', () => {
        const indices = _getSelectedIndices();
        if (indices.length === 0) { alert('请先勾选需要批量设置的行'); return; }
        const bgScale = parseInt(container.querySelector('#rbt-batch-bgscale')?.value) || 100;
        const bgDurScale = parseInt(container.querySelector('#rbt-batch-bgdurscale')?.value) || 100;
        const audioDurScale = parseInt(container.querySelector('#rbt-batch-audiodurscale')?.value) || 100;
        const parts = [];
        for (const idx of indices) {
            const task = window._reelsState.tasks[idx];
            if (!task) continue;
            task.bgScale = bgScale;
            task.bgDurScale = bgDurScale;
            task.audioDurScale = audioDurScale;
        }
        if (bgScale !== 100) parts.push(`背景缩放${bgScale}%`);
        if (bgDurScale !== 100) parts.push(`背景时长${bgDurScale}%`);
        if (audioDurScale !== 100) parts.push(`音频时长${audioDurScale}%`);
        _renderBatchTable();
        alert(`✅ 已将缩放设置 ${parts.length > 0 ? '(' + parts.join(', ') + ')' : '(100%)'} 应用到 ${indices.length} 行`);
    });

    // Add row
    container.querySelector('#rbt-add-row-btn')?.addEventListener('click', () => {
        _batchAddEmptyRow();
        _renderBatchTable();
    });

    // Clear
    container.querySelector('#rbt-clear-btn')?.addEventListener('click', () => {
        if (confirm('确定清空所有任务？')) {
            window._reelsState.tasks = [];
            _renderBatchTable();
        }
    });

    // Column visibility settings
    container.querySelector('#rbt-col-settings-btn')?.addEventListener('click', (e) => {
        _showColumnSettingsPopup(e.target);
    });

    // 循环填充素材
    container.querySelector('#rbt-cycle-fill-btn')?.addEventListener('click', () => {
        _showCycleFillDialog();
    });

    // Paste
    container.querySelector('#rbt-paste-btn')?.addEventListener('click', () => {
        _batchPasteFromSheet();
    });
    container.querySelector('#rbt-paste-scroll-btn')?.addEventListener('click', () => {
        _batchPasteScrollFromSheet();
    });

    // Save config
    container.querySelector('#rbt-save-config-btn')?.addEventListener('click', () => {
        _batchExportConfig();
    });

    // Load config
    container.querySelector('#rbt-load-config-btn')?.addEventListener('click', () => {
        container.querySelector('#rbt-file-config').click();
    });
    container.querySelector('#rbt-file-config')?.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) _batchImportConfig(file);
        e.target.value = '';
    });

    // ── 读取文件夹 ──
    container.querySelector('#rbt-upload-folder')?.addEventListener('click', () => {
        container.querySelector('#rbt-file-folder').click();
    });
    container.querySelector('#rbt-file-folder')?.addEventListener('change', (e) => {
        _batchImportFolder(Array.from(e.target.files));
        e.target.value = '';
    });

    // ── 批量上传按钮 ──
    container.querySelector('#rbt-upload-bg')?.addEventListener('click', () => {
        container.querySelector('#rbt-file-bg').click();
    });
    container.querySelector('#rbt-upload-audio')?.addEventListener('click', () => {
        container.querySelector('#rbt-file-audio').click();
    });
    container.querySelector('#rbt-upload-srt')?.addEventListener('click', () => {
        container.querySelector('#rbt-file-srt').click();
    });
    container.querySelector('#rbt-upload-txt')?.addEventListener('click', () => {
        container.querySelector('#rbt-file-txt').click();
    });
    container.querySelector('#rbt-paste-txtcontent')?.addEventListener('click', () => {
        _batchPasteTxtContent();
    });
    container.querySelector('#rbt-align-all-btn')?.addEventListener('click', () => {
        _batchAlignAllTasks();
    });
    container.querySelector('#rbt-video-drop-route-enabled')?.addEventListener('change', (e) => {
        _batchTableState.videoDropRouteEnabled = !!e.target.checked;
        const modeEl = container.querySelector('#rbt-video-drop-route-mode');
        if (modeEl) {
            modeEl.disabled = !_batchTableState.videoDropRouteEnabled;
            modeEl.style.opacity = _batchTableState.videoDropRouteEnabled ? '' : '.6';
        }
    });
    container.querySelector('#rbt-video-drop-route-mode')?.addEventListener('change', (e) => {
        _batchTableState.videoDropRouteMode = e.target.value === 'audio' ? 'audio' : 'bg';
    });

    // ── 批量文件选完后按顺序分配到各行 ──
    container.querySelector('#rbt-file-bg')?.addEventListener('change', (e) => {
        _batchAssignFiles(Array.from(e.target.files), 'bg');
        e.target.value = '';
    });
    container.querySelector('#rbt-file-audio')?.addEventListener('change', (e) => {
        _batchAssignFiles(Array.from(e.target.files), 'audio');
        e.target.value = '';
    });
    container.querySelector('#rbt-file-srt')?.addEventListener('change', (e) => {
        _batchAssignFiles(Array.from(e.target.files), 'srt');
        e.target.value = '';
    });
    container.querySelector('#rbt-file-txt')?.addEventListener('change', (e) => {
        _batchAssignTxtFiles(Array.from(e.target.files));
        e.target.value = '';
    });

    // ── 配乐批量设置 ──
    container.querySelector('#rbt-set-bgm-btn')?.addEventListener('click', () => {
        // 点击“设置配乐”按钮，先选文件，然后应用到勾选的行
        _batchTableState._bgmBatchMode = true;
        container.querySelector('#rbt-file-bgm').click();
    });
    container.querySelector('#rbt-file-bgm')?.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const filePath = (typeof getFileNativePath === 'function') ? getFileNativePath(file) : (file.path || file.name);
        const state = window._reelsState;

        if (_batchTableState._bgmBatchMode) {
            // 批量模式：应用到所有勾选的行
            const checks = container.querySelectorAll('.rbt-bgm-check:checked');
            let targets;
            if (checks.length > 0) {
                targets = Array.from(checks).map(c => state.tasks[parseInt(c.dataset.idx)]).filter(Boolean);
            } else {
                // 没有勾选任何行，应用到全部
                targets = state.tasks;
            }
            for (const t of targets) {
                t.bgmPath = filePath;
            }
            _renderBatchTable();
            if (typeof _renderTaskList === 'function') _renderTaskList();
            alert(`✅ 已将配乐设置到 ${targets.length} 个任务`);
        } else if (_batchTableState._bgmSingleIdx != null) {
            // 单行模式
            const task = state.tasks[_batchTableState._bgmSingleIdx];
            if (task) {
                task.bgmPath = filePath;
                _renderBatchTable();
                if (typeof _renderTaskList === 'function') _renderTaskList();
            }
        }
        _batchTableState._bgmBatchMode = false;
        _batchTableState._bgmSingleIdx = null;
        e.target.value = '';
    });

    // ── 单行文件选择 ──
    container.querySelector('#rbt-file-single')?.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const { idx, field } = _batchTableState._pendingSingle || {};
        if (idx != null && field) {
            _assignSingleFile(idx, field, file);
        }
        _batchTableState._pendingSingle = null;
        e.target.value = '';
    });

    // ── 拖拽文件到批量表格 ──
    const panel = container.querySelector('.rbt-panel') || container;
    let _dropOverlay = null;
    const _showDropOverlay = () => {
        if (_dropOverlay) return;
        _dropOverlay = document.createElement('div');
        _dropOverlay.style.cssText = 'position:absolute;inset:0;z-index:99;background:rgba(80,140,255,0.15);border:3px dashed #4a9eff;border-radius:12px;display:flex;align-items:center;justify-content:center;pointer-events:none;';
        _dropOverlay.innerHTML = '<span style="font-size:24px;color:#4a9eff;font-weight:700;text-shadow:0 2px 8px rgba(0,0,0,0.5);">📂 拖放文件到这里</span>';
        panel.style.position = 'relative';
        panel.appendChild(_dropOverlay);
    };
    const _hideDropOverlay = () => {
        if (_dropOverlay) { _dropOverlay.remove(); _dropOverlay = null; }
    };
    let _dragCounter = 0;
    panel.addEventListener('dragenter', (e) => { e.preventDefault(); _dragCounter++; _showDropOverlay(); });
    panel.addEventListener('dragleave', (e) => { e.preventDefault(); _dragCounter--; if (_dragCounter <= 0) { _dragCounter = 0; _hideDropOverlay(); } });
    panel.addEventListener('dragover', (e) => {
        e.preventDefault();
        // 内部行拖拽时不覆盖 dropEffect
        if (_dragSrcIdx == null) e.dataTransfer.dropEffect = 'copy';
    });
    panel.addEventListener('drop', (e) => {
        _dragCounter = 0;
        _hideDropOverlay();
        // 如果是内部行拖拽排序（无文件），不拦截，让 tbody 的 drop 处理
        const files = Array.from(e.dataTransfer.files || []);
        if (!files.length) return;
        e.preventDefault();

        // 按扩展名分类
        const bgExts = new Set(['mp4', 'mov', 'mkv', 'avi', 'wmv', 'flv', 'webm', 'jpg', 'jpeg', 'png', 'webp']);
        const audioExts = new Set(['mp3', 'wav', 'm4a', 'aac', 'flac', 'ogg', 'wma']);
        const groups = { hook: [], bg: [], audio: [], srt: [], txt: [] };
        const forcedVideoRoute = _batchTableState.videoDropRouteEnabled
            ? _batchTableState.videoDropRouteMode
            : null;

        for (const f of files) {
            const ext = (f.name || '').split('.').pop().toLowerCase();
            if (ext === 'srt') groups.srt.push(f);
            else if (ext === 'txt') groups.txt.push(f);
            else if (audioExts.has(ext)) groups.audio.push(f);
            else if (_VOICE_VIDEO_EXTS.has(ext) && forcedVideoRoute === 'audio') groups.audio.push(f);
            else if (_VOICE_VIDEO_EXTS.has(ext) && forcedVideoRoute === 'bg') groups.bg.push(f);
            else if (_VOICE_VIDEO_EXTS.has(ext) && forcedVideoRoute === 'hook') groups.hook.push(f);
            else if (_VOICE_VIDEO_EXTS.has(ext) && _looksLikeVoiceTrack(f.name || '')) groups.audio.push(f);
            else if (bgExts.has(ext)) {
                if (forcedVideoRoute === 'hook') groups.hook.push(f);
                else groups.bg.push(f);
            }
        }

        // 按类型依次分配（最多的类型先处理，其他自动填充）
        const assignments = [];
        if (groups.hook.length) assignments.push(['hook', groups.hook]);
        if (groups.bg.length) assignments.push(['bg', groups.bg]);
        if (groups.audio.length) assignments.push(['audio', groups.audio]);
        if (groups.srt.length) assignments.push(['srt', groups.srt]);
        if (groups.txt.length) assignments.push(['txt', groups.txt]);

        if (assignments.length === 0) {
            alert('未识别到支持的文件类型');
            return;
        }

        // 依次批量分配（异步串行）
        (async () => {
            for (const [field, fieldFiles] of assignments) {
                await _batchAssignFiles(fieldFiles, field);
            }
        })();
    });

    // ── 清空覆层文案 ──
    container.querySelector('#rbt-clear-overlay-btn')?.addEventListener('click', () => {
        const state = window._reelsState;
        if (!state || !state.tasks) return;
        let indices = _getSelectedIndices();
        const targetLabel = indices.length > 0 ? `勾选的 ${indices.length} 行` : '所有行';
        if (indices.length === 0) {
            indices = state.tasks.map((_, i) => i);
        }
        if (!confirm(`确定清空${targetLabel}的覆层文案（标题+内容+结尾）？`)) return;
        for (const idx of indices) {
            const task = state.tasks[idx];
            if (!task) continue;
            if (task.overlays && task.overlays.length > 0) {
                for (const ov of task.overlays) {
                    ov.title_text = '';
                    ov.body_text = '';
                    ov.footer_text = '';
                    ov.scroll_title = '';
                    ov.content = '';
                }
            }
        }
        _renderBatchTable();
        if (typeof showToast === 'function') showToast(`✅ 已清空${targetLabel}的覆层文案`, 'success');
    });

    // 仅导出选中行（表格数据不变）
    container.querySelector('#rbt-export-selected-btn')?.addEventListener('click', () => {
        const selected = _batchTableState.selectedRows;
        if (selected.size === 0) return;
        _applyBatchTableChanges(); // 先同步输入框到 tasks
        const allTasks = window._reelsState.tasks;
        const exported = [...selected].sort((a, b) => a - b).map(i => allTasks[i]).filter(Boolean);
        // 临时替换 tasks 为选中行，关闭表格
        const backup = allTasks.slice();
        window._reelsState.tasks = exported;
        window._reelsState.selectedIdx = 0;
        reelsToggleBatchTable();
        if (typeof _renderTaskList === 'function') _renderTaskList();
        // 恢复完整 tasks 到 tab（表格下次打开还是全部）
        const tab = _getActiveTab();
        if (tab) tab.tasks = _serializeTasks(backup);
        if (typeof showToast === 'function') showToast(`📤 已导出 ${exported.length} 行到主面板（表格数据不变）`, 'success');
    });

    // ── 拖拽排序 ──
    _dragSrcIdx = null;
    let _dragMouseTarget = null; // 追踪实际点击的元素
    const tbody2 = container.querySelector('#rbt-tbody');
    if (tbody2) {
        // 记录鼠标按下的实际元素，dragstart时用它判断是否从手柄发起
        tbody2.addEventListener('mousedown', (e) => {
            _dragMouseTarget = e.target;
        });
        tbody2.addEventListener('dragstart', (e) => {
            const row = e.target.closest('.rbt-row');
            if (!row) { e.preventDefault(); return; }
            // Only allow drag from the handle
            if (!_dragMouseTarget || !_dragMouseTarget.closest('.rbt-drag-handle')) { e.preventDefault(); return; }
            _dragSrcIdx = parseInt(row.dataset.idx);
            row.classList.add('rbt-dragging');
            e.dataTransfer.effectAllowed = 'move';
            e.dataTransfer.setData('text/plain', String(_dragSrcIdx));
        });
        tbody2.addEventListener('dragend', (e) => {
            const row = e.target.closest('.rbt-row');
            if (row) row.classList.remove('rbt-dragging');
            tbody2.querySelectorAll('.rbt-drag-over-row').forEach(r => r.classList.remove('rbt-drag-over-row'));
            _dragSrcIdx = null;
        });
        tbody2.addEventListener('dragover', (e) => {
            if (_dragSrcIdx == null) return;
            e.preventDefault();
            e.dataTransfer.dropEffect = 'move';
            const row = e.target.closest('.rbt-row');
            tbody2.querySelectorAll('.rbt-drag-over-row').forEach(r => r.classList.remove('rbt-drag-over-row'));
            if (row) row.classList.add('rbt-drag-over-row');
        });
        tbody2.addEventListener('dragleave', (e) => {
            const row = e.target.closest('.rbt-row');
            if (row) row.classList.remove('rbt-drag-over-row');
        });
        tbody2.addEventListener('drop', (e) => {
            e.preventDefault();
            const row = e.target.closest('.rbt-row');
            if (!row || _dragSrcIdx == null) return;
            const dstIdx = parseInt(row.dataset.idx);
            if (dstIdx === _dragSrcIdx) return;
            const tasks = window._reelsState.tasks;
            if (!tasks) return;
            const [moved] = tasks.splice(_dragSrcIdx, 1);
            tasks.splice(dstIdx, 0, moved);
            _batchTableState.selectedRows = new Set();
            _dragSrcIdx = null;
            // 保存滚动位置，渲染后恢复
            const scrollWrap = container.querySelector('.rbt-table-wrap');
            const scrollTop = scrollWrap ? scrollWrap.scrollTop : 0;
            _renderBatchTable();
            const scrollWrap2 = container.querySelector('.rbt-table-wrap');
            if (scrollWrap2) scrollWrap2.scrollTop = scrollTop;
            if (typeof showToast === 'function') showToast(`已移动到第 ${dstIdx + 1} 行`, 'info');
        });
    }

    // Per-row events (delegated)
    const tbody = container.querySelector('#rbt-tbody');
    if (tbody) {
        // 音量滑块实时更新标签
        tbody.addEventListener('input', (e) => {
            if (e.target.classList.contains('rbt-bgm-vol')) {
                const label = e.target.parentElement.querySelector('.rbt-bgm-vol-label');
                if (label) label.textContent = e.target.value + '%';
                // Sync volume to task for real-time preview
                const idx = parseInt(e.target.dataset.idx);
                const task = window._reelsState && window._reelsState.tasks[idx];
                if (task) {
                    task.bgmVolume = parseInt(e.target.value) || 0;
                    // If this is the selected task, update preview audio immediately
                    if (idx === window._reelsState.selectedIdx && typeof _applyPreviewAudioMix === 'function') {
                        _applyPreviewAudioMix();
                    }
                }
            }
            // ── 背景缩放 slider↔input 同步 ──
            if (e.target.classList.contains('rbt-bgscale-slider')) {
                const numInput = e.target.parentElement.querySelector('.rbt-bgscale-input');
                if (numInput) numInput.value = e.target.value;
                const idx = parseInt(e.target.dataset.idx);
                const task = window._reelsState && window._reelsState.tasks[idx];
                if (task) { task.bgScale = parseInt(e.target.value) || 100; console.log(`[Scale] row${idx} bgScale=${task.bgScale}`); }
            }
            if (e.target.classList.contains('rbt-bgscale-input')) {
                const slider = e.target.closest('td').querySelector('.rbt-bgscale-slider');
                if (slider) slider.value = e.target.value;
                const idx = parseInt(e.target.dataset.idx);
                const task = window._reelsState && window._reelsState.tasks[idx];
                if (task) { task.bgScale = parseInt(e.target.value) || 100; console.log(`[Scale] row${idx} bgScale=${task.bgScale}`); }
            }
            // ── 背景时长缩放 slider↔input 同步 ──
            if (e.target.classList.contains('rbt-bgdurscale-slider')) {
                const numInput = e.target.parentElement.querySelector('.rbt-bgdurscale-input');
                if (numInput) numInput.value = e.target.value;
                const idx = parseInt(e.target.dataset.idx);
                const task = window._reelsState && window._reelsState.tasks[idx];
                if (task) { task.bgDurScale = parseInt(e.target.value) || 100; console.log(`[Scale] row${idx} bgDurScale=${task.bgDurScale}`); }
            }
            if (e.target.classList.contains('rbt-bgdurscale-input')) {
                const slider = e.target.closest('td').querySelector('.rbt-bgdurscale-slider');
                if (slider) slider.value = e.target.value;
                const idx = parseInt(e.target.dataset.idx);
                const task = window._reelsState && window._reelsState.tasks[idx];
                if (task) { task.bgDurScale = parseInt(e.target.value) || 100; console.log(`[Scale] row${idx} bgDurScale=${task.bgDurScale}`); }
            }
            // ── 音频变速 slider↔input 同步 ──
            if (e.target.classList.contains('rbt-audiodurscale-slider')) {
                const numInput = e.target.parentElement.querySelector('.rbt-audiodurscale-input');
                if (numInput) numInput.value = e.target.value;
                const idx = parseInt(e.target.dataset.idx);
                const task = window._reelsState && window._reelsState.tasks[idx];
                if (task) { task.audioDurScale = parseInt(e.target.value) || 100; console.log(`[Scale] row${idx} audioDurScale=${task.audioDurScale}`); }
            }
            if (e.target.classList.contains('rbt-audiodurscale-input')) {
                const slider = e.target.closest('td').querySelector('.rbt-audiodurscale-slider');
                if (slider) slider.value = e.target.value;
                const idx = parseInt(e.target.dataset.idx);
                const task = window._reelsState && window._reelsState.tasks[idx];
                if (task) { task.audioDurScale = parseInt(e.target.value) || 100; console.log(`[Scale] row${idx} audioDurScale=${task.audioDurScale}`); }
            }
        });
        // Row checkbox change handler
        tbody.addEventListener('change', (e) => {
            if (e.target.classList.contains('rbt-row-check')) {
                const idx = parseInt(e.target.dataset.idx);
                if (e.target.checked) {
                    _batchTableState.selectedRows.add(idx);
                } else {
                    _batchTableState.selectedRows.delete(idx);
                }
                _updateBatchSelectCount();
            }
        });
        tbody.addEventListener('click', (e) => {
            // 预览按钮
            const selectBtn = e.target.closest('.rbt-select-btn');
            if (selectBtn) {
                const idx = parseInt(selectBtn.dataset.idx);
                _applyBatchTableChanges();
                reelsToggleBatchTable();
                if (typeof reelsSelectTask === 'function') reelsSelectTask(idx);
                return;
            }
            // 复制行按钮
            const cloneBtn = e.target.closest('.rbt-clone-btn');
            if (cloneBtn) {
                const idx = parseInt(cloneBtn.dataset.idx);
                const src = window._reelsState.tasks[idx];
                if (src) {
                    const copy = JSON.parse(JSON.stringify(src));
                    copy.baseName = (copy.baseName || 'copy') + '_copy';
                    copy.fileName = copy.baseName + '.mp4';
                    // 清掉不可序列化的引用
                    delete copy._video;
                    delete copy._bgThumb;
                    window._reelsState.tasks.splice(idx + 1, 0, copy);
                    _renderBatchTable();
                }
                return;
            }
            // 删除按钮
            const delBtn = e.target.closest('.rbt-delete-btn');
            if (delBtn) {
                const idx = parseInt(delBtn.dataset.idx);
                window._reelsState.tasks.splice(idx, 1);
                _renderBatchTable();
                return;
            }
            // 单字段清除按钮
            const clearBtn = e.target.closest('.rbt-field-clear');
            if (clearBtn) {
                e.stopPropagation();
                const idx = parseInt(clearBtn.dataset.idx);
                const field = clearBtn.dataset.field;
                const task = window._reelsState.tasks[idx];
                if (!task) return;
                switch (field) {
                    case 'hook': 
                        task.hookFile = ''; 
                        task.hookSpeed = 1; 
                        task.hookTransition = 'none'; 
                        break;
                    case 'bg': task.bgPath = ''; task.videoPath = ''; task.bgSrcUrl = ''; break;
                    case 'audio': task.audioPath = ''; break;
                    case 'srt': task.srtPath = ''; task.aligned = false; break;
                    case 'txt': task.txtContent = ''; task.aligned = false; break;
                    case 'bgm': task.bgmPath = ''; break;
                }
                _renderBatchTable();
                return;
            }
        });
        // 双击放大编辑 及 双击文件单元格
        tbody.addEventListener('dblclick', (e) => {
            if (e.target.tagName === 'TEXTAREA' && e.target.classList.contains('rbt-textarea')) {
                _showTextEditorModal(e.target);
                return;
            }
            const hookSet = e.target.closest('.rbt-hook-set');
            if (hookSet) {
                const row = hookSet.closest('.rbt-row');
                if (row) {
                    _openHookModal(parseInt(row.dataset.idx));
                }
                return;
            }
            // 双击文件单元格 → 选择文件
            const cell = e.target.closest('.rbt-droppable');
            if (cell && cell.dataset.field !== 'hook') {
                const row = cell.closest('.rbt-row');
                if (!row) return;
                const idx = parseInt(row.dataset.idx);
                const field = cell.dataset.field;
                _pickSingleFile(idx, field);
                return;
            }
            // 双击配乐单元格 → 选择配乐文件
            const bgmPick = e.target.closest('.rbt-bgm-pick');
            if (bgmPick) {
                const idx = parseInt(bgmPick.dataset.idx);
                _batchTableState._bgmBatchMode = false;
                _batchTableState._bgmSingleIdx = idx;
                container.querySelector('#rbt-file-bgm').click();
                return;
            }
        });
    }
}

// ═══════════════════════════════════════════════════════
// 5.5 Text Editor Modal
// ═══════════════════════════════════════════════════════

function _showTextEditorModal(textareaEl) {
    let editorWrap = document.getElementById('rbt-text-editor-modal');
    if (!editorWrap) {
        editorWrap = document.createElement('div');
        editorWrap.id = 'rbt-text-editor-modal';
        editorWrap.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;z-index:99999;background:rgba(0,0,0,0.85);display:none;align-items:center;justify-content:center;backdrop-filter:blur(4px);';
        
        const modalContent = document.createElement('div');
        modalContent.style.cssText = 'width:80%;max-width:800px;background:#1a1a2e;border:1px solid #4a6a8a;border-radius:12px;padding:20px;display:flex;flex-direction:column;gap:16px;box-shadow:0 12px 32px rgba(0,0,0,0.8);';
        
        const header = document.createElement('div');
        header.style.cssText = 'display:flex;justify-content:space-between;align-items:center;';
        
        const title = document.createElement('h3');
        title.id = 'rbt-text-editor-title';
        title.style.cssText = 'margin:0;color:#00D4FF;font-size:16px;';
        title.innerText = '大屏幕文案编辑';
        
        const closeBtn = document.createElement('button');
        closeBtn.innerHTML = '✕';
        closeBtn.style.cssText = 'background:transparent;border:none;color:#aaa;font-size:18px;cursor:pointer;padding:4px;';
        closeBtn.onclick = () => { editorWrap.style.display = 'none'; };
        
        header.appendChild(title);
        header.appendChild(closeBtn);
        
        const ta = document.createElement('textarea');
        ta.id = 'rbt-text-editor-textarea';
        ta.style.cssText = 'width:100%;height:450px;background:#111;color:#eee;border:1px solid #333;border-radius:6px;padding:12px;font-size:14px;line-height:1.6;box-sizing:border-box;font-family:monospace;resize:vertical;';
        
        const footer = document.createElement('div');
        footer.style.cssText = 'display:flex;justify-content:flex-end;gap:12px;align-items:center;';
        
        const hint = document.createElement('span');
        hint.style.cssText = 'color:#666;font-size:12px;flex:1;';
        hint.innerText = '小提示：按 Esc 键可退出，Ctrl+Enter 可快速保存';
        
        const cancelBtn = document.createElement('button');
        cancelBtn.innerText = '取消 (Esc)';
        cancelBtn.style.cssText = 'padding:8px 20px;background:#333;color:#ccc;border:1px solid #444;border-radius:6px;cursor:pointer;font-size:13px;';
        cancelBtn.onclick = () => { editorWrap.style.display = 'none'; };
        
        const saveBtn = document.createElement('button');
        saveBtn.innerText = '保存内容 (Ctrl+Enter)';
        saveBtn.id = 'rbt-text-editor-save';
        saveBtn.style.cssText = 'padding:8px 24px;background:#00D4FF;color:#000;border:none;border-radius:6px;cursor:pointer;font-size:13px;font-weight:bold;';
        
        footer.appendChild(hint);
        footer.appendChild(cancelBtn);
        footer.appendChild(saveBtn);
        
        modalContent.appendChild(header);
        modalContent.appendChild(ta);
        modalContent.appendChild(footer);
        editorWrap.appendChild(modalContent);
        document.body.appendChild(editorWrap);
        
        ta.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                e.preventDefault();
                editorWrap.style.display = 'none';
            }
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                saveBtn.click();
            }
        });
        
        // Block clicks from closing modal if clicking inside model content
        modalContent.addEventListener('click', (e) => e.stopPropagation());
        // Close modal if clicking outside
        editorWrap.addEventListener('click', () => { editorWrap.style.display = 'none'; });
    }

    const modalTitle = document.getElementById('rbt-text-editor-title');
    const modalTa = document.getElementById('rbt-text-editor-textarea');
    const saveBtn = document.getElementById('rbt-text-editor-save');
    
    let titleStr = "大屏幕文案编辑";
    if (textareaEl.classList.contains('rbt-txtcontent-input')) titleStr = "编辑【字幕文本】";
    else if (textareaEl.classList.contains('rbt-title-input')) titleStr = "编辑【覆层标题】";
    else if (textareaEl.classList.contains('rbt-body-input')) titleStr = "编辑【覆层内容】";
    else if (textareaEl.classList.contains('rbt-footer-input')) titleStr = "编辑【覆层结尾】";
    else if (textareaEl.classList.contains('rbt-scroll-title-input')) titleStr = "编辑【滚动标题】";
    else if (textareaEl.classList.contains('rbt-scroll-body-input')) titleStr = "编辑【滚动内容】";
    modalTitle.innerText = titleStr;
    
    modalTa.value = textareaEl.value;
    editorWrap.style.display = 'flex';
    modalTa.focus();
    // highlight text gently
    modalTa.setSelectionRange(modalTa.value.length, modalTa.value.length);
    
    saveBtn.onclick = () => {
        textareaEl.value = modalTa.value;
        // manually dispatch events so batch system picks it up immediately
        textareaEl.dispatchEvent(new Event('input', { bubbles: true }));
        textareaEl.dispatchEvent(new Event('change', { bubbles: true }));
        editorWrap.style.display = 'none';
    };
}

// ═══════════════════════════════════════════════════════
// 6. Apply changes from table → tasks
// ═══════════════════════════════════════════════════════

function _applyBatchTableChanges() {
    const container = _batchTableState.container;
    if (!container) return;
    const state = window._reelsState;
    if (!state) return;

    const titleInputs = container.querySelectorAll('.rbt-title-input');
    const bodyInputs = container.querySelectorAll('.rbt-body-input');
    const footerInputs = container.querySelectorAll('.rbt-footer-input');
    const subTplSelects = container.querySelectorAll('.rbt-sub-tpl-select');
    const cardTplSelects = container.querySelectorAll('.rbt-card-tpl-select');

    titleInputs.forEach(el => {
        const idx = parseInt(el.dataset.idx);
        const task = state.tasks[idx];
        if (!task) return;
        if (!task.overlays || !task.overlays[0]) {
            // Only create overlay if user actually typed something
            const titleVal = el.value.trim();
            if (!titleVal) return;
            const ReelsOverlay = window.ReelsOverlay;
            if (ReelsOverlay) {
                task.overlays = [ReelsOverlay.createTextCardOverlay({
                    title_text: titleVal,
                    body_text: '',
                    start: 0, end: 9999,
                })];
            }
        } else {
            task.overlays[0].title_text = el.value;
        }
    });

    bodyInputs.forEach(el => {
        const idx = parseInt(el.dataset.idx);
        const task = state.tasks[idx];
        if (!task || !task.overlays || !task.overlays[0]) return;
        task.overlays[0].body_text = el.value;
    });

    footerInputs.forEach(el => {
        const idx = parseInt(el.dataset.idx);
        const task = state.tasks[idx];
        if (!task || !task.overlays || !task.overlays[0]) return;
        task.overlays[0].footer_text = el.value;
    });

    // ── 滚动字幕列 ──
    const scrollTitleInputs = container.querySelectorAll('.rbt-scroll-title-input');
    const scrollBodyInputs = container.querySelectorAll('.rbt-scroll-body-input');

    scrollTitleInputs.forEach(el => {
        const idx = parseInt(el.dataset.idx);
        const task = state.tasks[idx];
        if (!task) return;
        const val = el.value.trim();
        let scrollOv = (task.overlays || []).find(o => o.type === 'scroll');
        if (!scrollOv && val) {
            const ReelsOverlay = window.ReelsOverlay;
            if (ReelsOverlay) {
                scrollOv = ReelsOverlay.createScrollOverlay({
                    scroll_title: val, content: '', start: 0, end: 9999,
                });
                if (!task.overlays) task.overlays = [];
                task.overlays.push(scrollOv);
            }
        } else if (scrollOv) {
            scrollOv.scroll_title = el.value;
        }
    });

    scrollBodyInputs.forEach(el => {
        const idx = parseInt(el.dataset.idx);
        const task = state.tasks[idx];
        if (!task) return;
        const val = el.value.trim();
        let scrollOv = (task.overlays || []).find(o => o.type === 'scroll');
        if (!scrollOv && val) {
            const ReelsOverlay = window.ReelsOverlay;
            if (ReelsOverlay) {
                scrollOv = ReelsOverlay.createScrollOverlay({
                    scroll_title: '', content: val, start: 0, end: 9999,
                });
                if (!task.overlays) task.overlays = [];
                task.overlays.push(scrollOv);
            }
        } else if (scrollOv) {
            scrollOv.content = el.value;
        }
    });

    // 自定义时长
    const durInputs = container.querySelectorAll('.rbt-dur-input');
    durInputs.forEach(el => {
        const idx = parseInt(el.dataset.idx);
        const task = state.tasks[idx];
        if (!task) return;
        const v = parseFloat(el.value) || 0;
        task.customDuration = v > 0 ? v : 0;
    });

    // 背景缩放
    container.querySelectorAll('.rbt-bgscale-input').forEach(el => {
        const idx = parseInt(el.dataset.idx);
        const task = state.tasks[idx];
        if (!task) return;
        task.bgScale = parseInt(el.value) || 100;
    });

    // 背景时长缩放
    container.querySelectorAll('.rbt-bgdurscale-input').forEach(el => {
        const idx = parseInt(el.dataset.idx);
        const task = state.tasks[idx];
        if (!task) return;
        task.bgDurScale = parseInt(el.value) || 100;
    });

    // 音频时长缩放
    container.querySelectorAll('.rbt-audiodurscale-input').forEach(el => {
        const idx = parseInt(el.dataset.idx);
        const task = state.tasks[idx];
        if (!task) return;
        task.audioDurScale = parseInt(el.value) || 100;
    });

    // 字幕文本（用于字幕对齐）
    const txtContentInputs = container.querySelectorAll('.rbt-txtcontent-input');
    txtContentInputs.forEach(el => {
        const idx = parseInt(el.dataset.idx);
        const task = state.tasks[idx];
        if (!task) return;
        const newContent = el.value.trim();
        if (newContent !== (task.txtContent || '').trim()) {
            task.txtContent = newContent;
            task.aligned = false; // 内容变了重置对齐状态
        }
    });

    // 字幕模板
    subTplSelects.forEach(el => {
        const idx = parseInt(el.dataset.idx);
        const task = state.tasks[idx];
        if (!task) return;
        task._subtitlePreset = el.value || '';
    });

    // 覆盖层组预设
    cardTplSelects.forEach(el => {
        const idx = parseInt(el.dataset.idx);
        const task = state.tasks[idx];
        if (!task) return;
        const presetName = el.value;
        if (presetName) {
            _applyOverlayGroupPresetToTask(task, presetName);
        }
    });

    // 配乐音量
    const bgmVols = container.querySelectorAll('.rbt-bgm-vol');
    bgmVols.forEach(el => {
        const idx = parseInt(el.dataset.idx);
        const task = state.tasks[idx];
        if (!task) return;
        task.bgmVolume = parseInt(el.value) || 0;
    });

    // 刷新任务列表
    if (typeof _renderTaskList === 'function') _renderTaskList();

    // 同步当前选中任务的覆层到 overlayMgr
    const selIdx = state.selectedIdx;
    const selTask = state.tasks[selIdx];
    if (selTask && state.overlayProxy && state.overlayProxy.overlayMgr) {
        const mgr = state.overlayProxy.overlayMgr;
        mgr.overlays = selTask.overlays ? [...selTask.overlays] : [];
        if (state.overlayPanel) {
            state.overlayPanel.deselectOverlay();
            state.overlayPanel._refreshList();
        }
    }
}

// ═══════════════════════════════════════════════════════
// 7. Paste from Google Sheets
// ═══════════════════════════════════════════════════════

async function _batchPasteFromSheet() {
    // ── 第一步：让用户选模式 ──
    const mode = await _showPasteModeDialog();
    if (!mode) return;

    // ── 第二步：获取文本 ──
    let raw = '';
    try {
        raw = await navigator.clipboard.readText();
    } catch (e) {
        raw = await _showPasteDialog();
    }
    if (!raw || !raw.trim()) return;

    const rows = _parseBatchTSV(raw);
    if (!rows.length) {
        alert('未检测到有效数据');
        return;
    }

    const state = window._reelsState;
    const ReelsOverlay = window.ReelsOverlay;
    if (!state || !ReelsOverlay) return;

    // 解析每行数据
    const entries = [];
    for (const row of rows) {
        let title = '', body = '';
        if (row.length >= 3) {
            // 三列格式：名称 | 标题 | 内容（名称暂不用于此版本）
            title = row[1] || '';
            body = row[2] || '';
        } else {
            title = row[0] || '';
            body = row[1] || '';
        }
        if (!title && !body) continue;
        entries.push({ title, body });
    }

    if (!entries.length) {
        alert('未检测到有效文案数据');
        return;
    }

    let filled = 0, created = 0;

    if (mode === 'fill') {
        // ═══ 补全模式：只填没有文案的空行 ═══
        let entryIdx = 0;
        for (let i = 0; i < state.tasks.length && entryIdx < entries.length; i++) {
            const task = state.tasks[i];
            const ov = (task.overlays && task.overlays.length > 0) ? task.overlays[0] : null;
            // 判断是不是空文案（title 和 body 都空，或者是默认占位符）
            const titleVal = ov ? (ov.title_text || '') : '';
            const bodyVal = ov ? (ov.body_text || '') : '';
            const isDefault = (t) => !t || t === '标题文字' || t === '内容文字';
            const isEmpty = isDefault(titleVal) && isDefault(bodyVal);

            if (isEmpty) {
                _setTaskText(task, entries[entryIdx].title, entries[entryIdx].body, ReelsOverlay);
                entryIdx++;
                filled++;
            }
        }
        // 剩余的追加为新行
        while (entryIdx < entries.length) {
            _createNewTextRow(state, entries[entryIdx].title, entries[entryIdx].body, ReelsOverlay);
            entryIdx++;
            created++;
        }

    } else if (mode === 'new') {
        // ═══ 添加新行模式：全部新建 ═══
        for (const entry of entries) {
            _createNewTextRow(state, entry.title, entry.body, ReelsOverlay);
            created++;
        }

    } else if (mode === 'overwrite') {
        // ═══ 覆盖模式：从第1行开始往下覆盖 ═══
        for (let i = 0; i < entries.length; i++) {
            if (i < state.tasks.length) {
                _setTaskText(state.tasks[i], entries[i].title, entries[i].body, ReelsOverlay);
                filled++;
            } else {
                _createNewTextRow(state, entries[i].title, entries[i].body, ReelsOverlay);
                created++;
            }
        }
    }

    _renderBatchTable();
    if (typeof _renderTaskList === 'function') _renderTaskList();

    const parts = [];
    if (filled) parts.push(`填充 ${filled} 行`);
    if (created) parts.push(`新建 ${created} 行`);
    alert(`✅ 粘贴完成：${parts.join('，')}`);
}

async function _batchPasteScrollFromSheet() {
    const mode = await _showPasteModeDialog();
    if (!mode) return;

    let raw = '';
    try {
        raw = await navigator.clipboard.readText();
    } catch (e) {
        raw = await _showPasteDialog();
    }
    if (!raw || !raw.trim()) return;

    const rows = _parseBatchTSV(raw);
    if (!rows.length) {
        alert('未检测到有效数据');
        return;
    }

    const state = window._reelsState;
    const ReelsOverlay = window.ReelsOverlay;
    if (!state || !ReelsOverlay) return;

    // 解析每行数据：3列(名, 标题, 内容), 2列(标题, 内容), 1列(内容)
    const entries = [];
    for (const row of rows) {
        let title = '', body = '';
        if (row.length >= 3) {
            title = row[1] || '';
            body = row[2] || '';
        } else if (row.length === 2) {
            title = row[0] || '';
            body = row[1] || '';
        } else {
            body = row[0] || '';
        }
        if (!title && !body) continue;
        entries.push({ title, body });
    }

    if (!entries.length) {
        alert('未检测到有效滚动字幕数据');
        return;
    }

    let filled = 0, created = 0;

    if (mode === 'fill') {
        let entryIdx = 0;
        for (let i = 0; i < state.tasks.length && entryIdx < entries.length; i++) {
            const task = state.tasks[i];
            const scrollOv = (task.overlays || []).find(o => o.type === 'scroll');
            const isEmpty = !scrollOv || (!(scrollOv.scroll_title || '').trim() && !(scrollOv.content || '').trim());
            if (isEmpty) {
                _setTaskScrollText(task, entries[entryIdx].title, entries[entryIdx].body, ReelsOverlay);
                entryIdx++;
                filled++;
            }
        }
        while (entryIdx < entries.length) {
            _createNewScrollRow(state, entries[entryIdx].title, entries[entryIdx].body, ReelsOverlay);
            entryIdx++;
            created++;
        }
    } else if (mode === 'new') {
        for (const entry of entries) {
            _createNewScrollRow(state, entry.title, entry.body, ReelsOverlay);
            created++;
        }
    } else if (mode === 'overwrite') {
        for (let i = 0; i < entries.length; i++) {
            if (i < state.tasks.length) {
                _setTaskScrollText(state.tasks[i], entries[i].title, entries[i].body, ReelsOverlay);
                filled++;
            } else {
                _createNewScrollRow(state, entries[i].title, entries[i].body, ReelsOverlay);
                created++;
            }
        }
    }

    _renderBatchTable();
    if (typeof _renderTaskList === 'function') _renderTaskList();

    const parts = [];
    if (filled) parts.push(`填充 ${filled} 行`);
    if (created) parts.push(`新建 ${created} 行`);
    alert(`✅ 滚动字幕粘贴完成：${parts.join('，')}`);
}

// ═══════════════════════════════════════════════════════
// Column visibility settings (per tab, saved to localStorage)
// ═══════════════════════════════════════════════════════

const _RBT_COLUMNS = [
    { key: 'hook', label: '🪝 前置Hook', default: true },
    { key: 'bg', label: '🖼 背景素材', default: true },
    { key: 'bgscale', label: '🔍 背景缩放', default: true },
    { key: 'bgdurscale', label: '⏱🖼 背景时长', default: true },
    { key: 'audio', label: '🔊 音频', default: true },
    { key: 'audiodurscale', label: '⏱🔊 音频变速', default: true },
    { key: 'srt', label: '📝 字幕SRT', default: true },
    { key: 'txtcontent', label: '📃 字幕文本', default: true },
    { key: 'bgm', label: '🎵 配乐', default: true },
    { key: 'title', label: '📋 覆层标题', default: true },
    { key: 'body', label: '📋 覆层内容', default: true },
    { key: 'footer', label: '📋 覆层结尾', default: true },
    { key: 'scroll-title', label: '🔄 滚动标题', default: true },
    { key: 'scroll-body', label: '🔄 滚动内容', default: true },
    { key: 'dur', label: '⏱ 时长', default: true },
    { key: 'tpl', label: '🎬 字幕/覆层预设', default: true },
];

function _getColVisStorageKey() {
    const tabId = _batchTableState.activeTabId || 'default';
    return `rbt-col-vis-${tabId}`;
}

function _getColVisibility() {
    const key = _getColVisStorageKey();
    const saved = localStorage.getItem(key);
    // 始终以 _RBT_COLUMNS 的 default 为基准，合并已保存的设置
    const vis = {};
    for (const col of _RBT_COLUMNS) vis[col.key] = col.default;
    if (saved) {
        try {
            const parsed = JSON.parse(saved);
            for (const col of _RBT_COLUMNS) {
                if (col.key in parsed) vis[col.key] = parsed[col.key];
            }
            // 一次性迁移：旧版本中缩放列默认隐藏，新版本默认显示
            const migKey = key + '-scale-migrated';
            if (!localStorage.getItem(migKey)) {
                vis['bgscale'] = true;
                vis['bgdurscale'] = true;
                vis['audiodurscale'] = true;
                localStorage.setItem(migKey, '1');
                _saveColVisibility(vis);
            }
        } catch (e) { }
    }
    return vis;
}

function _saveColVisibility(vis) {
    const key = _getColVisStorageKey();
    localStorage.setItem(key, JSON.stringify(vis));
}

function _applyColVisibility() {
    const vis = _getColVisibility();
    // Remove old style
    let styleEl = document.getElementById('rbt-col-vis-style');
    if (!styleEl) {
        styleEl = document.createElement('style');
        styleEl.id = 'rbt-col-vis-style';
        document.head.appendChild(styleEl);
    }
    const rules = [];
    for (const col of _RBT_COLUMNS) {
        if (col.key === 'tpl') {
            // tpl maps to two columns (字幕模板 + 覆层预设), both share .rbt-col-tpl
            if (!vis[col.key]) rules.push(`.rbt-col-tpl { display: none !important; }`);
        } else {
            if (!vis[col.key]) rules.push(`.rbt-col-${col.key} { display: none !important; }`);
        }
    }
    styleEl.textContent = rules.join('\n');
}

function _showColumnSettingsPopup(anchor) {
    // Remove existing popup
    const existing = document.getElementById('rbt-col-settings-popup');
    if (existing) { existing.remove(); return; }

    const vis = _getColVisibility();
    const popup = document.createElement('div');
    popup.id = 'rbt-col-settings-popup';
    popup.style.cssText = 'position:fixed;z-index:100000;background:#1a1a2e;border:1px solid #4a4a6a;border-radius:8px;padding:12px;min-width:200px;box-shadow:0 8px 24px rgba(0,0,0,0.6);';

    // Position near anchor
    const rect = anchor.getBoundingClientRect();
    popup.style.top = (rect.bottom + 4) + 'px';
    popup.style.right = (window.innerWidth - rect.right) + 'px';

    let html = '<div style="font-size:13px;font-weight:600;color:#ccc;margin-bottom:8px;">⚙️ 显示列设置</div>';
    for (const col of _RBT_COLUMNS) {
        const checked = vis[col.key] !== false ? 'checked' : '';
        html += `<label style="display:flex;align-items:center;gap:6px;padding:3px 0;font-size:12px;color:#bbb;cursor:pointer;">
            <input type="checkbox" class="rbt-col-vis-chk" data-col="${col.key}" ${checked} style="accent-color:#00D4FF;">
            ${col.label}
        </label>`;
    }
    html += `<div style="margin-top:8px;padding-top:8px;border-top:1px solid #333;display:flex;gap:8px;">
        <button id="rbt-col-vis-reset" style="font-size:11px;padding:4px 10px;border-radius:4px;border:1px solid #555;background:transparent;color:#aaa;cursor:pointer;">恢复默认</button>
        <button id="rbt-col-vis-close" style="font-size:11px;padding:4px 10px;border-radius:4px;border:none;background:#00D4FF;color:#000;cursor:pointer;font-weight:600;">完成</button>
    </div>`;
    popup.innerHTML = html;
    document.body.appendChild(popup);

    // Event: checkbox change
    popup.querySelectorAll('.rbt-col-vis-chk').forEach(chk => {
        chk.addEventListener('change', () => {
            const colKey = chk.dataset.col;
            vis[colKey] = chk.checked;
            _saveColVisibility(vis);
            _applyColVisibility();
        });
    });

    // Event: reset
    popup.querySelector('#rbt-col-vis-reset').addEventListener('click', () => {
        for (const col of _RBT_COLUMNS) vis[col.key] = col.default;
        _saveColVisibility(vis);
        _applyColVisibility();
        popup.querySelectorAll('.rbt-col-vis-chk').forEach(chk => {
            chk.checked = true;
        });
    });

    // Event: close
    popup.querySelector('#rbt-col-vis-close').addEventListener('click', () => {
        popup.remove();
    });

    // Click outside to close
    setTimeout(() => {
        const handler = (e) => {
            if (!popup.contains(e.target) && e.target !== anchor) {
                popup.remove();
                document.removeEventListener('mousedown', handler);
            }
        };
        document.addEventListener('mousedown', handler);
    }, 100);
}

// ═══════════════════════════════════════════════════════
// Cycle Fill — 循环填充素材（加权交错分配）
// ═══════════════════════════════════════════════════════

function _showCycleFillDialog() {
    const state = window._reelsState;
    if (!state) return;
    const tasks = state.tasks || [];

    // 移除已有弹窗
    document.getElementById('rbt-cycle-fill-overlay')?.remove();

    const overlay = document.createElement('div');
    overlay.id = 'rbt-cycle-fill-overlay';
    overlay.style.cssText = 'position:fixed;inset:0;z-index:99999;background:rgba(0,0,0,0.7);display:flex;align-items:center;justify-content:center;';

    const box = document.createElement('div');
    box.style.cssText = 'background:#1e1e2e;border:1px solid #333;border-radius:14px;padding:24px;min-width:480px;max-width:600px;max-height:80vh;overflow-y:auto;box-shadow:0 10px 40px rgba(0,0,0,0.6);';

    // 收集已有素材（去重）
    const existingBg = [...new Set(tasks.map(t => t.bgPath || t.videoPath).filter(Boolean))];
    const existingAudio = [...new Set(tasks.map(t => t.audioPath).filter(Boolean))];
    const existingBgm = [...new Set(tasks.map(t => t.bgmPath).filter(Boolean))];

    // 文件扩展名映射
    const EXT_MAP = {
        bg: new Set(['mp4', 'mov', 'mkv', 'avi', 'wmv', 'flv', 'webm', 'jpg', 'jpeg', 'png', 'webp', 'gif', 'bmp']),
        audio: new Set(['mp3', 'wav', 'm4a', 'aac', 'flac', 'ogg', 'wma', 'mp4', 'mov', 'mkv', 'avi', 'wmv', 'flv', 'webm']),
        bgm: new Set(['mp3', 'wav', 'm4a', 'aac', 'flac', 'ogg']),
    };

    const fieldOptions = [
        { key: 'bg', label: '🖼 背景素材', items: [...existingBg], origItems: existingBg },
        { key: 'audio', label: '🔊 音频', items: [...existingAudio], origItems: existingAudio },
        { key: 'bgm', label: '🎵 配乐', items: [...existingBgm], origItems: existingBgm },
    ];

    // 素材池持久化
    const POOL_STORAGE_KEY = 'rbt-cycle-pool';
    function _loadSavedPool() {
        try { return JSON.parse(localStorage.getItem(POOL_STORAGE_KEY) || '{}'); } catch (e) { return {}; }
    }
    function _savePool(fieldKey, paths) {
        const pools = _loadSavedPool();
        pools[fieldKey] = paths;
        localStorage.setItem(POOL_STORAGE_KEY, JSON.stringify(pools));
    }

    let selectedField = fieldOptions.find(f => f.items.length > 0)?.key || 'bg';
    let sourceMode = 'existing'; // 'existing' | 'folder'

    // 如果有已保存的素材池，自动切换到 folder 模式
    const savedPools = _loadSavedPool();
    if (savedPools[selectedField] && savedPools[selectedField].length > 0) {
        sourceMode = 'folder';
        const field = fieldOptions.find(f => f.key === selectedField);
        if (field) field.items = savedPools[selectedField];
    }

    function _shortN(p) { return (p || '').replace(/\\/g, '/').split('/').pop() || ''; }

    // 隐藏文件选择器
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.multiple = true;
    fileInput.style.display = 'none';
    fileInput.addEventListener('change', (e) => {
        const files = Array.from(e.target.files || []);
        if (!files.length) return;
        const exts = EXT_MAP[selectedField];
        const paths = files
            .filter(f => exts.has((f.name || '').split('.').pop().toLowerCase()))
            .map(f => (typeof getFileNativePath === 'function') ? getFileNativePath(f) : (f.path || f.name))
            .filter(p => {
                // 必须是绝对路径，防止仅存文件名导致 FFmpeg 找不到文件
                if (!p || (!p.startsWith('/') && !/^[A-Z]:\\/i.test(p))) {
                    console.warn('[CycleFill] 跳过无完整路径的文件:', p);
                    return false;
                }
                return true;
            });
        if (paths.length === 0) {
            alert('所选文件中没有匹配该字段类型的文件');
            return;
        }
        const field = fieldOptions.find(f => f.key === selectedField);
        if (field) {
            field.items = paths;
            _savePool(selectedField, paths); // 保存到 localStorage
            renderContent();
        }
        e.target.value = '';
    });
    box.appendChild(fileInput);

    function renderContent() {
        const field = fieldOptions.find(f => f.key === selectedField);
        const items = field ? field.items : [];
        const totalRows = tasks.length || 0;
        const emptyRows = field ? tasks.filter(t => {
            if (field.key === 'bg') return !t.bgPath && !t.videoPath;
            if (field.key === 'audio') return !t.audioPath;
            if (field.key === 'bgm') return !t.bgmPath;
            return true;
        }).length : 0;

        // 保留fileInput
        const existingFileInput = box.querySelector('input[type="file"]');

        box.innerHTML = `
            <div style="font-size:16px;font-weight:700;margin-bottom:16px;color:#00D4FF;">🔄 循环填充素材</div>
            <div style="margin-bottom:12px;">
                <span style="font-size:12px;color:#888;">选择字段：</span>
                ${fieldOptions.map(f => `
                    <label style="display:inline-flex;align-items:center;gap:4px;margin-right:12px;font-size:13px;color:${f.key === selectedField ? '#00D4FF' : '#aaa'};cursor:pointer;">
                        <input type="radio" name="rbt-cf-field" value="${f.key}" ${f.key === selectedField ? 'checked' : ''} style="accent-color:#00D4FF;">
                        ${f.label} (${f.items.length})
                    </label>
                `).join('')}
            </div>
            <div style="margin-bottom:10px;">
                <span style="font-size:12px;color:#888;">素材来源：</span>
                <label style="display:inline-flex;align-items:center;gap:4px;margin-right:12px;font-size:13px;color:${sourceMode === 'existing' ? '#00D4FF' : '#aaa'};cursor:pointer;">
                    <input type="radio" name="rbt-cf-source" value="existing" ${sourceMode === 'existing' ? 'checked' : ''} style="accent-color:#00D4FF;">
                    从已有行提取
                </label>
                <label style="display:inline-flex;align-items:center;gap:4px;margin-right:8px;font-size:13px;color:${sourceMode === 'folder' ? '#00D4FF' : '#aaa'};cursor:pointer;">
                    <input type="radio" name="rbt-cf-source" value="folder" ${sourceMode === 'folder' ? 'checked' : ''} style="accent-color:#00D4FF;">
                    从文件选取
                </label>
                ${sourceMode === 'folder' ? `<button id="rbt-cf-pick-files" style="font-size:11px;padding:3px 10px;border-radius:4px;border:1px solid #555;background:#2a2a3a;color:#ccc;cursor:pointer;">📂 选择文件...</button>` : ''}
                ${sourceMode === 'folder' && items.length > 0 ? `<button id="rbt-cf-clear-pool" style="font-size:11px;padding:3px 10px;border-radius:4px;border:1px solid #555;background:#3a2a2a;color:#f88;cursor:pointer;margin-left:4px;">✕ 清空素材池</button>` : ''}
            </div>
            <div style="font-size:11px;color:#666;margin-bottom:8px;">
                共 ${totalRows} 行，其中 ${emptyRows} 行该字段为空 | 素材池: ${items.length} 个
            </div>
            ${items.length === 0 ? `
                <div style="text-align:center;padding:30px;color:#666;">
                    ${sourceMode === 'existing' ? '当前没有该类型的素材。<br>请先通过批量添加素材到表格中。' : '请点击「选择文件」按钮添加素材文件。'}
                </div>
            ` : `
                <div style="border:1px solid #333;border-radius:8px;overflow:hidden;margin-bottom:12px;">
                    <div style="display:flex;padding:6px 12px;background:#12122a;font-size:11px;color:#888;border-bottom:1px solid #333;">
                        <span style="flex:1;">素材文件</span>
                        <span style="width:100px;text-align:center;">权重</span>
                        <span style="width:60px;text-align:center;">预计次数</span>
                    </div>
                    <div id="rbt-cf-items" style="max-height:200px;overflow-y:auto;">
                        ${items.map((item, i) => `
                            <div style="display:flex;align-items:center;padding:6px 12px;border-bottom:1px solid #222;font-size:12px;color:#ccc;" data-idx="${i}">
                                <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${_escHtml(item)}">${_shortN(item)}</span>
                                <span style="width:100px;text-align:center;">
                                    <input type="number" class="rbt-cf-weight" data-idx="${i}" value="1" min="1" max="20"
                                        style="width:50px;text-align:center;background:#1a1a3a;color:#ddd;border:1px solid #444;border-radius:4px;padding:2px 4px;font-size:12px;">
                                </span>
                                <span class="rbt-cf-count" data-idx="${i}" style="width:60px;text-align:center;color:#888;">—</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
                <div id="rbt-cf-preview" style="font-size:11px;color:#666;margin-bottom:12px;max-height:60px;overflow-y:auto;line-height:1.6;"></div>
            `}
            <div style="display:flex;gap:8px;justify-content:flex-end;margin-top:12px;">
                <button id="rbt-cf-cancel" style="padding:8px 16px;border-radius:6px;border:1px solid #555;background:transparent;color:#aaa;cursor:pointer;font-size:13px;">取消</button>
                ${items.length > 0 && emptyRows > 0 ? `
                    <button id="rbt-cf-fill-empty" style="padding:8px 16px;border-radius:6px;border:none;background:#2a5a3a;color:#8f8;cursor:pointer;font-size:13px;font-weight:600;">填充空行 (${emptyRows}行)</button>
                ` : ''}
                ${items.length > 0 ? `
                    <button id="rbt-cf-fill-all" style="padding:8px 16px;border-radius:6px;border:none;background:#00D4FF;color:#000;cursor:pointer;font-size:13px;font-weight:600;">覆盖全部 (${totalRows}行)</button>
                ` : ''}
            </div>
        `;

        // 重新挂 fileInput
        box.appendChild(fileInput);

        // Bind events
        box.querySelectorAll('input[name="rbt-cf-field"]').forEach(r => {
            r.addEventListener('change', () => {
                selectedField = r.value;
                const field = fieldOptions.find(f => f.key === selectedField);
                if (sourceMode === 'existing' && field) {
                    field.items = [...field.origItems];
                } else if (sourceMode === 'folder' && field) {
                    // 加载已保存的素材池
                    const saved = _loadSavedPool()[selectedField];
                    field.items = saved && saved.length > 0 ? saved : [];
                }
                renderContent();
            });
        });
        box.querySelectorAll('input[name="rbt-cf-source"]').forEach(r => {
            r.addEventListener('change', () => {
                sourceMode = r.value;
                const field = fieldOptions.find(f => f.key === selectedField);
                if (sourceMode === 'existing' && field) {
                    field.items = [...field.origItems];
                } else if (sourceMode === 'folder' && field) {
                    // 加载已保存的素材池
                    const saved = _loadSavedPool()[selectedField];
                    field.items = saved && saved.length > 0 ? saved : [];
                }
                renderContent();
            });
        });
        box.querySelector('#rbt-cf-pick-files')?.addEventListener('click', () => {
            const exts = [...EXT_MAP[selectedField]].map(e => '.' + e).join(',');
            fileInput.accept = exts;
            fileInput.click();
        });
        box.querySelector('#rbt-cf-clear-pool')?.addEventListener('click', () => {
            const field = fieldOptions.find(f => f.key === selectedField);
            if (field) field.items = [];
            _savePool(selectedField, []);
            renderContent();
        });
        box.querySelectorAll('.rbt-cf-weight').forEach(w => {
            w.addEventListener('input', updatePreview);
        });
        box.querySelector('#rbt-cf-cancel')?.addEventListener('click', () => overlay.remove());
        box.querySelector('#rbt-cf-fill-empty')?.addEventListener('click', () => {
            const field = fieldOptions.find(f => f.key === selectedField);
            _doCycleFill(selectedField, field.items, _getWeights(), 'empty');
            overlay.remove();
        });
        box.querySelector('#rbt-cf-fill-all')?.addEventListener('click', () => {
            const field = fieldOptions.find(f => f.key === selectedField);
            _doCycleFill(selectedField, field.items, _getWeights(), 'all');
            overlay.remove();
        });

        updatePreview();
    }

    function _getWeights() {
        const weights = [];
        box.querySelectorAll('.rbt-cf-weight').forEach(w => {
            weights[parseInt(w.dataset.idx)] = Math.max(1, parseInt(w.value) || 1);
        });
        return weights;
    }

    function updatePreview() {
        const field = fieldOptions.find(f => f.key === selectedField);
        if (!field || !field.items.length) return;
        const weights = _getWeights();
        const totalWeight = weights.reduce((s, w) => s + w, 0);
        const totalRows = tasks.length || 1;

        // 更新各素材预计次数
        box.querySelectorAll('.rbt-cf-count').forEach(span => {
            const i = parseInt(span.dataset.idx);
            const count = Math.round(totalRows * (weights[i] || 1) / totalWeight);
            span.textContent = count + '次';
        });

        // 小预览
        const preview = box.querySelector('#rbt-cf-preview');
        if (preview) {
            const previewCount = Math.min(totalRows, 30);
            const seq = _generateCycleSequence(field.items, weights, previewCount);
            const names = seq.map(idx => _shortN(field.items[idx]));
            const truncated = totalRows > 30 ? ' ...' : '';
            preview.textContent = '预览: ' + names.join(' → ') + truncated;
        }
    }

    overlay.appendChild(box);
    document.body.appendChild(overlay);
    overlay.addEventListener('click', (e) => { if (e.target === overlay) overlay.remove(); });
    renderContent();
}

/**
 * 加权交错分配算法 — 优先队列法，保证不相邻重复
 * @returns {number[]} 素材索引序列
 */
function _generateCycleSequence(items, weights, count) {
    if (items.length === 0) return [];
    if (items.length === 1) return new Array(count).fill(0);

    const totalWeight = weights.reduce((s, w) => s + w, 0);
    const seq = [];
    // 每个素材的剩余份额
    const remaining = weights.map((w, i) => ({ idx: i, rem: (w / totalWeight) * count }));
    let lastIdx = -1;

    for (let i = 0; i < count; i++) {
        // 找剩余份额最大的（排除上一个用过的）
        let best = -1, bestRem = -Infinity;
        for (const r of remaining) {
            if (r.idx === lastIdx) continue;
            if (r.rem > bestRem) { bestRem = r.rem; best = r.idx; }
        }
        if (best === -1) best = remaining[0].idx; // fallback
        seq.push(best);
        remaining.find(r => r.idx === best).rem -= 1;
        lastIdx = best;
    }

    return seq;
}

/**
 * 执行循环填充
 */
function _doCycleFill(fieldKey, items, weights, mode) {
    const state = window._reelsState;
    if (!state || !items.length) return;

    let targetIndices;
    if (mode === 'empty') {
        targetIndices = state.tasks.map((t, i) => {
            if (fieldKey === 'bg' && !t.bgPath && !t.videoPath) return i;
            if (fieldKey === 'audio' && !t.audioPath) return i;
            if (fieldKey === 'bgm' && !t.bgmPath) return i;
            return -1;
        }).filter(i => i >= 0);
    } else {
        targetIndices = state.tasks.map((_, i) => i);
    }

    if (targetIndices.length === 0) {
        alert('没有需要填充的行');
        return;
    }

    const seq = _generateCycleSequence(items, weights, targetIndices.length);
    const fieldLabel = fieldKey === 'bg' ? '背景' : fieldKey === 'audio' ? '音频' : '配乐';

    for (let i = 0; i < targetIndices.length; i++) {
        const task = state.tasks[targetIndices[i]];
        const filePath = items[seq[i]];
        if (fieldKey === 'bg') {
            const isImg = /\.(jpg|jpeg|png|webp|gif|bmp)$/i.test(filePath);
            if (isImg) {
                task.bgPath = filePath;
                task.videoPath = '';
            } else {
                task.videoPath = filePath;
                task.bgPath = filePath;
            }
            if (window.electronAPI && window.electronAPI.toFileUrl) {
                task.bgSrcUrl = window.electronAPI.toFileUrl(filePath);
            }
        } else if (fieldKey === 'audio') {
            task.audioPath = filePath;
        } else if (fieldKey === 'bgm') {
            task.bgmPath = filePath;
        }
    }

    _renderBatchTable();
    if (typeof _renderTaskList === 'function') _renderTaskList();
    alert(`✅ 已循环填充 ${targetIndices.length} 行的${fieldLabel}`);
}

/**
 * 从剪贴板批量粘贴文案到「字幕文本」列 (txtContent)
 * 支持：
 * - 单列：每行一条文案
 * - 多列表格：取第一列
 * - Google 表格 / Excel 复制过来的都支持
 */
async function _batchPasteTxtContent() {
    let raw = '';
    try {
        raw = await navigator.clipboard.readText();
    } catch (e) {
        raw = await _showPasteDialog();
    }
    if (!raw || !raw.trim()) return;

    // 解析行 — 支持 TSV 表格格式（取第一列）和纯文本（每行一条）
    const lines = raw.split(/\n/).map(line => {
        // 如果是 TSV，取第一列
        const cols = line.split('\t');
        return cols[0].trim();
    }).filter(s => s.length > 0);

    if (!lines.length) {
        alert('未检测到有效文案数据');
        return;
    }

    const state = window._reelsState;
    if (!state) return;

    let filled = 0, created = 0;

    for (let i = 0; i < lines.length; i++) {
        if (i < state.tasks.length) {
            // 填充已有行
            state.tasks[i].txtContent = lines[i];
            state.tasks[i].aligned = false;
            filled++;
        } else {
            // 新建行
            const taskName = `card_${String(state.tasks.length + 1).padStart(3, '0')}`;
            state.tasks.push({
                baseName: taskName,
                fileName: `${taskName}.mp4`,
                bgPath: null, bgSrcUrl: null,
                audioPath: null, srtPath: null,
                segments: [],
                videoPath: null, srcUrl: null,
                overlays: [],
                txtContent: lines[i],
                aligned: false,
            });
            created++;
        }
    }

    _renderBatchTable();
    if (typeof _renderTaskList === 'function') _renderTaskList();

    const parts = [];
    if (filled) parts.push(`填充 ${filled} 行`);
    if (created) parts.push(`新建 ${created} 行`);
    alert(`✅ 字幕文本粘贴完成：${parts.join('，')}`);
}

// ── 辅助：设置一行的文案 ──
function _setTaskText(task, title, body, ReelsOverlay) {
    if (!task.overlays) task.overlays = [];
    if (!task.overlays[0]) {
        task.overlays[0] = ReelsOverlay.createTextCardOverlay({
            title_text: '', body_text: '',
            start: 0, end: 9999,
        });
    }
    task.overlays[0].title_text = title;
    task.overlays[0].body_text = body;
}

// ── 辅助：新建一行并填入文案 ──
function _createNewTextRow(state, title, body, ReelsOverlay) {
    const taskName = `card_${String(state.tasks.length + 1).padStart(3, '0')}`;
    state.tasks.push({
        baseName: taskName,
        fileName: `${taskName}.mp4`,
        bgPath: null, bgSrcUrl: null,
        audioPath: null, srtPath: null,
        segments: [],
        videoPath: null, srcUrl: null,
        overlays: [ReelsOverlay.createTextCardOverlay({
            title_text: title,
            body_text: body,
            start: 0, end: 9999,
        })],
    });
}

// ── 辅助：设置滚动字幕文案 ──
function _setTaskScrollText(task, title, body, ReelsOverlay) {
    if (!task.overlays) task.overlays = [];
    let scrollOv = task.overlays.find(o => o.type === 'scroll');
    if (!scrollOv) {
        scrollOv = ReelsOverlay.createScrollOverlay({
            scroll_title: title, content: body,
            start: 0, end: 9999,
        });
        task.overlays.push(scrollOv);
    } else {
        scrollOv.scroll_title = title;
        scrollOv.content = body;
    }
}

// ── 辅助：新建一行并填入滚动字幕 ──
function _createNewScrollRow(state, title, body, ReelsOverlay) {
    const taskName = `scroll_${String(state.tasks.length + 1).padStart(3, '0')}`;
    state.tasks.push({
        baseName: taskName,
        fileName: `${taskName}.mp4`,
        bgPath: null, bgSrcUrl: null,
        audioPath: null, srtPath: null,
        segments: [],
        videoPath: null, srcUrl: null,
        overlays: [ReelsOverlay.createScrollOverlay({
            scroll_title: title,
            content: body,
            start: 0, end: 9999,
        })],
    });
}

// ── 模式选择弹窗 ──
function _showPasteModeDialog() {
    return new Promise(resolve => {
        const overlay = document.createElement('div');
        overlay.style.cssText = 'position:fixed;inset:0;z-index:99999;background:rgba(0,0,0,0.7);display:flex;align-items:center;justify-content:center;';

        const box = document.createElement('div');
        box.style.cssText = 'background:#1e1e2e;border:1px solid #333;border-radius:14px;padding:24px;min-width:360px;box-shadow:0 10px 40px rgba(0,0,0,0.6);';
        box.innerHTML = `
            <div style="font-size:16px;font-weight:700;margin-bottom:16px;color:#00D4FF;">📋 选择粘贴模式</div>
            <div style="display:flex;flex-direction:column;gap:10px;">
                <button class="rbt-paste-mode-btn" data-mode="fill" style="padding:12px 16px;border-radius:8px;border:1px solid #2a6b3a;background:#1a3a2a;color:#8f8;cursor:pointer;text-align:left;font-size:13px;">
                    <div style="font-weight:700;font-size:14px;margin-bottom:4px;">🔄 补全模式</div>
                    <div style="color:#aaa;font-size:12px;">只填充空行（没有文案的行），已有文案的行保持不动。多出的文案追加为新行。</div>
                </button>
                <button class="rbt-paste-mode-btn" data-mode="overwrite" style="padding:12px 16px;border-radius:8px;border:1px solid #b48b00;background:#3a3020;color:#ffd700;cursor:pointer;text-align:left;font-size:13px;">
                    <div style="font-weight:700;font-size:14px;margin-bottom:4px;">⚡ 覆盖模式</div>
                    <div style="color:#aaa;font-size:12px;">从第1行开始逐行覆盖文案（不改变背景/音频等其它数据）。多出的追加为新行。</div>
                </button>
                <button class="rbt-paste-mode-btn" data-mode="new" style="padding:12px 16px;border-radius:8px;border:1px solid #4466aa;background:#1a2540;color:#88bbff;cursor:pointer;text-align:left;font-size:13px;">
                    <div style="font-weight:700;font-size:14px;margin-bottom:4px;">➕ 新行模式</div>
                    <div style="color:#aaa;font-size:12px;">全部作为新行添加到末尾，不影响任何现有行。</div>
                </button>
            </div>
            <div style="margin-top:14px;text-align:right;">
                <button class="rbt-paste-cancel" style="padding:6px 16px;border-radius:6px;border:1px solid #555;background:transparent;color:#aaa;cursor:pointer;font-size:13px;">取消</button>
            </div>
        `;

        overlay.appendChild(box);
        document.body.appendChild(overlay);

        const close = (val) => {
            if (overlay.parentNode) overlay.parentNode.removeChild(overlay);
            resolve(val);
        };

        box.querySelectorAll('.rbt-paste-mode-btn').forEach(btn => {
            btn.addEventListener('click', () => close(btn.dataset.mode));
            btn.addEventListener('mouseenter', () => { btn.style.opacity = '0.85'; btn.style.transform = 'scale(1.01)'; });
            btn.addEventListener('mouseleave', () => { btn.style.opacity = '1'; btn.style.transform = 'scale(1)'; });
        });
        box.querySelector('.rbt-paste-cancel').addEventListener('click', () => close(null));
        overlay.addEventListener('click', (e) => { if (e.target === overlay) close(null); });
    });
}

function _showPasteDialog() {
    return new Promise(resolve => {
        const overlay = document.createElement('div');
        overlay.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.8);z-index:10001;display:flex;align-items:center;justify-content:center;';
        overlay.innerHTML = `
            <div style="background:#1a1a3e;border-radius:12px;padding:24px;width:500px;">
                <h3 style="margin:0 0 12px;color:#00D4FF;">📋 粘贴表格数据</h3>
                <textarea id="rbt-paste-area" style="width:100%;height:200px;background:#12122a;border:1px solid #2a2a4a;border-radius:8px;color:#ddd;font-size:13px;padding:12px;" placeholder="从 Google 表格粘贴..."></textarea>
                <div style="display:flex;gap:8px;margin-top:12px;justify-content:flex-end;">
                    <button id="rbt-paste-cancel" style="padding:8px 16px;background:#333;border:1px solid #555;border-radius:6px;color:#ccc;cursor:pointer;">取消</button>
                    <button id="rbt-paste-ok" style="padding:8px 16px;background:#00D4FF;border:none;border-radius:6px;color:#000;font-weight:bold;cursor:pointer;">导入</button>
                </div>
            </div>
        `;
        document.body.appendChild(overlay);
        const close = v => { document.body.removeChild(overlay); resolve(v); };
        overlay.querySelector('#rbt-paste-cancel').onclick = () => close(null);
        overlay.querySelector('#rbt-paste-ok').onclick = () => close(overlay.querySelector('#rbt-paste-area').value);
        overlay.addEventListener('click', e => { if (e.target === overlay) close(null); });
        setTimeout(() => overlay.querySelector('#rbt-paste-area')?.focus(), 100);
    });
}

// ═══════════════════════════════════════════════════════
// 8. TSV Parser (same as overlay panel)
// ═══════════════════════════════════════════════════════

function _parseBatchTSV(raw) {
    const rows = [];
    let i = 0;
    const len = raw.length;
    while (i < len) {
        const cells = [];
        while (i < len) {
            let cell = '';
            if (raw[i] === '"') {
                i++;
                while (i < len) {
                    if (raw[i] === '"') {
                        if (i + 1 < len && raw[i + 1] === '"') { cell += '"'; i += 2; }
                        else { i++; break; }
                    } else { cell += raw[i++]; }
                }
            } else {
                while (i < len && raw[i] !== '\t' && raw[i] !== '\n' && raw[i] !== '\r') { cell += raw[i++]; }
            }
            cells.push(cell);
            if (i < len && raw[i] === '\t') { i++; continue; }
            if (i < len && (raw[i] === '\n' || raw[i] === '\r')) {
                if (raw[i] === '\r' && i + 1 < len && raw[i + 1] === '\n') i++;
                i++;
                break;
            }
        }
        if (cells.length > 0 && cells.some(c => c.trim())) {
            rows.push([cells[0] || '', cells[1] || '']);
        }
    }
    return rows;
}

// ═══════════════════════════════════════════════════════
// 9. Template helpers
// ═══════════════════════════════════════════════════════

function _getSubtitlePresetList() {
    try {
        if (window.ReelsStyleEngine && typeof ReelsStyleEngine.loadSubtitlePresets === 'function') {
            const data = ReelsStyleEngine.loadSubtitlePresets();
            return Object.keys(data.presets || {});
        }
    } catch (e) { }
    return [];
}

function _getOverlayGroupPresetList() {
    try {
        const stored = localStorage.getItem('reels_overlay_group_presets');
        if (stored) {
            const obj = JSON.parse(stored);
            return Object.keys(obj).map(name => ({
                name,
                count: Array.isArray(obj[name]) ? obj[name].length : 0,
            }));
        }
    } catch (e) { }
    // Fallback: also include single-card templates as 1-layer presets
    try {
        const stored = localStorage.getItem('reels_card_templates');
        if (stored) {
            const obj = JSON.parse(stored);
            return Object.keys(obj).map(name => ({ name, count: 1 }));
        }
    } catch (e) { }
    return [];
}

function _applyOverlayGroupPresetToTask(task, presetName) {
    try {
        // Try group presets first
        const groupStored = localStorage.getItem('reels_overlay_group_presets');
        if (groupStored) {
            const presets = JSON.parse(groupStored);
            if (presets[presetName] && Array.isArray(presets[presetName])) {
                const layers = presets[presetName];
                const oldOverlays = task.overlays || [];
                task.overlays = layers.map((layerData, i) => {
                    const clone = JSON.parse(JSON.stringify(layerData));
                    clone.id = 'ov_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8);
                    // For non-fixed layers, preserve existing text from the task row
                    if (!clone.fixed_text) {
                        const old = oldOverlays[i];
                        if (old) {
                            if (old.title_text) clone.title_text = old.title_text;
                            if (old.body_text) clone.body_text = old.body_text;
                            if (old.footer_text) clone.footer_text = old.footer_text;
                            if (old.content) clone.content = old.content;
                        }
                    }
                    // Fixed layers already have text from preset — use as-is
                    return clone;
                });
                task._overlayPresetName = presetName;
                return;
            }
        }
        // Fallback: single card template
        const cardStored = localStorage.getItem('reels_card_templates');
        if (cardStored) {
            const templates = JSON.parse(cardStored);
            const tpl = templates[presetName];
            if (tpl) {
                if (!task.overlays || !task.overlays[0]) {
                    const ReelsOverlay = window.ReelsOverlay;
                    if (ReelsOverlay) {
                        task.overlays = [ReelsOverlay.createTextCardOverlay({
                            title_text: '', body_text: '',
                            start: 0, end: 9999,
                        })];
                    }
                }
                if (task.overlays && task.overlays[0]) {
                    const keepKeys = ['id', 'type', 'title_text', 'body_text', 'start', 'end'];
                    for (const [k, v] of Object.entries(tpl)) {
                        if (!keepKeys.includes(k)) task.overlays[0][k] = v;
                    }
                }
                task._overlayPresetName = presetName;
            }
        }
    } catch (e) { }
}

function _batchAddEmptyRow() {
    const state = window._reelsState;
    const ReelsOverlay = window.ReelsOverlay;
    if (!state || !ReelsOverlay) return;
    const taskName = `card_${String(state.tasks.length + 1).padStart(3, '0')}`;
    state.tasks.push({
        baseName: taskName,
        fileName: `${taskName}.mp4`,
        bgPath: null, bgSrcUrl: null,
        audioPath: null, srtPath: null,
        segments: [],
        videoPath: null, srcUrl: null,
        overlays: [],
    });
}

// ═══════════════════════════════════════════════════════
// 9a. Folder import (auto-classify + match)
// ═══════════════════════════════════════════════════════

const _BG_EXTS = new Set(['mp4', 'mov', 'mkv', 'avi', 'wmv', 'flv', 'webm', 'jpg', 'jpeg', 'png', 'webp', 'gif', 'bmp']);
const _AUDIO_EXTS = new Set(['mp3', 'wav', 'm4a', 'aac', 'flac', 'ogg', 'wma']);
const _SRT_EXTS = new Set(['srt']);
const _TXT_EXTS = new Set(['txt']);

/**
 * 读取文件夹 → 自动按类型分类 → 同名配对 → 创建任务行
 */
function _batchImportFolder(files) {
    if (!files || !files.length) return;
    const state = window._reelsState;
    const ReelsOverlay = window.ReelsOverlay;
    if (!state) return;

    // 按类型分桶
    const bgFiles = [];
    const audioFiles = [];
    const srtFiles = [];
    const txtFiles = [];

    for (const f of files) {
        const name = f.name || '';
        // 跳过隐藏文件和系统文件
        if (name.startsWith('.') || name === 'Thumbs.db' || name === 'desktop.ini') continue;
        const ext = name.split('.').pop().toLowerCase();
        if (_AUDIO_EXTS.has(ext)) audioFiles.push(f);
        else if (_VOICE_VIDEO_EXTS.has(ext) && _looksLikeVoiceTrack(name)) audioFiles.push(f);
        else if (_BG_EXTS.has(ext)) bgFiles.push(f);
        else if (_SRT_EXTS.has(ext)) srtFiles.push(f);
        else if (_TXT_EXTS.has(ext)) txtFiles.push(f);
    }

    if (!bgFiles.length && !audioFiles.length && !srtFiles.length && !txtFiles.length) {
        alert('文件夹中未找到支持的媒体文件');
        return;
    }

    // 提取基础匹配键（去除扩展名并应用 stopwords 过滤）
    function matchKey(f) {
        if (typeof window._buildAudioSubtitleMatchKey === 'function') {
            return window._buildAudioSubtitleMatchKey(f.name || '');
        }
        // 如果外部函数不可用，回退到基础去头尾处理
        const name = f.name || '';
        const dot = name.lastIndexOf('.');
        return (dot > 0 ? name.substring(0, dot) : name).trim().toLowerCase();
    }

    // 建立同名配对映射: matchKey → { bg, audio, srt, txt }
    const groups = new Map();

    function addToGroup(file, field) {
        const key = matchKey(file);
        if (!groups.has(key)) groups.set(key, { bg: null, audio: null, srt: null, txt: null });
        // 如果已经有该字段，优先保留第一个遇到的以稳定排序
        if (!groups.get(key)[field]) {
            groups.get(key)[field] = file;
        }
    }

    bgFiles.forEach(f => addToGroup(f, 'bg'));
    audioFiles.forEach(f => addToGroup(f, 'audio'));
    srtFiles.forEach(f => addToGroup(f, 'srt'));
    txtFiles.forEach(f => addToGroup(f, 'txt'));

    // 按名称排序
    const sortedKeys = [...groups.keys()].sort((a, b) => a.localeCompare(b));

    let created = 0;
    for (const bn of sortedKeys) {
        const g = groups.get(bn);
        const taskName = bn || `task_${String(state.tasks.length + 1).padStart(3, '0')}`;

        const task = {
            baseName: taskName,
            fileName: `${taskName}.mp4`,
            bgPath: g.bg ? ((typeof getFileNativePath === 'function') ? getFileNativePath(g.bg) : (g.bg.path || g.bg.name)) : null,
            bgSrcUrl: null,
            audioPath: g.audio ? ((typeof getFileNativePath === 'function') ? getFileNativePath(g.audio) : (g.audio.path || g.audio.name)) : null,
            srtPath: g.srt ? ((typeof getFileNativePath === 'function') ? getFileNativePath(g.srt) : (g.srt.path || g.srt.name)) : null,
            txtPath: g.txt ? ((typeof getFileNativePath === 'function') ? getFileNativePath(g.txt) : (g.txt.path || g.txt.name)) : null,
            txtContent: null,
            aligned: false,
            segments: [],
            videoPath: g.bg ? ((typeof getFileNativePath === 'function') ? getFileNativePath(g.bg) : (g.bg.path || g.bg.name)) : null,
            srcUrl: null,
            overlays: [],
        };

        // 生成预览 URL
        if (g.bg) {
            try { task.bgSrcUrl = URL.createObjectURL(g.bg); } catch (e) { }
        }

        // 读取 SRT 内容
        if (g.srt) {
            _readSrtFileToTask(task, g.srt);
        }

        // 读取 TXT 内容
        if (g.txt) {
            _readTxtFileToTask(task, g.txt);
        }

        state.tasks.push(task);
        created++;
    }

    _renderBatchTable();
    if (typeof _renderTaskList === 'function') _renderTaskList();

    const summary = [];
    if (bgFiles.length) summary.push(`${bgFiles.length} 个背景`);
    if (audioFiles.length) summary.push(`${audioFiles.length} 个音频`);
    if (srtFiles.length) summary.push(`${srtFiles.length} 个字幕`);
    if (txtFiles.length) summary.push(`${txtFiles.length} 个TXT文案`);
    alert(`✅ 从文件夹导入 ${created} 个任务\n检测到: ${summary.join(', ')}\n同名文件已自动配对`);
}

// ═══════════════════════════════════════════════════════
// 9b. File assignment helpers
// ═══════════════════════════════════════════════════════

function _openHookModal(idx) {
    const container = _batchTableState.container;
    if (!container) return;
    const hookModal = container.querySelector('#rbt-hook-modal');
    if (!hookModal) return;

    hookModal.dataset.editIdx = String(idx);

    // Initial clear
    const pathInput = container.querySelector('#rbt-hook-path');
    const startInput = container.querySelector('#rbt-hook-start');
    const endInput = container.querySelector('#rbt-hook-end');
    const speedInput = container.querySelector('#rbt-hook-speed');
    const transInput = container.querySelector('#rbt-hook-transition');
    const durInput = container.querySelector('#rbt-hook-trans-dur');

    if (pathInput) pathInput.value = '';
    if (startInput) startInput.value = '';
    if (endInput) endInput.value = '';
    if (speedInput) speedInput.value = '1';
    if (transInput) transInput.value = 'none';
    if (durInput) durInput.value = '0.5';

    let targetTask = null;
    if (idx >= 0) {
        targetTask = window._reelsState.tasks[idx];
    } else {
        const indices = _getSelectedIndices();
        if (indices.length > 0) {
            targetTask = window._reelsState.tasks[indices[0]];
        }
    }

    if (targetTask) {
        if (pathInput) pathInput.value = targetTask.hookFile || '';
        if (startInput) startInput.value = targetTask.hookTrimStart != null ? targetTask.hookTrimStart : '';
        if (endInput) endInput.value = targetTask.hookTrimEnd != null ? targetTask.hookTrimEnd : '';
        if (speedInput) speedInput.value = targetTask.hookSpeed || '1';
        if (transInput) transInput.value = targetTask.hookTransition || 'none';
        if (durInput) durInput.value = targetTask.hookTransDuration || '0.5';
    }

    _updateHookPreview();
    hookModal.style.display = 'flex';
}

function _updateHookPreview() {
    const container = _batchTableState.container;
    if (!container) return;
    const pathInput = container.querySelector('#rbt-hook-path');
    const previewContainer = container.querySelector('#rbt-hook-preview-container');
    const videoPreview = container.querySelector('#rbt-hook-preview-video');
    const imgPreview = container.querySelector('#rbt-hook-preview-img');
    const startInput = container.querySelector('#rbt-hook-start');
    
    if (!pathInput || !previewContainer || !videoPreview || !imgPreview) return;
    
    const filePath = pathInput.value.trim();
    if (!filePath) {
        previewContainer.style.display = 'none';
        videoPreview.src = '';
        imgPreview.src = '';
        return;
    }
    
    previewContainer.style.display = 'block';
    const fileUrl = 'file://' + filePath;
    const ext = filePath.split('.').pop().toLowerCase();
    
    if (['mp4', 'webm', 'ogg', 'mov', 'mkv'].includes(ext)) {
        imgPreview.style.display = 'none';
        videoPreview.style.display = 'block';
        if (videoPreview.src !== fileUrl) {
            videoPreview.src = fileUrl;
            videoPreview.onloadedmetadata = () => {
                const startTime = parseFloat(startInput?.value) || 0;
                videoPreview.currentTime = startTime;
            };
        } else {
            const startTime = parseFloat(startInput?.value) || 0;
            videoPreview.currentTime = startTime;
        }
    } else if (['jpg', 'jpeg', 'png', 'gif', 'webp'].includes(ext)) {
        videoPreview.style.display = 'none';
        imgPreview.style.display = 'block';
        imgPreview.src = fileUrl;
    } else {
        previewContainer.style.display = 'none';
    }
}

/**
 * 显示批量分配模式对话框
 */
function _showBatchModeDialog(fileCount, field) {
    const fieldLabel = { bg: '背景素材', audio: '音频', srt: '字幕', txt: 'TXT' }[field] || field;
    return new Promise((resolve) => {
        const overlay = document.createElement('div');
        overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:99999;display:flex;align-items:center;justify-content:center;';
        const dialog = document.createElement('div');
        dialog.style.cssText = 'background:#1e1e2e;border:1px solid #444;border-radius:12px;padding:24px;min-width:320px;color:#eee;font-size:14px;';
        const isSingle = fileCount === 1;
        dialog.innerHTML = `
            <h3 style="margin:0 0 14px;font-size:16px;">📂 批量${fieldLabel} · ${fileCount} 个文件</h3>
            <div style="display:flex;flex-direction:column;gap:8px;">
                <button data-mode="fill" style="padding:10px 16px;border:1px solid #555;border-radius:8px;background:#2a3a4a;color:#cce;cursor:pointer;text-align:left;font-size:13px;">
                    📥 <b>补全</b> — 填入空位，多余的添加新行
                </button>
                <button data-mode="overwrite" style="padding:10px 16px;border:1px solid #555;border-radius:8px;background:#4a2a2a;color:#ecc;cursor:pointer;text-align:left;font-size:13px;">
                    ✏️ <b>覆盖</b> — 从第1行开始覆盖
                </button>
                <button data-mode="append" style="padding:10px 16px;border:1px solid #555;border-radius:8px;background:#2a4a2a;color:#cec;cursor:pointer;text-align:left;font-size:13px;">
                    ➕ <b>添加新行</b> — 全部作为新行追加
                </button>
                ${isSingle ? `<button data-mode="applyall" style="padding:10px 16px;border:1px solid #7b5aab;border-radius:8px;background:#3a2a5a;color:#d0b0ff;cursor:pointer;text-align:left;font-size:13px;">
                    🔄 <b>应用到全部行</b> — 所有行都用这个文件
                </button>` : ''}
                ${field === 'bg' ? `<button data-mode="fill_hook" style="padding:10px 16px;border:1px solid #7a6a4a;border-radius:8px;background:#3a3a2a;color:#e0d0a0;cursor:pointer;text-align:left;font-size:13px;margin-bottom:8px;">
                    🪝 <b>填充到前置Hook</b> — 将视频/图片填入Hook列
                </button>
                <button data-mode="fill_audio" style="padding:10px 16px;border:1px solid #5a8a5a;border-radius:8px;background:#2a4a3a;color:#a0e0c0;cursor:pointer;text-align:left;font-size:13px;">
                    🎙 <b>填充到音频</b> — 将视频填入音频列
                </button>` : ''}
            </div>
            <button data-mode="cancel" style="margin-top:12px;padding:6px 16px;border:1px solid #555;border-radius:6px;background:#333;color:#aaa;cursor:pointer;width:100%;font-size:13px;">取消</button>
        `;
        dialog.querySelectorAll('button').forEach(btn => {
            btn.addEventListener('click', () => {
                overlay.remove();
                resolve(btn.dataset.mode);
            });
        });
        overlay.appendChild(dialog);
        overlay.addEventListener('click', (e) => { if (e.target === overlay) { overlay.remove(); resolve('cancel'); } });
        document.body.appendChild(overlay);
    });
}

/** 将文件应用到任务 */
function _assignFileToTask(task, file, field) {
    const filePath = (typeof getFileNativePath === 'function') ? getFileNativePath(file) : (file.path || file.name);
    if (field === 'hook') {
        task.hookFile = filePath;
        if (task.hookSpeed == null) task.hookSpeed = 1;
        if (task.hookTransition == null) task.hookTransition = 'none';
        if (task.hookTransDuration == null) task.hookTransDuration = 0.5;
    } else if (field === 'bg') {
        task.bgPath = filePath;
        task.videoPath = filePath;
        try { task.bgSrcUrl = URL.createObjectURL(file); } catch (e) { }
    } else if (field === 'audio') {
        task.audioPath = filePath;
    } else if (field === 'srt') {
        task.srtPath = filePath;
        _readSrtFileToTask(task, file);
    } else if (field === 'txt') {
        task.txtPath = filePath;
        _readTxtFileToTask(task, file);
    }
}

/** 创建空行 */
function _createEmptyTask() {
    const state = window._reelsState;
    const ReelsOverlay = window.ReelsOverlay;
    const taskName = `card_${String((state.tasks || []).length + 1).padStart(3, '0')}`;
    return {
        baseName: taskName,
        fileName: `${taskName}.mp4`,
        bgPath: null, bgSrcUrl: null,
        audioPath: null, srtPath: null,
        segments: [],
        videoPath: null, srcUrl: null,
        overlays: [],
    };
}

/**
 * 批量分配文件到各行 — 支持多种模式
 */
async function _batchAssignFiles(files, field) {
    const state = window._reelsState;
    if (!state || !files || files.length === 0) return;

    files.sort((a, b) => (a.name || '').localeCompare(b.name || ''));

    const mode = await _showBatchModeDialog(files.length, field);
    if (mode === 'cancel') return;

    if (mode === 'applyall') {
        // 应用单个文件到所有现有行
        const file = files[0];
        for (const task of state.tasks) {
            _assignFileToTask(task, file, field);
        }
    } else if (mode === 'fill') {
        // 补全：填入空位
        let taskIdx = 0;
        for (const file of files) {
            // 找空位
            while (taskIdx < state.tasks.length) {
                const tk = state.tasks[taskIdx];
                let occupied = false;
                if (field === 'hook') occupied = !!tk.hookFile;
                else if (field === 'bg') occupied = !!(tk.bgPath || tk.videoPath);
                else if (field === 'audio') occupied = !!tk.audioPath;
                else if (field === 'srt') occupied = !!tk.srtPath;
                else if (field === 'txt') occupied = !!tk.txtPath;
                if (!occupied) break;
                taskIdx++;
            }
            if (taskIdx >= state.tasks.length) {
                const newTask = _createEmptyTask();
                state.tasks.push(newTask);
            }
            _assignFileToTask(state.tasks[taskIdx], file, field);
            taskIdx++;
        }
    } else if (mode === 'overwrite') {
        // 覆盖：从第1行开始
        for (let i = 0; i < files.length; i++) {
            if (i >= state.tasks.length) {
                state.tasks.push(_createEmptyTask());
            }
            _assignFileToTask(state.tasks[i], files[i], field);
        }
    } else if (mode === 'append') {
        // 添加新行
        for (const file of files) {
            const newTask = _createEmptyTask();
            _assignFileToTask(newTask, file, field);
            state.tasks.push(newTask);
        }
    } else if (mode === 'fill_audio') {
        // 将视频文件填充到音频列（补全空位）
        let taskIdx = 0;
        for (const file of files) {
            while (taskIdx < state.tasks.length) {
                if (!state.tasks[taskIdx].audioPath) break;
                taskIdx++;
            }
            if (taskIdx >= state.tasks.length) {
                state.tasks.push(_createEmptyTask());
            }
            _assignFileToTask(state.tasks[taskIdx], file, 'audio');
            taskIdx++;
        }
    } else if (mode === 'fill_hook') {
        // 将文件填充到Hook列（补全空位）
        let taskIdx = 0;
        for (const file of files) {
            while (taskIdx < state.tasks.length) {
                if (!state.tasks[taskIdx].hookFile) break;
                taskIdx++;
            }
            if (taskIdx >= state.tasks.length) {
                state.tasks.push(_createEmptyTask());
            }
            _assignFileToTask(state.tasks[taskIdx], file, 'hook');
            taskIdx++;
        }
    }

    _renderBatchTable();
    if (typeof _renderTaskList === 'function') _renderTaskList();
}

/**
 * 批量分配 TXT 文件到各行
 * 1. 先按 matchKey 尝试匹配已有行（同名视频/音频）
 * 2. 没匹配到则顺序找「有背景但还没有 txtContent」的行
 * 3. 还没有就新建行
 */
function _batchAssignTxtFiles(files) {
    const state = window._reelsState;
    if (!state) return;
    const ReelsOverlay = window.ReelsOverlay;

    function getMatchKey(name) {
        if (typeof window._buildAudioSubtitleMatchKey === 'function') {
            return window._buildAudioSubtitleMatchKey(name || '');
        }
        const n = String(name || '');
        const dot = n.lastIndexOf('.');
        return (dot > 0 ? n.substring(0, dot) : n).trim().toLowerCase();
    }

    files.sort((a, b) => (a.name || '').localeCompare(b.name || ''));

    let fallbackIdx = 0;
    for (const file of files) {
        const filePath = (typeof getFileNativePath === 'function') ? getFileNativePath(file) : (file.path || file.name);
        const key = getMatchKey(file.name);

        // 尝试按名字匹配已有行（用 matchKey 比较）
        let targetTask = state.tasks.find(t => {
            const bgKey = getMatchKey(t.bgPath || t.videoPath || '');
            const audioKey = getMatchKey(t.audioPath || '');
            const baseKey = getMatchKey(t.baseName || '');
            return (bgKey && bgKey === key) || (audioKey && audioKey === key) || (baseKey && baseKey === key);
        });

        // 没匹配到：顺序找「有背景但还没有 txtContent」的行
        if (!targetTask) {
            while (fallbackIdx < state.tasks.length) {
                const tk = state.tasks[fallbackIdx];
                if (!tk.txtContent && !tk.srtPath) {
                    targetTask = tk;
                    break;
                }
                fallbackIdx++;
            }
        }

        // 还没找到则新建行
        if (!targetTask) {
            const baseName = key || `card_${String(state.tasks.length + 1).padStart(3, '0')}`;
            targetTask = {
                baseName: baseName,
                fileName: `${baseName}.mp4`,
                bgPath: null, bgSrcUrl: null,
                audioPath: null, srtPath: null,
                segments: [],
                videoPath: null, srcUrl: null,
                overlays: [],
            };
            state.tasks.push(targetTask);
            fallbackIdx = state.tasks.length;
        }

        targetTask.txtPath = filePath;
        targetTask.aligned = false;

        // 异步读取文件内容
        _readTxtFileToTask(targetTask, file);

        fallbackIdx++;
    }

    _renderBatchTable();
    if (typeof _renderTaskList === 'function') _renderTaskList();
    alert(`✅ 已分配 ${files.length} 个 TXT 文案文件`);
}

/**
 * 点击单元格 → 选择一个文件
 */
function _pickSingleFile(idx, field) {
    const container = _batchTableState.container;
    if (!container) return;
    const fileInput = container.querySelector('#rbt-file-single');
    if (!fileInput) return;

    // 根据字段设置 accept
    if (field === 'bg') {
        fileInput.accept = '.mp4,.mov,.mkv,.avi,.wmv,.flv,.webm,.jpg,.jpeg,.png,.webp';
    } else if (field === 'audio') {
        fileInput.accept = '.mp3,.wav,.m4a,.aac,.flac,.ogg,.wma,.mp4,.mov,.mkv,.avi,.wmv,.flv,.webm';
    } else if (field === 'srt') {
        fileInput.accept = '.srt';
    } else if (field === 'txt') {
        fileInput.accept = '.txt';
    }

    _batchTableState._pendingSingle = { idx, field };
    fileInput.click();
}

/**
 * 分配单个文件到指定行
 */
function _assignSingleFile(idx, field, file) {
    const state = window._reelsState;
    if (!state || !state.tasks[idx]) return;
    const task = state.tasks[idx];
    const filePath = (typeof getFileNativePath === 'function') ? getFileNativePath(file) : (file.path || file.name);

    console.log(`[BatchTable] _assignSingleFile idx=${idx} field=${field} filePath=${filePath} file.name=${file.name}`);

    if (field === 'bg') {
        task.bgPath = filePath;
        task.videoPath = filePath;
        try { task.bgSrcUrl = URL.createObjectURL(file); } catch (e) { }
    } else if (field === 'audio') {
        task.audioPath = filePath;
    } else if (field === 'srt') {
        task.srtPath = filePath;
        _readSrtFileToTask(task, file);
        console.log(`[BatchTable] After _readSrtFileToTask: segments.length=${(task.segments||[]).length}`);
    } else if (field === 'txt') {
        task.txtPath = filePath;
        _readTxtFileToTask(task, file);
    }

    _renderBatchTable();
    if (typeof _renderTaskList === 'function') _renderTaskList();
}

/**
 * 读取 SRT 文件内容到 task.segments
 */
function _readSrtFileToTask(task, file) {
    const filePath = task.srtPath;
    console.log('[SRT] _readSrtFileToTask called, filePath:', filePath, 'hasFile:', !!file);

    // 优先用 FileReader — 直接读 File 对象的 blob 内容，不依赖路径，Electron 和浏览器通用
    if (file) {
        const reader = new FileReader();
        reader.onload = (ev) => {
            const content = ev.target.result;
            console.log('[SRT] FileReader result, content length:', content?.length);
            if (!content) { console.warn('[SRT] FileReader returned empty'); return; }
            _parseSrtContent(task, content);
            console.log('[SRT] ✅ loaded via FileReader, segments:', task.segments?.length);
            if (_batchTableState.visible) _renderBatchTable();
            if (typeof _renderTaskList === 'function') _renderTaskList();
        };
        reader.onerror = (err) => {
            console.warn('[SRT] FileReader error:', err, '— trying electronAPI fallback');
            _readSrtViaElectronAPI(task, filePath);
        };
        reader.readAsText(file);
        return;
    }

    // 回退：用 Electron 同步 API（当只有路径没有 File 对象时）
    _readSrtViaElectronAPI(task, filePath);
}

function _readSrtViaElectronAPI(task, filePath) {
    if (filePath && window.electronAPI && window.electronAPI.readFileText) {
        try {
            const content = window.electronAPI.readFileText(filePath);
            if (content) {
                _parseSrtContent(task, content);
                console.log('[SRT] ✅ loaded via electronAPI:', filePath, 'segments:', task.segments?.length);
                if (_batchTableState.visible) _renderBatchTable();
                if (typeof _renderTaskList === 'function') _renderTaskList();
                return;
            }
        } catch (e) {
            console.warn('[SRT] electronAPI.readFileText error:', e);
        }
    }
    console.warn('[SRT] ❌ Failed to load SRT:', filePath);
}

function _parseSrtContent(task, content) {
    // parseSRT 定义在 reels-canvas-renderer.js，暴露在 window.parseSRT
    const parser = (typeof parseSRT === 'function') ? parseSRT : window.parseSRT;
    if (parser) {
        const rawSegs = parser(content).map(seg => ({ ...seg, _timeUnit: 'sec' }));
        task.segments = window.ReelsSubtitleProcessor
            ? ReelsSubtitleProcessor.srtToSegmentsWithWords(rawSegs)
            : rawSegs;
    } else {
        console.error('[SRT] parseSRT function not found!');
        task.segments = [];
    }
}

/**
 * 读取 TXT 文件内容到 task.txtContent
 */
function _readTxtFileToTask(task, file) {
    const reader = new FileReader();
    reader.onload = (ev) => {
        task.txtContent = ev.target.result || '';
        task.aligned = false;
        // 刷新表格显示加载的文案
        if (_batchTableState.visible) _renderBatchTable();
    };
    reader.readAsText(file);
}

/**
 * 智能断行 (批量表格内用)
 * 英文 ~5 词/行，中文 ~16 字/行，已有换行保留
 */
function _rbtSmartLineBreak(text) {
    if (!text || typeof text !== 'string') return text;
    // 按行处理，保留已有换行
    const lines = text.split('\n');
    const result = [];
    for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.length <= 10) { result.push(trimmed); continue; }

        const cjkCount = (trimmed.match(/[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]/g) || []).length;
        const isCJK = cjkCount / trimmed.length > 0.3;

        if (isCJK) {
            const maxChars = 16;
            let pos = 0;
            while (pos < trimmed.length) {
                let end = Math.min(pos + maxChars, trimmed.length);
                if (end < trimmed.length) {
                    const chunk = trimmed.slice(pos, end + 4);
                    const breakAt = chunk.search(/[，。！？；、\s,\.!?;]/g);
                    if (breakAt > maxChars * 0.5) end = pos + breakAt + 1;
                }
                result.push(trimmed.slice(pos, end).trim());
                pos = end;
                while (pos < trimmed.length && trimmed[pos] === ' ') pos++;
            }
        } else {
            const words = trimmed.split(/\s+/);
            if (words.length <= 5) { result.push(trimmed); continue; }
            for (let i = 0; i < words.length; i += 5) {
                result.push(words.slice(i, i + 5).join(' '));
            }
        }
    }
    return result.filter(l => l).join('\n');
}

// ═══════════════════════════════════════════════════════
// 10. Batch Alignment (对齐字幕)
// ═══════════════════════════════════════════════════════

async function _batchAlignAllTasks() {
    // 先同步表格输入到 task
    _applyBatchTableChanges();

    const state = window._reelsState;
    if (!state) return;

    const alignSource = document.getElementById('rbt-align-source')?.value || 'audio';
    const forceRealign = document.getElementById('rbt-force-realign')?.checked || false;

    // 筛选需要对齐的任务: 有字幕文本 + 有音频源 + (未对齐 或 强制)
    const tasksToAlign = state.tasks.filter(t => {
        if (!t.txtContent || !t.txtContent.trim()) return false;
        if (t.aligned && !forceRealign) return false;
        const hasAudio = alignSource === 'video'
            ? (t.bgPath || t.videoPath)
            : t.audioPath;
        return !!hasAudio;
    });

    if (tasksToAlign.length === 0) {
        alert('没有需要对齐的任务。\n请确认：\n1. 字幕文本列有文案\n2. 有音频/视频文件\n3. 尚未对齐');
        return;
    }

    const progressEl = document.getElementById('rbt-align-progress');
    const alignBtn = document.getElementById('rbt-align-all-btn');
    if (alignBtn) { alignBtn.disabled = true; alignBtn.textContent = '⏳ 对齐中...'; }

    let ok = 0, fail = 0;
    const failDetails = [];
    const language = document.getElementById('rbt-align-lang')?.value || '英语';

    for (let i = 0; i < tasksToAlign.length; i++) {
        const task = tasksToAlign[i];
        const taskName = task.baseName || task.fileName || `任务${i + 1}`;
        if (progressEl) progressEl.textContent = `🔗 对齐 ${i + 1}/${tasksToAlign.length}: ${taskName}`;

        try {
            // 确定音频路径
            const audioPath = alignSource === 'video'
                ? (task.bgPath || task.videoPath)
                : task.audioPath;

            if (!audioPath) throw new Error('没有音频文件');

            // 自动断行文案
            task.txtContent = _rbtSmartLineBreak(task.txtContent);

            // 调用对齐API（使用 apiFetch + API_BASE）
            const audioDir = audioPath.replace(/[\\/][^\\/]+$/, '');
            const resp = await apiFetch(`${API_BASE}/subtitle/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    audio_path: audioPath,
                    source_text: task.txtContent,
                    language: language,
                    audio_cut_length: 5.0,
                    output_dir: audioDir,
                    force: task.aligned ? true : false,  // 已对齐过则强制重新转录
                }),
            });

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.error || '对齐失败');
            }

            const data = await resp.json();

            // 保存 SRT 路径并解析 segments
            if (data.files && data.files.length > 0) {
                const srtFile = data.files.find(f => f.endsWith('_source.srt')) || data.files[0];
                task.srtPath = srtFile;

                if (window.electronAPI && window.electronAPI.readFileText) {
                    const srtContent = await window.electronAPI.readFileText(srtFile);
                    const rawSegs = (typeof parseSRT === 'function' ? parseSRT(srtContent) : []).map(seg => ({ ...seg, _timeUnit: 'sec' }));
                    task.segments = window.ReelsSubtitleProcessor
                        ? ReelsSubtitleProcessor.srtToSegmentsWithWords(rawSegs)
                        : rawSegs;
                }
            }

            task.aligned = true;
            ok++;
        } catch (err) {
            console.error('[BatchAlign] Failed:', taskName, err);
            failDetails.push(`${taskName}: ${err.message}`);
            fail++;
        }
    }

    // 完成
    if (alignBtn) { alignBtn.disabled = false; alignBtn.textContent = '🔗 对齐字幕'; }
    if (progressEl) {
        progressEl.textContent = fail > 0
            ? `⚠️ 对齐完成 ${ok}/${tasksToAlign.length}，失败 ${fail}`
            : `✅ 全部对齐完成 (${ok}个)`;
        setTimeout(() => { if (progressEl) progressEl.textContent = ''; }, 8000);
    }

    _renderBatchTable();
    if (typeof _renderTaskList === 'function') _renderTaskList();

    if (fail > 0) {
        alert(`对齐完成 ${ok}/${tasksToAlign.length}\n\n失败:\n${failDetails.slice(0, 5).join('\n')}`);
    } else {
        alert(`✅ 字幕对齐完成 ${ok} 个任务\nSRT 文件已自动保存到音频文件所在目录`);
    }
}

// ═══════════════════════════════════════════════════════
// 10b. Batch selection helpers
// ═══════════════════════════════════════════════════════

function _getSelectedIndices() {
    return Array.from(_batchTableState.selectedRows).sort((a, b) => a - b);
}

function _updateBatchSelectCount() {
    const container = _batchTableState.container;
    if (!container) return;
    const el = container.querySelector('#rbt-selected-count');
    const exportBtn = container.querySelector('#rbt-export-selected-btn');
    const count = _batchTableState.selectedRows.size;
    if (el) el.textContent = count > 0 ? `已选 ${count} 行` : '';
    if (exportBtn) exportBtn.style.display = count > 0 ? '' : 'none';
}

// ═══════════════════════════════════════════════════════
// 10c. Material folder refresh
// ═══════════════════════════════════════════════════════

const _MAT_BG_EXTS = new Set(['mp4', 'mov', 'mkv', 'avi', 'wmv', 'flv', 'webm', 'jpg', 'jpeg', 'png', 'webp', 'gif', 'bmp']);
const _MAT_AUDIO_EXTS = new Set(['mp3', 'wav', 'm4a', 'aac', 'flac', 'ogg', 'wma']);
const _MAT_SRT_EXTS = new Set(['srt']);
const _MAT_TXT_EXTS = new Set(['txt']);

async function _refreshMaterialFolder() {
    const tab = _getActiveTab();
    if (!tab.materialDir) {
        alert('请先设置素材文件夹');
        return;
    }

    const refreshBtn = _batchTableState.container?.querySelector('#rbt-refresh-mat');
    if (refreshBtn) {
        refreshBtn.disabled = true;
        refreshBtn.textContent = '⏳ 扫描中...';
    }

    try {
        let files;
        if (window.electronAPI && window.electronAPI.scanDirectory) {
            files = await window.electronAPI.scanDirectory(tab.materialDir);
        } else {
            alert('请在桌面应用中使用此功能');
            return;
        }

        if (!files || files.length === 0) {
            alert('文件夹为空或无法读取');
            return;
        }

        // Classify files
        const classified = { bg: [], audio: [], srt: [], txt: [] };
        for (const f of files) {
            const ext = (f.name || '').split('.').pop().toLowerCase();
            if (_MAT_AUDIO_EXTS.has(ext)) classified.audio.push(f);
            else if (_VOICE_VIDEO_EXTS.has(ext) && _looksLikeVoiceTrack(f.name || '')) classified.audio.push(f);
            else if (_MAT_BG_EXTS.has(ext)) classified.bg.push(f);
            else if (_MAT_SRT_EXTS.has(ext)) classified.srt.push(f);
            else if (_MAT_TXT_EXTS.has(ext)) classified.txt.push(f);
        }

        const state = window._reelsState;
        const tasks = state.tasks || [];

        // Build a map: baseName → task index for existing tasks
        const existingMap = new Map();
        tasks.forEach((t, i) => {
            const bgBase = _baseFileName(t.bgPath || t.videoPath || '');
            const audioBase = _baseFileName(t.audioPath || '');
            if (bgBase) existingMap.set(bgBase, i);
            if (audioBase) existingMap.set(audioBase, i);
        });

        let newCount = 0, updateCount = 0;

        // Match by baseName: group files
        const groups = new Map(); // baseName → { bg, audio, srt, txt }
        const allFiles = [
            ...classified.bg.map(f => ({ ...f, type: 'bg' })),
            ...classified.audio.map(f => ({ ...f, type: 'audio' })),
            ...classified.srt.map(f => ({ ...f, type: 'srt' })),
            ...classified.txt.map(f => ({ ...f, type: 'txt' })),
        ];
        for (const f of allFiles) {
            const base = _baseFileName(f.name);
            if (!groups.has(base)) groups.set(base, {});
            groups.get(base)[f.type] = f;
        }

        // Process each group
        for (const [base, group] of groups) {
            let taskIdx = existingMap.get(base);
            let task;

            if (taskIdx != null) {
                task = tasks[taskIdx];
                // Update existing task if files changed
                let changed = false;
                if (group.bg && task.bgPath !== group.bg.path) {
                    task.bgPath = group.bg.path; task.videoPath = group.bg.path; task.bgSrcUrl = ''; changed = true;
                }
                if (group.audio && task.audioPath !== group.audio.path) {
                    task.audioPath = group.audio.path; changed = true;
                }
                if (group.srt && task.srtPath !== group.srt.path) {
                    task.srtPath = group.srt.path; changed = true;
                }
                if (changed) {
                    task._justRefreshed = true;
                    updateCount++;
                }
            } else {
                // Create new task
                task = _createEmptyTask();
                task.baseName = base;
                if (group.bg) { task.bgPath = group.bg.path; task.videoPath = group.bg.path; }
                if (group.audio) { task.audioPath = group.audio.path; }
                if (group.srt) { task.srtPath = group.srt.path; }
                task._justRefreshed = true;
                tasks.push(task);
                newCount++;
            }
        }

        tab.lastRefreshTime = Date.now();
        state.tasks = tasks;

        _renderBatchTable();

        // Clear _justRefreshed flags after animation
        setTimeout(() => {
            for (const t of state.tasks) delete t._justRefreshed;
        }, 3500);

        const summary = [];
        if (newCount > 0) summary.push(`🆕 新增 ${newCount} 行`);
        if (updateCount > 0) summary.push(`🔄 更新 ${updateCount} 行`);
        if (summary.length === 0) summary.push('✅ 没有新变化');
        alert(`刷新完成\n${summary.join('\n')}\n\n文件夹: ${tab.materialDir}\n共 ${files.length} 个文件`);

    } catch (err) {
        console.error('[BatchTable] Refresh error:', err);
        alert(`刷新失败: ${err.message}`);
    } finally {
        if (refreshBtn) {
            refreshBtn.disabled = false;
            refreshBtn.textContent = '🔄 一键刷新';
        }
    }
}

function _baseFileName(filePath) {
    if (!filePath) return '';
    const name = filePath.replace(/\\/g, '/').split('/').pop() || '';
    return name.replace(/\.[^.]+$/, '').toLowerCase().trim();
}

// ═══════════════════════════════════════════════════════
// 11. Utility
// ═══════════════════════════════════════════════════════

function _shortName(path) {
    if (!path) return '';
    const parts = path.replace(/\\/g, '/').split('/');
    const name = parts[parts.length - 1] || '';
    return name.length > 20 ? name.slice(0, 18) + '…' : name;
}

function _escHtml(str) {
    return String(str || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ═══════════════════════════════════════════════════════
// 11. CSS injection
// ═══════════════════════════════════════════════════════

let _batchTableCSSInjected = false;
function _injectBatchTableCSS() {
    if (_batchTableCSSInjected) return;
    _batchTableCSSInjected = true;
    const style = document.createElement('style');
    style.textContent = `
        .rbt-panel {
            display:flex; flex-direction:column; width:95vw; max-width:1600px; height:90vh;
            background:#0d0d2b; border:1px solid #1a1a4a; border-radius:16px;
            box-shadow:0 30px 80px rgba(0,0,0,0.7); margin:auto; overflow:hidden;
        }
        .rbt-header {
            display:flex; justify-content:space-between; align-items:flex-start;
            padding:16px 24px; border-bottom:1px solid #1a1a4a; background:#12123a;
            gap:12px;
        }
        .rbt-header h2 { margin:0; font-size:18px; color:#00D4FF; white-space:nowrap; flex-shrink:0; }
        .rbt-actions { display:none; /* overwritten by inline styles now */ }
        .rbt-btn {
            padding:3px 8px; border-radius:4px; border:1px solid #333; background:#222;
            color:#ccc; font-size:11px; cursor:pointer; transition:all .15s;
        }
        .rbt-btn:hover { background:#333; color:#fff; }
        .rbt-btn-accent { background:#00D4FF; color:#000; border-color:#00D4FF; font-weight:bold; }
        .rbt-btn-accent:hover { background:#00BBDD; }
        .rbt-btn-primary { background:#00D4FF; color:#000; border:none; font-weight:bold; padding:8px 24px; font-size:14px; }
        .rbt-btn-primary:hover { background:#00BBDD; }
        .rbt-btn-danger { border-color:#f44; color:#f44; }
        .rbt-btn-danger:hover { background:#f44; color:#fff; }
        .rbt-btn-folder { background:#2a6b3a; color:#8f8; border-color:#3a8b4a; font-weight:600; }
        .rbt-btn-folder:hover { background:#3a8b4a; color:#fff; }
        .rbt-btn-voice { background:#182838; border-color:#2a4058; color:#90b8d8; }
        .rbt-btn-voice:hover { background:#2a4058; color:#c0daf0; }
        .rbt-btn-other { background:#2a2038; border-color:#403058; color:#b8a0d0; }
        .rbt-btn-other:hover { background:#403058; color:#d8c8f0; }
        .rbt-table-wrap {
            flex:1; overflow:auto; padding:0;
        }
        .rbt-table {
            width:100%; border-collapse:collapse; font-size:12px;
        }
        .rbt-table thead { position:sticky; top:0; z-index:2; }
        .rbt-table th {
            padding:10px 8px; background:#1a1a4a; color:#aaa; font-weight:600;
            text-align:left; border-bottom:2px solid #2a2a6a; white-space:nowrap;
        }
        .rbt-table td {
            padding:6px 8px; border-bottom:1px solid #1a1a3a; vertical-align:top;
        }
        .rbt-row:hover { background:rgba(0,212,255,0.05); }
        .rbt-row-selected { background:rgba(0,212,255,0.1) !important; }
        .rbt-col-num { width:40px; text-align:center; color:#666; font-weight:bold; }
        .rbt-col-hook { width:140px; }
        .rbt-col-bg { width:140px; }
        .rbt-col-audio { width:120px; }
        .rbt-col-srt { width:120px; }
        .rbt-col-txtcontent { min-width:180px; max-width:280px; }
        .rbt-col-bgm { width:140px; }
        .rbt-col-title { width:200px; }
        .rbt-col-body { min-width:250px; }
        .rbt-col-footer { min-width:180px; }
        .rbt-col-scroll-title { width:200px; }
        .rbt-col-scroll-body { min-width:250px; }
        .rbt-col-dur { width:70px; text-align:center; }
        .rbt-col-tpl { width:120px; }
        .rbt-col-act { width:70px; white-space:nowrap; }
        .rbt-file-name {
            display:block; padding:4px 6px; background:#12122a; border:1px dashed #2a2a4a;
            border-radius:4px; color:#8888cc; font-size:11px; min-height:28px;
            overflow:hidden; text-overflow:ellipsis; white-space:nowrap; cursor:default;
        }
        .rbt-placeholder { color:#444; font-style:italic; }
        .rbt-textarea {
            width:100%; padding:4px 6px; background:#12122a; border:1px solid #2a2a4a;
            border-radius:4px; color:#ddd; font-size:12px; resize:vertical; min-height:40px;
            font-family:inherit;
        }
        .rbt-textarea:focus { border-color:#00D4FF; outline:none; }
        .rbt-select {
            width:100%; padding:4px; background:#12122a; border:1px solid #2a2a4a;
            border-radius:4px; color:#ddd; font-size:11px;
        }
        .rbt-row-btn {
            padding:2px 6px; background:none; border:1px solid transparent;
            cursor:pointer; font-size:14px; border-radius:4px;
        }
        .rbt-row-btn:hover { background:rgba(255,255,255,0.1); }
        .rbt-field-clear {
            background:none; border:none; color:#888; cursor:pointer;
            font-size:11px; padding:1px 4px; border-radius:3px; line-height:1;
            flex-shrink:0; opacity:0.6; transition:all 0.15s;
        }
        .rbt-field-clear:hover { color:#ff6b6b; opacity:1; background:rgba(255,107,107,0.15); }
        .rbt-footer {
            display:flex; justify-content:space-between; align-items:center;
            padding:12px 24px; border-top:1px solid #1a1a4a; background:#12123a;
        }
        #rbt-count { color:#888; font-size:13px; }
        .rbt-droppable.rbt-drag-over {
            border-color:#00D4FF !important; background:rgba(0,212,255,0.1) !important;
        }
        .rbt-hidden-input { display:none !important; }
        .rbt-sep {
            display:inline-block; width:1px; height:20px; background:#333; margin:0 4px;
        }
        .rbt-group-label {
            font-size:10px; color:#888; font-weight:600; white-space:nowrap;
            border-bottom:1px solid #555; padding-bottom:1px; margin-right:2px;
        }
        .rbt-file-name { cursor:pointer !important; }
        .rbt-file-name:hover { border-color:#00D4FF !important; color:#00D4FF !important; }
        .rbt-footer-hint { font-size:11px; color:#555; }

        /* ══ Tab bar ══ */
        .rbt-tabbar {
            display:flex; align-items:stretch; background:#08081a; border-bottom:2px solid #1a1a4a;
            padding:0 8px; min-height:36px; overflow-x:auto; flex-shrink:0;
        }
        .rbt-tabs-scroll {
            display:flex; align-items:stretch; gap:2px; flex:1; min-width:0;
        }
        .rbt-tab {
            display:flex; align-items:center; gap:6px; padding:6px 16px; cursor:pointer;
            font-size:12px; color:#888; border:1px solid transparent; border-bottom:none;
            border-radius:8px 8px 0 0; transition:all .15s; position:relative; white-space:nowrap;
            background:transparent; user-select:none;
        }
        .rbt-tab:hover { color:#ccc; background:rgba(255,255,255,0.05); }
        .rbt-tab-active {
            color:#00D4FF !important; background:#12123a !important;
            border-color:#1a1a4a #1a1a4a transparent; font-weight:600;
        }
        .rbt-tab-active::after {
            content:''; position:absolute; bottom:-2px; left:0; right:0; height:2px; background:#00D4FF;
        }
        .rbt-tab-close {
            font-size:14px; line-height:1; opacity:0.4; transition:all .15s; padding:0 2px;
        }
        .rbt-tab-close:hover { opacity:1; color:#ff6b6b; }
        .rbt-tab-add {
            color:#555; font-size:16px; font-weight:700; padding:6px 12px;
        }
        .rbt-tab-add:hover { color:#00D4FF; }

        /* ══ Material folder bar ══ */
        .rbt-material-bar {
            display:flex; align-items:center; gap:8px; padding:6px 16px;
            background:#0a0a22; border-bottom:1px solid #1a1a3a; flex-shrink:0;
        }
        .rbt-mat-path {
            font-size:12px; color:#a0a0d0; background:#12122a; padding:3px 10px;
            border-radius:4px; border:1px solid #2a2a4a; max-width:300px;
            overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
        }
        .rbt-btn-refresh {
            background:linear-gradient(135deg, #1a6b3a, #2a8b4a) !important;
            color:#8f8 !important; border-color:#3a8b4a !important; font-weight:600;
            transition:all .2s;
        }
        .rbt-btn-refresh:hover:not(:disabled) {
            background:linear-gradient(135deg, #2a8b4a, #3aab5a) !important;
            box-shadow:0 0 12px rgba(0,255,100,0.3);
        }
        .rbt-btn-refresh:disabled { opacity:0.4; cursor:not-allowed; }
        .rbt-refresh-time { font-size:10px; color:#666; margin-left:4px; }

        /* ══ Batch actions bar ══ */
        .rbt-batch-actions-bar {
            display:flex; align-items:center; gap:8px; padding:6px 16px;
            background:#0e0e28; border-bottom:1px solid #1a1a3a; flex-shrink:0;
        }
        .rbt-batch-label {
            display:flex; align-items:center; gap:4px; font-size:11px; color:#aaa; cursor:pointer;
        }

        /* ══ Row refresh animation ══ */
        .rbt-row-refreshed {
            animation: rbt-refresh-glow 3s ease-out;
        }
        @keyframes rbt-refresh-glow {
            0%   { background:rgba(0,255,100,0.25); box-shadow:inset 0 0 20px rgba(0,255,100,0.15); }
            30%  { background:rgba(0,255,100,0.15); }
            100% { background:transparent; box-shadow:none; }
        }

        /* ══ Drag handle & reorder ══ */
        .rbt-col-drag { width:24px; text-align:center; padding:0 2px !important; }
        .rbt-drag-handle {
            cursor:grab; color:#555; font-size:14px; user-select:none;
            display:inline-block; padding:4px 2px; line-height:1; transition:color .15s;
        }
        .rbt-drag-handle:hover { color:#00D4FF; }
        .rbt-drag-handle:active { cursor:grabbing; }
        .rbt-row.rbt-dragging {
            opacity:0.4; background:rgba(0,212,255,0.05) !important;
        }
        .rbt-row.rbt-drag-over-row {
            border-top:2px solid #00D4FF !important;
            background:rgba(0,212,255,0.08) !important;
        }
    `;
    document.head.appendChild(style);
}

// ═══════════════════════════════════════════════════════
// 12. Config Save / Load / Auto-save
// ═══════════════════════════════════════════════════════

const BATCH_CONFIG_KEY = 'reels_batch_config_autosave';

/** 序列化任务数据（去掉不可序列化的部分） */
function _serializeTasks(tasks) {
    return (tasks || []).map(t => {
        const o = { ...t };
        // 去掉 DOM / Video 等不可序列化的引用
        delete o._video;
        delete o._bgThumb;
        // 去掉 blob: URL（页面刷新后失效，渲染时会从 bgPath 重新生成 file:// URL）
        if (o.bgSrcUrl && String(o.bgSrcUrl).startsWith('blob:')) o.bgSrcUrl = null;
        if (o.srcUrl && String(o.srcUrl).startsWith('blob:')) o.srcUrl = null;
        return o;
    });
}

/** 自动保存到 localStorage (含所有标签页) */
function _batchAutoSave() {
    try {
        // Sync current tasks to active tab first
        _syncTasksToActiveTab();
        const data = {
            timestamp: new Date().toISOString(),
            version: '2.0',
            activeTabId: _batchTableState.activeTabId,
            nextTabId: _batchTableState.nextTabId,
            tabs: _batchTableState.tabs.map(tab => ({
                id: tab.id,
                name: tab.name,
                materialDir: tab.materialDir || '',
                lastRefreshTime: tab.lastRefreshTime || null,
                tasks: _serializeTasks(tab.tasks),
            })),
        };
        localStorage.setItem(BATCH_CONFIG_KEY, JSON.stringify(data));
    } catch (e) {
        console.warn('[BatchTable] Auto-save failed:', e.message);
    }
}

/** 自动恢复 */
function _batchAutoRestore() {
    try {
        const raw = localStorage.getItem(BATCH_CONFIG_KEY);
        if (!raw) return;
        const data = JSON.parse(raw);

        if (data.version === '2.0' && data.tabs && data.tabs.length > 0) {
            // v2: multi-tab format
            _batchTableState.tabs = data.tabs.map(t => ({
                id: t.id,
                name: t.name,
                materialDir: t.materialDir || '',
                lastRefreshTime: t.lastRefreshTime || null,
                tasks: t.tasks || [],
            }));
            _batchTableState.activeTabId = data.activeTabId || _batchTableState.tabs[0].id;
            _batchTableState.nextTabId = data.nextTabId || _batchTableState.tabs.length + 1;
            // Load active tab's tasks
            const activeTab = _getActiveTab();
            if (activeTab && activeTab.tasks.length > 0) {
                const existing = window._reelsState.tasks || [];
                if (existing.length === 0) {
                    window._reelsState.tasks = activeTab.tasks.map(t => ({ ...t }));
                }
            }
            console.log(`[BatchTable] Auto-restored ${_batchTableState.tabs.length} tabs from ${data.timestamp}`);
        } else if (data.tasks && data.tasks.length > 0) {
            // v1: legacy single-list format — migrate to tab
            const existing = window._reelsState.tasks || [];
            if (existing.length === 0) {
                window._reelsState.tasks = data.tasks;
                _batchTableState.tabs[0].tasks = data.tasks;
                console.log(`[BatchTable] Auto-restored ${data.tasks.length} tasks (v1) from ${data.timestamp}`);
            }
        }
    } catch (e) {
        console.warn('[BatchTable] Auto-restore failed:', e.message);
    }
}

/** 导出所有标签页为 JSON 工程文件 */
function _batchExportConfig() {
    _syncTasksToActiveTab();
    const totalTasks = _batchTableState.tabs.reduce((sum, t) => sum + (t.tasks || []).length, 0);
    if (totalTasks === 0) {
        alert('没有任务可以保存');
        return;
    }
    const data = {
        version: '2.0',
        timestamp: new Date().toISOString(),
        activeTabId: _batchTableState.activeTabId,
        nextTabId: _batchTableState.nextTabId,
        tabs: _batchTableState.tabs.map(tab => ({
            id: tab.id,
            name: tab.name,
            materialDir: tab.materialDir || '',
            lastRefreshTime: tab.lastRefreshTime || null,
            tasks: _serializeTasks(tab.tasks),
        })),
    };
    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `batch_project_${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
    console.log(`[BatchTable] Exported project: ${_batchTableState.tabs.length} tabs, ${totalTasks} tasks`);
}

/** 从 JSON 文件导入工程配置 */
function _batchImportConfig(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            const data = JSON.parse(e.target.result);

            if (data.version === '2.0' && data.tabs && data.tabs.length > 0) {
                // v2: multi-tab project
                if (!confirm(`将加载 ${data.tabs.length} 个标签页的工程文件。\n\n确定 = 替换全部\n取消 = 放弃`)) return;
                _batchTableState.tabs = data.tabs.map(t => ({
                    id: t.id,
                    name: t.name,
                    materialDir: t.materialDir || '',
                    lastRefreshTime: t.lastRefreshTime || null,
                    tasks: t.tasks || [],
                }));
                _batchTableState.activeTabId = data.activeTabId || _batchTableState.tabs[0].id;
                _batchTableState.nextTabId = data.nextTabId || _batchTableState.tabs.length + 1;
                const activeTab = _getActiveTab();
                _loadTabTasks(activeTab);
                _renderBatchTable();
                const total = _batchTableState.tabs.reduce((s, t) => s + (t.tasks || []).length, 0);
                alert(`✅ 成功加载工程: ${_batchTableState.tabs.length} 个标签页, ${total} 个任务\n(${data.timestamp || ''})`);
            } else {
                // v1: legacy single-task-list
                const tasks = data.tasks;
                if (!Array.isArray(tasks) || tasks.length === 0) {
                    alert('配置文件中没有任务数据');
                    return;
                }
                const mode = (window._reelsState.tasks || []).length > 0
                    ? confirm(`当前有 ${window._reelsState.tasks.length} 个任务。\n\n确定 = 替换全部\n取消 = 追加到末尾`)
                        ? 'replace'
                        : 'append'
                    : 'replace';

                if (mode === 'replace') {
                    window._reelsState.tasks = tasks;
                } else {
                    window._reelsState.tasks = (window._reelsState.tasks || []).concat(tasks);
                }
                _syncTasksToActiveTab();
                _renderBatchTable();
                alert(`成功加载 ${tasks.length} 个任务 (${data.timestamp || ''})`);
            }
        } catch (err) {
            alert(`配置文件解析失败: ${err.message}`);
        }
    };
    reader.readAsText(file);
}

// ═══════════════════════════════════════════════════════
// 13. Auto-initialize
// ═══════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    _batchAutoRestore();
    _initBatchTable();

    // 每30秒自动保存一次（安全网）
    setInterval(() => {
        if (window._reelsState && (window._reelsState.tasks || []).length > 0) {
            _batchAutoSave();
        }
    }, 30000);
});

// 关闭/刷新前保存
window.addEventListener('beforeunload', () => {
    if (window._reelsState && (window._reelsState.tasks || []).length > 0) {
        _batchAutoSave();
    }
});

// Expose
window.reelsToggleBatchTable = reelsToggleBatchTable;
