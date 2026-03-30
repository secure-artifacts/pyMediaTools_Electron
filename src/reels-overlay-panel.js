/**
 * reels-overlay-panel.js — 覆层属性面板
 *
 * 移植自 AutoSub_v8 属性面板 (PyQt6 prop_panel → HTML DOM)
 *
 * 功能:
 *   - 覆层列表 (添加/删除/选中)
 *   - 文本覆层属性: 内容、字体、大小、颜色、描边、阴影、背景、动画
 *   - 图片覆层属性: 源路径、缩放、翻转、混合模式
 *   - 时间范围: 起止时间编辑
 *   - 变换: X/Y/宽/高/旋转/不透明度
 */

/** 四舍五入到1位小数，避免浮点精度问题 (如 19.700000000000003) */
function _ropRound(v) {
    return Math.round((parseFloat(v) || 0) * 10) / 10;
}

class ReelsOverlayPanel {
    constructor(containerEl, videoCanvas) {
        this.container = containerEl;
        this.videoCanvas = videoCanvas;
        this._selectedOv = null;
        this._init();
    }

    _init() {
        this.container.innerHTML = `
        <div class="rop-panel">
            <!-- 覆层组预设 (多层) -->
            <div class="rop-section">
                <div class="rop-group">
                    <div class="rop-group-title">📦 覆层组预设</div>
                    <div style="display:flex;gap:4px;align-items:center;">
                        <select id="rop-group-preset-select" class="rop-select" style="flex:1;"></select>
                        <button class="btn btn-secondary rop-btn" id="rop-group-preset-load" style="padding:2px 8px;">加载</button>
                    </div>
                    <div style="display:flex;gap:4px;margin-top:4px;">
                        <button class="btn btn-secondary rop-btn" id="rop-group-preset-save" style="flex:1;">保存</button>
                        <button class="btn btn-secondary rop-btn" id="rop-group-preset-del" style="flex:1;">删除</button>
                        <button class="btn btn-secondary rop-btn" id="rop-group-preset-import" style="flex:1;">导入</button>
                        <button class="btn btn-secondary rop-btn" id="rop-group-preset-export" style="flex:1;">导出</button>
                    </div>
                </div>
            </div>

            <!-- 覆层列表 -->
            <div class="rop-section">
                <div class="rop-header">
                    <span>📐 覆层列表</span>
                    <div class="rop-header-actions">
                        <button class="btn btn-secondary rop-btn" id="rop-add-text" title="添加文本覆层">T+</button>
                        <button class="btn btn-secondary rop-btn" id="rop-add-textcard" title="添加文案卡片" style="background:#FFD700;color:#000;">📋+</button>
                        <button class="btn btn-secondary rop-btn" id="rop-add-image" title="添加图片覆层">🖼+</button>
                        <button class="btn btn-secondary rop-btn" id="rop-add-scroll" title="添加滚动字幕" style="background:#FF6B35;color:#fff;">🔄+</button>
                        <button class="btn btn-secondary rop-btn" id="rop-batch-import" title="从表格批量导入文案" style="background:#00D4FF;color:#000;">📋批量</button>
                    </div>
                </div>
                <div id="rop-overlay-list" class="rop-list"></div>
            </div>

            <!-- 属性编辑 -->
            <div id="rop-props" class="rop-section" style="display:none;">
                <div class="rop-header"><span>⚙️ 属性</span></div>

                <!-- 固定文案标记 -->
                <div id="rop-fixed-text-group" class="rop-group" style="display:none;">
                    <div style="display:flex;align-items:center;gap:8px;">
                        <input type="checkbox" id="rop-fixed-text" style="width:16px;height:16px;cursor:pointer;">
                        <label for="rop-fixed-text" style="font-size:12px;color:#ccc;cursor:pointer;">🔒 固定文案 <span style="color:#888;font-size:11px;">— 勾选后文案随预设保存/加载</span></label>
                    </div>
                </div>

                <!-- 变换 -->
                <div class="rop-group">
                    <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;">
                        <div class="rop-group-title" style="margin:0;">变换</div>
                        <div style="display:flex;gap:4px;">
                            <button class="rop-reset-all" id="rop-reset-transform" title="恢复默认位置和大小">↺ 默认</button>
                            <button class="rop-reset-all" id="rop-fill-screen" title="一键铺满画布">📐 全屏填充</button>
                        </div>
                    </div>
                    <div class="rop-grid" style="margin-top:6px;">
                        <label id="rop-xy-label-x">位置X</label><input type="number" id="rop-x" class="rop-input" step="1">
                        <label id="rop-xy-label-y">位置Y</label><input type="number" id="rop-y" class="rop-input" step="1">
                        <label id="rop-wh-label-w">宽度</label><input type="number" id="rop-w" class="rop-input" step="1">
                        <label id="rop-wh-label-h">高度</label><input type="number" id="rop-h" class="rop-input" step="1">
                        <label>旋转</label><input type="number" id="rop-rotation" class="rop-input" min="-360" max="360" value="0">
                        <label>不透明</label>
                        <div style="display:flex;align-items:center;gap:6px;">
                            <input type="range" id="rop-opacity" class="rop-range" min="0" max="100" value="100" style="flex:1;">
                            <span id="rop-opacity-val" style="min-width:36px;text-align:right;font-size:12px;color:#aaa;">100%</span>
                        </div>
                        <label id="rop-scale-label" style="display:none;">缩放%</label>
                        <div id="rop-scale-wrap" style="display:none;align-items:center;gap:6px;">
                            <input type="range" id="rop-scale" class="rop-range" min="10" max="1000" value="100" style="flex:1;">
                            <span id="rop-scale-val" style="min-width:44px;text-align:right;font-size:12px;color:#aaa;">100%</span>
                        </div>
                        <div id="rop-time-in-transform">
                        <label>开始(s)</label><input type="number" id="rop-start" class="rop-input" step="0.1" min="0">
                        <label>结束(s)</label><input type="number" id="rop-end" class="rop-input" step="0.1" min="0">
                        </div>
                    </div>
                </div>


                <!-- 文本属性 (文本覆层独有) -->
                <div id="rop-text-props" class="rop-group" style="display:none;">
                    <div class="rop-group-title">文本</div>
                    <textarea id="rop-content" class="rop-textarea" rows="3" placeholder="覆层文本内容"></textarea>
                    <div class="rop-grid">
                        <label>字体</label>
                        <select id="rop-font" class="rop-select">
                            <option>Arial</option><option>Helvetica</option><option>Impact</option>
                            <option>Roboto</option><option>Open Sans</option>
                        </select>
                        <label>字号</label><input type="number" id="rop-fontsize" class="rop-input" min="8" max="300" value="40">
                        <label>颜色</label><input type="color" id="rop-color" class="rop-color" value="#ffffff">
                        <label>粗体</label><input type="checkbox" id="rop-bold">
                        <label>字重</label>
                        <select id="rop-font-weight" class="rop-select">
                            <option value="100">Thin</option><option value="200">ExtraLight</option><option value="300">Light</option>
                            <option value="400">Regular</option><option value="500">Medium</option><option value="600">SemiBold</option>
                            <option value="700" selected>Bold</option><option value="800">ExtraBold</option><option value="900">Black</option>
                        </select>
                        <label>描边色</label><input type="color" id="rop-stroke-color" class="rop-color" value="#000000">
                        <label>描边宽</label><input type="number" id="rop-stroke-width" class="rop-input" min="0" max="20" step="0.5" value="0">
                        <label>阴影色</label><input type="color" id="rop-shadow-color" class="rop-color" value="#000000">
                        <label>阴影模糊</label><input type="number" id="rop-shadow-blur" class="rop-input" min="0" max="50" value="0">
                    </div>
                </div>

                <!-- 图片属性 (图片覆层独有) -->
                <div id="rop-image-props" class="rop-group" style="display:none;">
                    <div class="rop-group-title">图片</div>
                    <div class="rop-grid">
                        <label>水平翻转</label><input type="checkbox" id="rop-flip-h">
                        <label>垂直翻转</label><input type="checkbox" id="rop-flip-v">
                        <label>混合模式</label>
                        <select id="rop-blend" class="rop-select">
                            <option value="source-over">正常</option>
                            <option value="multiply">正片叠底</option>
                            <option value="screen">滤色</option>
                            <option value="overlay">叠加</option>
                        </select>
                    </div>
                </div>

                <!-- 动画 -->
                <div class="rop-group">
                    <div class="rop-group-title">动画</div>
                    <div class="rop-grid">
                        <label>入场</label>
                        <select id="rop-anim-in" class="rop-select">
                            <option value="none">无</option><option value="fade">淡入</option>
                            <option value="pop">弹出</option><option value="slide_up">上滑</option>
                            <option value="slide_down">下滑</option><option value="slide_left">左滑</option>
                            <option value="slide_right">右滑</option>
                        </select>
                        <label>出场</label>
                        <select id="rop-anim-out" class="rop-select">
                            <option value="none">无</option><option value="fade">淡出</option>
                            <option value="pop">弹出</option><option value="slide_up">上滑</option>
                            <option value="slide_down">下滑</option>
                        </select>
                        <label>入场时长</label><input type="number" id="rop-anim-in-dur" class="rop-input" min="0" max="5" step="0.05" value="0.3">
                        <label>出场时长</label><input type="number" id="rop-anim-out-dur" class="rop-input" min="0" max="5" step="0.05" value="0.3">
                    </div>
                </div>

                <!-- 滚动字幕属性 (scroll覆层独有) -->
                <div id="rop-scroll-props" class="rop-group" style="display:none;">
                    <div class="rop-group-title">🔄 滚动字幕</div>

                    <!-- ① 内容：标题在前，正文在后 -->
                    <div class="rop-group-title" style="margin-top:4px;">📌 标题</div>
                    <div class="rop-grid">
                        <label>标题文字</label><input type="text" id="rop-scroll-title" class="rop-input" placeholder="留空=无标题">
                        <label>固定标题</label>
                        <div style="display:flex;align-items:center;gap:6px;">
                            <input type="checkbox" id="rop-scroll-title-fixed" checked>
                            <span style="font-size:11px;color:#888;">标题不参与滚动</span>
                        </div>
                        <label>标题字号</label><input type="number" id="rop-scroll-title-fontsize" class="rop-input" min="8" max="300" value="56">
                        <label>标题颜色</label><input type="color" id="rop-scroll-title-color" class="rop-color" value="#ffffff">
                        <label>标题字重</label>
                        <select id="rop-scroll-title-weight" class="rop-select">
                            <option value="400">Regular</option><option value="500">Medium</option>
                            <option value="600">SemiBold</option><option value="700" selected>Bold</option>
                            <option value="800">ExtraBold</option><option value="900">Black</option>
                        </select>
                        <label>标题间距</label><input type="number" id="rop-scroll-title-gap" class="rop-input" min="0" max="200" step="5" value="20">
                    </div>

                    <div class="rop-group-title" style="margin-top:6px;">📝 正文</div>
                    <textarea id="rop-scroll-content" class="rop-textarea" rows="4" placeholder="滚动文字内容（正文）"></textarea>
                    <div class="rop-grid">
                        <label>字体</label>
                        <select id="rop-scroll-font" class="rop-select"></select>
                        <label>字号</label><input type="number" id="rop-scroll-fontsize" class="rop-input" min="8" max="300" value="40">
                        <label>颜色</label><input type="color" id="rop-scroll-color" class="rop-color" value="#ffffff">
                        <label>粗体</label><input type="checkbox" id="rop-scroll-bold">
                        <label>字重</label>
                        <select id="rop-scroll-weight" class="rop-select">
                            <option value="100">Thin</option><option value="200">ExtraLight</option><option value="300">Light</option>
                            <option value="400" selected>Regular</option><option value="500">Medium</option><option value="600">SemiBold</option>
                            <option value="700">Bold</option><option value="800">ExtraBold</option><option value="900">Black</option>
                        </select>
                        <label>对齐</label>
                        <select id="rop-scroll-align" class="rop-select">
                            <option value="center">居中</option><option value="left">左对齐</option><option value="right">右对齐</option>
                        </select>
                        <label>行距</label><input type="number" id="rop-scroll-linespacing" class="rop-input" min="0" max="50" step="1" value="6">
                        <label>文本宽度</label><input type="number" id="rop-scroll-textw" class="rop-input" min="100" max="1920" step="10" value="900">
                    </div>

                    <!-- ② 文字效果：描边 + 阴影 -->
                    <div class="rop-group-title" style="margin-top:8px;">✨ 文字效果</div>
                    <div class="rop-grid">
                        <label>描边色</label><input type="color" id="rop-scroll-stroke-color" class="rop-color" value="#000000">
                        <label>描边宽</label><input type="number" id="rop-scroll-stroke-width" class="rop-input" min="0" max="20" step="0.5" value="0">
                        <label>阴影</label><input type="checkbox" id="rop-scroll-shadow">
                        <label>阴影色</label><input type="color" id="rop-scroll-shadow-color" class="rop-color" value="#000000">
                        <label>阴影模糊</label><input type="number" id="rop-scroll-shadow-blur" class="rop-input" min="0" max="50" value="4">
                    </div>

                    <!-- ③ 滚动运动：位置参数 -->
                    <div class="rop-group-title" style="margin-top:8px;">📍 滚动运动
                        <button id="rop-scroll-show-end" class="btn btn-secondary" style="float:right;padding:1px 8px;font-size:11px;border-radius:4px;">👁 显示终点</button>
                    </div>
                    <div class="rop-grid">
                        <label>开始(s)</label><input type="number" id="rop-scroll-start-time" class="rop-input" step="0.1" min="0">
                        <label>结束(s)</label><input type="number" id="rop-scroll-end-time" class="rop-input" step="0.1" min="0">
                        <label title="文字开始向上滚动时的初始Y坐标">起始Y</label><input type="number" id="rop-scroll-from-y" class="rop-input" step="10" value="960">
                        <label title="文字向上滚动最终消失或停留的结束Y坐标">结束Y</label><input type="number" id="rop-scroll-to-y" class="rop-input" step="10" value="-200">
                        <label>X位置</label><input type="number" id="rop-scroll-from-x" class="rop-input" step="10">
                        <!-- 速度参数已移除：速度由 距离÷时间 自动决定 -->
                        <input type="hidden" id="rop-scroll-speed" value="1">
                        <input type="hidden" id="rop-scroll-speed-num" value="1">
                    </div>

                    <!-- ④ 智能适配 -->
                    <div class="rop-group-title" style="margin-top:8px;">🧠 智能适配</div>
                    <div class="rop-grid">
                        <label>自动停止</label>
                        <div style="display:flex;align-items:center;gap:6px;">
                            <input type="checkbox" id="rop-scroll-auto-stop">
                            <span style="font-size:11px;color:#888;">文字全显示后停止滚动</span>
                        </div>
                        <label>字号自适应</label>
                        <div style="display:flex;align-items:center;gap:6px;">
                            <input type="checkbox" id="rop-scroll-auto-fit">
                            <span style="font-size:11px;color:#888;">自动缩小确保全部显示</span>
                        </div>
                        <label>最小字号</label><input type="number" id="rop-scroll-min-fontsize" class="rop-input" min="8" max="200" value="16">
                    </div>

                    <!-- ⑤ 羽化 -->
                    <div class="rop-group-title" style="margin-top:8px;">✂️ 羽化</div>
                    <div class="rop-grid">
                        <label>上羽化</label><input type="number" id="rop-scroll-feather-top" class="rop-input" min="0" max="500" step="10" value="80">
                        <label>下羽化</label><input type="number" id="rop-scroll-feather-bottom" class="rop-input" min="0" max="500" step="10" value="80">
                    </div>

                    <!-- ⑥ 卡片背景 -->
                    <div class="rop-group-title" style="margin-top:8px;">📋 卡片背景</div>
                    <div class="rop-grid">
                        <label>启用</label><input type="checkbox" id="rop-scroll-bg-enabled">
                        <label>颜色</label><input type="color" id="rop-scroll-bg-color" class="rop-color" value="#000000">
                        <label>透明度%</label><input type="number" id="rop-scroll-bg-opacity" class="rop-input" min="0" max="100" value="75">
                        <label>圆角</label><input type="number" id="rop-scroll-bg-radius" class="rop-input" min="0" max="200" value="12">
                        <label>上边距</label><input type="number" id="rop-scroll-bg-pad-top" class="rop-input" min="0" max="500" value="55">
                        <label>下边距</label><input type="number" id="rop-scroll-bg-pad-bottom" class="rop-input" min="0" max="500" value="55">
                        <label>左边距</label><input type="number" id="rop-scroll-bg-pad-left" class="rop-input" min="0" max="500" value="16">
                        <label>右边距</label><input type="number" id="rop-scroll-bg-pad-right" class="rop-input" min="0" max="500" value="16">
                        <label>全屏蒙版</label>
                        <div style="display:flex;align-items:center;gap:6px;">
                            <input type="checkbox" id="rop-scroll-bg-fullscreen">
                            <span style="font-size:11px;color:#888;">背景铺满整个画面</span>
                        </div>
                    </div>
                </div>

                <!-- 文案卡片属性 (textcard覆层独有) -->
                <div id="rop-textcard-props" class="rop-group" style="display:none;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div class="rop-group-title" style="margin:0;">📋 卡片背景</div>
                        <button class="rop-reset-all" id="rop-card-reset-all" title="全部恢复默认">↺ 默认</button>
                    </div>
                    <div class="rop-grid" style="margin-top:6px;">
                        <label>颜色</label><input type="color" id="rop-card-color" class="rop-color rop-defaultable" data-default="#ffffff" value="#ffffff">
                        <label>透明度 %</label>
                        <div class="rop-slider-combo">
                            <input type="range" id="rop-card-opacity" class="rop-range rop-defaultable" data-default="80" min="0" max="100" value="80">
                            <input type="number" class="rop-num-readout" data-link="rop-card-opacity" min="0" max="100" value="80">
                            <button class="rop-reset-btn" data-target="rop-card-opacity" title="恢复默认">↺</button>
                        </div>
                        <label>全屏蒙版</label><input type="checkbox" id="rop-fullscreen-mask" class="rop-defaultable" data-default="false">
                    </div>
                    <div class="rop-group-title" style="margin-top:8px;">⬰ 圆角</div>
                    <div class="rop-grid">
                        <label>圆角</label>
                        <div class="rop-slider-combo"><input type="range" id="rop-radius-all" class="rop-range rop-defaultable" data-default="33" min="0" max="200" value="33"><input type="number" class="rop-num-readout" data-link="rop-radius-all" min="0" max="200" value="33"><button class="rop-reset-btn" data-target="rop-radius-all" title="恢复默认">↺</button></div>
                    </div>
                    <div class="rop-group-title" style="margin-top:8px;">🔤 标题</div>
                    <textarea id="rop-title-text" class="rop-textarea" rows="2" placeholder="标题文字"></textarea>
                    <div class="rop-grid">
                        <label>字体</label>
                        <select id="rop-title-font" class="rop-select rop-defaultable" data-default="Crimson Pro">
                        </select>
                        <label>字号</label>
                        <div class="rop-slider-combo"><input type="range" id="rop-title-fontsize" class="rop-range rop-defaultable" data-default="60" min="12" max="200" value="60"><input type="number" class="rop-num-readout" data-link="rop-title-fontsize" min="12" max="200" value="60"><button class="rop-reset-btn" data-target="rop-title-fontsize" title="恢复默认">↺</button></div>
                        <label>颜色</label><input type="color" id="rop-title-color" class="rop-color rop-defaultable" data-default="#000000" value="#000000">
                        <label>粗体</label><input type="checkbox" id="rop-title-bold" class="rop-defaultable" data-default="true" checked>
                        <label>字重</label>
                        <select id="rop-title-weight" class="rop-select rop-defaultable" data-default="900">
                            <option value="100">Thin</option><option value="200">ExtraLight</option><option value="300">Light</option>
                            <option value="400">Regular</option><option value="500">Medium</option><option value="600">SemiBold</option>
                            <option value="700">Bold</option><option value="800">ExtraBold</option><option value="900" selected>Black</option>
                        </select>
                        <label>大写</label><input type="checkbox" id="rop-title-uppercase" class="rop-defaultable" data-default="true" checked>
                        <label>对齐</label>
                        <select id="rop-title-align" class="rop-select rop-defaultable" data-default="center">
                            <option value="center">居中</option><option value="left">左对齐</option><option value="right">右对齐</option>
                        </select>
                    </div>
                    <div class="rop-group-title" style="margin-top:8px;">📝 内容</div>
                    <textarea id="rop-body-text" class="rop-textarea" rows="4" placeholder="内容文字"></textarea>
                    <div class="rop-grid">
                        <label>字体</label>
                        <select id="rop-body-font" class="rop-select rop-defaultable" data-default="Arial">
                        </select>
                        <label>字号</label>
                        <div class="rop-slider-combo"><input type="range" id="rop-body-fontsize" class="rop-range rop-defaultable" data-default="40" min="8" max="200" value="40"><input type="number" class="rop-num-readout" data-link="rop-body-fontsize" min="8" max="200" value="40"><button class="rop-reset-btn" data-target="rop-body-fontsize" title="恢复默认">↺</button></div>
                        <label>颜色</label><input type="color" id="rop-body-color" class="rop-color rop-defaultable" data-default="#000000" value="#000000">
                        <label>粗体</label><input type="checkbox" id="rop-body-bold" class="rop-defaultable" data-default="false">
                        <label>字重</label>
                        <select id="rop-body-weight" class="rop-select rop-defaultable" data-default="400">
                            <option value="100">Thin</option><option value="200">ExtraLight</option><option value="300">Light</option>
                            <option value="400" selected>Regular</option><option value="500">Medium</option><option value="600">SemiBold</option>
                            <option value="700">Bold</option><option value="800">ExtraBold</option><option value="900">Black</option>
                        </select>
                        <label>行距</label>
                        <div class="rop-slider-combo"><input type="range" id="rop-body-linespacing" class="rop-range rop-defaultable" data-default="6" min="0" max="50" value="6"><input type="number" class="rop-num-readout" data-link="rop-body-linespacing" min="0" max="50" value="6"><button class="rop-reset-btn" data-target="rop-body-linespacing" title="恢复默认">↺</button></div>
                        <label>对齐</label>
                        <select id="rop-body-align" class="rop-select rop-defaultable" data-default="center">
                            <option value="center">居中</option><option value="left">左对齐</option><option value="right">右对齐</option>
                        </select>
                    </div>
                    <div class="rop-group-title" style="margin-top:8px;">📎 结尾</div>
                    <textarea id="rop-footer-text" class="rop-textarea" rows="2" placeholder="结尾文字（可选）"></textarea>
                    <div class="rop-grid">
                        <label>字体</label>
                        <select id="rop-footer-font" class="rop-select rop-defaultable" data-default="Arial">
                        </select>
                        <label>字号</label>
                        <div class="rop-slider-combo"><input type="range" id="rop-footer-fontsize" class="rop-range rop-defaultable" data-default="32" min="8" max="200" value="32"><input type="number" class="rop-num-readout" data-link="rop-footer-fontsize" min="8" max="200" value="32"><button class="rop-reset-btn" data-target="rop-footer-fontsize" title="恢复默认">↺</button></div>
                        <label>颜色</label><input type="color" id="rop-footer-color" class="rop-color rop-defaultable" data-default="#666666" value="#666666">
                        <label>粗体</label><input type="checkbox" id="rop-footer-bold" class="rop-defaultable" data-default="false">
                        <label>字重</label>
                        <select id="rop-footer-weight" class="rop-select rop-defaultable" data-default="400">
                            <option value="100">Thin</option><option value="200">ExtraLight</option><option value="300">Light</option>
                            <option value="400" selected>Regular</option><option value="500">Medium</option><option value="600">SemiBold</option>
                            <option value="700">Bold</option><option value="800">ExtraBold</option><option value="900">Black</option>
                        </select>
                        <label>对齐</label>
                        <select id="rop-footer-align" class="rop-select rop-defaultable" data-default="center">
                            <option value="center">居中</option><option value="left">左对齐</option><option value="right">右对齐</option>
                        </select>
                    </div>
                    <div class="rop-group-title" style="margin-top:8px;">✨ 文字效果</div>
                    <div class="rop-grid">
                        <label>描边颜色</label><input type="color" id="rop-text-stroke-color" class="rop-color rop-defaultable" data-default="#000000" value="#000000">
                        <label>描边宽度</label>
                        <div class="rop-slider-combo"><input type="range" id="rop-text-stroke-width" class="rop-range rop-defaultable" data-default="0" min="0" max="20" value="0"><input type="number" class="rop-num-readout" data-link="rop-text-stroke-width" min="0" max="20" value="0"><button class="rop-reset-btn" data-target="rop-text-stroke-width" title="恢复默认">↺</button></div>
                        <label>阴影颜色</label><input type="color" id="rop-text-shadow-color" class="rop-color rop-defaultable" data-default="#000000" value="#000000">
                        <label>阴影模糊</label>
                        <div class="rop-slider-combo"><input type="range" id="rop-text-shadow-blur" class="rop-range rop-defaultable" data-default="0" min="0" max="30" value="0"><input type="number" class="rop-num-readout" data-link="rop-text-shadow-blur" min="0" max="30" value="0"><button class="rop-reset-btn" data-target="rop-text-shadow-blur" title="恢复默认">↺</button></div>
                        <label>阴影偏移X</label>
                        <div class="rop-slider-combo"><input type="range" id="rop-text-shadow-x" class="rop-range rop-defaultable" data-default="2" min="-20" max="20" value="2"><input type="number" class="rop-num-readout" data-link="rop-text-shadow-x" min="-20" max="20" value="2"><button class="rop-reset-btn" data-target="rop-text-shadow-x" title="恢复默认">↺</button></div>
                        <label>阴影偏移Y</label>
                        <div class="rop-slider-combo"><input type="range" id="rop-text-shadow-y" class="rop-range rop-defaultable" data-default="2" min="-20" max="20" value="2"><input type="number" class="rop-num-readout" data-link="rop-text-shadow-y" min="-20" max="20" value="2"><button class="rop-reset-btn" data-target="rop-text-shadow-y" title="恢复默认">↺</button></div>
                    </div>
                    <div class="rop-group-title" style="margin-top:8px;">📐 布局</div>
                    <div class="rop-grid">
                        <label>蒙版宽度</label>
                        <div class="rop-slider-combo"><input type="range" id="rop-card-width" class="rop-range rop-defaultable" data-default="910" min="100" max="1080" value="910"><input type="number" class="rop-num-readout" data-link="rop-card-width" min="100" max="1080" value="910"><button class="rop-reset-btn" data-target="rop-card-width" title="恢复默认">↺</button></div>
                        <label>自动适配</label><input type="checkbox" id="rop-auto-fit" class="rop-defaultable" data-default="true" checked>
                        <label>垂直居中</label><input type="checkbox" id="rop-auto-center" class="rop-defaultable" data-default="true" checked>
                        <label>垂直偏移</label>
                        <div class="rop-slider-combo"><input type="range" id="rop-offset-y" class="rop-range rop-defaultable" data-default="0" min="-500" max="500" value="0"><input type="number" class="rop-num-readout" data-link="rop-offset-y" min="-500" max="500" value="0"><button class="rop-reset-btn" data-target="rop-offset-y" title="恢复默认">↺</button></div>
                        <label>标题间距</label>
                        <div class="rop-slider-combo"><input type="range" id="rop-title-body-gap" class="rop-range rop-defaultable" data-default="42" min="0" max="100" value="42"><input type="number" class="rop-num-readout" data-link="rop-title-body-gap" min="0" max="100" value="42"><button class="rop-reset-btn" data-target="rop-title-body-gap" title="恢复默认">↺</button></div>
                        <label>上边距</label>
                        <div class="rop-slider-combo"><input type="range" id="rop-pad-top" class="rop-range rop-defaultable" data-default="20" min="0" max="200" value="20"><input type="number" class="rop-num-readout" data-link="rop-pad-top" min="0" max="200" value="20"><button class="rop-reset-btn" data-target="rop-pad-top" title="恢复默认">↺</button></div>
                        <label>下边距</label>
                        <div class="rop-slider-combo"><input type="range" id="rop-pad-bottom" class="rop-range rop-defaultable" data-default="40" min="0" max="200" value="40"><input type="number" class="rop-num-readout" data-link="rop-pad-bottom" min="0" max="200" value="40"><button class="rop-reset-btn" data-target="rop-pad-bottom" title="恢复默认">↺</button></div>
                        <label>左边距</label>
                        <div class="rop-slider-combo"><input type="range" id="rop-pad-left" class="rop-range rop-defaultable" data-default="40" min="0" max="200" value="40"><input type="number" class="rop-num-readout" data-link="rop-pad-left" min="0" max="200" value="40"><button class="rop-reset-btn" data-target="rop-pad-left" title="恢复默认">↺</button></div>
                        <label>右边距</label>
                        <div class="rop-slider-combo"><input type="range" id="rop-pad-right" class="rop-range rop-defaultable" data-default="40" min="0" max="200" value="40"><input type="number" class="rop-num-readout" data-link="rop-pad-right" min="0" max="200" value="40"><button class="rop-reset-btn" data-target="rop-pad-right" title="恢复默认">↺</button></div>
                    </div>
                    <div class="rop-group-title" style="margin-top:8px;">📏 自动缩放</div>
                    <div class="rop-grid">
                        <label>启用缩放</label><input type="checkbox" id="rop-auto-shrink" class="rop-defaultable" data-default="true" checked>
                        <label>最大高度</label>
                        <div class="rop-slider-combo"><input type="range" id="rop-max-height" class="rop-range rop-defaultable" data-default="1400" min="200" max="1920" value="1400"><input type="number" class="rop-num-readout" data-link="rop-max-height" min="200" max="1920" value="1600"><button class="rop-reset-btn" data-target="rop-max-height" title="恢复默认">↺</button></div>
                        <label>标题缩放行</label>
                        <div class="rop-slider-combo"><input type="range" id="rop-title-max-lines" class="rop-range rop-defaultable" data-default="3" min="1" max="10" value="3"><input type="number" class="rop-num-readout" data-link="rop-title-max-lines" min="1" max="10" value="3"><button class="rop-reset-btn" data-target="rop-title-max-lines" title="恢复默认">↺</button></div>
                        <label>最小字号</label>
                        <div class="rop-slider-combo"><input type="range" id="rop-min-fontsize" class="rop-range rop-defaultable" data-default="16" min="8" max="40" value="16"><input type="number" class="rop-num-readout" data-link="rop-min-fontsize" min="8" max="40" value="16"><button class="rop-reset-btn" data-target="rop-min-fontsize" title="恢复默认">↺</button></div>
                    </div>
                    <div class="rop-group-title" style="margin-top:8px;">💾 卡片模板</div>
                    <div style="display:flex;gap:4px;margin-top:4px;">
                        <select id="rop-card-tpl-select" class="rop-select" style="flex:1;">
                            <option value="">-- 选择模板 --</option>
                        </select>
                        <button class="btn btn-secondary rop-btn" id="rop-card-load-tpl" style="padding:2px 8px;">加载</button>
                    </div>
                    <div style="display:flex;gap:4px;margin-top:4px;">
                        <button class="btn btn-secondary rop-btn" id="rop-card-save-tpl" style="flex:1;">保存</button>
                        <button class="btn btn-secondary rop-btn" id="rop-card-del-tpl" style="flex:1;">删除</button>
                        <button class="btn btn-secondary rop-btn" id="rop-card-import-tpl" style="flex:1;">导入</button>
                        <button class="btn btn-secondary rop-btn" id="rop-card-export-tpl" style="flex:1;">导出</button>
                    </div>
                </div>

                <!-- 操作 -->
                <div class="rop-actions">
                    <button class="btn btn-secondary rop-btn-full" id="rop-duplicate">📋 复制覆层</button>
                    <button class="btn btn-secondary rop-btn-full rop-btn-danger" id="rop-delete">🗑️ 删除覆层</button>
                </div>
            </div>
        </div>
        `;

        this._bindEvents();
    }

    _bindEvents() {
        // 添加覆层
        this.container.querySelector('#rop-add-text').addEventListener('click', () => this._addTextOverlay());
        this.container.querySelector('#rop-add-textcard').addEventListener('click', () => this._addTextCardOverlay());
        this.container.querySelector('#rop-add-image').addEventListener('click', () => this._addImageOverlay());
        this.container.querySelector('#rop-add-scroll').addEventListener('click', () => this._addScrollOverlay());
        this.container.querySelector('#rop-batch-import').addEventListener('click', () => this._batchImportTextCards());
        this.container.querySelector('#rop-duplicate').addEventListener('click', () => this._duplicateOverlay());
        this.container.querySelector('#rop-delete').addEventListener('click', () => this._deleteOverlay());

        // "显示终点" toggle
        const showEndBtn = this.container.querySelector('#rop-scroll-show-end');
        if (showEndBtn) {
            showEndBtn.addEventListener('click', () => {
                const active = showEndBtn.classList.toggle('active');
                showEndBtn.style.background = active ? '#FF6B35' : '';
                showEndBtn.style.color = active ? '#fff' : '';
                showEndBtn.textContent = active ? '👁 终点预览中' : '👁 显示终点';
                // Set global flag for render loop
                if (window._reelsState) window._reelsState._scrollPreviewEnd = active;
            });
        }

        // Overlay group presets
        this.container.querySelector('#rop-group-preset-save')?.addEventListener('click', () => this._saveOverlayGroupPreset());
        this.container.querySelector('#rop-group-preset-load')?.addEventListener('click', () => this._loadOverlayGroupPreset());
        this.container.querySelector('#rop-group-preset-del')?.addEventListener('click', () => this._deleteOverlayGroupPreset());
        this.container.querySelector('#rop-group-preset-import')?.addEventListener('click', () => this._importOverlayGroupPresets());
        this.container.querySelector('#rop-group-preset-export')?.addEventListener('click', () => this._exportOverlayGroupPresets());
        this._refreshOverlayGroupPresetSelect();

        // Reset to default button
        this.container.querySelector('#rop-reset-transform')?.addEventListener('click', () => {
            if (!this._selectedOv) return;
            const ov = this._selectedOv;
            ov.w = 300;
            ov.h = 300;
            ov.x = (1080 - ov.w) / 2;  // 居中
            ov.y = (1920 - ov.h) / 2;
            ov.rotation = 0;
            ov.opacity = 255;
            if (ov.type === 'image') ov.scale = 1;
            this._syncFromOverlay(ov);
            if (this.videoCanvas) this.videoCanvas.render();
        });

        // Fill screen button
        this.container.querySelector('#rop-fill-screen')?.addEventListener('click', () => {
            if (!this._selectedOv) return;
            const ov = this._selectedOv;
            ov.x = 0;
            ov.y = 0;
            ov.w = 1080;
            ov.h = 1920;
            if (ov.type === 'image') ov.scale = 1;
            this._syncFromOverlay(ov);
            if (this.videoCanvas) this.videoCanvas.render();
        });

        // 使用 FontManager 填充字体下拉框（和字幕面板一致）
        if (window.getFontManager) {
            const fm = getFontManager();
            fm.refreshFontSelect('rop-font', 'Arial');
            fm.refreshFontSelect('rop-title-font', 'Crimson Pro');
            fm.refreshFontSelect('rop-body-font', 'Arial');
            fm.refreshFontSelect('rop-footer-font', 'Arial');
            fm.refreshFontSelect('rop-scroll-font', 'Arial');
            if (fm && typeof fm.loadGoogleFont === 'function') {
                fm.loadGoogleFont('Crimson Pro').catch(() => { });
            }
        }

        // 属性变更
        const fields = [
            'rop-x', 'rop-y', 'rop-w', 'rop-h', 'rop-rotation', 'rop-opacity',
            'rop-start', 'rop-end', 'rop-content', 'rop-font', 'rop-fontsize',
            'rop-color', 'rop-bold', 'rop-font-weight', 'rop-stroke-color', 'rop-stroke-width',
            'rop-shadow-color', 'rop-shadow-blur', 'rop-scale', 'rop-flip-h',
            'rop-flip-v', 'rop-blend', 'rop-anim-in', 'rop-anim-out',
            'rop-anim-in-dur', 'rop-anim-out-dur',
            // Text card fields
            'rop-card-color', 'rop-card-opacity', 'rop-radius-all',
            'rop-title-text', 'rop-title-font', 'rop-title-fontsize',
            'rop-title-color', 'rop-title-bold', 'rop-title-weight', 'rop-title-uppercase', 'rop-title-align',
            'rop-body-text', 'rop-body-font', 'rop-body-fontsize',
            'rop-body-color', 'rop-body-bold', 'rop-body-weight', 'rop-body-linespacing', 'rop-body-align',
            'rop-footer-text', 'rop-footer-font', 'rop-footer-fontsize',
            'rop-footer-color', 'rop-footer-bold', 'rop-footer-weight', 'rop-footer-align',
            'rop-text-stroke-color', 'rop-text-stroke-width',
            'rop-text-shadow-color', 'rop-text-shadow-blur', 'rop-text-shadow-x', 'rop-text-shadow-y',
            'rop-auto-fit', 'rop-auto-center', 'rop-fullscreen-mask', 'rop-title-body-gap', 'rop-offset-y',
            'rop-pad-top', 'rop-pad-bottom', 'rop-pad-left', 'rop-pad-right',
            'rop-card-width',
            'rop-auto-shrink', 'rop-max-height', 'rop-title-max-lines', 'rop-min-fontsize',
            // Scroll fields
            'rop-scroll-content',
            'rop-scroll-title', 'rop-scroll-title-fontsize', 'rop-scroll-title-color',
            'rop-scroll-title-weight', 'rop-scroll-title-gap', 'rop-scroll-title-fixed',
            'rop-scroll-font', 'rop-scroll-fontsize',
            'rop-scroll-color', 'rop-scroll-bold', 'rop-scroll-weight',
            'rop-scroll-align', 'rop-scroll-linespacing', 'rop-scroll-textw',
            'rop-scroll-stroke-color', 'rop-scroll-stroke-width',
            'rop-scroll-shadow', 'rop-scroll-shadow-color', 'rop-scroll-shadow-blur',
            'rop-scroll-from-y', 'rop-scroll-to-y', 'rop-scroll-from-x', 'rop-scroll-to-x',
            'rop-scroll-final-y', 'rop-scroll-start-offset',
            'rop-scroll-start-time', 'rop-scroll-end-time',
            'rop-scroll-speed',
            'rop-scroll-auto-stop',
            'rop-scroll-auto-fit', 'rop-scroll-min-fontsize',
            'rop-scroll-feather-top', 'rop-scroll-feather-bottom',
            'rop-scroll-bg-enabled', 'rop-scroll-bg-color', 'rop-scroll-bg-opacity',
            'rop-scroll-bg-radius', 'rop-scroll-bg-pad-top', 'rop-scroll-bg-pad-bottom',
            'rop-scroll-bg-pad-left', 'rop-scroll-bg-pad-right', 'rop-scroll-bg-fullscreen',
        ];
        for (const fid of fields) {
            const el = this.container.querySelector('#' + fid);
            if (!el) continue;
            el.addEventListener('input', () => this._syncToOverlay());
            el.addEventListener('change', () => this._syncToOverlay());
        }

        // 滚动字幕时间字段 ↔ 主时间字段 双向同步
        const scrollStartTime = this.container.querySelector('#rop-scroll-start-time');
        const scrollEndTime = this.container.querySelector('#rop-scroll-end-time');
        const mainStart = this.container.querySelector('#rop-start');
        const mainEnd = this.container.querySelector('#rop-end');
        if (scrollStartTime && mainStart) {
            scrollStartTime.addEventListener('input', () => { mainStart.value = scrollStartTime.value; });
            scrollStartTime.addEventListener('change', () => { mainStart.value = scrollStartTime.value; });
        }
        if (scrollEndTime && mainEnd) {
            scrollEndTime.addEventListener('input', () => { mainEnd.value = scrollEndTime.value; });
            scrollEndTime.addEventListener('change', () => { mainEnd.value = scrollEndTime.value; });
        }

        // 移除导致死循环的自动修正 "起始偏移" 逻辑（允许独立调节保证顺畅）
        // Live value display for sliders
        const opSlider = this.container.querySelector('#rop-opacity');
        const opVal = this.container.querySelector('#rop-opacity-val');
        if (opSlider && opVal) {
            opSlider.addEventListener('input', () => { opVal.textContent = opSlider.value + '%'; });
        }
        const scSlider = this.container.querySelector('#rop-scale');
        const scVal = this.container.querySelector('#rop-scale-val');
        if (scSlider && scVal) {
            scSlider.addEventListener('input', () => { scVal.textContent = scSlider.value + '%'; });
        }

        // Scroll speed slider removed — speed is auto-calculated from distance ÷ time

        const syncBoldToWeight = (boldId, weightId, boldValue = '700', normalValue = '400') => {
            const boldEl = this.container.querySelector('#' + boldId);
            const weightEl = this.container.querySelector('#' + weightId);
            if (!boldEl || !weightEl) return;
            boldEl.addEventListener('change', () => {
                const target = boldEl.checked ? boldValue : normalValue;
                if (Array.from(weightEl.options).some(o => o.value === target)) {
                    weightEl.value = target;
                }
                this._syncToOverlay();
            });
            weightEl.addEventListener('change', () => {
                const w = parseInt(weightEl.value || normalValue, 10);
                boldEl.checked = Number.isFinite(w) ? w >= 600 : false;
                this._syncToOverlay();
            });
        };
        syncBoldToWeight('rop-bold', 'rop-font-weight', '700', '400');
        syncBoldToWeight('rop-title-bold', 'rop-title-weight', '900', '400');
        syncBoldToWeight('rop-body-bold', 'rop-body-weight', '700', '400');
        syncBoldToWeight('rop-footer-bold', 'rop-footer-weight', '700', '400');
        syncBoldToWeight('rop-scroll-bold', 'rop-scroll-weight', '700', '400');

        const fontWeightPairs = [
            ['rop-font', 'rop-font-weight'],
            ['rop-title-font', 'rop-title-weight'],
            ['rop-body-font', 'rop-body-weight'],
            ['rop-footer-font', 'rop-footer-weight'],
            ['rop-scroll-font', 'rop-scroll-weight'],
        ];
        for (const [fontId, weightId] of fontWeightPairs) {
            const fontEl = this.container.querySelector('#' + fontId);
            if (!fontEl) continue;
            fontEl.addEventListener('change', async () => {
                if (window.getFontManager) {
                    const fm = getFontManager();
                    if (fm && typeof fm.loadGoogleFont === 'function') {
                        try { await fm.loadGoogleFont(fontEl.value); } catch (_) { }
                    }
                }
                this._refreshWeightOptions(weightId, fontEl.value);
            });
        }

        // Slider ↔ Number readout bidirectional linking
        this.container.querySelectorAll('.rop-num-readout').forEach(numEl => {
            const linkId = numEl.dataset.link;
            if (!linkId) return;
            const rangeEl = this.container.querySelector('#' + linkId);
            if (!rangeEl) return;
            // Range → Number
            rangeEl.addEventListener('input', () => { numEl.value = rangeEl.value; });
            // Number → Range
            numEl.addEventListener('input', () => { rangeEl.value = numEl.value; this._syncToOverlay(); });
            numEl.addEventListener('change', () => { rangeEl.value = numEl.value; this._syncToOverlay(); });
        });

        // Drag-to-scrub on number inputs (click+drag horizontally to adjust value)
        this.container.querySelectorAll('input.rop-input[type="number"]').forEach(el => {
            el.style.cursor = 'ew-resize';
            let dragging = false, startX = 0, startVal = 0;
            el.addEventListener('mousedown', (e) => {
                // Allow clicking into input to type when already focused
                if (document.activeElement === el) return;
                dragging = true;
                startX = e.clientX;
                startVal = parseFloat(el.value) || 0;
                e.preventDefault();
                const onMove = (me) => {
                    if (!dragging) return;
                    const dx = me.clientX - startX;
                    const speed = me.shiftKey ? 0.1 : 1;
                    const step = parseFloat(el.step) || 1;
                    el.value = Math.round((startVal + dx * speed * step) / step) * step;
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                };
                const onUp = () => {
                    dragging = false;
                    document.removeEventListener('mousemove', onMove);
                    document.removeEventListener('mouseup', onUp);
                };
                document.addEventListener('mousemove', onMove);
                document.addEventListener('mouseup', onUp);
            });
            // Double-click to focus for manual typing
            el.addEventListener('dblclick', (e) => {
                e.preventDefault();
                el.focus();
                el.select();
                el.style.cursor = 'text';
            });
            el.addEventListener('blur', () => {
                el.style.cursor = 'ew-resize';
            });
            el.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') { el.blur(); }
            });
        });


        // Per-parameter ↺ reset buttons
        this.container.querySelectorAll('.rop-reset-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                const targetId = btn.dataset.target;
                const el = this.container.querySelector('#' + targetId);
                if (!el) return;
                const def = el.dataset.default;
                if (def == null) return;
                el.value = def;
                // Sync linked number readout
                const numReadout = this.container.querySelector(`.rop-num-readout[data-link="${targetId}"]`);
                if (numReadout) numReadout.value = def;
                this._syncToOverlay();
            });
        });

        // "Reset all" button — resets all rop-defaultable controls
        const resetAllBtn = this.container.querySelector('#rop-card-reset-all');
        if (resetAllBtn) {
            resetAllBtn.addEventListener('click', () => {
                this.container.querySelectorAll('.rop-defaultable').forEach(el => {
                    const def = el.dataset.default;
                    if (def == null) return;
                    if (el.type === 'checkbox') {
                        el.checked = def === 'true';
                    } else {
                        el.value = def;
                    }
                    // Sync linked number readout
                    if (el.id) {
                        const numReadout = this.container.querySelector(`.rop-num-readout[data-link="${el.id}"]`);
                        if (numReadout) numReadout.value = def;
                    }
                });
                this._syncToOverlay();
            });
        }

        // Card template buttons
        const saveTpl = this.container.querySelector('#rop-card-save-tpl');
        const loadTpl = this.container.querySelector('#rop-card-load-tpl');
        const delTpl = this.container.querySelector('#rop-card-del-tpl');
        const importTpl = this.container.querySelector('#rop-card-import-tpl');
        const exportTpl = this.container.querySelector('#rop-card-export-tpl');

        if (saveTpl) saveTpl.addEventListener('click', () => this._saveCardTemplate());
        if (loadTpl) loadTpl.addEventListener('click', () => this._loadCardTemplate());
        if (delTpl) delTpl.addEventListener('click', () => this._deleteCardTemplate());
        if (importTpl) importTpl.addEventListener('click', () => this._importCardTemplates());
        if (exportTpl) exportTpl.addEventListener('click', () => this._exportCardTemplates());

        this._refreshCardTemplateSelect();

        // VideoCanvas 回调
        if (this.videoCanvas) {
            this.videoCanvas.onSelect = (ov) => this.selectOverlay(ov);
            this.videoCanvas.onDeselect = () => this.deselectOverlay();
            this.videoCanvas.onOverlayChange = (ov) => this._syncFromOverlay(ov);
        }
    }

    // ═══════════════════════════════════════════════
    // 覆层 CRUD
    // ═══════════════════════════════════════════════

    _addTextCardOverlay() {
        const ReelsOverlay = window.ReelsOverlay;
        if (!ReelsOverlay) return;
        const ov = ReelsOverlay.createTextCardOverlay({
            title_text: 'IN MARCH, READ THIS JUST ONCE AND IT WILL COME TO PASS IMMEDIATELY',
            body_text: 'Lord, in the name of Jesus, March has begun. I rebuke every spiritual curse, evil eye, jealousy, sickness, and confusion coming against me and my family! I cut off every hidden opening and destroy every trap set by the enemy.\nI declare: the precious blood of Jesus covers my spouse, my children, and everyone I love. Angels guard every door and window—the enemy cannot come near, not even one step! Darkness cannot enter.\nLord, bring breakthrough to everyone who writes "Amen" and render every curse powerless!',
            x: 40, y: 200, w: 910,
            start: 0, end: 9999,
        });
        if (this.videoCanvas) this.videoCanvas.addOverlay(ov);
        this._refreshList();
        this.selectOverlay(ov);
    }

    /**
     * 批量导入文案卡片 — 从 Google Sheets 粘贴 TSV
     * 格式: 第一列=标题, 第二列=内容 (支持单元格内换行)
     * 每行自动创建一个任务 + 文案卡片覆层
     */
    async _batchImportTextCards() {
        const ReelsOverlay = window.ReelsOverlay;
        if (!ReelsOverlay) return;

        // 弹窗让用户粘贴表格数据
        const result = await this._showBatchImportDialog();
        if (!result || !result.data || !result.data.trim()) return;

        const importType = result.type || 'textcard';

        // 解析 TSV (支持引号内换行)
        const rows = this._parseTSV(result.data);
        if (!rows.length) {
            alert('未检测到有效数据，请确保每行至少有一列内容。');
            return;
        }

        // 读取当前覆层模板样式
        const templateProps = importType === 'scroll'
            ? this._getCurrentScrollTemplate()
            : this._getCurrentCardTemplate();

        // 获取 _reelsState 来创建任务
        const state = window._reelsState;
        if (!state) {
            alert('批量Reels模块未初始化');
            return;
        }

        let created = 0;
        for (let i = 0; i < rows.length; i++) {
            const row = rows[i];
            let rawName = '', title = '', body = '';

            if (importType === 'scroll') {
                // 滚动字幕：3列(命名, 标题, 正文), 2列(标题, 正文), 1列(正文)
                if (row.length >= 3) {
                    rawName = row[0] || '';
                    title = row[1] || '';
                    body = row[2] || '';
                } else if (row.length === 2) {
                    title = row[0] || '';
                    body = row[1] || '';
                } else {
                    body = row[0] || '';
                }
            } else {
                // 文案卡片：3列(命名, 标题, 内容), 2列(标题, 内容)
                if (row.length >= 3) {
                    rawName = row[0] || '';
                    title = row[1] || '';
                    body = row[2] || '';
                } else {
                    title = row[0] || '';
                    body = row[1] || '';
                }
            }
            if (!title && !body && !rawName) continue; // 跳过空行

            // 命名
            const baseNameInput = rawName.trim() || `batch_${importType}_${String(i + 1).padStart(3, '0')}`;

            let task = null;
            if (typeof _getOrCreateTaskByBase === 'function' && typeof _normalizeBaseName === 'function') {
                const normBase = _normalizeBaseName(baseNameInput);
                task = _getOrCreateTaskByBase(normBase, baseNameInput);
            } else {
                const taskName = baseNameInput;
                task = state.tasks.find(t => t.baseName === taskName);
                if (!task) {
                    task = {
                        baseName: taskName,
                        fileName: `${taskName}.mp4`,
                        bgPath: null, bgSrcUrl: null, audioPath: null, srtPath: null,
                        segments: [], videoPath: null, srcUrl: null,
                        overlays: [],
                    };
                    state.tasks.push(task);
                }
            }

            if (!task.overlays) task.overlays = [];

            // 创建覆层
            if (importType === 'scroll') {
                const ovOpts = Object.assign({}, templateProps, {
                    scroll_title: title,
                    content: body,
                    start: 0,
                    end: 9999,
                });
                const ov = ReelsOverlay.createScrollOverlay(ovOpts);
                task.overlays.push(ov);
            } else {
                const ovOpts = Object.assign({}, templateProps, {
                    title_text: title,
                    body_text: body,
                    start: 0,
                    end: 9999,
                });
                const ov = ReelsOverlay.createTextCardOverlay(ovOpts);
                task.overlays.push(ov);
            }

            created++;
        }

        // 刷新任务列表
        if (typeof _renderTaskList === 'function') _renderTaskList();

        // 自动匹配素材
        if (typeof reelsAutoMatchFiles === 'function') reelsAutoMatchFiles();

        const typeLabel = importType === 'scroll' ? '滚动字幕' : '文案卡片';
        alert(`✅ 成功导入 ${created} 条${typeLabel}！\n\n请添加背景素材、音频等，系统将自动配对。`);
    }

    /**
     * 解析 TSV 数据 (支持引号内换行的 Google Sheets 格式)
     * 返回 [[title, body], ...]
     */
    _parseTSV(raw) {
        const rows = [];
        let i = 0;
        const len = raw.length;

        while (i < len) {
            const cells = [];
            // 解析一行的所有单元格
            while (i < len) {
                let cell = '';
                if (raw[i] === '"') {
                    // 引号字段 — 可能包含换行和tab
                    i++; // 跳过开头引号
                    while (i < len) {
                        if (raw[i] === '"') {
                            if (i + 1 < len && raw[i + 1] === '"') {
                                cell += '"'; // 转义引号
                                i += 2;
                            } else {
                                i++; // 跳过结尾引号
                                break;
                            }
                        } else {
                            cell += raw[i++];
                        }
                    }
                } else {
                    // 非引号字段
                    while (i < len && raw[i] !== '\t' && raw[i] !== '\n' && raw[i] !== '\r') {
                        cell += raw[i++];
                    }
                }
                cells.push(cell);
                // Tab = 下一列
                if (i < len && raw[i] === '\t') { i++; continue; }
                // 换行 = 下一行
                if (i < len && (raw[i] === '\n' || raw[i] === '\r')) {
                    if (raw[i] === '\r' && i + 1 < len && raw[i + 1] === '\n') i++;
                    i++;
                    break;
                }
            }
            if (cells.length > 0 && cells.some(c => c.trim())) {
                rows.push(cells); // 返回所有列，以支持 3 列解析
            }
        }
        return rows;
    }

    /**
     * 获取当前卡片模板的属性 (不含文本内容)
     */
    _getCurrentCardTemplate() {
        const props = {};
        if (this._selectedOv && this._selectedOv.type === 'textcard') {
            const ov = this._selectedOv;
            const keys = [
                'card_color', 'card_opacity',
                'radius_tl', 'radius_tr', 'radius_bl', 'radius_br',
                'title_font_family', 'title_fontsize', 'title_font_weight', 'title_bold', 'title_italic',
                'title_color', 'title_align', 'title_uppercase', 'title_letter_spacing',
                'body_font_family', 'body_fontsize', 'body_font_weight', 'body_bold', 'body_italic',
                'body_color', 'body_align', 'body_line_spacing',
                'auto_fit', 'auto_center_v',
                'padding_top', 'padding_bottom', 'padding_left', 'padding_right',
                'title_body_gap', 'w',
            ];
            for (const k of keys) {
                if (ov[k] !== undefined) props[k] = ov[k];
            }
        }
        return props;
    }

    _getCurrentScrollTemplate() {
        const props = {};
        if (this._selectedOv && this._selectedOv.type === 'scroll') {
            const ov = this._selectedOv;
            const keys = [
                'font_family', 'fontsize', 'font_weight', 'bold', 'italic',
                'color', 'text_align', 'line_spacing', 'text_width',
                'use_stroke', 'stroke_color', 'stroke_width',
                'shadow_enabled', 'shadow_color', 'shadow_blur', 'shadow_opacity',
                'shadow_offset_x', 'shadow_offset_y',
                'scroll_from_x', 'scroll_from_y', 'scroll_to_x', 'scroll_to_y',
                'scroll_speed', 'scroll_auto_stop', 'scroll_auto_fit', 'scroll_min_fontsize',
                'scroll_title_fontsize', 'scroll_title_font_family', 'scroll_title_font_weight',
                'scroll_title_bold', 'scroll_title_color', 'scroll_title_align', 'scroll_title_gap', 'scroll_title_fixed',
                'feather_top', 'feather_bottom',
                'bg_enabled', 'bg_color', 'bg_opacity', 'bg_radius',
                'bg_padding_top', 'bg_padding_bottom', 'bg_padding_left', 'bg_padding_right', 'bg_fullscreen',
                'x', 'y', 'w', 'h',
            ];
            for (const k of keys) {
                if (ov[k] !== undefined) props[k] = ov[k];
            }
        }
        return props;
    }

    /**
     * 批量导入弹窗
     */
    _showBatchImportDialog() {
        return new Promise(resolve => {
            const overlay = document.createElement('div');
            overlay.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.7);z-index:10000;display:flex;align-items:center;justify-content:center;';
            overlay.innerHTML = `
                <div style="background:#1a1a3e;border:1px solid #2a2a5a;border-radius:12px;padding:24px;width:600px;max-height:80vh;display:flex;flex-direction:column;box-shadow:0 20px 60px rgba(0,0,0,0.5);">
                    <h3 style="margin:0 0 8px 0;color:#00D4FF;font-size:16px;">📋 批量导入文案</h3>
                    <div style="display:flex;gap:8px;margin-bottom:10px;align-items:center;">
                        <label style="font-size:12px;color:#aaa;">覆层类型：</label>
                        <select id="rop-batch-type" style="padding:4px 8px;background:#12122a;border:1px solid #2a2a4a;border-radius:4px;color:#ddd;font-size:12px;">
                            <option value="textcard">📝 文案卡片</option>
                            <option value="scroll">🔄 滚动字幕</option>
                        </select>
                    </div>
                    <p id="rop-batch-help" style="margin:0 0 12px 0;color:#999;font-size:12px;line-height:1.5;">
                        从 Google 表格复制粘贴到下方。<br>
                        <b>格式选项</b>：<br>
                        • 3列格式：第一列 = <b>命名</b>，第二列 = 标题，第三列 = 内容<br>
                        • 2列格式：第一列 = 标题，第二列 = 内容（没有命名）<br>
                        如果您提供了命名，系统将按照外面导入素材的同名逻辑<b>自动合并</b>到对应行。
                    </p>
                    <textarea id="rop-batch-data" style="flex:1;min-height:300px;padding:12px;background:#12122a;border:1px solid #2a2a4a;border-radius:8px;color:#ddd;font-size:13px;font-family:monospace;resize:vertical;" placeholder="从 Google 表格复制粘贴到这里&#10;&#10;命名1&#9;标题1&#9;内容1&#10;命名2&#9;标题2&#9;内容2&#10;这是标题3(没命名)&#9;这是内容3"></textarea>
                    <div style="display:flex;gap:8px;margin-top:12px;justify-content:flex-end;">
                        <button id="rop-batch-cancel" style="padding:8px 20px;background:#333;border:1px solid #555;border-radius:6px;color:#ccc;cursor:pointer;">取消</button>
                        <button id="rop-batch-ok" style="padding:8px 20px;background:#00D4FF;border:none;border-radius:6px;color:#000;font-weight:bold;cursor:pointer;">导入</button>
                    </div>
                </div>
            `;
            document.body.appendChild(overlay);

            // 动态更新帮助文字
            const typeSelect = overlay.querySelector('#rop-batch-type');
            const helpEl = overlay.querySelector('#rop-batch-help');
            const textareaEl = overlay.querySelector('#rop-batch-data');
            typeSelect.addEventListener('change', () => {
                if (typeSelect.value === 'scroll') {
                    helpEl.innerHTML = `从 Google 表格复制粘贴到下方。<br>
                        <b>格式选项</b>：<br>
                        • 3列格式：第一列 = <b>命名</b>，第二列 = 滚动标题，第三列 = 滚动内容<br>
                        • 2列格式：第一列 = 滚动标题，第二列 = 滚动内容（没有命名）<br>
                        • 1列格式：只填滚动内容（没有标题和命名）<br>
                        滚动字幕将使用当前面板的样式设置。`;
                    textareaEl.placeholder = '从 Google 表格复制粘贴到这里\n\n命名1\t标题1\t正文内容1\n命名2\t标题2\t正文内容2\n这是标题(没命名)\t这是正文内容';
                } else {
                    helpEl.innerHTML = `从 Google 表格复制粘贴到下方。<br>
                        <b>格式选项</b>：<br>
                        • 3列格式：第一列 = <b>命名</b>，第二列 = 标题，第三列 = 内容<br>
                        • 2列格式：第一列 = 标题，第二列 = 内容（没有命名）<br>
                        如果您提供了命名，系统将按照外面导入素材的同名逻辑<b>自动合并</b>到对应行。`;
                    textareaEl.placeholder = '从 Google 表格复制粘贴到这里\n\n命名1\t标题1\t内容1\n命名2\t标题2\t内容2\n这是标题3(没命名)\t这是内容3';
                }
            });

            const close = (val) => {
                document.body.removeChild(overlay);
                resolve(val);
            };
            overlay.querySelector('#rop-batch-cancel').onclick = () => close(null);
            overlay.querySelector('#rop-batch-ok').onclick = () => {
                const data = overlay.querySelector('#rop-batch-data').value;
                const type = typeSelect.value;
                close({ data, type });
            };
            overlay.addEventListener('click', (e) => {
                if (e.target === overlay) close(null);
            });
            // 自动聚焦
            setTimeout(() => overlay.querySelector('#rop-batch-data').focus(), 100);
        });
    }

    _addTextOverlay() {
        const ReelsOverlay = window.ReelsOverlay;
        if (!ReelsOverlay) return;
        const ov = ReelsOverlay.createTextOverlay({
            content: '新文本',
            x: 200, y: 800, w: 680, h: 120,
            fontsize: 74, color: '#ffffff',
            start: 0, end: 5,
        });
        if (this.videoCanvas) this.videoCanvas.addOverlay(ov);
        this._refreshList();
        this.selectOverlay(ov);
    }

    _addScrollOverlay() {
        const ReelsOverlay = window.ReelsOverlay;
        if (!ReelsOverlay) return;
        const ov = ReelsOverlay.createScrollOverlay({
            content: '滚动字幕示例\n第二行\n第三行\n第四行\n第五行',
            start: 0, end: 10,
        });
        if (this.videoCanvas) this.videoCanvas.addOverlay(ov);
        this._refreshList();
        this.selectOverlay(ov);
    }

    _addImageOverlay() {
        // 弹出文件选择器
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = 'image/*';
        input.onchange = (e) => {
            const file = e.target.files[0];
            if (!file) return;
            // Electron 桌面版: 用 getFileNativePath + toFileUrl (与其他模块一致)
            // Web 模式回退: 用 blob URL
            let url;
            const nativePath = getFileNativePath(file);
            if (nativePath && nativePath !== file.name && window.electronAPI && window.electronAPI.toFileUrl) {
                url = window.electronAPI.toFileUrl(nativePath);
            }
            if (!url) {
                url = URL.createObjectURL(file);
            }
            const ReelsOverlay = window.ReelsOverlay;
            if (!ReelsOverlay) return;
            const ov = ReelsOverlay.createImageOverlay({
                content: url, x: 390, y: 810, w: 300, h: 300,
                start: 0, end: 5,
            });
            if (this.videoCanvas) this.videoCanvas.addOverlay(ov);
            this._refreshList();
            this.selectOverlay(ov);
        };
        input.click();
    }

    _duplicateOverlay() {
        if (!this._selectedOv) return;
        const clone = JSON.parse(JSON.stringify(this._selectedOv));
        clone.id = 'ov_' + Date.now();
        clone.x += 30;
        clone.y += 30;
        if (this.videoCanvas) this.videoCanvas.addOverlay(clone);
        this._refreshList();
        this.selectOverlay(clone);
    }

    _deleteOverlay() {
        if (!this._selectedOv) return;
        if (this.videoCanvas) this.videoCanvas.removeOverlay(this._selectedOv.id);
        this._selectedOv = null;
        this._refreshList();
        this.container.querySelector('#rop-props').style.display = 'none';
    }

    // ═══════════════════════════════════════════════
    // 列表
    // ═══════════════════════════════════════════════

    _refreshList() {
        const list = this.container.querySelector('#rop-overlay-list');
        if (!list || !this.videoCanvas) return;

        const overlays = this.videoCanvas.overlayMgr.overlays || [];
        if (overlays.length === 0) {
            list.innerHTML = '<div class="rop-empty">暂无覆层，点击上方按钮添加</div>';
            return;
        }

        list.innerHTML = overlays.map(ov => {
            const isSelected = this._selectedOv?.id === ov.id;
            const icon = ov.type === 'text' ? '📝' : (ov.type === 'textcard' ? '📋' : (ov.type === 'scroll' ? '🔄' : '🖼️'));
            const lockIcon = ov.fixed_text ? '🔒' : '';
            const label = ov.type === 'textcard'
                ? (ov.title_text || '').slice(0, 15) || '文案卡片'
                : (ov.type === 'scroll' ? '滚动: ' + (ov.content || '').split('\n')[0].slice(0, 12)
                : (ov.type === 'text' ? (ov.content || '').slice(0, 15) : (ov.name || '图片')));
            return `<div class="rop-list-item ${isSelected ? 'selected' : ''}" data-id="${ov.id}">
                <span class="rop-list-arrow">${isSelected ? '▼' : '▶'}</span>
                ${icon} <span class="rop-list-label">${lockIcon}${label}</span>
                <span class="rop-list-time">${ov.start?.toFixed(1) || 0}s–${(ov.end >= 9999 ? '全程' : (ov.end?.toFixed(1) || 0) + 's')}</span>
                <button class="rop-list-del" data-id="${ov.id}" title="删除此覆层">✕</button>
            </div>`;
        }).join('');

        list.querySelectorAll('.rop-list-item').forEach(el => {
            // Click label to select/toggle
            el.addEventListener('click', (e) => {
                if (e.target.classList.contains('rop-list-del')) return; // don't select when clicking delete
                const ov = (this.videoCanvas?.overlayMgr?.overlays || []).find(o => o.id === el.dataset.id);
                if (!ov) return;
                if (this._selectedOv?.id === ov.id) {
                    // Click again to collapse
                    this.deselectOverlay();
                } else {
                    this.selectOverlay(ov);
                }
                this._refreshList();
            });
        });

        // Delete buttons
        list.querySelectorAll('.rop-list-del').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const id = btn.dataset.id;
                const ov = (this.videoCanvas?.overlayMgr?.overlays || []).find(o => o.id === id);
                const label = ov ? (ov.name || ov.title_text || ov.content || '').slice(0, 20) || ov.type : id;
                if (!confirm(`确定删除覆层「${label}」吗？`)) return;
                this.videoCanvas.overlayMgr.removeOverlay(id);
                if (this._selectedOv?.id === id) {
                    this._selectedOv = null;
                    this.container.querySelector('#rop-props').style.display = 'none';
                }
                this._refreshList();
            });
        });
    }

    // ═══════════════════════════════════════════════
    // 选中 / 属性同步
    // ═══════════════════════════════════════════════

    selectOverlay(ov) {
        // 清除之前选中的标记
        if (this._selectedOv) this._selectedOv._selected = false;
        this._selectedOv = ov;
        ov._selected = true;
        this.container.querySelector('#rop-props').style.display = 'block';
        this.container.querySelector('#rop-text-props').style.display = ov.type === 'text' ? 'block' : 'none';
        this.container.querySelector('#rop-image-props').style.display = ov.type === 'image' ? 'block' : 'none';
        this.container.querySelector('#rop-textcard-props').style.display = ov.type === 'textcard' ? 'block' : 'none';
        this.container.querySelector('#rop-scroll-props').style.display = ov.type === 'scroll' ? 'block' : 'none';
        // Show fixed_text toggle for text, textcard, and scroll overlays
        const hasText = ov.type === 'text' || ov.type === 'textcard' || ov.type === 'scroll';
        this.container.querySelector('#rop-fixed-text-group').style.display = hasText ? 'block' : 'none';

        // Image overlays: show scale, hide W/H. Others: show W/H, hide scale.
        const isImg = ov.type === 'image';
        this.container.querySelector('#rop-wh-label-w').style.display = isImg ? 'none' : '';
        this.container.querySelector('#rop-w').style.display = isImg ? 'none' : '';
        this.container.querySelector('#rop-wh-label-h').style.display = isImg ? 'none' : '';
        this.container.querySelector('#rop-h').style.display = isImg ? 'none' : '';
        this.container.querySelector('#rop-scale-label').style.display = isImg ? '' : 'none';
        this.container.querySelector('#rop-scale-wrap').style.display = isImg ? '' : 'none';

        // Dynamic labels for scroll overlay (x/y/w/h = clip region)
        const isScroll = ov.type === 'scroll';
        this.container.querySelector('#rop-xy-label-x').textContent = isScroll ? '裁切X' : '位置X';
        this.container.querySelector('#rop-xy-label-y').textContent = isScroll ? '裁切Y' : '位置Y';
        this.container.querySelector('#rop-wh-label-w').textContent = isScroll ? '裁切宽' : '宽度';
        this.container.querySelector('#rop-wh-label-h').textContent = isScroll ? '裁切高' : '高度';

        // 滚动覆层: 时间字段移到滚动运动区
        const timeInTransform = this.container.querySelector('#rop-time-in-transform');
        if (timeInTransform) timeInTransform.style.display = isScroll ? 'none' : '';

        this._syncFromOverlay(ov);
        this._refreshList();
    }

    deselectOverlay() {
        if (this._selectedOv) this._selectedOv._selected = false;
        this._selectedOv = null;
        this.container.querySelector('#rop-props').style.display = 'none';
        this._refreshList();
    }

    _val(id, v) {
        const el = this.container.querySelector('#' + id);
        if (!el) return;
        if (el.type === 'checkbox') el.checked = !!v;
        else el.value = v ?? '';
        // Sync linked number readout
        const numReadout = this.container.querySelector(`.rop-num-readout[data-link="${id}"]`);
        if (numReadout) numReadout.value = v ?? '';
    }

    _get(id) {
        const el = this.container.querySelector('#' + id);
        if (!el) return undefined;
        if (el.type === 'checkbox') return el.checked;
        if (el.type === 'number' || el.type === 'range') return parseFloat(el.value) || 0;
        return el.value;
    }

    _refreshWeightOptions(weightSelectId, fontFamily) {
        const select = this.container.querySelector('#' + weightSelectId);
        if (!select) return;
        const current = String(select.value || '700');
        let entries = [
            { value: '100', label: 'Thin' },
            { value: '200', label: 'ExtraLight' },
            { value: '300', label: 'Light' },
            { value: '400', label: 'Regular' },
            { value: '500', label: 'Medium' },
            { value: '600', label: 'SemiBold' },
            { value: '700', label: 'Bold' },
            { value: '800', label: 'ExtraBold' },
            { value: '900', label: 'Black' },
        ];
        if (window.getFontManager) {
            const fm = getFontManager();
            if (fm && typeof fm.getFontWeightEntries === 'function') {
                const list = fm.getFontWeightEntries(fontFamily, 'normal');
                if (Array.isArray(list) && list.length > 0) {
                    entries = list.map(item => {
                        const value = String(item.value || '400');
                        const label = String(item.label || value);
                        return { value, label };
                    });
                }
            } else if (fm && typeof fm.getFontWeightOptions === 'function') {
                const list = fm.getFontWeightOptions(fontFamily);
                if (Array.isArray(list) && list.length > 0) {
                    entries = list.map(v => ({ value: String(v), label: String(v) }));
                }
            }
        }
        const weights = entries.map(x => x.value);
        select.innerHTML = entries.map(x => `<option value="${x.value}">${x.label}</option>`).join('');
        if (weights.includes(current)) select.value = current;
        else if (weights.includes('700')) select.value = '700';
        else select.value = weights[weights.length - 1] || '700';
    }

    _syncFromOverlay(ov) {
        if (!ov) return;
        this._val('rop-x', Math.round(ov.x));
        this._val('rop-y', Math.round(ov.y));
        this._val('rop-w', Math.round(ov.w));
        this._val('rop-h', Math.round(ov.h));
        this._val('rop-rotation', ov.rotation || 0);
        const opacityPct = Math.round((ov.opacity ?? 255) / 255 * 100);
        this._val('rop-opacity', opacityPct);
        const opValEl = this.container.querySelector('#rop-opacity-val');
        if (opValEl) opValEl.textContent = opacityPct + '%';
        // 清理源数据浮点精度 + 显示
        ov.start = _ropRound(ov.start || 0);
        this._val('rop-start', ov.start);
        // 9999 = 全程，面板显示实际时长但不修改数据
        let displayEnd = ov.end || 0;
        if (displayEnd >= 9999) {
            const mediaEl = document.getElementById('reels-preview-video') || document.querySelector('#reels-preview video');
            if (mediaEl && mediaEl.duration && isFinite(mediaEl.duration)) {
                displayEnd = _ropRound(mediaEl.duration);
            } else {
                displayEnd = 9999; // 保持原值
            }
        } else {
            ov.end = _ropRound(displayEnd);
            displayEnd = ov.end;
        }
        this._val('rop-end', displayEnd);
        // 同步到滚动字幕专用的时间字段
        this._val('rop-scroll-start-time', ov.start);
        this._val('rop-scroll-end-time', displayEnd);

        if (ov.type === 'text') {
            this._val('rop-content', ov.content || '');
            this._val('rop-font', ov.font_family || 'Arial');
            this._refreshWeightOptions('rop-font-weight', ov.font_family || 'Arial');
            this._val('rop-fontsize', ov.fontsize || 40);
            this._val('rop-color', ov.color || '#ffffff');
            const fw = Math.max(100, Math.min(900, parseInt(ov.font_weight || (ov.bold ? 700 : 400), 10) || 400));
            this._val('rop-font-weight', fw);
            this._val('rop-bold', fw >= 600);
            this._val('rop-stroke-color', ov.stroke_color || '#000000');
            this._val('rop-stroke-width', ov.stroke_width || 0);
            this._val('rop-shadow-color', ov.shadow_color || '#000000');
            this._val('rop-shadow-blur', ov.shadow_blur || 0);
        }

        if (ov.type === 'image') {
            const scalePct = Math.round((ov.scale || 1) * 100);
            this._val('rop-scale', scalePct);
            const scValEl = this.container.querySelector('#rop-scale-val');
            if (scValEl) scValEl.textContent = scalePct + '%';
            this._val('rop-flip-h', ov.flip_x || false);
            this._val('rop-flip-v', ov.flip_y || false);
            this._val('rop-blend', ov.blend_mode || 'source-over');
        }

        if (ov.type === 'textcard') {
            this._val('rop-card-color', ov.card_color || '#ffffff');
            this._val('rop-card-opacity', ov.card_opacity ?? 80);
            this._val('rop-radius-all', ov.radius_tl ?? 33);
            this._val('rop-title-text', ov.title_text || '');
            this._val('rop-title-font', ov.title_font_family || 'Crimson Pro');
            this._refreshWeightOptions('rop-title-weight', ov.title_font_family || 'Crimson Pro');
            this._val('rop-title-fontsize', ov.title_fontsize ?? 60);
            this._val('rop-title-color', ov.title_color || '#000000');
            const tw = Math.max(100, Math.min(900, parseInt(ov.title_font_weight || ((ov.title_bold !== false) ? 900 : 400), 10) || 900));
            this._val('rop-title-weight', tw);
            this._val('rop-title-bold', tw >= 600);
            this._val('rop-title-uppercase', ov.title_uppercase !== false);
            this._val('rop-title-align', ov.title_align || 'center');
            this._val('rop-body-text', ov.body_text || '');
            this._val('rop-body-font', ov.body_font_family || 'Arial');
            this._refreshWeightOptions('rop-body-weight', ov.body_font_family || 'Arial');
            this._val('rop-body-fontsize', ov.body_fontsize ?? 40);
            this._val('rop-body-color', ov.body_color || '#000000');
            const bw = Math.max(100, Math.min(900, parseInt(ov.body_font_weight || (ov.body_bold ? 700 : 400), 10) || 400));
            this._val('rop-body-weight', bw);
            this._val('rop-body-bold', bw >= 600);
            this._val('rop-body-linespacing', ov.body_line_spacing ?? 6);
            this._val('rop-body-align', ov.body_align || 'center');
            // Footer
            this._val('rop-footer-text', ov.footer_text || '');
            this._val('rop-footer-font', ov.footer_font_family || 'Arial');
            this._refreshWeightOptions('rop-footer-weight', ov.footer_font_family || 'Arial');
            this._val('rop-footer-fontsize', ov.footer_fontsize ?? 32);
            this._val('rop-footer-color', ov.footer_color || '#666666');
            const ftw = Math.max(100, Math.min(900, parseInt(ov.footer_font_weight || (ov.footer_bold ? 700 : 400), 10) || 400));
            this._val('rop-footer-weight', ftw);
            this._val('rop-footer-bold', ftw >= 600);
            this._val('rop-footer-align', ov.footer_align || 'center');
            // Text effects
            this._val('rop-text-stroke-color', ov.text_stroke_color || '#000000');
            this._val('rop-text-stroke-width', ov.text_stroke_width ?? 0);
            this._val('rop-text-shadow-color', ov.text_shadow_color || '#000000');
            this._val('rop-text-shadow-blur', ov.text_shadow_blur ?? 0);
            this._val('rop-text-shadow-x', ov.text_shadow_x ?? 2);
            this._val('rop-text-shadow-y', ov.text_shadow_y ?? 2);
            this._val('rop-auto-fit', ov.auto_fit !== false);
            this._val('rop-auto-center', ov.auto_center_v !== false);
            this._val('rop-offset-y', ov.offset_y ?? 0);
            this._val('rop-fullscreen-mask', ov.fullscreen_mask || false);
            this._val('rop-title-body-gap', ov.title_body_gap ?? 42);
            this._val('rop-pad-top', ov.padding_top ?? 20);
            this._val('rop-pad-bottom', ov.padding_bottom ?? 40);
            this._val('rop-pad-left', ov.padding_left ?? 40);
            this._val('rop-pad-right', ov.padding_right ?? 40);
            this._val('rop-card-width', ov.w || 910);
            this._val('rop-auto-shrink', ov.auto_shrink !== false);
            this._val('rop-max-height', ov.max_height ?? 1400);
            this._val('rop-title-max-lines', ov.title_max_lines ?? 3);
            this._val('rop-min-fontsize', ov.min_fontsize ?? 16);
        }

        if (ov.type === 'scroll') {
            this._val('rop-scroll-content', ov.content || '');
            // 标题
            this._val('rop-scroll-title', ov.scroll_title || '');
            this._val('rop-scroll-title-fontsize', ov.scroll_title_fontsize ?? 56);
            this._val('rop-scroll-title-color', ov.scroll_title_color || ov.color || '#ffffff');
            this._val('rop-scroll-title-weight', ov.scroll_title_font_weight ?? 700);
            this._val('rop-scroll-title-gap', ov.scroll_title_gap ?? 20);
            this._val('rop-scroll-title-fixed', ov.scroll_title_fixed !== false);
            // 正文
            this._val('rop-scroll-font', ov.font_family || 'Arial');
            this._refreshWeightOptions('rop-scroll-weight', ov.font_family || 'Arial');
            this._val('rop-scroll-fontsize', ov.fontsize || 40);
            this._val('rop-scroll-color', ov.color || '#ffffff');
            const sw = Math.max(100, Math.min(900, parseInt(ov.font_weight || (ov.bold ? 700 : 400), 10) || 400));
            this._val('rop-scroll-weight', sw);
            this._val('rop-scroll-bold', sw >= 600);
            this._val('rop-scroll-align', ov.text_align || 'center');
            this._val('rop-scroll-linespacing', ov.line_spacing ?? 6);
            this._val('rop-scroll-textw', ov.text_width ?? 900);
            this._val('rop-scroll-stroke-color', ov.stroke_color || '#000000');
            this._val('rop-scroll-stroke-width', ov.stroke_width || 0);
            this._val('rop-scroll-shadow', ov.shadow_enabled || false);
            this._val('rop-scroll-shadow-color', ov.shadow_color || '#000000');
            this._val('rop-scroll-shadow-blur', ov.shadow_blur || 4);
            // 独立同步 起始Y 和 结束Y
            this._val('rop-scroll-from-y', ov.scroll_from_y ?? 960);
            this._val('rop-scroll-to-y', ov.scroll_to_y ?? -200);
            this._val('rop-scroll-from-x', ov.scroll_from_x ?? 90);
            // scroll_speed 已移除，速度由 距离÷时间 自动决定
            this._val('rop-scroll-feather-top', ov.feather_top ?? 80);
            this._val('rop-scroll-feather-bottom', ov.feather_bottom ?? 80);
            this._val('rop-scroll-auto-stop', ov.scroll_auto_stop === true);
            this._val('rop-scroll-auto-fit', ov.scroll_auto_fit === true);
            this._val('rop-scroll-min-fontsize', ov.scroll_min_fontsize ?? 16);
            // 卡片背景
            this._val('rop-scroll-bg-enabled', ov.bg_enabled || false);
            this._val('rop-scroll-bg-color', ov.bg_color || '#000000');
            this._val('rop-scroll-bg-opacity', Math.round((ov.bg_opacity ?? 191) / 255 * 100));
            this._val('rop-scroll-bg-radius', ov.bg_radius ?? 12);
            this._val('rop-scroll-bg-pad-top', ov.bg_padding_top ?? 55);
            this._val('rop-scroll-bg-pad-bottom', ov.bg_padding_bottom ?? 55);
            this._val('rop-scroll-bg-pad-left', ov.bg_padding_left ?? 16);
            this._val('rop-scroll-bg-pad-right', ov.bg_padding_right ?? 16);
            this._val('rop-scroll-bg-fullscreen', ov.bg_fullscreen || false);
        }

        this._val('rop-anim-in', ov.anim_in_type || 'none');
        this._val('rop-anim-out', ov.anim_out_type || 'none');
        this._val('rop-anim-in-dur', ov.anim_in_duration || 0.3);
        this._val('rop-anim-out-dur', ov.anim_out_duration || 0.3);

        // Fixed text flag
        this._val('rop-fixed-text', ov.fixed_text || false);
    }

    _syncToOverlay() {
        const ov = this._selectedOv;
        if (!ov) return;

        ov.x = this._get('rop-x');
        ov.y = this._get('rop-y');
        ov.w = this._get('rop-w');
        ov.h = this._get('rop-h');
        ov.rotation = this._get('rop-rotation');
        ov.opacity = Math.round(this._get('rop-opacity') / 100 * 255);
        ov.start = _ropRound(this._get('rop-start'));
        ov.end = _ropRound(this._get('rop-end'));

        if (ov.type === 'text') {
            ov.content = this._get('rop-content');
            ov.font_family = this._get('rop-font');
            ov.fontsize = this._get('rop-fontsize');
            ov.color = this._get('rop-color');
            const fw = Math.max(100, Math.min(900, parseInt(this._get('rop-font-weight') || (this._get('rop-bold') ? 700 : 400), 10) || 400));
            ov.font_weight = fw;
            ov.bold = fw >= 600;
            ov.stroke_color = this._get('rop-stroke-color');
            ov.stroke_width = this._get('rop-stroke-width');
            ov.shadow_color = this._get('rop-shadow-color');
            ov.shadow_blur = this._get('rop-shadow-blur');
        }

        if (ov.type === 'image') {
            ov.scale = this._get('rop-scale') / 100;
            ov.flip_x = this._get('rop-flip-h');
            ov.flip_y = this._get('rop-flip-v');
            ov.blend_mode = this._get('rop-blend');
        }

        if (ov.type === 'textcard') {
            ov.card_color = this._get('rop-card-color');
            ov.card_opacity = this._get('rop-card-opacity');
            const radius = this._get('rop-radius-all');
            ov.radius_tl = radius;
            ov.radius_tr = radius;
            ov.radius_bl = radius;
            ov.radius_br = radius;
            ov.title_text = this._get('rop-title-text');
            ov.title_font_family = this._get('rop-title-font');
            ov.title_fontsize = this._get('rop-title-fontsize');
            ov.title_color = this._get('rop-title-color');
            const tw = Math.max(100, Math.min(900, parseInt(this._get('rop-title-weight') || (this._get('rop-title-bold') ? 900 : 400), 10) || 900));
            ov.title_font_weight = tw;
            ov.title_bold = tw >= 600;
            ov.title_uppercase = this._get('rop-title-uppercase');
            ov.title_align = this._get('rop-title-align');
            ov.body_text = this._get('rop-body-text');
            ov.body_font_family = this._get('rop-body-font');
            ov.body_fontsize = this._get('rop-body-fontsize');
            ov.body_color = this._get('rop-body-color');
            const bw = Math.max(100, Math.min(900, parseInt(this._get('rop-body-weight') || (this._get('rop-body-bold') ? 700 : 400), 10) || 400));
            ov.body_font_weight = bw;
            ov.body_bold = bw >= 600;
            ov.body_line_spacing = this._get('rop-body-linespacing');
            ov.body_align = this._get('rop-body-align');
            // Footer
            ov.footer_text = this._get('rop-footer-text');
            ov.footer_font_family = this._get('rop-footer-font');
            ov.footer_fontsize = this._get('rop-footer-fontsize');
            ov.footer_color = this._get('rop-footer-color');
            const ftw = Math.max(100, Math.min(900, parseInt(this._get('rop-footer-weight') || (this._get('rop-footer-bold') ? 700 : 400), 10) || 400));
            ov.footer_font_weight = ftw;
            ov.footer_bold = ftw >= 600;
            ov.footer_align = this._get('rop-footer-align');
            // Text effects
            ov.text_stroke_color = this._get('rop-text-stroke-color');
            ov.text_stroke_width = this._get('rop-text-stroke-width');
            ov.text_shadow_color = this._get('rop-text-shadow-color');
            ov.text_shadow_blur = this._get('rop-text-shadow-blur');
            ov.text_shadow_x = this._get('rop-text-shadow-x');
            ov.text_shadow_y = this._get('rop-text-shadow-y');
            ov.auto_fit = this._get('rop-auto-fit');
            ov.auto_center_v = this._get('rop-auto-center');
            ov.offset_y = this._get('rop-offset-y');
            ov.fullscreen_mask = this._get('rop-fullscreen-mask');
            ov.title_body_gap = this._get('rop-title-body-gap');
            ov.padding_top = this._get('rop-pad-top');
            ov.padding_bottom = this._get('rop-pad-bottom');
            ov.padding_left = this._get('rop-pad-left');
            ov.padding_right = this._get('rop-pad-right');
            const cardW = this._get('rop-card-width');
            if (cardW > 0) ov.w = cardW;
            ov.auto_shrink = this._get('rop-auto-shrink');
            ov.max_height = this._get('rop-max-height');
            ov.title_max_lines = this._get('rop-title-max-lines');
            ov.min_fontsize = this._get('rop-min-fontsize');
        }

        if (ov.type === 'scroll') {
            ov.content = this._get('rop-scroll-content');
            // 标题
            ov.scroll_title = this._get('rop-scroll-title') || '';
            ov.scroll_title_fontsize = this._get('rop-scroll-title-fontsize') || 56;
            ov.scroll_title_font_weight = this._get('rop-scroll-title-weight') || 700;
            ov.scroll_title_bold = (ov.scroll_title_font_weight >= 600);
            ov.scroll_title_color = this._get('rop-scroll-title-color') || '';
            ov.scroll_title_gap = this._get('rop-scroll-title-gap') ?? 20;
            ov.scroll_title_fixed = this._get('rop-scroll-title-fixed');
            // 正文
            ov.font_family = this._get('rop-scroll-font');
            ov.fontsize = this._get('rop-scroll-fontsize');
            ov.color = this._get('rop-scroll-color');
            const sw = Math.max(100, Math.min(900, parseInt(this._get('rop-scroll-weight') || (this._get('rop-scroll-bold') ? 700 : 400), 10) || 400));
            ov.font_weight = sw;
            ov.bold = sw >= 600;
            ov.text_align = this._get('rop-scroll-align');
            ov.line_spacing = this._get('rop-scroll-linespacing');
            ov.text_width = this._get('rop-scroll-textw');
            ov.stroke_color = this._get('rop-scroll-stroke-color');
            ov.stroke_width = this._get('rop-scroll-stroke-width');
            ov.use_stroke = (ov.stroke_width || 0) > 0;
            ov.shadow_enabled = this._get('rop-scroll-shadow');
            ov.shadow_color = this._get('rop-scroll-shadow-color');
            ov.shadow_blur = this._get('rop-scroll-shadow-blur');
            // 独立获取设置，解耦拖动逻辑防止双重偏移修改死循环
            ov.scroll_from_y = this._get('rop-scroll-from-y') ?? 960;
            ov.scroll_to_y = this._get('rop-scroll-to-y') ?? -200;
            ov.scroll_from_x = this._get('rop-scroll-from-x');
            ov.scroll_to_x = ov.scroll_from_x;  // X stays the same
            ov.scroll_speed = 1;  // 固定为1，速度由 距离÷时间 自动决定
            ov.feather_top = this._get('rop-scroll-feather-top');
            ov.feather_bottom = this._get('rop-scroll-feather-bottom');
            ov.scroll_auto_stop = this._get('rop-scroll-auto-stop');
            ov.scroll_auto_fit = this._get('rop-scroll-auto-fit');
            ov.scroll_min_fontsize = this._get('rop-scroll-min-fontsize');
            // 卡片背景
            ov.bg_enabled = this._get('rop-scroll-bg-enabled');
            ov.bg_color = this._get('rop-scroll-bg-color');
            ov.bg_opacity = Math.round(this._get('rop-scroll-bg-opacity') / 100 * 255);
            ov.bg_radius = this._get('rop-scroll-bg-radius');
            ov.bg_padding_top = this._get('rop-scroll-bg-pad-top');
            ov.bg_padding_bottom = this._get('rop-scroll-bg-pad-bottom');
            ov.bg_padding_left = this._get('rop-scroll-bg-pad-left');
            ov.bg_padding_right = this._get('rop-scroll-bg-pad-right');
            ov.bg_fullscreen = this._get('rop-scroll-bg-fullscreen');
        }

        ov.anim_in_type = this._get('rop-anim-in');
        ov.anim_out_type = this._get('rop-anim-out');
        ov.anim_in_duration = this._get('rop-anim-in-dur');
        ov.anim_out_duration = this._get('rop-anim-out-dur');

        // Fixed text flag
        ov.fixed_text = this._get('rop-fixed-text');

        // Re-render canvas to reflect changes
        if (this.videoCanvas) this.videoCanvas.render();
    }

    // ═══════════════════════════════════════════════
    // 卡片模板管理
    // ═══════════════════════════════════════════════

    _getCardTemplates() {
        try {
            return JSON.parse(localStorage.getItem('reels_card_templates') || '{}');
        } catch (e) { return {}; }
    }

    _setCardTemplates(data) {
        localStorage.setItem('reels_card_templates', JSON.stringify(data));
    }

    _refreshCardTemplateSelect() {
        if (!this.container) return;
        const select = this.container.querySelector('#rop-card-tpl-select');
        if (!select) return;
        const current = select.value;
        const templates = this._getCardTemplates();
        select.innerHTML = '<option value="">-- 选择模板 --</option>';
        for (const name of Object.keys(templates)) {
            const opt = document.createElement('option');
            opt.value = name;
            opt.textContent = name;
            select.appendChild(opt);
        }
        if (current && templates[current]) select.value = current;
    }

    _extractCardStyle(ov) {
        // Extract only card-related style from overlay (not position/text content)
        const keys = [
            'w',
            'card_color', 'card_opacity',
            'radius_tl', 'radius_tr', 'radius_bl', 'radius_br',
            'title_font_family', 'title_fontsize', 'title_font_weight', 'title_bold', 'title_italic',
            'title_color', 'title_align', 'title_uppercase', 'title_letter_spacing',
            'body_font_family', 'body_fontsize', 'body_font_weight', 'body_bold', 'body_italic',
            'body_color', 'body_align', 'body_line_spacing',
            'auto_fit', 'auto_center_v',
            'padding_top', 'padding_bottom', 'padding_left', 'padding_right',
            'title_body_gap',
            'max_height', 'auto_shrink', 'title_max_lines', 'min_fontsize', 'fullscreen_mask',
        ];
        const result = {};
        for (const k of keys) {
            if (ov[k] !== undefined) result[k] = ov[k];
        }
        return result;
    }

    _showCardTemplateNameDialog(defaultName = '') {
        return new Promise((resolve) => {
            const overlay = document.createElement('div');
            overlay.style.cssText = [
                'position:fixed', 'inset:0', 'z-index:99999',
                'background:rgba(0,0,0,0.6)',
                'display:flex', 'align-items:center', 'justify-content:center',
            ].join(';');

            const box = document.createElement('div');
            box.style.cssText = [
                'background:var(--bg-primary,#1e1e2e)',
                'border:1px solid var(--border-color,#444)',
                'border-radius:12px',
                'padding:20px',
                'min-width:320px',
                'box-shadow:0 8px 32px rgba(0,0,0,0.5)',
            ].join(';');

            box.innerHTML = `
                <div style="font-size:15px;font-weight:600;margin-bottom:12px;color:var(--text-primary,#fff);">保存卡片模板</div>
                <input type="text" class="rop-tpl-name-input" placeholder="请输入模板名称"
                    style="width:100%;box-sizing:border-box;padding:8px 12px;border-radius:6px;border:1px solid var(--border-color,#555);background:var(--bg-secondary,#2a2a3e);color:var(--text-primary,#fff);font-size:14px;outline:none;">
                <div style="display:flex;gap:10px;justify-content:flex-end;margin-top:14px;">
                    <button class="rop-tpl-cancel" style="padding:6px 16px;border-radius:6px;border:1px solid var(--border-color,#555);background:transparent;color:var(--text-secondary,#aaa);cursor:pointer;font-size:13px;">取消</button>
                    <button class="rop-tpl-ok" style="padding:6px 16px;border-radius:6px;border:none;background:var(--accent-primary,#5b6abf);color:#fff;cursor:pointer;font-size:13px;">保存</button>
                </div>
            `;

            overlay.appendChild(box);
            document.body.appendChild(overlay);

            const input = box.querySelector('.rop-tpl-name-input');
            const okBtn = box.querySelector('.rop-tpl-ok');
            const cancelBtn = box.querySelector('.rop-tpl-cancel');
            input.value = defaultName || '';

            const close = (val) => {
                if (overlay.parentNode) overlay.parentNode.removeChild(overlay);
                resolve(val);
            };

            okBtn.onclick = () => close((input.value || '').trim() || null);
            cancelBtn.onclick = () => close(null);
            overlay.onclick = (e) => { if (e.target === overlay) close(null); };
            input.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') close((input.value || '').trim() || null);
                if (e.key === 'Escape') close(null);
            });
            setTimeout(() => input.focus(), 30);
        });
    }

    async _saveCardTemplate() {
        if (!this._selectedOv || this._selectedOv.type !== 'textcard') {
            alert('请先选择一个文案卡片覆层');
            return;
        }
        const select = this.container.querySelector('#rop-card-tpl-select');
        const defaultName = select ? select.value : '';
        const name = await this._showCardTemplateNameDialog(defaultName);
        if (!name) return;
        const templates = this._getCardTemplates();
        templates[name] = this._extractCardStyle(this._selectedOv);
        this._setCardTemplates(templates);
        this._refreshCardTemplateSelect();
        if (select) select.value = name;
        alert(`✅ 卡片模板 "${name}" 已保存`);
    }

    async _loadCardTemplate() {
        if (!this._selectedOv || this._selectedOv.type !== 'textcard') {
            alert('请先选择一个文案卡片覆层');
            return;
        }
        const select = this.container.querySelector('#rop-card-tpl-select');
        if (!select || !select.value) {
            alert('请先在下拉列表中选择一个模板');
            return;
        }
        const name = select.value;
        const templates = this._getCardTemplates();
        if (!templates[name]) return;

        Object.assign(this._selectedOv, templates[name]);
        this._syncFromOverlay(this._selectedOv);
        if (this.videoCanvas) this.videoCanvas.render();
    }

    async _deleteCardTemplate() {
        const select = this.container.querySelector('#rop-card-tpl-select');
        if (!select || !select.value) {
            alert('请先在下拉列表中选择要删除的模板');
            return;
        }
        const name = select.value;
        if (!confirm(`确定要删除模板 "${name}" 吗？`)) return;

        const templates = this._getCardTemplates();
        delete templates[name];
        this._setCardTemplates(templates);
        this._refreshCardTemplateSelect();
    }

    _exportCardTemplates() {
        const templates = this._getCardTemplates();
        if (Object.keys(templates).length === 0) {
            alert('您还没有保存任何自定义模板！');
            return;
        }
        const str = JSON.stringify(templates, null, 2);
        const blob = new Blob([str], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `textcard_templates_${new Date().getTime()}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }

    _importCardTemplates() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.json';
        input.onchange = (e) => {
            const file = e.target.files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const data = JSON.parse(e.target.result);
                    const templates = this._getCardTemplates();
                    const count = Object.keys(data).length;
                    Object.assign(templates, data); // 合并模板
                    this._setCardTemplates(templates);
                    this._refreshCardTemplateSelect();
                    alert(`✅ 成功导入了 ${count} 个模板`);
                } catch (err) {
                    console.error('导入模板出错:', err);
                    alert('导入失败，不是有效的模板 JSON 文件。');
                }
            };
            reader.readAsText(file);
        };
        input.click();
    }

    // ═══════════════════════════════════════════════
    // 覆层组预设管理 (多层)
    // ═══════════════════════════════════════════════

    _getOverlayGroupPresets() {
        try {
            return JSON.parse(localStorage.getItem('reels_overlay_group_presets') || '{}');
        } catch (e) { return {}; }
    }

    _setOverlayGroupPresets(data) {
        localStorage.setItem('reels_overlay_group_presets', JSON.stringify(data));
    }

    _refreshOverlayGroupPresetSelect() {
        if (!this.container) return;
        const select = this.container.querySelector('#rop-group-preset-select');
        if (!select) return;
        const current = select.value;
        const presets = this._getOverlayGroupPresets();
        select.innerHTML = '<option value="">-- 选择预设 --</option>';
        for (const name of Object.keys(presets)) {
            const layers = presets[name];
            const count = Array.isArray(layers) ? layers.length : 0;
            const opt = document.createElement('option');
            opt.value = name;
            opt.textContent = `${name} (${count}层)`;
            select.appendChild(opt);
        }
        if (current && presets[current]) select.value = current;
    }

    async _saveOverlayGroupPreset() {
        if (!this.videoCanvas) {
            alert('没有可用的覆层管理器');
            return;
        }
        const overlays = this.videoCanvas.overlayMgr?.overlays || [];
        if (overlays.length === 0) {
            alert('当前没有覆层，请先添加覆层再保存');
            return;
        }
        const name = await this._showCardTemplateNameDialog('');
        if (!name) return;

        // Deep clone overlays, strip runtime-only keys and text content
        const serialized = overlays.map(ov => {
            const clone = JSON.parse(JSON.stringify(ov));
            delete clone._img;
            delete clone._imgLoaded;
            delete clone._templateName;
            // Only keep text content for layers marked as fixed
            if (!clone.fixed_text) {
                delete clone.title_text;
                delete clone.body_text;
                delete clone.footer_text;
                delete clone.content;  // plain text overlay
            }
            return clone;
        });

        const presets = this._getOverlayGroupPresets();
        presets[name] = serialized;
        this._setOverlayGroupPresets(presets);
        this._refreshOverlayGroupPresetSelect();
        const select = this.container.querySelector('#rop-group-preset-select');
        if (select) select.value = name;
        alert(`✅ 覆层组预设 "${name}" 已保存 (${serialized.length} 层)`);
    }

    _loadOverlayGroupPreset() {
        const select = this.container.querySelector('#rop-group-preset-select');
        if (!select || !select.value) {
            alert('请先在下拉列表中选择一个预设');
            return;
        }
        const name = select.value;
        const presets = this._getOverlayGroupPresets();
        const layers = presets[name];
        if (!Array.isArray(layers) || layers.length === 0) {
            alert('该预设为空或格式不正确');
            return;
        }
        if (!this.videoCanvas) return;

        const mgr = this.videoCanvas.overlayMgr;
        if (!mgr) return;

        // Confirm if there are existing overlays
        if (mgr.overlays.length > 0) {
            if (!confirm(`当前有 ${mgr.overlays.length} 个覆层，加载预设将替换全部。继续？`)) return;
        }

        // Save existing overlays' text before clearing
        const oldOverlays = [...mgr.overlays];

        // Clear existing
        mgr.overlays = [];

        // Deep-clone and add each layer with new IDs, preserving existing text
        for (let i = 0; i < layers.length; i++) {
            const layerData = layers[i];
            const clone = JSON.parse(JSON.stringify(layerData));
            clone.id = 'ov_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8);
            // For non-fixed layers, preserve text from corresponding old overlay
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
            mgr.overlays.push(clone);
        }

        // Refresh UI
        this._selectedOv = mgr.overlays[0] || null;
        this._refreshOverlayList();
        if (this._selectedOv) this._syncFromOverlay(this._selectedOv);
        if (this.videoCanvas) this.videoCanvas.render();
        alert(`✅ 已加载预设 "${name}" (${layers.length} 层)`);
    }

    _deleteOverlayGroupPreset() {
        const select = this.container.querySelector('#rop-group-preset-select');
        if (!select || !select.value) {
            alert('请先在下拉列表中选择要删除的预设');
            return;
        }
        const name = select.value;
        if (!confirm(`确定要删除预设 "${name}" 吗？`)) return;
        const presets = this._getOverlayGroupPresets();
        delete presets[name];
        this._setOverlayGroupPresets(presets);
        this._refreshOverlayGroupPresetSelect();
    }

    _importOverlayGroupPresets() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.json';
        input.onchange = (e) => {
            const file = e.target.files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = (ev) => {
                try {
                    const data = JSON.parse(ev.target.result);
                    const presets = this._getOverlayGroupPresets();
                    const count = Object.keys(data).length;
                    Object.assign(presets, data);
                    this._setOverlayGroupPresets(presets);
                    this._refreshOverlayGroupPresetSelect();
                    alert(`✅ 成功导入了 ${count} 个覆层组预设`);
                } catch (err) {
                    console.error('导入预设出错:', err);
                    alert('导入失败，不是有效的预设 JSON 文件。');
                }
            };
            reader.readAsText(file);
        };
        input.click();
    }

    _exportOverlayGroupPresets() {
        const presets = this._getOverlayGroupPresets();
        const keys = Object.keys(presets);
        if (keys.length === 0) {
            alert('暂无覆层组预设可导出');
            return;
        }
        const json = JSON.stringify(presets, null, 2);
        const blob = new Blob([json], { type: 'application/json' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `overlay_group_presets_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(a.href);
    }
}

// ═══════════════════════════════════════════════
// CSS
// ═══════════════════════════════════════════════

(function injectOverlayPanelStyles() {
    if (document.getElementById('rop-styles')) return;
    const s = document.createElement('style');
    s.id = 'rop-styles';
    s.textContent = `
        .rop-panel { font-size: 12px; color: #ccc; }
        .rop-section { margin-bottom: 12px; }
        .rop-header { display:flex; justify-content:space-between; align-items:center;
                      padding: 8px 10px; background: var(--bg-secondary, #1e1e3a);
                      border-radius: 6px 6px 0 0; font-weight: bold; font-size: 13px; }
        .rop-header-actions { display:flex; gap:4px; }
        .rop-btn { padding:3px 8px !important; font-size:11px !important; min-width:unset !important; }
        .rop-btn-full { width:100%; margin-top:4px; }
        .rop-btn-danger { color:#ff6b6b !important; }
        .rop-list { background:var(--bg-tertiary, #0f0f2e); border-radius:0 0 6px 6px; max-height:150px; overflow-y:auto; }
        .rop-list-item { display:flex; align-items:center; gap:6px; padding:6px 10px; cursor:pointer;
                         border-bottom: 1px solid rgba(255,255,255,0.05); transition: background 0.15s; }
        .rop-list-item:hover { background:rgba(255,255,255,0.06); }
        .rop-list-item.selected { background:rgba(0,212,255,0.12); border-left:3px solid #00D4FF; }
        .rop-list-arrow { font-size:9px; color:#666; width:10px; flex-shrink:0; }
        .rop-list-label { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        .rop-list-time { font-size:10px; color:#888; font-family:monospace; white-space:nowrap; }
        .rop-list-del { background:none; border:none; color:#666; cursor:pointer; font-size:13px; padding:0 4px;
                        line-height:1; flex-shrink:0; transition:color 0.15s; }
        .rop-list-del:hover { color:#ff6b6b; }
        .rop-empty { padding:16px; text-align:center; color:#555; font-style:italic; }
        .rop-group { padding:8px 10px; background:var(--bg-tertiary, #0f0f2e); border-radius:6px; margin-top:8px; }
        .rop-group-title { font-weight:bold; font-size:11px; color:#8899bb; margin-bottom:6px; text-transform:uppercase; letter-spacing:0.5px; }
        .rop-grid { display:grid; grid-template-columns: auto 1fr; gap:4px 8px; align-items:center; }
        .rop-grid label { font-size:11px; color:#999; text-align:right; }
        .rop-input { width:100%; padding:3px 6px; background:var(--bg-primary, #12122a); border:1px solid var(--border-color, #2a2a4a);
                     border-radius:4px; color:#ddd; font-size:11px; font-family:monospace; }
        .rop-input:focus { border-color:#00D4FF; outline:none; }
        .rop-range { width:100%; }
        .rop-select { width:100%; padding:3px 6px; background:var(--bg-primary, #12122a); border:1px solid var(--border-color, #2a2a4a);
                      border-radius:4px; color:#ddd; font-size:11px; }
        .rop-color { width:100%; height:24px; padding:0; border:1px solid var(--border-color, #2a2a4a); border-radius:4px; cursor:pointer; }
        .rop-textarea { width:100%; padding:6px; background:var(--bg-primary, #12122a); border:1px solid var(--border-color, #2a2a4a);
                        border-radius:4px; color:#ddd; font-size:11px; resize:vertical; margin-bottom:6px; font-family:system-ui; }
        .rop-actions { padding:8px 10px; }
        .rop-slider-combo { display:flex; align-items:center; gap:6px; width:100%; }
        .rop-slider-combo .rop-range { flex:1; min-width:0; }
        .rop-num-readout { width:52px!important; flex-shrink:0; padding:2px 4px; background:var(--bg-primary, #12122a);
                           border:1px solid var(--border-color, #2a2a4a); border-radius:4px; color:#ddd;
                           font-size:11px; font-family:monospace; text-align:center; }
        .rop-num-readout:focus { border-color:#00D4FF; outline:none; }
        .rop-reset-btn { flex-shrink:0; width:22px; height:22px; padding:0; border:1px solid rgba(255,255,255,0.1);
                         background:rgba(255,255,255,0.05); border-radius:4px; color:#888; font-size:12px;
                         cursor:pointer; display:flex; align-items:center; justify-content:center; transition:all 0.15s; }
        .rop-reset-btn:hover { background:rgba(0,212,255,0.15); color:#00D4FF; border-color:rgba(0,212,255,0.3); }
        .rop-reset-all { padding:2px 8px; border:1px solid rgba(255,255,255,0.1); background:rgba(255,255,255,0.05);
                         border-radius:4px; color:#888; font-size:10px; cursor:pointer; transition:all 0.15s; white-space:nowrap; }
        .rop-reset-all:hover { background:rgba(0,212,255,0.15); color:#00D4FF; border-color:rgba(0,212,255,0.3); }
    `;
    document.head.appendChild(s);
})();

// Export
if (typeof window !== 'undefined') window.ReelsOverlayPanel = ReelsOverlayPanel;
