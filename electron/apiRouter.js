/**
 * IPC API 路由器
 * 将所有前端 API 调用路由到对应的 Node.js 服务
 * 替代 Python Flask 后端的所有 HTTP 端点
 */
const { ipcMain } = require('electron');
const path = require('path');
const fs = require('fs');
const os = require('os');

// 服务模块
const ffmpegService = require('./services/ffmpeg');
const elevenlabsService = require('./services/elevenlabs');
const settingsService = require('./services/settings');
const subtitleService = require('./services/subtitle');
const fcpxmlService = require('./services/fcpxml');

const ytdlpService = require('./services/ytdlp');
const gladiaService = require('./services/gladia');
const imageClassifyService = require('./services/imageClassify');
const workflowService = require('./services/workflow');
const subtitleUtils = require('./services/subtitleUtils');
const { audioSubtitleSearchDifferentStrong } = require('./services/subtitleAlignment');
const wav2lipService = require('./services/wav2lip');

/**
 * 注册所有 IPC API 路由
 */
function registerAPIHandlers() {
    // ==================== 通用 API 调用接口 ====================
    ipcMain.handle('api-call', async (event, endpoint, data) => {
        try {
            const result = await routeAPI(endpoint, data || {});
            return { success: true, data: result };
        } catch (error) {
            console.error(`[API Error] ${endpoint}:`, error.message);
            return { success: false, error: error.message };
        }
    });

    // ==================== 文件上传（需要特殊处理 Buffer） ====================
    ipcMain.handle('api-upload', async (event, endpoint, fileBuffer, fileName, formData) => {
        try {
            const result = await routeUpload(endpoint, fileBuffer, fileName, formData || {});
            return { success: true, data: result };
        } catch (error) {
            console.error(`[Upload Error] ${endpoint}:`, error.message);
            return { success: false, error: error.message };
        }
    });
}

/**
 * API 路由分发
 */
async function routeAPI(endpoint, data) {
    // 去除前导斜杠
    const ep = endpoint.replace(/^\/?(api\/)?/, '');

    switch (ep) {
        // ==================== 健康检查 ====================
        case 'health':
            return { status: 'ok', uptime: process.uptime(), backend: 'nodejs' };

        // ==================== 设置 ====================
        case 'settings/gladia-keys':
            if (data._method === 'GET') return settingsService.loadGladiaKeys();
            settingsService.saveGladiaKeys(data);
            return { message: '保存成功' };

        case 'settings/elevenlabs':
            if (data._method === 'GET') return settingsService.loadElevenLabsSettings();
            return settingsService.saveElevenLabsSettings(data);

        case 'settings/elevenlabs-keys-status':
            if (data._method === 'GET') return { keys: settingsService.loadElevenLabsKeysWithStatus() };
            settingsService.saveElevenLabsKeysWithStatus(data.keys || data);
            return { message: '保存成功' };

        case 'settings/replace-rules':
            if (data._method === 'GET') return settingsService.loadReplaceRules();
            settingsService.saveReplaceRules(data);
            return { message: '保存成功' };

        case 'languages':
            return { languages: settingsService.getLanguages() };

        // ==================== 文件操作 ====================
        case 'open-folder':
            return await settingsService.openFolder(data.path);

        // ==================== ElevenLabs ====================
        case 'elevenlabs/voices': {
            const keys = elevenlabsService.loadKeys();
            if (!keys || keys.length === 0) return { voices: [], error: '未配置 API Key' };
            const apiKey = elevenlabsService.selectKey(keys, data.key_index);
            if (!apiKey) return { voices: [], error: '无可用 API Key' };
            const voices = await elevenlabsService.getVoices(apiKey);
            return { voices };
        }

        case 'elevenlabs/search': {
            const keys = elevenlabsService.loadKeys();
            if (!keys || keys.length === 0) return { voices: [], error: '未配置 API Key' };
            const apiKey = elevenlabsService.selectKey(keys, data.key_index);
            const voices = await elevenlabsService.searchVoices(apiKey, data.search_term);
            return { voices };
        }

        case 'elevenlabs/add-voice': {
            const keys = elevenlabsService.loadKeys();
            if (!keys || keys.length === 0) throw new Error('未配置 API Key');
            const apiKey = elevenlabsService.selectKey(keys, data.key_index);
            return await elevenlabsService.addVoice(apiKey, data.public_voice_id, data.name || 'My Voice', data.auto_delete !== false);
        }

        case 'elevenlabs/delete-voice': {
            const keys = elevenlabsService.loadKeys();
            if (!keys || keys.length === 0) throw new Error('未配置 API Key');
            const apiKey = elevenlabsService.selectKey(keys, data.key_index);
            return await elevenlabsService.deleteVoice(apiKey, data.voice_id);
        }

        case 'elevenlabs/quota': {
            const keys = elevenlabsService.loadKeys();
            if (!keys || keys.length === 0) return { usage: -1, limit: -1, error: '未配置 API Key' };
            const apiKey = elevenlabsService.selectKey(keys, data.key_index);
            return await elevenlabsService.getQuota(apiKey);
        }

        case 'elevenlabs/all-quotas':
            return await elevenlabsService.getAllQuotas();

        case 'elevenlabs/tts': {
            const keys = elevenlabsService.loadKeys();
            if (!keys || keys.length === 0) throw new Error('未配置 API Key');
            if (!data.text || !data.voice_id) throw new Error('缺少必需参数');

            let stabilityVal = parseFloat(data.stability || 0.5);
            if (stabilityVal > 1) stabilityVal /= 100;
            stabilityVal = Math.max(0, Math.min(1, stabilityVal));

            const { audio, usedKey } = await elevenlabsService.requestTTSWithRotation(
                keys, data.voice_id, data.text,
                data.model_id || 'eleven_multilingual_v2',
                stabilityVal,
                data.output_format || 'mp3_44100_128',
                data.key_index
            );

            let savePath = data.save_path;
            if (!savePath) {
                savePath = elevenlabsService.buildTTSSavePath(data.text, data.output_format || 'mp3_44100_128', 'tts');
            }
            fs.writeFileSync(savePath, audio);
            return { message: '生成成功', file_path: savePath };
        }

        case 'elevenlabs/tts-batch': {
            const keys = elevenlabsService.loadKeys();
            if (!keys || keys.length === 0) throw new Error('未配置 API Key');
            const items = data.items || [];
            if (items.length === 0) throw new Error('缺少生成项目');

            const modelId = data.model_id || 'eleven_multilingual_v2';
            let stabilityVal = parseFloat(data.stability || 0.5);
            if (stabilityVal > 1) stabilityVal /= 100;
            const outputFormat = data.output_format || 'mp3_44100_128';

            const results = [];
            for (let i = 0; i < items.length; i++) {
                const item = items[i];
                try {
                    const { audio, usedKey } = await elevenlabsService.requestTTSWithRotation(
                        keys, item.voice_id || data.voice_id, item.text,
                        modelId, stabilityVal, outputFormat, item.key_index || data.key_index
                    );
                    let savePath = item.save_path;
                    if (!savePath) {
                        savePath = elevenlabsService.buildTTSSavePath(item.text, outputFormat, 'batch', `${String(i + 1).padStart(3, '0')}`);
                    }
                    fs.writeFileSync(savePath, audio);
                    results.push({ index: i, success: true, file_path: savePath });
                } catch (e) {
                    results.push({ index: i, success: false, error: e.message });
                }
            }
            return { results, total: items.length, success_count: results.filter(r => r.success).length };
        }

        case 'elevenlabs/tts-workflow':
            return await workflowService.ttsWorkflow(data);

        case 'elevenlabs/sfx': {
            const keys = elevenlabsService.loadKeys();
            if (!keys || keys.length === 0) throw new Error('未配置 API Key');
            if (!data.prompt) throw new Error('缺少音效描述');
            const apiKey = elevenlabsService.selectKey(keys, data.key_index);
            const audio = await elevenlabsService.generateSFX(apiKey, data.prompt, data.duration || 5);
            let savePath = data.save_path;
            if (!savePath) {
                savePath = elevenlabsService.buildTTSSavePath(data.prompt, 'mp3_44100_128', 'sfx');
            }
            fs.writeFileSync(savePath, audio);
            return { message: '生成成功', file_path: savePath };
        }

        case 'elevenlabs/toggle-key': {
            // 必须加载包含已停用的 key，否则停用后找不到目标 key 无法重新启用
            const allKeys = elevenlabsService.loadKeys(true);
            if (!allKeys || allKeys.length === 0) throw new Error('未配置 API Key');
            const targetKey = data.api_key;
            if (!targetKey) throw new Error('缺少 api_key');
            elevenlabsService.setKeyEnabled(targetKey, data.enabled !== false, data.reason || '', data.source || 'manual');
            return { message: '已更新' };
        }

        // ==================== 字幕操作 ====================
        case 'srt/adjust':
            if (!data.src_path) throw new Error('缺少必需参数: src_path');
            return subtitleService.adjustSRT(data.src_path, {
                intervalTime: data.interval_time,
                charTime: data.char_time,
                minCharCount: data.min_char_count,
                scale: data.scale,
                ignore: data.ignore,
            });

        case 'srt/seamless':
            if (!data.src_path) throw new Error('缺少必需参数: src_path');
            return subtitleService.seamlessSRT(data.src_path);

        case 'srt/compute-char-time':
            if (!data.ref_path) throw new Error('缺少必需参数: ref_path');
            return subtitleService.computeCharTime(data.ref_path, data.interval_time);

        // ==================== 媒体操作 ====================
        case 'media/info': {
            if (!data.file_path) throw new Error('缺少文件路径');
            const [duration, frameRate, resolution] = await Promise.all([
                ffmpegService.getDuration(data.file_path),
                ffmpegService.getFrameRate(data.file_path),
                ffmpegService.getResolution(data.file_path)
            ]);
            return { duration, frame_rate: frameRate, resolution };
        }

        case 'media/waveform': {
            if (!data.file_path) throw new Error('缺少文件路径');
            return await ffmpegService.getWaveformBinary(data.file_path, data.num_peaks || 300);
        }

        case 'media/trim':
            if (!data.file_path) throw new Error('缺少文件路径');
            return await ffmpegService.mediaTrim(
                data.file_path,
                parseFloat(data.start_time ?? data.start),
                parseFloat(data.end_time ?? data.end),
                data.output_dir,
                data.precise !== false
            );

        case 'media/scene-detect':
            if (!data.file_path) throw new Error('缺少文件路径');
            return await ffmpegService.sceneDetect(
                data.file_path,
                parseFloat(data.threshold || 0.3),
                parseFloat(data.min_interval || 0.5)
            );

        case 'media/scene-split':
            if (!data.file_path || !data.segments) throw new Error('缺少参数');
            return await ffmpegService.sceneSplit(data.file_path, data.segments, data.output_dir);

        case 'media/batch-cut':
            if (!data.file_path) throw new Error('缺少文件路径');
            if (!data.segments || data.segments.length === 0) throw new Error('缺少剪辑片段');
            return await ffmpegService.batchCut(
                data.file_path,
                data.segments,
                data.output_dir,
                data.precise !== false
            );

        case 'media/export-fcpxml-timeline': {
            if (!data.file_path) throw new Error('缺少文件路径');
            if (!data.segments || data.segments.length === 0) throw new Error('缺少剪辑片段');
            const outDir = data.output_dir || path.dirname(data.file_path);
            const baseName = path.basename(data.file_path, path.extname(data.file_path));
            const fcpxmlPath = path.join(outDir, `${baseName}_timeline.fcpxml`);
            return fcpxmlService.segmentsToFcpxml(
                data.file_path,
                data.segments,
                data.duration || 0,
                data.fps || 30,
                data.resolution || '1920x1080',
                fcpxmlPath,
                data.subtitle_style || null
            );
        }




        case 'media/convert': {
            const files = data.files || [data.file_path];
            const mode = data.mode || 'mp3';
            const outDir = data.output_dir || path.dirname(files[0]);
            const allResults = [];

            for (const file of files) {
                try {
                    const results = await ffmpegService.mediaConvert(file, mode, outDir, data);
                    allResults.push(...results);
                } catch (e) {
                    allResults.push({ error: e.message, file });
                }
            }
            return { message: `转换完成: ${allResults.length} 个文件`, converted: allResults };
        }

        case 'media/batch-thumbnail': {
            if (!data.files || data.files.length === 0) throw new Error('缺少文件列表');
            const outDir = data.output_dir || path.join(os.tmpdir(), 'thumbnails');
            const results = await ffmpegService.batchThumbnail(
                data.files, outDir,
                data.format || 'jpg',
                data.quality || 2
            );
            return {
                message: `截图完成: ${results.filter(r => r.success).length}/${results.length}`,
                results,
                output_dir: outDir,
            };
        }

        case 'media/smart-split':
            if (!data.file_path) throw new Error('缺少文件路径');
            return await ffmpegService.smartSplitAnalyze(data.file_path, data.max_duration || 29);

        case 'media/image-classify':
            if (!data.input_dir) throw new Error('缺少输入目录');
            return await imageClassifyService.imageClassify(
                data.input_dir,
                data.output_dir || path.join(data.input_dir, '_classified'),
                {
                    threshold: data.threshold || 5,
                    moveMode: data.move_mode || false,
                }
            );

        // ==================== 字幕生成（Gladia + 完整对齐） ====================
        case 'subtitle/generate': {
            if (!data.audio_path && !data.audio_file_path) throw new Error('缺少音频文件路径');
            const audioPath = data.audio_path || data.audio_file_path;
            const gladiaKeysData = settingsService.loadGladiaKeys();
            const gladiaKeys = gladiaKeysData.keys || [];
            if (gladiaKeys.length === 0) throw new Error('未配置 Gladia API Key');

            // 确定语言
            const langInput = data.language || 'english';
            let currentLanguage = langInput;
            // 如果传入中文名称，转为语言代码
            for (const [code, info] of Object.entries(subtitleUtils.LANGUAGES)) {
                if (info.name === langInput) { currentLanguage = code; break; }
            }
            const langEnName = subtitleUtils.getLanguage(currentLanguage);

            // 构建日志路径
            const fileName = path.parse(audioPath).name;
            const logDir = path.join(os.tmpdir(), 'pymediatools_log');
            fs.mkdirSync(logDir, { recursive: true });
            const jsonPath = path.join(logDir, `${currentLanguage}_${fileName}_audio_text_whittime.json`);
            const txtPath = path.join(logDir, `${currentLanguage}_${fileName}_finally.txt`);

            // JSON 文件输入
            let generationSubtitleArray;
            let generationSubtitleText;

            if (audioPath.toLowerCase().endsWith('.json')) {
                // 直接读取 JSON 结果
                const audioJson = JSON.parse(fs.readFileSync(audioPath, 'utf-8'));
                const transcription = audioJson.result?.transcription || audioJson.transcription || {};
                const wordTimeInfo = transcription.utterances || [];
                let allText = '';
                generationSubtitleArray = [];
                for (const single of wordTimeInfo) {
                    const newSingle = { audio_start: single.start, audio_end: single.end, text: single.text, words: [] };
                    for (const word of (single.words || [])) {
                        allText += ' ' + word.word.trim();
                        newSingle.words.push({ word: word.word.trim(), start: word.start, end: word.end, score: word.confidence || 0 });
                    }
                    generationSubtitleArray.push(newSingle);
                }
                generationSubtitleText = allText.trimStart();
                fs.writeFileSync(jsonPath, JSON.stringify(generationSubtitleArray, null, 4), 'utf-8');
                fs.writeFileSync(txtPath, generationSubtitleText, 'utf-8');
            } else if (!fs.existsSync(jsonPath)) {
                // Gladia 转录
                const cutLength = parseFloat(data.audio_cut_length || 5.0);
                const result = await gladiaService.transcribeAudioFull(
                    audioPath, gladiaKeys, langEnName, jsonPath, txtPath, cutLength
                );
                generationSubtitleArray = result.wordTimeInfo;
                generationSubtitleText = result.fullText;
            } else {
                generationSubtitleArray = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
                generationSubtitleText = fs.readFileSync(txtPath, 'utf-8').trim();
            }

            // 如果提供了原文本，执行完整对齐
            if (data.source_text) {
                const sourceTextPath = path.join(os.tmpdir(), `source_${Date.now()}.txt`);
                fs.writeFileSync(sourceTextPath, data.source_text, 'utf-8');

                const translateTextDict = {};
                if (data.translate_text) {
                    const translatePath = path.join(os.tmpdir(), `translate_${Date.now()}.txt`);
                    fs.writeFileSync(translatePath, data.translate_text, 'utf-8');
                    translateTextDict['翻译文本'] = {
                        filename: '翻译文本',
                        filepath: translatePath,
                        translate_text_with_info: subtitleUtils.readTextWithGoogleDoc(translatePath),
                        trans_srt: '',
                    };
                }

                const sourceTextWithInfo = subtitleUtils.readTextWithGoogleDoc(sourceTextPath);

                const dateStr = new Date().toISOString().replace(/[T:.]/g, '').slice(0, 15);
                const outputDir = data.output_dir || path.join(os.homedir(), 'Desktop', `字幕输出_${dateStr}`);
                fs.mkdirSync(outputDir, { recursive: true });

                const genMergeSrt = data.gen_merge_srt === true || data.gen_merge_srt === 'true';
                const sourceUpOrder = data.source_up_order === true || data.source_up_order === 'true';
                const exportFcpxml = data.export_fcpxml === true || data.export_fcpxml === 'true';
                const seamlessFcpxml = data.seamless_fcpxml === true || data.seamless_fcpxml === 'true';

                const alignResult = audioSubtitleSearchDifferentStrong(
                    currentLanguage, outputDir, fileName,
                    generationSubtitleArray, generationSubtitleText,
                    sourceTextWithInfo, translateTextDict,
                    genMergeSrt, sourceUpOrder, exportFcpxml, seamlessFcpxml
                );

                // 收集生成的文件
                const generatedFiles = [];
                const sourceSrt = path.join(outputDir, `${fileName}_${currentLanguage}_source.srt`);
                if (fs.existsSync(sourceSrt)) generatedFiles.push(sourceSrt);
                for (const k of Object.keys(translateTextDict)) {
                    const transSrt = path.join(outputDir, `${fileName}_${currentLanguage}_${k.replace('.txt', '')}_translate.srt`);
                    if (fs.existsSync(transSrt)) generatedFiles.push(transSrt);
                }
                const mergeSrt = path.join(outputDir, `${fileName}_${currentLanguage}_merge.srt`);
                if (fs.existsSync(mergeSrt)) generatedFiles.push(mergeSrt);
                const fcpxmlFile = path.join(outputDir, `${fileName}_${currentLanguage}.fcpxml`);
                if (fs.existsSync(fcpxmlFile)) generatedFiles.push(fcpxmlFile);

                // 清理
                try { fs.unlinkSync(sourceTextPath); } catch { }

                return {
                    message: '处理完成',
                    result: alignResult,
                    files: generatedFiles,
                    output_dir: outputDir,
                };
            }

            return {
                message: '转录完成',
                word_time_info: generationSubtitleArray,
                full_text: generationSubtitleText,
            };
        }

        case 'subtitle/generate-with-file':
            // 文件上传版本在 routeUpload 中处理
            throw new Error('文件上传请使用 api-upload 通道');

        // ==================== 视频下载 ====================
        case 'video/analyze':
            if (!data.url) throw new Error('缺少视频链接');
            return await ytdlpService.analyzeVideo(data.url);

        case 'video/download':
            if (!data.url) throw new Error('缺少视频链接');
            return await ytdlpService.downloadVideo(data.url, {
                quality: data.quality,
                outputDir: data.output_dir,
                downloadSubtitle: data.download_subtitle,
            });

        case 'video/download-batch':
            if (!data.items || data.items.length === 0) throw new Error('没有要下载的视频');
            return await ytdlpService.downloadBatch(data.items, {
                outputDir: data.output_dir || undefined,
                audioOnly: data.options?.audio_only,
                ext: data.options?.ext,
                quality: data.options?.quality,
                subtitles: data.options?.subtitles,
                subLang: data.options?.sub_lang,
            });

        // ==================== 别名兼容 ====================
        case 'settings/elevenlabs/keys':
            if (data._method === 'GET') return { keys: settingsService.loadElevenLabsKeysWithStatus() };
            settingsService.saveElevenLabsKeysWithStatus(data.keys || data);
            return { message: '保存成功' };

        case 'audio/smart-split-analyze':
            if (!data.file_path) throw new Error('缺少文件路径');
            return await ffmpegService.smartSplitAnalyze(data.file_path, data.max_duration || 29);

        case 'file/upload':
            // 简易文件上传（非 FormData 方式，直接传路径）
            return { message: '请使用 api-upload 通道上传文件' };

        case 'file/open-folder':
            return await settingsService.openFolder(data.path || data.folder_path);

        case 'file/download-zip':
        case 'subtitle/download-zip': {
            if (!data.files || data.files.length === 0) throw new Error('缺少文件列表');
            const zipPath = path.join(os.tmpdir(), `download_${Date.now()}.zip`);
            await settingsService.createZip(data.files, zipPath);
            return { message: '打包完成', zip_path: zipPath };
        }

        case 'media/batch-thumbnail-progress':
            // 进度查询（TODO: 如需实时进度可通过 WebSocket 或 IPC 事件）
            return { status: 'completed', progress: 100 };

        case 'status':
            return { status: 'ok', backend: 'nodejs', uptime: process.uptime() };

        // ==================== Wav2Lip 口型同步 ====================
        case 'wav2lip/check':
            return await wav2lipService.checkEnvironment();

        case 'wav2lip/run': {
            if (!data.face_path) throw new Error('缺少视频/图片路径');
            if (!data.audio_path) throw new Error('缺少音频路径');

            // 获取窗口以发送进度事件
            const { BrowserWindow } = require('electron');
            const wins = BrowserWindow.getAllWindows();

            return await wav2lipService.lipSync({
                facePath: data.face_path,
                audioPath: data.audio_path,
                outputPath: data.output_path || '',
                pads: data.pads || [0, 10, 0, 0],
                resizeFactor: parseInt(data.resize_factor) || 1,
                batchSize: parseInt(data.batch_size) || 32,
                onProgress: (percent, message) => {
                    // 通过 IPC 事件发送进度到渲染进程
                    for (const win of wins) {
                        try {
                            win.webContents.send('wav2lip-progress', { percent, message });
                        } catch { /* window closed */ }
                    }
                },
            });
        }

        default:
            throw new Error(`未知接口: ${ep}`);
    }
}

/**
 * 文件上传路由
 */
async function routeUpload(endpoint, fileBuffer, fileName, formData) {
    const ep = endpoint.replace(/^\/?(api\/)?/, '');

    switch (ep) {
        case 'upload-image':
        case 'upload': {
            const result = settingsService.uploadFile(Buffer.from(fileBuffer), fileName);
            return result;
        }

        case 'subtitle/generate-with-file': {
            // 保存临时文件
            const tempDir = os.tmpdir();
            const tempPath = path.join(tempDir, `batch_${Date.now()}_${fileName}`);
            fs.writeFileSync(tempPath, Buffer.from(fileBuffer));

            try {
                const gladiaKeysData = settingsService.loadGladiaKeys();
                let gladiaKeys = gladiaKeysData.keys || [];
                if (formData.gladia_keys) {
                    try { gladiaKeys = JSON.parse(formData.gladia_keys); } catch { }
                }

                const sourceText = formData.source_text || '';
                const translateText = formData.translate_text || '';
                const langInput = formData.language || 'en';

                if (!sourceText) throw new Error('缺少原文本');

                // 确定语言
                let currentLanguage = langInput;
                for (const [code, info] of Object.entries(subtitleUtils.LANGUAGES)) {
                    if (info.name === langInput || info.code === langInput) { currentLanguage = code; break; }
                }
                const langEnName = subtitleUtils.getLanguage(currentLanguage);

                // 构建日志路径
                const baseName = path.parse(fileName).name;
                const logDir = path.join(os.tmpdir(), 'pymediatools_log');
                fs.mkdirSync(logDir, { recursive: true });
                const jsonPath = path.join(logDir, `${currentLanguage}_${baseName}_audio_text_whittime.json`);
                const txtPath = path.join(logDir, `${currentLanguage}_${baseName}_finally.txt`);

                // Gladia 转录
                const cutLength = parseFloat(formData.audio_cut_length || 5.0);
                const result = await gladiaService.transcribeAudioFull(
                    tempPath, gladiaKeys, langEnName, jsonPath, txtPath, cutLength
                );

                // 写入原文本到临时文件
                const sourceTextPath = path.join(os.tmpdir(), `source_upload_${Date.now()}.txt`);
                fs.writeFileSync(sourceTextPath, sourceText, 'utf-8');
                const sourceTextWithInfo = subtitleUtils.readTextWithGoogleDoc(sourceTextPath);

                // 翻译文本
                const translateTextDict = {};
                if (translateText) {
                    const translatePath = path.join(os.tmpdir(), `translate_upload_${Date.now()}.txt`);
                    fs.writeFileSync(translatePath, translateText, 'utf-8');
                    translateTextDict['翻译文本'] = {
                        filename: '翻译文本',
                        filepath: translatePath,
                        translate_text_with_info: subtitleUtils.readTextWithGoogleDoc(translatePath),
                        trans_srt: '',
                    };
                }

                // 输出目录
                const dateStr = new Date().toISOString().replace(/[T:.]/g, '').slice(0, 15);
                const outputDir = formData.output_dir || path.join(os.homedir(), 'Desktop', `字幕输出_${dateStr}`);
                fs.mkdirSync(outputDir, { recursive: true });

                // 选项
                const genMergeSrt = formData.gen_merge_srt === 'true' || formData.gen_merge_srt === true;
                const sourceUpOrder = formData.source_up_order === 'true' || formData.source_up_order === true;
                const exportFcpxml = formData.export_fcpxml === 'true' || formData.export_fcpxml === true;
                const seamlessFcpxml = formData.seamless_fcpxml === 'true' || formData.seamless_fcpxml === true;

                // 完整对齐
                const alignResult = audioSubtitleSearchDifferentStrong(
                    currentLanguage, outputDir, baseName,
                    result.wordTimeInfo, result.fullText,
                    sourceTextWithInfo, translateTextDict,
                    genMergeSrt, sourceUpOrder, exportFcpxml, seamlessFcpxml
                );

                // 收集生成的文件
                const generatedFiles = [];
                const sourceSrt = path.join(outputDir, `${baseName}_${currentLanguage}_source.srt`);
                if (fs.existsSync(sourceSrt)) generatedFiles.push(sourceSrt);
                for (const k of Object.keys(translateTextDict)) {
                    const transSrt = path.join(outputDir, `${baseName}_${currentLanguage}_${k.replace('.txt', '')}_translate.srt`);
                    if (fs.existsSync(transSrt)) generatedFiles.push(transSrt);
                }
                const mergeSrt = path.join(outputDir, `${baseName}_${currentLanguage}_merge.srt`);
                if (fs.existsSync(mergeSrt)) generatedFiles.push(mergeSrt);
                const fcpxmlFile = path.join(outputDir, `${baseName}_${currentLanguage}.fcpxml`);
                if (fs.existsSync(fcpxmlFile)) generatedFiles.push(fcpxmlFile);

                // 清理临时文件
                try { fs.unlinkSync(tempPath); } catch { }
                try { fs.unlinkSync(sourceTextPath); } catch { }

                return {
                    message: '处理完成',
                    result: alignResult,
                    files: generatedFiles,
                    output_dir: outputDir,
                };
            } catch (e) {
                try { fs.unlinkSync(tempPath); } catch { }
                throw e;
            }
        }

        default:
            throw new Error(`未知上传接口: ${ep}`);
    }
}

module.exports = { registerAPIHandlers };
