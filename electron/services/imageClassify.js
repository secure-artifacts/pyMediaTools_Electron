/**
 * 图片/视频画面分类服务（感知哈希 dHash）
 * 替代 Python 的 _compute_dhash / _cluster_by_hash / image_classify
 * 使用 FFmpeg 提取首帧 + 简化的 dHash 实现
 */
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

// ==================== dHash 计算 ====================

/**
 * 用 FFmpeg 将图片/视频首帧缩放为 (hashSize+1) x hashSize 的灰度像素
 * 返回灰度像素数组
 */
function extractGrayPixels(filePath, hashSize = 8) {
    const w = hashSize + 1;
    const h = hashSize;
    return new Promise((resolve, reject) => {
        const ffmpegCmd = process.env.FFMPEG_PATH || 'ffmpeg';
        const proc = spawn(ffmpegCmd, [
            '-hide_banner', '-loglevel', 'error',
            '-i', filePath,
            '-vframes', '1',
            '-vf', `scale=${w}:${h},format=gray`,
            '-f', 'rawvideo', '-pix_fmt', 'gray',
            'pipe:1'
        ], { timeout: 30000 });

        const chunks = [];
        proc.stdout.on('data', c => chunks.push(c));
        proc.stderr.on('data', () => { });
        proc.on('close', code => {
            if (code !== 0) return reject(new Error(`FFmpeg 提取像素失败: ${filePath}`));
            const buf = Buffer.concat(chunks);
            if (buf.length < w * h) return reject(new Error(`像素数据不足: ${filePath}`));
            resolve(buf);
        });
        proc.on('error', reject);
    });
}

/**
 * 计算 dHash
 * @returns BigInt 类型的哈希值
 */
async function computeDHash(filePath, hashSize = 8) {
    const w = hashSize + 1;
    const h = hashSize;
    const pixels = await extractGrayPixels(filePath, hashSize);

    let hash = BigInt(0);
    for (let row = 0; row < h; row++) {
        for (let col = 0; col < hashSize; col++) {
            const left = pixels[row * w + col];
            const right = pixels[row * w + col + 1];
            hash = (hash << BigInt(1)) | (left > right ? BigInt(1) : BigInt(0));
        }
    }
    return hash;
}

/**
 * 计算汉明距离
 */
function hammingDistance(h1, h2) {
    let xor = h1 ^ h2;
    let dist = 0;
    while (xor > BigInt(0)) {
        dist += Number(xor & BigInt(1));
        xor >>= BigInt(1);
    }
    return dist;
}

// ==================== Union-Find 聚类 ====================

function clusterByHash(hashes, threshold) {
    const n = hashes.length;
    const parent = Array.from({ length: n }, (_, i) => i);
    const rank = new Array(n).fill(0);

    function find(x) {
        if (parent[x] !== x) parent[x] = find(parent[x]);
        return parent[x];
    }

    function union(a, b) {
        const ra = find(a), rb = find(b);
        if (ra === rb) return;
        if (rank[ra] < rank[rb]) parent[ra] = rb;
        else if (rank[ra] > rank[rb]) parent[rb] = ra;
        else { parent[rb] = ra; rank[ra]++; }
    }

    // O(n^2) 比较
    for (let i = 0; i < n; i++) {
        if (hashes[i] === null) continue;
        for (let j = i + 1; j < n; j++) {
            if (hashes[j] === null) continue;
            if (hammingDistance(hashes[i].hash, hashes[j].hash) <= threshold) {
                union(i, j);
            }
        }
    }

    // 收集聚类结果
    const clusters = {};
    for (let i = 0; i < n; i++) {
        const root = find(i);
        if (!clusters[root]) clusters[root] = [];
        clusters[root].push(i);
    }

    return Object.values(clusters);
}

// ==================== 主分类函数 ====================

const IMAGE_EXTS = new Set(['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tiff', '.tif']);
const VIDEO_EXTS = new Set(['.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv', '.m4v', '.mpg', '.mpeg', '.3gp']);

/**
 * 执行图片/视频画面分类
 */
async function imageClassify(inputDir, outputDir, options = {}) {
    const {
        threshold = 5,      // 汉明距离阈值
        moveMode = false,    // true = 移动, false = 复制
        onProgress = null,
    } = options;

    // 扫描文件
    const allFiles = [];
    function scanDir(dir) {
        for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
            if (entry.isDirectory()) {
                scanDir(path.join(dir, entry.name));
            } else {
                const ext = path.extname(entry.name).toLowerCase();
                if (IMAGE_EXTS.has(ext) || VIDEO_EXTS.has(ext)) {
                    allFiles.push(path.join(dir, entry.name));
                }
            }
        }
    }
    scanDir(inputDir);

    if (allFiles.length === 0) throw new Error('未找到图片或视频文件');
    const total = allFiles.length;

    // 计算 dHash
    const hashes = [];
    const errors = [];
    for (let i = 0; i < total; i++) {
        if (onProgress) onProgress({ step: 'hash', current: i + 1, total });
        try {
            const hash = await computeDHash(allFiles[i]);
            hashes.push({ index: i, hash, path: allFiles[i] });
        } catch (e) {
            hashes.push({ index: i, hash: null, path: allFiles[i] });
            errors.push({ path: allFiles[i], error: e.message });
        }
    }

    // 聚类
    if (onProgress) onProgress({ step: 'cluster', current: 0, total: 1 });
    const clusters = clusterByHash(hashes.map(h => h.hash !== null ? h : null), threshold);

    // 创建输出目录
    fs.mkdirSync(outputDir, { recursive: true });

    // 导出分组
    const groupResults = [];
    let groupIndex = 0;

    // 排序：大组优先
    clusters.sort((a, b) => b.length - a.length);

    for (const cluster of clusters) {
        groupIndex++;
        const count = cluster.length;

        let groupDir;
        if (count >= 2) {
            groupDir = path.join(outputDir, `group_${String(groupIndex).padStart(4, '0')}_${count}files`);
        } else {
            groupDir = path.join(outputDir, '_others');
        }
        fs.mkdirSync(groupDir, { recursive: true });

        const files = [];
        for (const idx of cluster) {
            const srcPath = hashes[idx].path;
            const destPath = path.join(groupDir, path.basename(srcPath));

            // 处理重名
            let finalDest = destPath;
            let counter = 1;
            while (fs.existsSync(finalDest)) {
                const parsed = path.parse(destPath);
                finalDest = path.join(parsed.dir, `${parsed.name}_${counter}${parsed.ext}`);
                counter++;
            }

            try {
                if (moveMode) {
                    fs.renameSync(srcPath, finalDest);
                } else {
                    fs.copyFileSync(srcPath, finalDest);
                }
                files.push(finalDest);
            } catch (e) {
                errors.push({ path: srcPath, error: e.message });
            }
        }

        groupResults.push({
            group: groupIndex,
            count,
            directory: groupDir,
            files,
        });
    }

    const largeGroups = groupResults.filter(g => g.count >= 2);
    const singleCount = clusters.filter(c => c.length === 1).length;

    return {
        message: `分类完成: ${total} 个文件 → ${groupResults.length} 个分组`,
        output_dir: outputDir,
        total_files: total,
        total_groups: groupResults.length,
        large_groups: largeGroups.length,
        single_files: singleCount,
        groups: groupResults,
        errors,
    };
}

module.exports = {
    computeDHash,
    hammingDistance,
    clusterByHash,
    imageClassify,
};
