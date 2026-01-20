# 🌍 pyMediaTools 跨平台打包完整指南

## 目录
1. [架构概述](#架构概述)
2. [平台特定依赖准备](#平台特定依赖准备)
3. [打包命令](#打包命令)
4. [main.js 跨平台检测逻辑](#mainjs-跨平台检测逻辑)
5. [常见问题](#常见问题)

---

## 架构概述

```
┌─────────────────────────────────────────────────────────────────┐
│                         pyMediaTools                             │
├─────────────────────────────────────────────────────────────────┤
│  Electron Shell (跨平台 UI)                                      │
│  ├── main.js (智能平台检测)                                       │
│  └── preload.js (安全桥接)                                       │
├─────────────────────────────────────────────────────────────────┤
│  Vite Frontend (静态资源)                                         │
│  └── dist/ (HTML/CSS/JS)                                         │
├─────────────────────────────────────────────────────────────────┤
│  Python Backend (Flask API)                                      │
│  └── backend/server.py                                           │
├─────────────────────────────────────────────────────────────────┤
│  Vendor Dependencies (平台特定)                                   │
│  ├── vendor/windows/ → Python Embedded + FFmpeg                  │
│  ├── vendor/darwin/  → FFmpeg (macOS 用系统 Python)              │
│  └── vendor/linux/   → Python Portable + FFmpeg                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 平台特定依赖准备

### 🪟 Windows (x64 + ARM64)

```bash
# 目录结构
vendor/windows/
├── python/                    # Python Embedded 3.11+
│   ├── python.exe
│   ├── python311.dll
│   └── Lib/site-packages/    # 预装依赖
│       ├── flask/
│       ├── pydub/
│       ├── requests/
│       └── ...
└── ffmpeg/
    └── bin/
        ├── ffmpeg.exe
        └── ffprobe.exe
```

**下载地址：**
- Python Embedded: https://www.python.org/downloads/windows/ (选择 "Windows embeddable package")
- FFmpeg: https://github.com/BtbN/FFmpeg-Builds/releases (ffmpeg-master-latest-win64-gpl.zip)

**安装依赖到 Embedded Python：**
```cmd
cd vendor/windows/python
python.exe -m pip install flask pydub requests --target Lib/site-packages
```

> ⚠️ **注意**：Python Embedded 默认禁用 pip，需要先下载 get-pip.py 并安装。

---

### 🍎 macOS (Apple Silicon M1/M2/M3 + Intel)

**推荐方案**：使用系统 Python3（macOS 自带），只需打包 FFmpeg。

```bash
# 目录结构
vendor/darwin/
└── ffmpeg/
    └── bin/
        ├── ffmpeg         # Universal Binary (arm64 + x64)
        └── ffprobe
```

**下载地址：**
- FFmpeg macOS: https://evermeet.cx/ffmpeg/ (选择 Static builds)
- 或使用 Homebrew: `brew install ffmpeg`

**创建 Universal Binary (可选)：**
```bash
# 如果有两个架构的 ffmpeg
lipo -create ffmpeg-arm64 ffmpeg-x64 -output ffmpeg
```

**修改 Python 依赖检测**：macOS 应使用系统 Python 或 pyenv，不需要嵌入。

---

### 🐧 Linux (x64 + ARM64)

```bash
# 目录结构  
vendor/linux/
├── python/                    # Python Portable (可选)
│   └── bin/python3
└── ffmpeg/
    └── bin/
        ├── ffmpeg
        └── ffprobe
```

**下载地址：**
- Python Portable: https://github.com/indygreg/python-build-standalone/releases
- FFmpeg Static: https://johnvansickle.com/ffmpeg/

**两种策略：**

1. **依赖系统 Python（推荐）**：
   - 在安装说明中要求用户安装 `python3` 和依赖
   - AppImage 启动时检测系统 Python

2. **完全自包含**：
   - 使用 python-build-standalone
   - 打包后体积较大 (~200MB+)

---

## 打包命令

### 单平台打包

```bash
# macOS Apple Silicon
npm run pack:mac-arm64

# macOS Intel
npm run pack:mac-x64

# macOS Universal (同时支持 M1 和 Intel)
npm run pack:mac-universal

# Windows x64
npm run pack:win-x64

# Windows ARM64
npm run pack:win-arm64

# Linux x64
npm run pack:linux-x64

# Linux ARM64
npm run pack:linux-arm64
```

### 批量打包

```bash
# 所有 macOS 版本
npm run pack:all-mac

# 所有 Windows 版本
npm run pack:all-win

# 所有 Linux 版本  
npm run pack:all-linux

# 全平台（需要在 macOS 上运行）
npm run pack:all
```

### 输出位置

```
dist-electron/
├── pyMediaTools-2.1.0-arm64.dmg      # macOS ARM64
├── pyMediaTools-2.1.0.dmg            # macOS x64
├── pyMediaTools-2.1.0-arm64-mac.zip
├── pyMediaTools-2.1.0-mac.zip
├── pyMediaTools-2.1.0-win.zip        # Windows x64
├── pyMediaTools-2.1.0-arm64-win.zip  # Windows ARM64
├── pyMediaTools-2.1.0.AppImage       # Linux x64
└── pyMediaTools-2.1.0-arm64.AppImage # Linux ARM64
```

---

## main.js 跨平台检测逻辑

将 `electron/main.js` 中的 Python/FFmpeg 路径检测改为：

```javascript
// 获取 vendor 路径 - 根据平台选择
function getVendorPath() {
    const platform = process.platform; // 'darwin', 'win32', 'linux'
    const platformMap = {
        'darwin': 'darwin',
        'win32': 'windows', 
        'linux': 'linux'
    };
    return path.join(getResourcePath('vendor'), platformMap[platform] || platform);
}

// 获取 Python 路径 - 跨平台
function getPythonPath() {
    const platform = process.platform;
    
    if (app.isPackaged) {
        if (platform === 'win32') {
            // Windows: 使用嵌入式 Python
            const vendorPython = path.join(getVendorPath(), 'python', 'python.exe');
            if (fs.existsSync(vendorPython)) return vendorPython;
        } else if (platform === 'linux') {
            // Linux: 可选使用打包的 Python
            const vendorPython = path.join(getVendorPath(), 'python', 'bin', 'python3');
            if (fs.existsSync(vendorPython)) return vendorPython;
        }
        // macOS 和 Linux fallback: 使用系统 Python
    }
    
    // 开发模式或 fallback
    return platform === 'win32' ? 'python' : 'python3';
}

// 获取 FFmpeg 路径 - 跨平台
function getFfmpegBinPath() {
    const vendorFfmpeg = path.join(getVendorPath(), 'ffmpeg', 'bin');
    if (fs.existsSync(vendorFfmpeg)) {
        return vendorFfmpeg;
    }
    return null; // 使用系统 FFmpeg
}

// 设置环境变量 - 跨平台
function getEnv() {
    const env = { ...process.env };
    const platform = process.platform;
    const ffmpegPath = getFfmpegBinPath();
    
    if (ffmpegPath) {
        env.PATH = `${ffmpegPath}${path.delimiter}${env.PATH || ''}`;
        
        const ext = platform === 'win32' ? '.exe' : '';
        env.FFMPEG_PATH = path.join(ffmpegPath, `ffmpeg${ext}`);
        env.FFPROBE_PATH = path.join(ffmpegPath, `ffprobe${ext}`);
    }
    
    return env;
}
```

---

## 常见问题

### Q1: 跨平台打包必须在目标平台上进行吗？

**部分是的：**
- ✅ **Windows → Windows**: 可以在 Windows 上打包
- ✅ **macOS → macOS/Windows/Linux**: macOS 可以打包所有平台（推荐）
- ❌ **Windows → macOS**: 无法打包 macOS（需要 Xcode 命令行工具）
- ⚠️ **Linux → Linux/Windows**: 可以，但不能打包 macOS

**建议**：使用 **macOS** 作为构建机器，或使用 **GitHub Actions** 进行 CI/CD 跨平台构建。

---

### Q2: Python 依赖如何管理？

**方法 1：预安装到 Embedded Python（Windows）**
```bash
# 在 Windows 机器上
cd vendor/windows/python
./python.exe -m pip install -r ../../backend/requirements.txt --target Lib/site-packages
```

**方法 2：首次启动时安装（需要网络）**
```javascript
// 在 main.js 中检测依赖
function checkPythonDependencies() {
    // 启动时运行 pip check 或导入测试
}
```

**方法 3：使用 PyInstaller 编译后端为独立二进制**
```bash
pyinstaller --onefile backend/server.py
# 然后在 Electron 中启动编译后的 server.exe / server
```

---

### Q3: 如何处理 macOS 签名和公证？

```bash
# 在 package.json 中添加
"mac": {
  "hardenedRuntime": true,
  "gatekeeperAssess": false,
  "entitlements": "build/entitlements.mac.plist",
  "entitlementsInherit": "build/entitlements.mac.plist"
}
```

创建 `build/entitlements.mac.plist`：
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>com.apple.security.cs.allow-jit</key>
  <true/>
  <key>com.apple.security.cs.allow-unsigned-executable-memory</key>
  <true/>
</dict>
</plist>
```

---

### Q4: GitHub Actions 自动化跨平台构建

创建 `.github/workflows/build.yml`：

```yaml
name: Build All Platforms

on:
  push:
    tags:
      - 'v*'

jobs:
  build-mac:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
      - run: npm ci
      - run: npm run pack:all-mac
      - uses: actions/upload-artifact@v4
        with:
          name: mac-builds
          path: dist-electron/*.dmg

  build-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
      - run: npm ci
      - run: npm run pack:win-x64
      - uses: actions/upload-artifact@v4
        with:
          name: windows-builds
          path: dist-electron/*.zip

  build-linux:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
      - run: npm ci
      - run: npm run pack:linux-x64
      - uses: actions/upload-artifact@v4
        with:
          name: linux-builds
          path: dist-electron/*.AppImage
```

---

## 总结清单

| 平台 | Python 来源 | FFmpeg 来源 | 打包格式 |
|------|-------------|-------------|----------|
| Windows x64 | Embedded 3.11 | vendor/windows/ffmpeg | ZIP |
| Windows ARM64 | Embedded 3.11-arm64 | vendor/windows/ffmpeg-arm64 | ZIP |
| macOS Intel | 系统 python3 | vendor/darwin/ffmpeg | DMG/ZIP |
| macOS M1/M2/M3 | 系统 python3 | vendor/darwin/ffmpeg | DMG/ZIP |
| Linux x64 | 系统 python3 | vendor/linux/ffmpeg | AppImage |
| Linux ARM64 | 系统 python3 | vendor/linux/ffmpeg-arm64 | AppImage |

---

**提示**：对于生产环境，强烈建议使用 **GitHub Actions** 或 **Azure DevOps** 进行自动化跨平台构建，确保每个平台都在原生环境中编译。
